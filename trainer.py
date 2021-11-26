import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder
from decoding_utils import greedy_decoding


def evaluate(subset, encoder, decoder, batch_size, max_len, device):
    """Evaluates (BLEU score) caption generation model on a given subset.

    Arguments:
        subset (Flickr8KDataset): Train/Val/Test subset
        encoder (nn.Module): CNN which generates image features
        decoder (nn.Module): Transformer Decoder which generates captions for images
        batch_size (int): Number of elements in each mini-batch
        max_len (int): Maximum length of generated captions
        device (torch.device): Device on which to port used tensors
    Returns:
        bleu (float): BLEU-4 score performance metric on the entire subset - corpus bleu
    """
    # Mapping from vocab index to string representation
    idx2word = subset._idx2word
    # Ids for special tokens
    sos_id = subset._start_idx
    eos_id = subset._end_idx
    pad_id = subset._pad_idx

    references_total = []
    predictions_total = []

    for x_img, y_caption in subset.inference_batch(batch_size):
        x_img = x_img.to(device)
        # Extract image features
        img_features = encoder(x_img)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        img_features = img_features.detach()

        # Get the caption prediction for each image in the mini-batch
        predictions = greedy_decoding(decoder, img_features, sos_id, eos_id, pad_id, idx2word, max_len, device)
        references_total += y_caption
        predictions_total += predictions

    bleu = corpus_bleu(references_total, predictions_total)
    return bleu


def train(config, writer, device):
    """Performs the training of the model.

    Arguments:
        config (object): Contains configuration of the pipeline
        writer: tensorboardX writer object
        device: device on which to map the model and data
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Define dataloader hyper-parameters
    train_hyperparams = {
        "batch_size": config["batch_size"]["train"],
        "shuffle": True,
        "drop_last": True
    }

    # Create dataloaders
    train_set = Flickr8KDataset(config, config["split_save"]["train"], training=True)
    valid_set = Flickr8KDataset(config, config["split_save"]["validation"], training=False)
    train_loader = DataLoader(train_set, **train_hyperparams)

    # Download pretrained CNN encoder
    encoder = models.resnet50(pretrained=True)
    # Extract only the convolutional backbone of the model
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    encoder = encoder.to(device)
    # Freeze encoder layers
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # Instantiate the decoder
    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)

    # Load training configuration
    train_config = config["train_config"]
    learning_rate = train_config["learning_rate"]

    # Prepare the model optimizer
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["l2_penalty"]
    )

    # Loss function
    loss_fcn = nn.CrossEntropyLoss()

    train_step = 0
    for epoch in range(train_config["num_of_epochs"]):
        print("Epoch:", epoch)
        decoder.train()

        for x_img, x_words, y, padding_mask, tgt_pos in train_loader:
            train_step += 1

            # Move the used tensors to defined device
            x_img, x_words = x_img.to(device), x_words.to(device)
            y, tgt_pos = y.to(device), tgt_pos.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()

            # Extract image features
            img_features = encoder(x_img)
            img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.detach()

            # Get the output of the decoder
            y_pred = decoder(x_words, img_features, padding_mask)
            # Extract the prediction of the target token
            tgt_pos = tgt_pos.view(-1)
            y_pred = y_pred[torch.arange(y.size(0)), tgt_pos]

            # Calculate the loss
            y = y.view(-1)
            loss = loss_fcn(y_pred, y)

            # Update model weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), train_config["gradient_clipping"])
            optimizer.step()

            writer.add_scalar("Train/Step-Loss", loss.item(), train_step)
            writer.add_scalar("Train/Learning-Rate", learning_rate, train_step)

        with torch.no_grad():
            encoder.eval()
            decoder.eval()

            train_bleu = evaluate(train_set, encoder, decoder, config["batch_size"]["eval"], config["max_len"], device)
            valid_bleu = evaluate(valid_set, encoder, decoder, config["batch_size"]["eval"], config["max_len"], device)
            writer.add_scalar("Train/BLEU-4", train_bleu, epoch)
            writer.add_scalar("Valid/BLEU-4", valid_bleu, epoch)

            decoder.train()
