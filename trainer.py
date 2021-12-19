import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder
from utils.decoding_utils import greedy_decoding
from utils.utils import save_checkpoint, log_gradient_norm, set_up_causal_mask


def evaluate(subset, encoder, decoder, config, device):
    """Evaluates (BLEU score) caption generation model on a given subset.

    Arguments:
        subset (Flickr8KDataset): Train/Val/Test subset
        encoder (nn.Module): CNN which generates image features
        decoder (nn.Module): Transformer Decoder which generates captions for images
        config (object): Contains configuration for the evaluation pipeline
        device (torch.device): Device on which to port used tensors
    Returns:
        bleu (float): BLEU-{1:4} scores performance metric on the entire subset - corpus bleu
    """
    batch_size = config["batch_size"]["eval"]
    max_len = config["max_len"]
    bleu_w = config["bleu_weights"]

    # Mapping from vocab index to string representation
    idx2word = subset._idx2word
    # Ids for special tokens
    sos_id = subset._start_idx
    eos_id = subset._end_idx
    pad_id = subset._pad_idx

    references_total = []
    predictions_total = []

    print("Evaluating model.")
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

    # Evaluate BLEU score of the generated captions
    bleu_1 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-1"]) * 100
    bleu_2 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-2"]) * 100
    bleu_3 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-3"]) * 100
    bleu_4 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-4"]) * 100
    bleu = [bleu_1, bleu_2, bleu_3, bleu_4]
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
        "num_workers": 1,
        "drop_last": True
    }

    # Create dataloaders
    train_set = Flickr8KDataset(config, config["split_save"]["train"], training=True)
    valid_set = Flickr8KDataset(config, config["split_save"]["validation"], training=False)
    train_loader = DataLoader(train_set, **train_hyperparams)

    #######################
    # Set up the encoder 
    #######################
    # Download pretrained CNN encoder
    encoder = models.resnet50(pretrained=True)
    # Extract only the convolutional backbone of the model
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    encoder = encoder.to(device)
    # Freeze encoder layers
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    ######################
    # Set up the decoder
    ######################
    # Instantiate the decoder
    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)

    if config["checkpoint"]["load"]:
        checkpoint_path = config["checkpoint"]["path"]
        decoder.load_state_dict(torch.load(checkpoint_path))
    decoder.train()

    # Set up causal mask for transformer decoder
    causal_mask = set_up_causal_mask(config["max_len"], device)

    # Load training configuration
    train_config = config["train_config"]
    learning_rate = train_config["learning_rate"]

    # Prepare the model optimizer
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["l2_penalty"]
    )
    # Loss function
    loss_fcn = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.strftime("%b-%d_%H-%M-%S")
    train_step = 0
    for epoch in range(train_config["num_of_epochs"]):
        print("Epoch:", epoch)
        decoder.train()

        for x_img, x_words, y, tgt_padding_mask in train_loader:
            optimizer.zero_grad()
            train_step += 1

            # Move the used tensors to defined device
            x_img, x_words = x_img.to(device), x_words.to(device)
            y = y.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)

            # Extract image features
            with torch.no_grad():
                img_features = encoder(x_img)
                img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
                img_features = img_features.permute(0, 2, 1)
                img_features = img_features.detach()

            # Get the prediction of the decoder
            y_pred = decoder(x_words, img_features, tgt_padding_mask, causal_mask)
            tgt_padding_mask = torch.logical_not(tgt_padding_mask)
            y_pred = y_pred[tgt_padding_mask]

            y = y[tgt_padding_mask]

            # Calculate the loss
            loss = loss_fcn(y_pred, y.long())

            # Update model weights
            loss.backward()
            log_gradient_norm(decoder, writer, train_step, "Before")
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), train_config["gradient_clipping"])
            log_gradient_norm(decoder, writer, train_step, "After")
            optimizer.step()

            writer.add_scalar("Train/Step-Loss", loss.item(), train_step)
            writer.add_scalar("Train/Learning-Rate", learning_rate, train_step)

        # Save the model and optimizer state
        save_checkpoint(decoder, optimizer, start_time, epoch)
        # Evaluate model performance
        if (epoch + 1) % train_config["eval_period"] == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                # Evaluate model performance on subsets
                train_bleu = evaluate(train_set, encoder, decoder, config, device)
                valid_bleu = evaluate(valid_set, encoder, decoder, config, device)

                # Log the evaluated BLEU score
                for i, t_b in enumerate(train_bleu):
                    writer.add_scalar(f"Train/BLEU-{i+1}", t_b, epoch)
                for i, v_b in enumerate(valid_bleu):
                    writer.add_scalar(f"Valid/BLEU-{i+1}", v_b, epoch)

                decoder.train()
            
        print()
