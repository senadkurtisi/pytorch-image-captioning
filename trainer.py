import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

from dataloader import Flickr8KDataset
from decoder import CaptionDecoder


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
    valid_hyperparams = {
        "batch_size": config["batch_size"]["validation"],
        "shuffle": False,
        "drop_last": True
    }

    # Create dataloaders
    train_set = Flickr8KDataset(config, config["split_save"]["train"])
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
