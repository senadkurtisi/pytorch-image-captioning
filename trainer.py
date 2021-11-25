import numpy as np
import torch
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
    train_set = Flickr8KDataset(config, config["dataset_path"]["train"])
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
