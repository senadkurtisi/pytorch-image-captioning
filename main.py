import json

import torch
from tensorboardX import SummaryWriter

from trainer import train


def main():
    # Load the pipeline configuration file
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    writer = SummaryWriter()
    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    train(config, writer, device)


if __name__ == "__main__":
    main()
