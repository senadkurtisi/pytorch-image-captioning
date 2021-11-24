import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class Flickr8KDataset(Dataset):
    """"Represents dataloader for the Flickr8k dataset.

    Data is stored in following format:
        image_name: associated caption.
    Each image has maximum 5 different captions.
    """

    def __init__(self, config, path):
        """Initializes the module.
        
        Arguments:
            config (object): Contains dataset configuration
            path (str): Location where image captions are stored
        """
        with open(path, "r") as f:
            self._data = [line.replace("\n", "") for line in f.readlines()]
        
        # Load the vocabulary mappings
        with open(config["word2idx_path"], "r", encoding="utf8") as f:
            self._word2idx = json.load(f)
        self._idx2word = {str(idx): word for word, idx in self._word2idx.items()}

        # Set the default value for the OOV tokens
        self._word2idx = defaultdict(
            lambda: self._word2idx[config["OOV_token"]],
            self._word2idx
        )

        self._start_idx = config["START_idx"]
        self._end_idx = config["END_idx"]

        self._PAD_token = config["PAD_token"]
        self._PAD_label = config["PAD_label"]

        self._max_len = config["max_len"]
        self._dataset_size = len(self.data)