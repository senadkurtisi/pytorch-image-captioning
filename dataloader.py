import os
import json
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class Flickr8KDataset(Dataset):
    """"Represents dataloader for the Flickr8k dataset.

    Data is stored in following format:
        image_name: associated caption
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

        self._image_transform = self._construct_image_transform(config["image_size"])

    def _construct_image_transform(self, image_size):
        """Constructs the image preprocessing transform object.

        Arguments:
            image_size (int): Size of the result image
        """
        # ImageNet normalization statistics
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

        return preprocessing

    def load_and_process_images(self, image_dir, image_names):
        """Loades dataset images and adapts them for the CNN.

        Arguments:
            image_dir (str): Directory where images are stored
            image_names (list of str): Names of image files in the dataset
        Returns:
            images_processed (list of torch.Tensor): "ImageNet-adapted" versions
                of loaded images
        """
        image_paths = [os.path.join(image_dir, fname) for fname in image_names]
        # Load images
        images_raw = [Image.open(path) for path in image_paths]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        images_processed = [self._image_transform(img) for img in images_raw]
        return images_processed
