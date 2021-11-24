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
        self._START_token = config["START_token"]
        self._END_token = config["END_token"]

        self._PAD_token = config["PAD_token"]
        self._PAD_label = config["PAD_label"]

        self._max_len = config["max_len"]
        self._dataset_size = len(self.data)

        # Transformation to apply to each image
        self._image_transform = self._construct_image_transform(config["image_size"])
        # All images that appear in the dataset
        self._image_names = list(set([line.split()[0].split("#")[0] for line in self._data]))
        # Preprocessed images
        self._images = self._load_and_process_images(config["image_dir"], self._image_names)

        # Create artificial samples
        self._data = self._create_artificial_samples()

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

    def _load_and_process_images(self, image_dir, image_names):
        """Loades dataset images and adapts them for the CNN.

        Arguments:
            image_dir (str): Directory where images are stored
            image_names (list of str): Names of image files in the dataset
        Returns:
            images_processed (dict): "ImageNet-adapted" versions of loaded images
                key: image name, dict: torch.Tensor of loaded and processed image
        """
        # TODO: Implement dict mapping -> image_id: torch tensor
        image_paths = [os.path.join(image_dir, fname) for fname in image_names]
        # Load images
        images_raw = [Image.open(path) for path in image_paths]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        image_tensors = [self._image_transform(img) for img in images_raw]

        images_processed = {img_name: img_tensor for img_name, img_tensor in zip(image_names, image_tensors)}
        return images_processed

    def _create_artificial_samples(self):
        """Augments the dataset with artificial samples.

        Example:
            - original sample: image_0 Two dogs playing outside
            - Generated samples:
                1. image_0 Two
                2. image_0 Two dogs
                3. image_0 Two dogs playing
                4. image_0 Two dogs playing outside

        Returns:
            augmented_data (list): Augmented dataset
                Each element is a tuple (image_name, input_words, label)
                - input_words (list of str): Words predicted until now
                - label (str): Correct prediction for the next word
        """
        augmented_data = []
        for line in self._data:
            line_split = line.split()
            image_name, caption = line_split[0], line_split[1:]
            # Clean image name entry
            image_name = image_name.split("#")[0]

            # Add tokens for start and end of the sequence
            # Start token is necessary for predicting the first word of the caption
            caption_words = [self._START_token] + caption.split() + [self._END_token]
            num_words = len(caption_words)
            for pos in range(1, num_words):
                # Input for the neural network: Words predicted until now
                new_input = caption_words[:pos]
                # Correct prediction for the next word
                label = caption_words[pos]
                augmented_data += (image_name, new_input, label)

        return augmented_data

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        pass
