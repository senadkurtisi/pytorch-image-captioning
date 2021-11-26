import os
import json

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

    def __init__(self, config, path, training=True):
        """Initializes the module.
        
        Arguments:
            config (object): Contains dataset configuration
            path (str): Location where image captions are stored
        """
        with open(path, "r") as f:
            self._data = [line.replace("\n", "") for line in f.readlines()]
        
        self._training = training

        # Create inference data
        self._inference_captions = self._group_captions(self._data)

        # Load the vocabulary mappings
        with open(config["word2idx_path"], "r", encoding="utf8") as f:
            self._word2idx = json.load(f)
        self._idx2word = {str(idx): word for word, idx in self._word2idx.items()}

        self._start_idx = config["START_idx"]
        self._end_idx = config["END_idx"]
        self._pad_idx = config["PAD_idx"]
        self._START_token = config["START_token"]
        self._END_token = config["END_token"]
        self._PAD_token = config["PAD_token"]

        self._max_len = config["max_len"]

        # Transformation to apply to each image
        self._image_specs = config["image_specs"]
        self._image_transform = self._construct_image_transform(self._image_specs["image_size"])
        # All images that appear in the dataset
        self._image_names = list(set([line.split()[0].split("#")[0] for line in self._data]))
        # Preprocessed images
        self._images = self._load_and_process_images(self._image_specs["image_dir"], self._image_names)

        # Create artificial samples
        self._data = self._create_artificial_samples() if self._training else None
        self._dataset_size = len(self._data)

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
        image_paths = [os.path.join(image_dir, fname) for fname in image_names]
        # Load images
        images_raw = [Image.open(path) for path in image_paths]
        # Adapt the images to CNN trained on ImageNet { PIL -> Tensor }
        image_tensors = [self._image_transform(img) for img in images_raw]

        images_processed = {img_name: img_tensor for img_name, img_tensor in zip(image_names, image_tensors)}
        return images_processed

    def _group_captions(self, data):
        """Groups captions which correspond to the same image.

        Main usage: Calculating BLEU score

        Arguments:
            data (list of str): Each element contains image name and corresponding caption
        Returns:
            grouped_captions (dict): Key - image name, Value - list of captions associated
                with that picture
        """
        grouped_captions = {}

        for line in data:
            caption_data = line.split()
            img_name, img_caption = caption_data[0].split("#")[0], caption_data[1:]
            if img_name not in grouped_captions:
                grouped_captions[img_name] = []

            grouped_captions[img_name].append(" ".join(img_caption))

        return grouped_captions

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
            caption_words = [self._START_token] + caption + [self._END_token]
            num_words = len(caption_words)
            for pos in range(1, num_words):
                # Input for the neural network: Words predicted until now
                new_input = caption_words[:pos]
                # Correct prediction for the next word
                label = caption_words[pos]
                augmented_data += [(image_name, new_input, label)]

        return augmented_data

    def inference_batch(self, batch_size):
        """Creates a mini batch dataloader for inference.

        During inference we generate caption from scratch and in each iteration
        we feed words generated previously by the model (i.e. no teacher forcing).
        We only need input image as well as the target caption.
        """
        caption_data_items = list(self._inference_captions.items())
        num_batches = len(caption_data_items) // batch_size
        for idx in range(num_batches):
            caption_samples = caption_data_items[idx * batch_size: (idx + 1) * batch_size]
            batch_imgs = []
            batch_captions = []

            # Increase index for the next batch
            idx += batch_size

            # Create a mini batch data
            for image_name, captions in caption_samples:
                batch_captions.append(captions)
                batch_imgs.append(self._images[image_name])

            # Batch image tensors
            batch_imgs = torch.stack(batch_imgs, dim=0)

            yield batch_imgs, batch_captions

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        # Extract the caption data
        image_id, input_tokens, label = self._data[index]

        # Extract image tensor
        image_tensor = self._images[image_id]

        # Number of words in the input token
        sample_size = len(input_tokens)

        # Pad the token and label sequences
        input_tokens = input_tokens[:self._max_len]
        padding_size = self._max_len - sample_size
        if padding_size > 0:
            input_tokens += [self._PAD_token for _ in range(padding_size)]

        # Apply the vocabulary mapping to the input tokens
        input_tokens = [token.strip().lower() for token in input_tokens]
        input_tokens = [self._word2idx[token] for token in input_tokens]
        input_tokens = torch.Tensor(input_tokens).long()

        # Next word label
        label = self._word2idx[label]
        label = torch.Tensor([label]).long()

        # Index from which to extract the model prediction
        tgt_pos = torch.Tensor([sample_size]).long()

        # Define the padding mask
        padding_mask = torch.ones([self._max_len, ])
        padding_mask[:sample_size] = 0.0

        return image_tensor, input_tokens, label, padding_mask, tgt_pos
