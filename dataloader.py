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

        # Auxiliary token indices
        self._start_idx = config["START_idx"]
        self._end_idx = config["END_idx"]
        self._pad_idx = config["PAD_idx"]
        self._UNK_idx = config["UNK_idx"]
        # Auxiliary token marks
        self._START_token = config["START_token"]
        self._END_token = config["END_token"]
        self._PAD_token = config["PAD_token"]
        self._UNK_token = config["UNK_token"]

        self._max_len = config["max_len"]

        # Transformation to apply to each image
        self._image_specs = config["image_specs"]
        self._image_transform = self._construct_image_transform(self._image_specs["image_size"])

        # Create paths to image files belonging to the subset
        subset = "train" if training else "validation"
        self.image_dir = self._image_specs["image_dir"][subset]

        # Create (X, Y) pairs
        self._data = self._create_input_label_mappings(self._data)

        self._dataset_size = len(self._data) if self._training else 0

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
                # We came across the first caption for this particular image
                grouped_captions[img_name] = []

            grouped_captions[img_name].append(img_caption)

        return grouped_captions

    def _create_input_label_mappings(self, data):
        """Creates (image, description) pairs.

        Arguments:
            data (list of str): Each element consists out of image file name and appropriate caption
                Elements are organized in the following format: 'image_name[SPACE]caption'
        Returns:
            processed_data (list of tuples): Each tuple is organized in following format: (image_name, caption)
        """
        processed_data = []
        for line in data:
            tokens = line.split()
            # Separate image name from the label tokens
            img_name, caption_words = tokens[0].split("#")[0], tokens[1:]
            # Construct (X, Y) pair
            pair = (img_name, caption_words)
            processed_data.append(pair)

        return processed_data

    def _load_and_prepare_image(self, image_name):
        """Performs image preprocessing.

        Images need to be prepared for the ResNet encoder.
        Arguments:
            image_name (str): Name of the image file located in the subset directory
        """
        image_path = os.path.join(self.image_dir, image_name)
        img_pil = Image.open(image_path).convert("RGB")
        image_tensor = self._image_transform(img_pil)
        return image_tensor

    def inference_batch(self, batch_size):
        """Creates a mini batch dataloader for inference.

        During inference we generate caption from scratch and in each iteration
        we feed words generated previously by the model (i.e. no teacher forcing).
        We only need input image as well as the target caption.
        """
        caption_data_items = list(self._inference_captions.items())
        # random.shuffle(caption_data_items)

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
                batch_imgs.append(self._load_and_prepare_image(image_name))

            # Batch image tensors
            batch_imgs = torch.stack(batch_imgs, dim=0)
            if batch_size == 1:
                batch_imgs = batch_imgs.unsqueeze(0)

            yield batch_imgs, batch_captions

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        # Extract the caption data
        image_id, tokens = self._data[index]

        # Load and preprocess image
        image_tensor = self._load_and_prepare_image(image_id)

        # Pad the token and label sequences
        tokens = tokens[:self._max_len]

        tokens = [token.strip().lower() for token in tokens]
        tokens = [self._START_token] + tokens + [self._END_token]
        # Extract input and target output
        input_tokens = tokens[:-1].copy()
        tgt_tokens = tokens[1:].copy()

        # Number of words in the input token
        sample_size = len(input_tokens)
        padding_size = self._max_len - sample_size

        if padding_size > 0:
            padding_vec = [self._PAD_token for _ in range(padding_size)]
            input_tokens += padding_vec.copy()
            tgt_tokens += padding_vec.copy()

        # Apply the vocabulary mapping to the input tokens
        input_tokens = [self._word2idx.get(token, self._UNK_idx) for token in input_tokens]
        tgt_tokens = [self._word2idx.get(token, self._UNK_idx) for token in tgt_tokens]

        input_tokens = torch.Tensor(input_tokens).long()
        tgt_tokens = torch.Tensor(tgt_tokens).long()

        # Index from which to extract the model prediction
        # Define the padding masks
        tgt_padding_mask = torch.ones([self._max_len, ])
        tgt_padding_mask[:sample_size] = 0.0
        tgt_padding_mask = tgt_padding_mask.bool()

        return image_tensor, input_tokens, tgt_tokens, tgt_padding_mask
