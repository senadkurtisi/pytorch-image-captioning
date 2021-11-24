import os
import json
import string

import numpy as np


def save_captions(image2caption, subset_imgs, save_path):
    """Saves captions for images which belong to subset to a file.

    Arguments:
        image2caption (dict): Mapping from image id to all
            captions of that image that occured in the dataset
        subset_imgs (list of str): List of image names which belong to subset
        save_path (str): Path to which to save extracted captions
    """
    captions = []
    for image_name in subset_imgs:
        image_id = os.path.splitext(image_name)[0]
        if image_id in image2caption:
            [captions.append("{} {}\n".format(image_name, caption)) for caption in image2caption[image_id]]

    # Save extracted captions
    with open(save_path, "w") as f:
        f.writelines(captions)


def split_dataset(image2caption, split_images_paths, save_paths):
    """Perfoms splitting of the dataset.

    Arguments:
        image2caption (dict): Mapping from image id to all
            captions of that image that occured in the dataset
        split_images_paths (list of str): Contains paths of files which contain
            names of images which belong to each subset in the split
        split_images_paths (list of str): Contains paths to which to save
            captions extracted for each image in specified subsets
    """
    for load_path, save_path in zip(split_images_paths, save_paths):
        # Load names of images which belong to current subset
        with open(load_path, "r") as f:
            subset_imgs = [fname.replace("\n", "") for fname in f.readlines()]
        # Save processed captions for those images in a separate file
        save_captions(image2caption, subset_imgs, save_path)


def extract_embeddings(config):
    """Extracts GloVe word embeddings for words in vocab.

    Arguments:
        config (object): Contains dataset & pipeline configuration info
    """
    embeddings_config = config["embeddings"]
    save_path_emb = embeddings_config["path"]
    embedding_dim = embeddings_config["size"]

    # Used for finding the embedding vector for each token
    vectors = []

    embedding_file_name = "glove.6B.{}d.txt".format(embedding_dim)
    embeddings_path = os.path.join(config["glove_dir"], embedding_file_name)
    with open(embeddings_path, "rb") as f:
        for line in f:
            line = line.decode().split()
            # Extract and pre-process the token
            word = line[0]
            word = word.strip().lower()
            # Remember the embedding vector for the word
            embedding_vec = np.array(line[1:], dtype="float")
            vectors += [embedding_vec]

    vectors = np.array(vectors)
    # Embedding vector for tokens used for padding the input sequence
    pad_embedding = np.zeros((embedding_dim,))
    # Embedding vector for tokens not present in the training set
    unk_embedding = vectors.mean(axis=0)

    vectors = np.vstack([unk_embedding, pad_embedding, vectors])
    # Save extracted embeddings
    np.savetxt(save_path_emb, vectors)

    print("\nExtracted GloVe embeddings for all tokens in the training set.") 
    print("Embedding vectors size:", embedding_dim)


def create_vocab(image2caption, save_path):
    """Creates a vocabulary of tokens in the dataset corpus.
    
    Arguments:
        image2caption (dict): Mapping from image id to all
            captions of that image that occured in the dataset
        save_path (str): Path to which to save genererated vocabulary
    """
    # Vocabulary dictionary
    word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2}
    # All possible words in the token
    words = set()
    # Extract all tokens from the image captions
    for captions in image2caption.values():
        current_words = [word for caption in captions for word in caption.split()]
        words.update(current_words)

    starting_len = len(word2idx)
    words = list(words)
    word2idx.update({(idx + starting_len): word for idx, word in enumerate(words)})

    # Save vocabulary to a file
    with open(save_path, "w", encoding="utf8") as f:
        json.dump(word2idx, f)


def clean_captions(image2caption):
    """Cleans the loaded image captions.
    
    Makes tokens lowercase. Removes punctuation. Removes some stop words.

    Arguments:
        image2caption (dict): Mapping from image id to all
            captions of that image that occured in the dataset
    Returns:
        image2caption_clean (dict): Mapping from image id to
            cleaned captions of that image
    """
    image2caption_clean = image2caption.copy()
    punct_table = str.maketrans("", "", string.punctuation)
    for image_id, captions in image2caption.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Extract separate tokens
            caption = caption.split()
            # Make tokens lowercase
            caption = [word.lower() for word in caption]
            # Remove punctuation
            caption = [word.translate(punct_table) for word in caption]
            # Remove trailing "'s" or "a"
            caption = [word for word in caption if len(word) > 1]
            # Remove tokens which contain number
            caption = [word for word in caption if word.isalpha()]
            # Save the cleaned caption
            image2caption_clean[image_id][i] =  " ".join(caption)

    return image2caption_clean


def load_captions(data):
    """Processes the loaded image captions.

    Image captions are saved in the following format:
        image_id[SPACE]Caption
    
    Arguments:
        data (str): Loaded image captions from .txt file
    Returns:
        image2desc (dict): Mapping from image id to all
            captions of that image that occured in the dataset
        train_images (list of str): Containes names of images in train set
    """
    image2caption = dict()
    for sample in data.split("\n"):
        tokens = sample.split()
        if len(sample) < 2:
            # Image has no description
            continue
		# First token is image id, remaining ones correspond to the caption
        image_name, image_caption = tokens[0], tokens[1:]

        image_id = image_name.split(".")[0]
        # Recreate the description
        image_caption = " ".join(image_caption)
        
        if image_id not in image2caption:
            image2caption[image_id] = list()
        # Save the description
        image2caption[image_id].append(image_caption)

    return image2caption
