import json

from utils import clean_captions, create_vocab, extract_embeddings, load_captions


if __name__ == "__main__":
    # Load the project pipeline configuration
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load image captions
    dataset_path = config["dataset_path"]
    with open(dataset_path, "r") as f:
        data = f.read()

    # Load the names of images in train set
    with open(config["split_images"]["train"], "r") as f:
        train_images = [fname.replace("\n", "") for fname in f.readlines()]

    image2caption = load_captions(data)
    image2caption = clean_captions(image2caption)
    create_vocab(image2caption, config["word2idx_path"])
    # Extract GloVe embeddings for tokens present in the training set vocab
    extract_embeddings(config)
