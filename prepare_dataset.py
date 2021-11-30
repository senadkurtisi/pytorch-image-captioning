import json

from utils.utils import clean_captions, create_vocab, extract_embeddings, load_captions, split_dataset


if __name__ == "__main__":
    # Load the project pipeline configuration
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load image captions
    dataset_path = config["dataset_path"]
    with open(dataset_path, "r") as f:
        data = f.read()

    image2caption = load_captions(data)
    image2caption = clean_captions(image2caption)
    # Create and save dataset corpus vocabulary
    create_vocab(image2caption, config["word2idx_path"])
    # Extract GloVe embeddings for tokens present in the training set vocab
    extract_embeddings(config)

    # Split dataset
    split_images_paths = list(config["split_images"].values())
    split_save_paths = list(config["split_save"].values())
    split_dataset(image2caption, split_images_paths, split_save_paths)
