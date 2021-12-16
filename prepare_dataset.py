import json

from utils.dataset_utils import clean_captions, create_vocab, extract_embeddings, split_dataset, load_captions


if __name__ == "__main__":
    # Load the project pipeline configuration
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load and clean the loaded captions
    dataset_path = config["dataset_path"]
    with open(dataset_path, "r") as f:
        data = f.read()

    image2caption = load_captions(data)
    image2caption = clean_captions(image2caption)

    # Create and save dataset corpus vocabulary
    vocab = create_vocab(image2caption)
    # Extract GloVe embeddings for tokens present in the training set vocab
    extract_embeddings(config, vocab)

    # Save info regarding the dataset split elements
    split_images_paths = list(config["split_images"].values())
    split_save_paths = list(config["split_save"].values())
    split_dataset(image2caption, split_images_paths, split_save_paths)
