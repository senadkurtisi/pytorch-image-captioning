import string


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
            caption = caption[i]
            # Extract separate tokens
            caption = caption.split()
            # Make tokens lowercase
            caption = [word.lower() for word in caption]
            # Remove punctuation
            caption = [word.translate(punct_table) for word in caption]
            # Remove trailing "'s" or "a"
            caption = [word for word in caption if len(word) > 1]
            # Save the cleaned caption
            image2caption_clean[image_id][i] =  caption

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
    """
    image2caption = dict()
    for sample in data.split('\n'):
        tokens = sample.split()
        if len(sample) < 2:
            # Image has no description
            continue
		# First token is image id, remaining ones correspond to the caption
        image_id, image_caption = tokens[0], tokens[1:]
		# Extract only the filename from the image id
        image_id = image_id.split('.')[0]
		# Recreate the description
        image_caption = " ".join(image_caption)
        
        if image_id not in image2caption:
            image2caption[image_id] = list()
		# Save the description
        image2caption[image_id].append(image_caption)

	return image2caption
