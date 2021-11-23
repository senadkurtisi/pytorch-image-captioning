
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
