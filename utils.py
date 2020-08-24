from skimage import io


def load_image(image_path):
    """
        Loads specified image.

        Args:
            image_path (str): path to specific image in the dataset
    """
    image = io.imread(image_path)
    return image
