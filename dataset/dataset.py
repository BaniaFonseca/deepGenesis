import pathlib
import tensorflow as tf
import random

class Dataset:

    def __init__(self, data_root=str, image_width=int, image_height=int):
        self.data_root = pathlib.Path(data_root)
        self.image_width = image_width
        self.image_height = image_height
        self.all_image_paths = [str(path) for path in  list(self.data_root.glob('*/*'))]
        random.shuffle(self.all_image_paths, random.seed(7*(len(self.all_image_paths))))



    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [self.image_width, self.image_height])
        image /= 255.0 # normalize to [0, 1] ramge
        return image

    def load_and_preprocess_image(self, path):
        image = tf.read_file(path)
        return self.preprocess_image(image)
