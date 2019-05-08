import pathlib
import tensorflow as tf
import random
from dataset.exception import *

class Dataset:

    def __init__(self, data_root=str, image_width=int, image_height=int):
        self.data_root = pathlib.Path(data_root)
        self.image_width = image_width
        self.image_height = image_height
        self.build_dataset()

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.image_width, self.image_height])
        image /= 255.0 # normalize to [0, 1] ramge
        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def build_dataset(self):

        try:
            if len(list(self.data_root.glob('*/*')))  is 0:
                raise ImageNotFound

            self.all_image_paths = [str(path) for path in list(self.data_root.glob('*/*'))]
            random.shuffle(self.all_image_paths, random.seed(7 * (len(self.all_image_paths))))
            self.all_image_labes = [pathlib.Path(path).parent.name for path in self.all_image_paths]
            self.path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
            self.image_ds = self.path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.labels_ds = tf.data.Dataset.from_tensor_slices(self.all_image_labes)
            self.image_label_ds = tf.data.Dataset.zip((self.image_ds, self.labels_ds))
        except ImageNotFound:
            print('images not found at director: [{}]'.format(self.data_root.absolute()))