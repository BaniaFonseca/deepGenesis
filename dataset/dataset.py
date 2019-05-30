import pathlib
import tensorflow as tf
import random
from dataset.exception import *

class Dataset:
    def __init__(self):
        pass

    def generate_dataset(self, path):
        try:
            path = pathlib.Path(path)

            if len(list(path.glob('*/*')))  is 0:
                raise ImageNotFound

            all_image_paths = [str(path) for path in list(path.glob('*/*'))]
            random.shuffle(all_image_paths, random.seed())
            label_names = sorted(item.name for item in path.glob('*/') if item.is_dir())
            label_to_index = dict((name, index) for index, name in enumerate(label_names))
            all_image_labes = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
            path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
            image_ds = path_ds.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            labels_ds = tf.data.Dataset.from_tensor_slices(all_image_labes)
            image_label_ds = tf.data.Dataset.zip((image_ds, labels_ds))
            return image_label_ds
        except ImageNotFound:
            print('images not found at director: [{}]'.format(self.data_root.absolute()))

    def load_image(self, path):
        image = tf.io.read_file(path)
        return tf.image.decode_jpeg(image, channels=3)
