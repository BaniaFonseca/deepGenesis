from config import *
import pathlib
import tensorflow as tf
import random
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from dataset.exception import *
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self):
        pass

    def get_testset(self):
        try:
            path = TEST_DATA
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            test_dataset = self.generate_dataset(path)
            iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
            next_element = iterator.get_next()
            images = []
            labels = []

            with tf.compat.v1.Session() as session:
                for n in range(len(list(path.glob('*/*')))):
                    try:
                        image, label = session.run(next_element)
                        images.append(image)
                        labels.append(label)
                    except tf.errors.OutOfRangeError:
                        break

                np.save(str(TEST_DATA) + '_images', images)
                np.save(str(TEST_DATA) + '_labels', labels)

                images = np.load(str(TEST_DATA) + '_images.npy', allow_pickle=True)
                labels = np.load(str(TEST_DATA) + '_labels.npy', allow_pickle=True)

            return images, labels
        except ImageNotFound:
            print('images not found at director: [{}]'.format(TEST_DATA.absolute()))

    def get_trainset(self):
        try:
            path = TRAIN_DATA
            size = len(list(path.glob('*/*')))
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            train_dataset = self.generate_dataset(path)
            iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
            next_element = iterator.get_next()
            images = []
            labels = []

            with tf.compat.v1.Session() as session:
                for n in range(size):
                    try:
                        image, label = session.run(next_element)
                        images.append(image)
                        labels.append(label)
                    except tf.errors.OutOfRangeError:
                        break

                np.save(str(TRAIN_DATA) + '_images', images)
                np.save(str(TRAIN_DATA) + '_labels', labels)

                images = np.load(str(TRAIN_DATA) + '_images.npy', allow_pickle=True)
                labels = np.load(str(TRAIN_DATA) + '_labels.npy', allow_pickle=True)

            return images, labels
        except ImageNotFound:
            print('images not found at director: [{}]'.format(TRAIN_DATA.absolute()))

    def generate_dataset(self, path):
        all_image_paths = [str(path) for path in list(path.glob('*/*'))]
        random.shuffle(all_image_paths)
        label_to_index = dict((name, index) for index, name in enumerate(LABEL_NAMES))
        labels = \
            [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        images_ds = \
            paths_ds.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        labels_ds = tf.data.Dataset.from_tensor_slices(labels)

        return tf.data.Dataset.zip((images_ds, labels_ds))

    def load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [HEIGHT, WIDTH])
        image /= 255.0
        return image

    def load_data(self):
        return (self.get_trainset(), self.get_testset())