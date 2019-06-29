from config import *
import pathlib
import tensorflow as tf
import random
from dataset.data_processing import DataProcessing
import numpy as np
from dataset.exception import *

class Dataset():

    def __init__(self,classe='*', name='*'):
        self.save_datasets_as_npy(classe, name)

    def save_datasets_as_npy(self, classe='*', name='*'):
        self.save_trainset_as_npy(classe, name)
        self.save_validationset_as_npy()
        self.save_testset_as_npy(classe, name)

    def save_validationset_as_npy(self):
        try:
            path = VALIDATION_DATA
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            path = path.joinpath()
            validation_dataset = self.generate_dataset(path)
            iterator = tf.compat.v1.data.make_one_shot_iterator(validation_dataset)
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

                np.save(str(VALIDATION_DATA)+'-images', images)
                np.save(str(VALIDATION_DATA) + '-labels', labels)
        except ImageNotFound:
            print('images not found at director: [{}]'.format(TEST_DATA.absolute()))

    def save_testset_as_npy(self, classe='*', name='*'):
        try:
            path = TEST_DATA
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            test_dataset = self.generate_dataset(path, classe, name)
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

                np.save(str(TEST_DATA)+'-images', images)
                np.save(str(TEST_DATA) + '-labels', labels)
        except ImageNotFound:
            print('images not found at director: [{}]'.format(TEST_DATA.absolute()))

    def save_trainset_as_npy(self, classe='*', name='*'):
        try:
            path = TRAIN_DATA
            size = len(list(path.glob('*/*')))
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            train_dataset = self.generate_dataset(path, classe ,name)
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

                np.save(str(TRAIN_DATA)+'-images', images)
                np.save(str(TRAIN_DATA)+'-labels', labels)
        except ImageNotFound:
            print('images not found at director: [{}]'.format(TRAIN_DATA.absolute()))

    def generate_dataset(self, path, classe='*', name='*'):
        all_image_paths = [str(path) for path in list(path.glob(classe+'/'+name))]
        all_image_paths.sort()

        # random.shuffle(all_image_paths, random.seed(7))
        label_to_index = dict((name, index) for index, name in enumerate(LABEL_NAMES))
        labels = \
            [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
        # print(labels)

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

    def load_trainset(self):
        images  = np.load(str(TRAIN_DATA)+'-images.npy', allow_pickle=True)
        labels = np.load(str(TRAIN_DATA)+'-labels.npy', allow_pickle=True)
        return images, labels

    def load_testset(self):
        images = np.load(str(TEST_DATA)+'-images.npy', allow_pickle=True)
        labels = np.load(str(TEST_DATA)+'-labels.npy', allow_pickle=True)
        return images, labels

    def load_validationtest(self):
        images = np.load(str(VALIDATION_DATA) + '-images.npy', allow_pickle=True)
        labels = np.load(str(VALIDATION_DATA)+'-labels.npy', allow_pickle=True)
        return images, labels
