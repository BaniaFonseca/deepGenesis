from config import *
from dataset.exception import *
import random
import cv2
import pathlib
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img, img_to_array
from keras.preprocessing.image import  load_img

class DataProcessing:

    def __init__(self):
        try:
            path = ALL_DATA
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound

            self.all_image_paths = \
                [str(path) for path in list(ALL_DATA.glob('*/*'))]
            random.shuffle(self.all_image_paths)
            self.processed_images = 0
            self.creat_folders()

        except ImageNotFound:
            print('images not found at director: [{}]'.format(ALL_DATA.absolute()))

    def process_data(self):
        for i in range(TEST_SIZE):
            self.processed_images = self.processed_images+1
            path = self.all_image_paths[i]
            img = load_img(path)
            data = img_to_array(img)
            img = self.resize(data)
            filename = TEST_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name,'original_' + str(i) + '.jpeg')
            save_img(str(filename), img)
            print(filename)

        for i in range(TEST_SIZE, DATASET_SIZE):
            self.processed_images = self.processed_images+1
            path = self.all_image_paths[i]
            img = load_img(path)
            data = img_to_array(img)
            img = self.resize(data)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'original_' + str(i) + '.jpeg')
            save_img(str(filename), img)
            print(filename)

        self.rotate()
        self.flip_horizontal()
        self.brightness()
        self.rescale()
        self.zoom()

        print("{} images was processed and saved sucessfuly"
              .format(self.processed_images))

    def rescale(self):
        for i in range(TEST_SIZE, DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images + 1
            img = load_img(path)
            data = img_to_array(img)
            datagen = ImageDataGenerator(rescale=2)
            samples = expand_dims(data, 0)
            it = datagen.flow(samples, batch_size=1)
            batch = it.next()
            image = batch[0].astype('uint8')
            image = self.resize(image)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'rescale_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def rotate(self):
        for i in range(TEST_SIZE, DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images+1
            img = load_img(path)
            data = img_to_array(img)
            datagen = ImageDataGenerator(rotation_range=45)
            samples = expand_dims(data, 0)
            it = datagen.flow(samples, batch_size=1)
            batch = it.next()
            image = batch[0].astype('uint8')
            image = self.resize(image)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'rotate_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def flip_horizontal(self):
        for i in range(TEST_SIZE, DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images + 1
            image = cv2.imread(path)
            image = cv2.flip(image, 1)
            image = self.resize(image)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'flip_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def brightness(self):
        for i in range(TEST_SIZE, DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images+1
            img = load_img(path)
            data = img_to_array(img)
            datagen = ImageDataGenerator(brightness_range=[2, 3])
            samples = expand_dims(data, 0)
            it = datagen.flow(samples, batch_size=1)
            batch = it.next()
            image = batch[0].astype('uint8')
            image = self.resize(image)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'brightness_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def zoom(self):
        for i in range(TEST_SIZE, DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images+1
            img = load_img(path)
            data = img_to_array(img)
            datagen = ImageDataGenerator(zoom_range=[1.5, 1.6])
            samples = expand_dims(data, 0)
            it = datagen.flow(samples, batch_size=1)
            batch = it.next()
            image = batch[0].astype('uint8')
            image = self.resize(image)
            filename = TRAIN_DATA
            filename = \
                filename.joinpath(pathlib.Path(path).parent.name, 'zoom_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def resize(self, img):
        return cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)

    def creat_folders(self):
        for dir_name in ALL_DATA.glob('*/'):
            path = TEST_DATA
            path = path.joinpath(dir_name.name)
            if not pathlib.Path(path).exists():
                pathlib.Path(path).mkdir()

        for dir_name in ALL_DATA.glob('*/'):
            path = TRAIN_DATA
            path = path.joinpath(dir_name.name)
            if not pathlib.Path(path).exists():
                pathlib.Path(path).mkdir()
