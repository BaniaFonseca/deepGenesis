import random
import cv2
import pathlib
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img, img_to_array
from keras.preprocessing.image import  load_img

class DataProcessing:

    def __init__(self, preprocessed_data_root, processed_data_root):
        self.preprocessed_data_root = pathlib.Path(preprocessed_data_root)
        self.DATASET_SIZE = len(list(self.preprocessed_data_root.glob('*/*')))
        self.TRAIN_SIZE = int(0.7 * self.DATASET_SIZE)
        self.TEST_SIZE = int(0.3 * self.DATASET_SIZE)
        self.HEIGHT = 256
        self.WIDTH = 256
        self.processed_data_root = pathlib.Path(processed_data_root)
        self.all_image_paths = \
            [str(path) for path in list(self.preprocessed_data_root.glob('*/*'))]
        random.shuffle(self.all_image_paths)
        self.processed_images = 0
        self.save_processed_data()

    def save_processed_data(self):
        for i in range(self.TEST_SIZE):
            self.processed_images = self.processed_images+1
            path = self.all_image_paths[i]
            img = load_img(path)
            data = img_to_array(img)
            img = self.resize(data)
            filename = self.processed_data_root
            filename = filename.joinpath('test',
                pathlib.Path(path).parent.name,'original_' + str(i) + '.jpeg')
            save_img(str(filename), img)
            print(filename)

        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
            self.processed_images = self.processed_images+1
            path = self.all_image_paths[i]
            img = load_img(path)
            data = img_to_array(img)
            img = self.resize(data)
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'original_' + str(i) + '.jpeg')
            save_img(str(filename), img)
            print(filename)

        self.rotate()
        self.flip_horizontal()
        self.brightness()
        self.rescale()
        self.zoom()

        print("{} images was processed and saved sucessfuly at: {}"
              .format(self.processed_images, self.processed_data_root))

    def rescale(self):
        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
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
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'rescale_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def rotate(self):
        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
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
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'rotate_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def flip_horizontal(self):
        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
            path = self.all_image_paths[i]
            self.processed_images = self.processed_images + 1
            image = cv2.imread(path)
            image = cv2.flip(image, 1)
            image = self.resize(image)
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'flip_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def brightness(self):
        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
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
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'brightness_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def zoom(self):
        for i in range(self.TEST_SIZE, self.DATASET_SIZE):
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
            filename = self.processed_data_root
            filename = filename.joinpath('train',
                pathlib.Path(path).parent.name, 'zoom_' + str(i) + '.jpeg')
            save_img(str(filename), image)
            print(filename)

    def resize(self, img):
        return cv2.resize(img, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_AREA)