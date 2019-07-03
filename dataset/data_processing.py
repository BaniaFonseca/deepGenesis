from config import *
from dataset.exception import *
import random
import cv2
import numpy as np
import pathlib
from keras.preprocessing.image import save_img
from keras.preprocessing.image import  load_img
import shutil
from skimage import img_as_float
from skimage import exposure
from skimage.util import random_noise
from skimage.util import invert
from skimage.transform import rotate
from scipy import ndimage

class DataProcessing:

    def __init__(self):
        self.processed_images = 0
        try:
            path = ALL_DATA
            if len(list(path.glob('*/*'))) is 0:
                raise ImageNotFound
        except ImageNotFound:
            print('images not found at director: [{}]'.format(ALL_DATA.absolute()))

    def process_and_save_data(self, labels=list(LABEL_NAMES)):
        self.creat_folders()
        if len(labels) > 0:
            label = labels[0]
            image_paths = \
                [str(path) for path in list(ALL_DATA.glob(label+'/*'))]

            random.shuffle(image_paths, random.seed())

            self.process_and_save_testdata(image_paths)
            self.process_and_save_validationdata(image_paths)
            self.process_and_save_traindata(image_paths)

            del labels[0]
            self.process_and_save_data(labels)

    def process_and_save_testdata(self, image_paths):
        for i in range(TEST_SIZE):
            path = image_paths[i]
            img = self.read_img_and_clear_noise(path)
            filename = self.build_filename(TEST_DATA, path)
            self.save_img(filename, img)
            print(filename)

    def process_and_save_validationdata(self, image_paths):
        for i in range(TEST_SIZE, TEST_SIZE + VALIDATION_SIZE):
            path = image_paths[i]
            img = self.read_img_and_clear_noise(path)
            filename = self.build_filename(VALIDATION_DATA, path)
            self.save_img(filename, img)
            print(filename)

    def process_and_save_traindata(self, image_paths):
        for i in range(TEST_SIZE+VALIDATION_SIZE, DATASET_SIZE):
            path = image_paths[i]
            img = self.read_img_and_clear_noise(path)
            filename = self.build_filename(TRAIN_DATA, path)
            self.save_img(filename, img)
            print(filename)

    def build_filename(self, DATA_ROOT, img_path):
        filename = DATA_ROOT
        self.processed_images = self.processed_images + 1
        filename = \
            filename.joinpath(pathlib.Path(img_path).parent.name,
                              str(self.processed_images) + '.jpeg')
        return str(filename)

    def save_img(self, filename, img):
        img = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
        save_img(str(filename), img)

    def creat_folders(self, root_paths=DATA_ROOTS):
        if len(root_paths) > 0:
            for dir_name in ALL_DATA.glob('*/'):
                path = root_paths[0]
                path = path.joinpath(dir_name.name)
                if pathlib.Path(path).exists():
                    shutil.rmtree(path)
                    pathlib.Path(path).mkdir()
                else:
                    pathlib.Path(path).mkdir()
            del root_paths[0]
            self.creat_folders(root_paths)
    #1
    def color_inversion(self, img):
        return invert(img)
    #2
    def rotate(self, img):
        return rotate(img, 45)
    #3
    def blur(self, img):
        return ndimage.uniform_filter(img, size=(21, 21, 1))
    #4
    def flip(self, img):
        return img[:, ::-1]
    #5
    def random_noise(self, img):
        return random_noise(img)

    def read_img_and_clear_noise(self, path):
        img = load_img(path)
        img = img_as_float(img)  # [0, 1]
        img = exposure.adjust_gamma(img, gamma=0.5, gain=1)
        v_min, v_max = np.percentile(img, (0.2, 99.8))
        better_contrast = exposure.rescale_intensity(img, in_range=(v_min, v_max))
        return better_contrast

    def augment(self, img, path, transformatios=list()):
        if len(transformatios) > 0:
            for i, transf in enumerate(transformatios):
                t_img = transf(img)
                filename = self.build_filename(TRAIN_DATA, path)
                self.save_img(filename, t_img)
                print(filename)
                remain_transformatios = list(transformatios)
                del remain_transformatios [i]
                self.augment(t_img, path, remain_transformatios)
