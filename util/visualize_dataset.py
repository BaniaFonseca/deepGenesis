from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pathlib
from matplotlib import pyplot as plt
from PIL import Image
from dataset.dataset import Dataset
import tensorflow as tf

class VisualizeDataset(Dataset):
    def __init__(self, data_root):
        super().__init__(data_root=data_root, image_width=256, image_height=256)

    def show_mean_aspect_ratio(self, preprocess):
        aspect_ratio = []
        for path in self.all_image_paths:
            if preprocess is True:
                h, w, d = self.load_and_preprocess_image(path).shape.as_list()
                aspect_ratio.append(float(w) / float(h))
            else:
               h, w, d = np.shape(Image.open(path))
               aspect_ratio.append(float(w) / float(h))
        images = [i for i in range(1, aspect_ratio.__len__()+1)]
        aspect_ratio.sort()
        plt.suptitle('Acpect Ratio')
        plt.title('mean aspect ratio = '+str(np.mean(aspect_ratio)))
        plt.xlabel('image')
        plt.ylabel('apspect ratio')
        plt.plot(images, aspect_ratio, 'r')
        plt.show()

    def show_mean_width_heigth(self, preprocess):
        width = []
        height = []

        for path in self.all_image_paths:
            if preprocess is True:
                h, w, d = self.load_and_preprocess_image(path).shape.as_list()
            else:
                image = Image.open(path)
                h, w, d = np.shape(image)
            width.append(w)
            height.append(h)

        height.sort()
        width.sort()
        plt.title('Mean height = ' + str(np.mean(height))+' and '+'Mean width = ' + str(np.mean(width)))
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.plot(width, height, '.r', label='Width x Height')
        plt.show()

    def show_images(self, augmentend_data_root):
        columns, rows, size= 5, 4, 32
        augmentend_data_root = pathlib.Path(augmentend_data_root)
        datagen = ImageDataGenerator()
        all_flip_image_paths = list(augmentend_data_root.glob('*/flip*'))
        all_zoom_image_paths = list(augmentend_data_root.glob('*/zoom*'))
        all_brightness_image_paths = list(augmentend_data_root.glob('*/brightness*'))
        all_rotate_image_paths = list(augmentend_data_root.glob('*/rotate*'))
        all_rescale_image_paths = list(augmentend_data_root.glob('*/rescale*'))
        fig = plt.figure(figsize=(40, 40))
        index = 0
        i = 0
        while i <= columns*rows:
            print(index)
            img = load_img(all_flip_image_paths[index])
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            img = load_img(all_rescale_image_paths[index])
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            img = load_img(all_rotate_image_paths[index])
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            img = load_img(all_brightness_image_paths[index])
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            img = load_img(all_zoom_image_paths[index])
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)

            index = index+1

            if index > 3:
                break
        plt.show()
