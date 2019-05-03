import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class Dataset:

    def __init__(self, train_dir=str, test_dir=str, width=int, height=int):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.WIDTH = width
        self.HEIGHT = height
        self.resize_func = lambda image: tf.image.resize_image_with_crop_or_pad(image, height, width)
        self.train_image_file_names = [train_dir + i for i in os.listdir(train_dir)]
        self.test_image_file_names = [test_dir + i for i in os.listdir(test_dir)]
        self.train_images = self.decode_image(self.train_image_file_names)
        self.test_images = self.decode_image(self.test_image_file_names)
        self.all_images = self.train_images + self.test_images

    def decode_image(self, image_file_names, resize_func=None):
        images = []

        graph = tf.Graph()
        with graph.as_default():
            file_name = tf.placeholder(dtype=tf.string)
            file = tf.read_file(file_name)
            image = tf.image.decode_jpeg(file)
            if resize_func != None:
                image = resize_func(image)

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            for i in range(len(image_file_names)):
                images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))

            session.close()

        return images


    def process_images(self):
        self.train_images = self.decode_image(self.train_image_file_names, resize_func=self.resize_func)
        self.test_images = self.decode_image(self.test_image_file_names, resize_func=self.resize_func)
        self.all_images = self.train_images + self.test_images

    def show_mean_aspect_ratio(self):
        aspect_ratio = []

        for image in self.all_images:
            h, w, d = np.shape(image)
            aspect_ratio.append(float(w) / float(h))

        images = [i for i in range(1, aspect_ratio.__len__()+1)]
        aspect_ratio.sort()
        plt.title('Acpect Ratio')
        plt.text(np.mean(aspect_ratio), np.min(aspect_ratio), 'mean aspect ratio = '+str(np.mean(aspect_ratio)), horizontalalignment='left', verticalalignment='bottom', fontsize=11, fontweight='bold')
        plt.xlabel('image')
        plt.ylabel('apspect ratio')
        plt.plot(images, aspect_ratio, 'r')
        plt.legend(loc='lower right')
        plt.show()

        plt.show()

    def show_mean_width_heigth(self):
        width = []
        height = []

        for image in self.all_images:
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

    def get_all_images(self):
        return self.all_images

    def get_train_images(self):
        return self.train_images

    def get_test_images(self):
        return self.test_images
