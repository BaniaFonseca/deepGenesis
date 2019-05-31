from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pathlib
from matplotlib import pyplot as plt
import tensorflow as tf

class VisualizeDataset:
    def __init__(self):
        pass

    def show_mean_aspect_ratio(self, path):
        path = pathlib.Path(path)
        all_image_paths = \
            [str(path) for path in list(path.glob('*/*'))]
        aspect_ratio = []

        for path in all_image_paths:
            img = load_img(path)
            img = img_to_array(img)
            h, w, d = img.shape
            aspect_ratio.append(float(w) / float(h))

        images = [i for i in range(1, aspect_ratio.__len__()+1)]
        aspect_ratio.sort()
        plt.suptitle('Acpect Ratio')
        plt.title('mean aspect ratio = '+str(np.mean(aspect_ratio)))
        plt.xlabel('image')
        plt.ylabel('apspect ratio')
        plt.plot(images, aspect_ratio, 'r')
        plt.show()

    def show_mean_width_heigth(self, path):
        path = pathlib.Path(path)
        all_image_paths = \
            [str(path) for path in list(path.glob('*/*'))]
        width = []
        height = []

        for path in all_image_paths:
            img = load_img(path)
            img = img_to_array(img)
            h, w, d = img.shape
            width.append(w)
            height.append(h)

        height.sort()
        width.sort()
        plt.title('Mean height = ' +
            str(np.mean(height))+' and '+'Mean width = ' + str(np.mean(width)))
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.plot(width, height, '.r', label='Width x Height')
        plt.show()

    def show_images(self, dataset, cols=6, rows=3):
        # create a one-shot iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        # extract an element
        next_element = iterator.get_next()
        with tf.compat.v1.Session() as session:
            plt.figure(figsize=(32, 32))
            try:
                for n in range(cols*rows):
                    image = session.run(next_element)
                    plt.subplot(rows, cols, n + 1)
                    plt.imshow(image[0])
                    plt.title(image[1])
                plt.show()
                session.close()
            finally:
                session.close()