import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from dataset.dataset import Dataset

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
