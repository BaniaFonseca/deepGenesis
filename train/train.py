import tensorflow as tf
from config import *
from dataset.dataset import Dataset
from threading import Thread
import matplotlib.pyplot as plt
from test.test_model import TestModel

class ROC(Thread):

    def __init__(self, model_dir):
        super(ROC, self).__init__()
        self.model_dir = model_dir
        self.test_model = TestModel()

    def run(self):
        self.test_model.test(self.model_dir)


class Train:

    def __init__(self):
        pass

    def start_training(self, model, retrain=False, model_dir=None):
        ds = Dataset()
        ds.save_trainset_as_npy(label='empty')
        train_images, train_labels = ds.load_trainset()
        batch_size = len(train_images)
        chunks = int((len(train_images)/batch_size))
        min = 0

        for i in range(chunks):
            max = ((i + 1) * batch_size)
            batch_images = train_images[min:max]
            batch_labels = train_labels[min:max]
            min = max

            print('batch {}: batch size: {}'.format(i, len(batch_images)))
            if i > 0:
                model.train(batch_images, batch_labels, retrain=True)
            else:
                model.train(batch_images, batch_labels, retrain)


class PlotTrainHistory(Thread):

    def __init__(self, history):
        super(PlotTrainHistory, self).__init__()
        self.history = history

    def run(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()