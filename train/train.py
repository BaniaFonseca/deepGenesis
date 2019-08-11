import tensorflow as tf
from config import *
from dataset.dataset import Dataset
from threading import Thread
from test.test_model import TestModel
from dataset.data_processing import DataProcessing
from util.visualize_dataset import VisualizeDataset
import matplotlib.pyplot as plt
from pathlib import Path
from test.test_model import TestModel
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np

class Train:

    def __init__(self):
        self.test_model = None
        self.plot_train_history = None
        self.history = None
        self.dp = DataProcessing()
        self.vs = VisualizeDataset()


    def start_training(self, model, retrain=False, model_dir=None):
        ds = Dataset()
        X, Y = ds.load_trainset()
        kfold_splits = 5

        # [(1, 7), (6, 1097), (11, 2017), (16, 2027), (21, 2087), (26, 2137)]
        # [6, 11, 16, 21, 26], [1097, 2017, 2027, 2087, 2137]
        for fold, seed in zip([6, 11, 16, 21, 26], [1097, 2017, 2027, 2087, 2137]):
            skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=seed)
            for k, (train, test) in enumerate(skf.split(X, Y), fold):
                prefix = "fold_" + str(k) + "_"
                print('fold:{}\n'.format(k))

                for j, (lr, epochs) in enumerate([(1e-4, 40)]):
                    if j == 0:
                        self.history = model.train(train_images=X[train], train_labels=Y[train],validation_images=X[test],
                                validation_labels=Y[test], retrain=retrain, prefix=prefix, lr=lr, epochs=epochs)

                        plt.close()
                        self.plot_train_history = PlotTrainHistory(history=self.history,                                                               model_dir=model_dir, prefix=prefix)
                        self.plot_train_history.run()
                    else:
                        self.history = model.train(train_images=X[train], train_labels=Y[train], validation_images=X[test],
                                validation_labels=Y[test], retrain=True, prefix=prefix, lr=lr, epochs=epochs)

                    self.test_model = TestModel(model_dir=model_dir,test_images=X[test],test_labels=Y[test], prefix=prefix)
                    self.test_model.run()
                    plt.close()
            print("Done! {}\n".format(fold))

class PlotTrainHistory(Thread):

    def __init__(self, history, model_dir=None, prefix=""):
        super(PlotTrainHistory, self).__init__()
        self.history = history
        self.model_dir = model_dir
        self.prefix = prefix

    def run(self):

        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(self.model_dir.joinpath(self.prefix + Path(self.model_dir).parent.name + '_acc.jpeg'))
        plt.close()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(self.model_dir.joinpath(self.prefix + Path(self.model_dir).parent.name + '_loss.jpeg'))
        plt.close()
