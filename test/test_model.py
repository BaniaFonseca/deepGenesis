from config import *
from dataset.dataset import Dataset
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import auc, average_precision_score, f1_score, confusion_matrix
from inspect import signature
from util.visualize_dataset import VisualizeDataset
import numpy as np
from pathlib import Path
from threading import Thread
import os

class TestModel(Thread):

    def __init__(self, model_dir, test_images, test_labels,label='*', prefix=""):
        super(TestModel, self).__init__()
        self.vs = VisualizeDataset()
        self.model_dir = model_dir
        self.test_images = test_images
        self.test_labels = test_labels
        self.label = label
        self.prefix =  prefix

    def run(self):
        loaded_model = None
        adam = Adam(lr=1e-5)

        with open(self.model_dir.joinpath('model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(str(self.model_dir.joinpath(self.prefix + "model.h5")))
        print("Loaded model from disk")
        loaded_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=adam, metrics=['accuracy'])
        probs = list()
        y_hat = list()

        for i in range(len(self.test_labels)):
            pred = loaded_model.predict(np.array([self.test_images[i], ]))
            label = LABEL_NAMES[self.test_labels[i]]

            print("%i-> %s: %.2f%%" % (i, label, float(pred[0][self.test_labels[i]] * 100)))

            if self.test_labels[i] == 0:
                if pred[0][self.test_labels[i]] > 0.5:
                    probs.append(0.49 * pred[0][self.test_labels[i]])
                    y_hat.append(0)
                else:
                    probs.append(pred[0][self.test_labels[i]] / 0.49)
                    y_hat.append(1)
                    # self.vs.show_img(test_images[i])
            else:
                if pred[0][self.test_labels[i]] > 0.5:
                    y_hat.append(1)
                    # self.vs.show_img(test_images[i])
                else:
                    y_hat.append(0)
                    # \self.vs.show_img(test_images[i])

                probs.append(pred[0][self.test_labels[i]])

        self.plot_pr_curve(self.test_labels, probs, y_hat, self.model_dir, self.prefix)
        os.remove(str(self.model_dir.joinpath(self.prefix + "model.h5")))

    def plot_pr_curve(self, true_labels, probs, y_hat, model_dir=None, prefix=""):
        fig = plt.figure()
        fig.suptitle(model_dir.joinpath(Path(model_dir)).parent.name, fontsize=12, fontweight='bold')
        ax = fig.add_subplot(111)

        precision, recall, thresholds = precision_recall_curve(true_labels, probs)
        ap = average_precision_score(true_labels, probs)
        p = precision_score(true_labels, y_hat)
        r = recall_score(true_labels, y_hat)
        pr_auc = auc(recall, precision)
        acc = accuracy_score(true_labels, y_hat)
        plt.title("Precision-Recall Curve")
        f1 = f1_score(true_labels, y_hat)
        cm = confusion_matrix(true_labels, y_hat)

        plt.plot(recall, precision, marker='o',
                 label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.plot([0, 1], [0.5, 0.5], linestyle='-', color='r')

        results = 'PR-AUC = {0:.3f}\n'.format(pr_auc)
        results += 'F1 = {0:.3f}\n'.format(f1)
        results += 'PRECISION = {0:.3f}\n'.format(p)
        results += 'RECALL = {0:.3f}\n'.format(r)
        results += 'ACC = {0:.3f}\n'.format(acc)
        results +='TN = {}, FP = {}, FN = {}, TP = {}\n'.format(cm[0,0], cm[0,1], cm[1,0], cm[1,1])

        plt.text(0.05, 0.05,results, fontsize=12,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform = ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7))

        plt.legend(loc="lower right")
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        if not (model_dir is None):
            plt.savefig(model_dir.joinpath( prefix+Path(model_dir).parent.name+'_PR_curve.jpeg'))

        # plt.show()
