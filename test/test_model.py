from config import *
from dataset.dataset import Dataset
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc, average_precision_score
from inspect import signature
from util.visualize_dataset import VisualizeDataset
import numpy as np

class TestModel:

    def __init__(self):
        self.vs = VisualizeDataset()

    def test(self, model_dir, label='*'):
        ds = Dataset(label)
        test_images, test_labels = ds.load_testset()

        loaded_model = None
        adam = Adam(lr=1e-6)

        with open(model_dir.joinpath('model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(str(model_dir.joinpath("model.h5")))
        print("Loaded model from disk")
        loaded_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=adam, metrics=['accuracy'])
        probs = list()
        y_hat = list()

        for i in range(len(test_labels)):
            pred = loaded_model.predict(np.array([test_images[i],]))
            label = LABEL_NAMES[test_labels[i]]

            print("%i-> %s: %.2f%%" %(i,label,float(pred[0][test_labels[i]]* 100)))

            if test_labels[i] == 0:
                if pred[0][test_labels[i]] > 0.5:
                    probs.append(0.49*pred[0][test_labels[i]])
                    y_hat.append(0)
                else:
                    probs.append(pred[0][test_labels[i]]/0.49)
                    y_hat.append(1)
                    # self.vs.show_img(test_images[i])
            else:
                if pred[0][test_labels[i]] > 0.5:
                    y_hat.append(1)
                else:
                    y_hat.append(0)
                    # self.vs.show_img(test_images[i])

                probs.append(pred[0][test_labels[i]])

        self.plot_pr_curve(test_labels, probs, y_hat)

    def plot_pr_curve(self, true_labels, probs, y_hat):
        precision, recall, thresholds = precision_recall_curve(true_labels, probs)
        ap = average_precision_score(true_labels, probs)
        prec = precision_score(true_labels, y_hat)
        rec = recall_score(true_labels, y_hat)
        area = auc(recall, precision)
        plt.plot(recall, precision, marker='o',
                 label='PR curve (area = %0.2f) \n(AP = %0.2f)' % (area, ap))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        # plot the roc curve for the model
        plt.title('Precision-Recall curve: PRECISON=%0.2f, RECALL=%0.2f' % (prec, rec))
        plt.legend(loc="lower right")
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.show()
