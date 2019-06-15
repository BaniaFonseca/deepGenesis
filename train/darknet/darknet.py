from config import *
from model.darknet.darknet import DarkNet
from dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
import tensorflow as tf
from keras.callbacks import *
from keras.optimizers import SGD
from keras.models import model_from_json

class TDarknet(DarkNet):

    def __init__(self):
        pass

    def train(self):
        ds = Dataset()
        train_images, train_labels = ds.get_trainset()
        validation_images, validation_labels = ds.get_validationset()

        mc = ModelCheckpoint(str(DARKNET_DIR_RES.joinpath('model.h5')), monitor='val_loss',
                              mode='auto', verbose=1, save_best_only=True)
        model = self.model()

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        plot_model(model, show_shapes=True, to_file=DARKNET_DIR_RES.joinpath('darknet_v3.png'))

        model_json = model.to_json()
        with open(DARKNET_DIR_RES.joinpath('model.json'), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

        # a = True
        # if a:
        #     return None

        history = model.fit(train_images,train_labels, epochs=80,shuffle=False,
                  validation_data=(validation_images, validation_labels), callbacks=[mc])
        #callbacks=[es, mc]
        self.test()

        #[3xrescale#1, 2xzoom#0, 3xbrightness#1, 3Xflip_horizontal-1, 1xrotate]
        #augmentation = '[zoom]'

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss:')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def test(self):
        ds = Dataset()
        test_images, test_labels = ds.get_testset()

        # load json and create model
        json_file = open(DARKNET_DIR_RES.joinpath('model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(DARKNET_DIR_RES.joinpath("model.h5"))
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(test_images, test_labels, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        preds = loaded_model.evaluate(test_images, test_labels)
        print('Loss = {}'.format(str(preds[0])))
        print('Test Accuracy = {}'.format(str(preds[1])))
