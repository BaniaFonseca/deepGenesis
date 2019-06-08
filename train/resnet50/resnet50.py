from config import *
from model.resnet50.resnet50 import ResNet50
from dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import model_from_json
import tensorflow as tf
from keras.callbacks import *

class TResNet50(ResNet50):

    def __init__(self):
        print(tf.__version__)

    def train(self):
        ds = Dataset()
        train_images, train_labels = ds.get_trainset()
        #train_labels = to_categorical(train_labels)
        test_images, test_labels = ds.get_testset()
        #test_labels = to_categorical(test_labels)

        mc = ModelCheckpoint(str(RESNET50_DIR_RES.joinpath('model.h5')), monitor='val_acc',
                             mode='auto', verbose=1, save_best_only=True)

        model = self.model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # # serialize model to JSON
        model_json = model.to_json()
        with open(RESNET50_DIR_RES.joinpath('model.json'), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

        model.summary()
        plot_model(model, show_shapes=True, to_file=RESNET50_DIR_RES.joinpath('resnet34.png'))

        # a = True
        # if a:
        #     return None

        history = model.fit(train_images, train_labels, epochs=40,
                  validation_data=(test_images, test_labels), callbacks=[mc])

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


    def test(self):
        ds = Dataset()
        test_images, test_labels = ds.get_testset()

        # load json and create model
        json_file = open(RESNET50_DIR_RES.joinpath('model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        loaded_model.load_weights(RESNET50_DIR_RES.joinpath("model.h5"))
        print("Loaded model from disk")
        # # evaluate loaded model on test data
        loaded_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(test_images, test_labels, verbose=1)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        preds = loaded_model.evaluate(test_images, test_labels)
        print('Loss = {}'.format(str(preds[0])))
        print('Test Accuracy = {}'.format(str(preds[1])))
