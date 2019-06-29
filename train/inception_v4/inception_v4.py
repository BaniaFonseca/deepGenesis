from config import *
from model.inception_v4.inception_v4 import Inception_v4
from dataset.dataset import Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam

class TInception_v4(Inception_v4):

    def __init__(self):
        pass

    def train(self):
        ds = Dataset(name='orig*')
        train_images, train_labels = ds.load_trainset()
        validation_images, validation_labels = ds.load_validationtest()
        mc =  ModelCheckpoint(str(INCEPTION_V4_DIR_RES.joinpath('model.h5')), monitor='val_loss',
                             mode='auto', verbose=1, save_best_only=True)

        model = self.model()

        adam = Adam(lr=1e-6)
        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


        # # serialize model to JSON
        model_json = model.to_json()
        with open(INCEPTION_V4_DIR_RES.joinpath('model.json'), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

        model.summary()
        plot_model(model, show_shapes=True, to_file=INCEPTION_V4_DIR_RES.joinpath('inception_v4.png'))

        # a = True
        # if a:
        #     return None

        history = model.fit(train_images, train_labels, epochs=200, shuffle=True, verbose=2,
                  validation_data=(validation_images, validation_labels), callbacks=[mc])

        self.test()

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
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()


    def test(self):
        ds = Dataset()
        test_images, test_labels = ds.load_testset()

        # load json and create model
        json_file = open(INCEPTION_V4_DIR_RES.joinpath('model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        loaded_model.load_weights(str(INCEPTION_V4_DIR_RES.joinpath("model.h5")))
        print("Loaded model from disk")
        # # evaluate loaded model on test data
        loaded_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate(test_images, test_labels, verbose=1)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        preds = loaded_model.evaluate(test_images, test_labels)
        print('Loss = {}'.format(str(preds[0])))
        print('Test Accuracy = {}'.format(str(preds[1])))
