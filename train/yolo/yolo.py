from config import *
from model.yolo.yolo import Yolo
from dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
import tensorflow as tf

class TYolo(Yolo):

    def __init__(self):
        print(tf.__version__)

    def train(self):
        ds = Dataset()
        train_images, train_labels = ds.get_trainset()
        train_labels = to_categorical(train_labels)
        test_images, test_labels = ds.get_testset()
        test_labels = to_categorical(test_labels)

        model = self.model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        plot_model(model, show_shapes=True, to_file=YOLO_DIR_RES.joinpath('yolo_v3.png'))

        a = True
        if a:
            return None

        history = model.fit(train_images, train_labels, epochs=3,
                  validation_data=(test_images, test_labels))

        # SVG(model_to_dot(model).create(prog='dot', format ='svg'))

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

        # serialize model to JSON
        model_json = model.to_json()
        with open(YOLO_DIR_RES.joinpath('model.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(YOLO_DIR_RES.joinpath('model.h5'))
        print("Saved model to disk")

        # later...

        # load json and create model
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights("model.h5")
        # print("Loaded model from disk")

        # evaluate loaded model on test data
        # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # score = loaded_model.evaluate(X, Y, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        # preds = model.evaluate(test_images, test_labels)
        # print('Loss = {}'.format(str(preds[0])))
        # print('Test Accuracy = {}'.format(str(preds[1])))

