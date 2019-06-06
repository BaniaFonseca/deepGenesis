from config import *
from model.retinanet.retinanet import RetinaNet
from dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import model_from_json
import tensorflow as tf
from keras.callbacks import *

_WEIGHT_DECAY = 0.0001
_LEAKY_RELU = 0.1
_LEARNING_RATE = 0.01
_MOMEMTUM = 0.9
_MODEL_SIZE = (HEIGHT, WIDTH)

class TRetinaNet(RetinaNet):

    def __init__(self):
        print(tf.__version__)

    def train(self):
        ds = Dataset()
        train_images, train_labels = ds.get_trainset()
        train_labels = to_categorical(train_labels)
        test_images, test_labels = ds.get_testset()
        test_labels = to_categorical(test_labels)

        mc = ModelCheckpoint(str(RETINANET_DIR_RES.joinpath('best_model.h5')), monitor='val_loss', mode='min', verbose=1)
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                           verbose=0, mode='auto', baseline=None)

        model = self.model()
        # sgd = optimizers.SGD(lr=_LEARNING_RATE, decay=_WEIGHT_DECAY,
        #                      momentum=_MOMEMTUM)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])


        model.summary()
        plot_model(model, show_shapes=True, to_file=RETINANET_DIR_RES.joinpath('retinanet.png'))

        # a = True
        # if a:
        #     return None

        history = model.fit(train_images, train_labels, epochs=10,
                  validation_data=(test_images, test_labels), callbacks=[es, mc])

        # SVG(model_to_dot(model).create(prog='dot', format ='svg'))
        #
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
        #
        # # serialize model to JSON
        model_json = model.to_json()
        with open(RETINANET_DIR_RES.joinpath('model.json'), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(RETINANET_DIR_RES.joinpath('model.h5'))
        print("Saved model to disk")

        # later...

        # # load json and create model
        # json_file = open(RETINANET_DIR_RES.joinpath('model.json'), 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # # load weights into new model
        # loaded_model.load_weights(RETINANET_DIR_RES.joinpath("model.h5"))
        # print("Loaded model from disk")
        #
        # # evaluate loaded model on test data
        # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # score = loaded_model.evaluate(test_images, test_labels, verbose=0)
        # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        # preds = model.evaluate(test_images, test_labels)
        # print('Loss = {}'.format(str(preds[0])))
        # print('Test Accuracy = {}'.format(str(preds[1])))

