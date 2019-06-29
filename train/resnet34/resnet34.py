from config import *
from model.resnet34.resnet34 import ResNet34
from dataset.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import model_from_json
import tensorflow as tf
from keras.callbacks import *
from keras.optimizers import Adam

class TResNet34(ResNet34):

    def __init__(self):
        pass

    def train(self, retrain=False, test=False):

        if test:
            self.test()
            return None

        ds = Dataset()
        train_images, train_labels = ds.load_trainset()
        validation_images, validation_labels = ds.load_validationtest()
        #test_labels = to_categorical(test_labels)

        mc = ModelCheckpoint(str(RESNET34_DIR_RES.joinpath('model.h5')), monitor='val_loss',
                             mode='auto', verbose=1, save_best_only=True)

        model = None
        adam = Adam(lr=1e-6)

        if not retrain:
            model = self.model()
            model_json = model.to_json()
            with open(RESNET34_DIR_RES.joinpath('model.json'), "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")
        else:
            print('retraining...')
            with open(RESNET34_DIR_RES.joinpath('model.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
                model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(RESNET34_DIR_RES.joinpath("model.h5"))
            print("Loaded model from disk")

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()
        plot_model(model, show_shapes=True, to_file=RESNET34_DIR_RES.joinpath('resnet34.png'))

        # a = True
        # if a:
        #     return None

        history = model.fit(train_images, train_labels, epochs=80,
                  validation_data=(validation_images, validation_labels), callbacks=[mc], verbose=2)

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


    def test(self, classe='*'):
        ds = Dataset(classe)
        test_images, test_labels = ds.load_testset()

        loaded_model = None
        adam = Adam(lr=1e-6)
        # load json and create model
        with open(RESNET34_DIR_RES.joinpath('model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        loaded_model.load_weights(RESNET34_DIR_RES.joinpath("model.h5"))
        print("Loaded model from disk")
        # # evaluate loaded model on test data
        loaded_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=adam, metrics=['accuracy'])
        score = loaded_model.evaluate(test_images, test_labels, verbose=1)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
        preds = loaded_model.evaluate(test_images, test_labels)
        print('Loss = {}'.format(str(preds[0])))
        print('Test Accuracy = {}'.format(str(preds[1])))


        for i in range(len(test_labels)):
            probs = loaded_model.predict(np.array([test_images[i],]))
            label = LABEL_NAMES[test_labels[i]]
            print('{} -> {}'.format(label, probs))