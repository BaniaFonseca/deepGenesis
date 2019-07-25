from config import *
from model.inception_v4.inception_v4 import Inception_v4
from dataset.dataset import Dataset
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.optimizers import Adam

class TInception_v4(Inception_v4):

    def __init__(self):
        pass

    def train(self, train_images, train_labels, retrain=False):
        ds = Dataset()
        validation_images, validation_labels = ds.load_validationtest()
        mc =  ModelCheckpoint(str(INCEPTION_V4_DIR_RES.joinpath('model.h5')), monitor='val_loss',
                             mode='auto', verbose=1, save_best_only=True)

        model = None
        adam = Adam(lr=1e-6)

        if not retrain:
            model = self.model()
            model_json = model.to_json()
            with open(INCEPTION_V4_DIR_RES.joinpath('model.json'), "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")
        else:
            print('retraining...')
            with open(INCEPTION_V4_DIR_RES.joinpath('model.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
                model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(str(INCEPTION_V4_DIR_RES.joinpath("model.h5")))
            print("Loaded model from disk")

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])


        # model.summary()
        # plot_model(model, show_shapes=True, to_file=INCEPTION_V4_DIR_RES.joinpath('inception_v4.png'))

        history = model.fit(train_images, train_labels, epochs=40, verbose=2,
                  validation_data=(validation_images, validation_labels), callbacks=[mc])
