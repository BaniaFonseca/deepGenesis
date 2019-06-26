from config import *
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Conv2D, \
    BatchNormalization, Flatten, MaxPooling2D, Input
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

class Inception_v4():

    def __init__(self):
        pass

    def model(self):
        X_input = Input((HEIGHT, WIDTH, CHANELS))
        X =  X_input
        X = self.inception_v4(X)
        X = Flatten()(X)
        X = Dense(CLASS_NUMBER, activation='softmax',
                       kernel_initializer='glorot_uniform')(X)
        model = Model(inputs=X_input, outputs=X, name='inception_v4')
        return model

    def inception_v4(self, inputs):
        return inputs