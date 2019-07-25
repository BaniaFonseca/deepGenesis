from config import *
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Conv2D, \
    BatchNormalization, Flatten, MaxPooling2D, Input, Add, AvgPool2D, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import tensorflow.python.keras.backend as backend

backend.set_image_data_format('channels_last')

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_N = 384
_K = 192
_M = 256
_I = 224

class Inception_v4():

    def __init__(self):
        pass

    def model(self):
        X_input = Input((HEIGHT, WIDTH, CHANELS))
        X =  X_input
        X = self.inception_v4(X)
        X = Dense(CLASS_NUMBER, activation='softmax',
                  kernel_initializer='glorot_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='inception_v4')
        return model

    def inception_v4(self, inputs):
        inputs = self.stem(inputs)

        for _ in range(4):
            inputs = self.inception_a(inputs)

        inputs = self.reduction_a(inputs)

        for _ in range(7):
            inputs = self.inception_b(inputs)

        inputs = self.reduction_b(inputs)

        for _ in range(3):
            inputs = self.inception_c(inputs)

        inputs = Flatten()(inputs)
        inputs = Dropout(0.2) (inputs)
        return inputs

    def stem(self, inputs):
        unit_1 = self.conv(inputs, filters=32, kernel_size=3, strides=2, padding='VALID')
        unit_1 = self.conv(unit_1, filters=32, kernel_size=3, padding='VALID')
        unit_1 = self.conv(unit_1, filters=64, kernel_size=3)

        unit_2  = self.max_poll(unit_1, pool_size=(3,3), strides=2, padding='VALID')
        unit_3 = self.conv(unit_1, filters=64, kernel_size=3, strides=2, padding='VALID')

        unit_2_plus_3 = Add() ([unit_2, unit_3])

        unit_4 = self.conv(unit_2_plus_3, filters=64, kernel_size=1)
        unit_4 = self.conv(unit_4, filters=96, kernel_size=3, padding='VALID')

        unit_5 = self.conv(unit_2_plus_3, filters=64, kernel_size=1)
        unit_5 = self.conv(unit_5, filters=64, kernel_size=(7,1))
        unit_5 = self.conv(unit_5, filters=64, kernel_size=(1, 7))
        unit_5 = self.conv(unit_5, filters=96, kernel_size=3, padding='VALID')

        unit_4_plus_5 = Add() ([unit_4, unit_5])

        unit_6 = self.conv(unit_4_plus_5, filters=96, kernel_size=3, strides=2, padding='VALID')
        unit_7 = self.max_poll(unit_4_plus_5, pool_size=(3,3), strides=2, padding='VALID')

        unit_6_plus_7 = Add()([unit_6, unit_7])

        return unit_6_plus_7

    def inception_a(self, inputs):
        unit_1 = AvgPool2D(padding='SAME', pool_size=(1,1))(inputs)
        unit_1 =  self.conv(unit_1, filters=96, kernel_size=1)

        unit_2 = self.conv(inputs, filters=96, kernel_size=1)

        unit_3 = self.conv(inputs, filters=64, kernel_size=1)
        unit_3 = self.conv(unit_3, filters=96, kernel_size=3)

        unit_4 = self.conv(inputs, filters=64, kernel_size=1)
        unit_4 = self.conv(unit_4, filters=96, kernel_size=3)
        unit_4 = self.conv(unit_4, filters=96, kernel_size=1)

        return Add () ([unit_1, unit_2, unit_3, unit_4])

    def reduction_a(self, inputs):
        unit_1 = self.max_poll(inputs, pool_size=(3,3), strides=2, padding='VALID')

        unit_2 = self.conv(inputs, kernel_size=3, strides=2, padding='VALID', filters=96)

        unit_3 = self.conv(inputs, kernel_size=1, filters=_K)
        unit_3 = self.conv(unit_3, kernel_size=3, filters=_I)
        unit_3 = self.conv(unit_3, kernel_size=3, strides=2, filters=96, padding='VALID')

        return Add() ([unit_1, unit_2, unit_3])

    def inception_b(self, inputs):
        unit_1 = AvgPool2D(padding='SAME', pool_size=(1,1))(inputs)
        unit_1 =  self.conv(unit_1, filters=256, kernel_size=1)

        unit_2 = self.conv(inputs, filters=256, kernel_size=1)

        unit_3 = self.conv(inputs, filters=192, kernel_size=1)
        unit_3 = self.conv(unit_3, filters=224, kernel_size=(1,7))
        unit_3 = self.conv(unit_3, filters=256, kernel_size=(1,7))

        unit_4 = self.conv(inputs, filters=192, kernel_size=1)
        unit_4 = self.conv(unit_4, filters=192, kernel_size=(1,7))
        unit_4 = self.conv(unit_4, filters=224, kernel_size=(7,1))
        unit_4 = self.conv(unit_4, filters=224, kernel_size=(1,7))
        unit_4 = self.conv(unit_4, filters=256, kernel_size=(7,1))

        return Add () ([unit_1, unit_2, unit_3, unit_4])

    def reduction_b(self, inputs):
        unit_1 = self.max_poll(inputs, pool_size=3, strides=2, padding='VALID')

        unit_2 = self.conv(inputs, filters=192, kernel_size=1)
        unit_2 = self.conv(unit_2, filters=256, kernel_size=3, strides=2, padding='VALID')

        unit_3 = self.conv(inputs, filters=256, kernel_size=1)
        unit_3 = self.conv(unit_3, filters=256, kernel_size=(1, 7))
        unit_3 = self.conv(unit_3, filters=320, kernel_size=(7, 1))
        unit_3 = self.conv(unit_3, filters=256, kernel_size=3, strides=2, padding='VALID')

        return Add()([unit_1, unit_2, unit_3])

    def inception_c(self, inputs):
        unit_1 = AvgPool2D(padding='SAME', pool_size=(1,1))(inputs)
        unit_1 =  self.conv(unit_1, filters=256, kernel_size=1)

        unit_2 = self.conv(inputs, filters=256, kernel_size=1)

        unit_3 = self.conv(inputs, filters=384, kernel_size=1)
        unit_3_1 = self.conv(unit_3, filters=256, kernel_size=(1,3))
        unit_3_2 = self.conv(unit_3, filters=256, kernel_size=(3,1))

        unit_4 = self.conv(inputs, filters=384, kernel_size=1)
        unit_4 = self.conv(unit_4, filters=448, kernel_size=(1,3))
        unit_4 = self.conv(unit_4, filters=512, kernel_size=(3,1))
        unit_4_1 = self.conv(unit_4, filters=256, kernel_size=(3, 1))
        unit_4_2 = self.conv(unit_4, filters=256, kernel_size=(1, 3))

        return Add () ([unit_1, unit_2, unit_3_1, unit_3_2, unit_4_1, unit_4_2])

    def conv(self, inputs, filters, kernel_size, strides=1, padding='SAME'):
        inputs =  Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer='glorot_uniform',
                        use_bias=False)(inputs)
        inputs = LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        return inputs

    def max_poll(self, inputs, pool_size, strides=1, padding='SAME'):
        inputs = MaxPooling2D(pool_size=pool_size, strides=strides,
                              padding=padding)(inputs)
        return inputs
