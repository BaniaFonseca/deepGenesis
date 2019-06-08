from config import *
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, \
    ZeroPadding2D, BatchNormalization, Flatten, MaxPooling2D, UpSampling2D
import keras.initializers
import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

class ResNet50():

    def __init__(self):
        pass

    def model(self):
        X_input = keras.layers.Input((HEIGHT, WIDTH, CHANELS))
        X = X_input
        X = self.resnet_50(X)
        X = Flatten(name='FC') (X)
        X = Dense(CLASS_NUMBER, activation='softmax',
                  kernel_initializer='glorot_uniform')(X)
        model = Model(inputs=X_input, outputs=X, name='RetinaNet')
        return model

    def resnet_50(self, inputs):
        inputs = Conv2D(filters=64, kernel_size=7, strides=2, padding='VALID', name='conv_1',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = MaxPooling2D(pool_size=(3,3), strides=2)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = Conv2D(filters=256, kernel_size=1, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_II(inputs)

        inputs = Conv2D(filters=512, kernel_size=3, strides=2, padding='VALID',
                        name='conv_II_POLL',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_III(inputs)

        inputs = Conv2D(filters=1024, kernel_size=3, strides=2, padding='VALID',
                        name='conv_III_POLL',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_IV(inputs)

        inputs = Conv2D(filters=2048, kernel_size=3, strides=2, padding='VALID',
                        name='conv_IV_POLL',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)

        inputs = self.batch_norm(inputs)

        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        inputs = self.conv_V(inputs)

        inputs = Conv2D(filters=2024, kernel_size=3, strides=2, padding='VALID',
                        name='conv_V_POLL',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        return inputs

    def conv_II(self, inputs):
        shortcut = inputs
        for i in range(3):
            inputs = Conv2D(filters=64, kernel_size=1, strides=1,  padding='SAME',
                            name='conv_3'+str(i+1) ,kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=256, kernel_size=1, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_III(self, inputs):
        shortcut = inputs
        for _ in range(4):
            inputs = Conv2D(filters=128, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=128, kernel_size=3, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=512, kernel_size=1, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_IV(self, inputs):
        shortcut = inputs
        for _ in range(6):
            inputs = Conv2D(filters=256, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=256, kernel_size=3, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = Conv2D(filters=1024, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_V(self, inputs):
        shortcut = inputs
        for _ in range(3):
            inputs = Conv2D(filters=512, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=512, kernel_size=3, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=2048, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        return keras.layers.BatchNormalization (axis=3, momentum=_BATCH_NORM_DECAY,
                                    epsilon=_BATCH_NORM_EPSILON, scale=True)(inputs)
