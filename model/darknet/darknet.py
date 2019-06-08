from config import *
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, ZeroPadding2D, BatchNormalization, Flatten
import keras.initializers
import keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

class DarkNet():

    def __init__(self):
        pass

    def model(self):
        X_input = keras.layers.Input((HEIGHT, WIDTH, CHANELS))
        X =  X_input
        X = self.darknet53(X)
        X = Flatten()(X)
        X = Dense(CLASS_NUMBER, activation='softmax',
                       kernel_initializer='glorot_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='darknet_53')
        return model

    def batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        return keras.layers.BatchNormalization (axis=3, momentum=_BATCH_NORM_DECAY,
                                    epsilon=_BATCH_NORM_EPSILON, scale=True)(inputs)

    def darknet53(self, inputs):
        inputs = Conv2D(filters=32, kernel_size=3, strides=1, padding='VALID',
                        name='CONV_OI', kernel_initializer='glorot_uniform',
                        use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = Conv2D(filters=64, kernel_size=3, strides=2, padding='VALID',
                        name='CONV_OII', kernel_initializer='glorot_uniform',
                        use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_I(inputs)

        inputs = Conv2D(filters=128, kernel_size=3, strides=2, padding='VALID',
                        kernel_initializer='glorot_uniform',use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_II(inputs)

        inputs = Conv2D(filters=256, kernel_size=3, strides=2, padding='VALID',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_III(inputs)

        inputs = Conv2D(filters=512, kernel_size=3, strides=2, padding='VALID',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_IV(inputs)

        inputs = Conv2D(filters=1024, kernel_size=3, strides=2, padding='VALID',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        inputs = self.conv_V(inputs)
        return inputs

    def conv_I(self, inputs):
        shortcut = inputs
        inputs = Conv2D(filters=32, kernel_size=1, strides=1,  padding='SAME',
                            name='CONV_I',kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        inputs = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.Add()([inputs, shortcut])
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        return inputs

    def conv_II(self, inputs):
        shortcut = inputs
        for i in range(2):
            inputs = Conv2D(filters=64, kernel_size=1, strides=1,  padding='SAME',
                            name='CONV_II'+str(i+1) ,kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_III(self, inputs):
        shortcut = inputs
        for i in range(8):
            inputs = Conv2D(filters=128, kernel_size=1, strides=1,  padding='SAME',
                            name='CONV_III'+str(i+1) ,kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_IV(self, inputs):
        shortcut = inputs
        for i in range(8):
            inputs = Conv2D(filters=256, kernel_size=1, strides=1,  padding='SAME',
                            name='CONV_IV'+str(i+1) ,kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs

    def conv_V(self, inputs):
        shortcut = inputs
        for i in range(4):
            inputs = Conv2D(filters=512, kernel_size=1, strides=1,  padding='SAME',
                            name='CONV_V'+str(i+1) ,kernel_initializer='glorot_uniform',
                            use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

            inputs = Conv2D(filters=1024, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs
        return inputs