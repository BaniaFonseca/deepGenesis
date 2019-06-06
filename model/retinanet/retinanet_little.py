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
_MODEL_SIZE = (HEIGHT, WIDTH)

class RetinaNet():

    def __init__(self):
        pass

    def model(self):
        X_input = keras.layers.Input((HEIGHT, WIDTH, CHANELS))
        X = X_input

        X = self.fpn(X)

        # Dense layer
        X = Flatten()(X)

        X = Dense(CLASS_NUMBER, activation='softmax',
                  kernel_initializer='glorot_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='RetinaNet')
        return model

    def fpn(self, inputs):

        c3, c4, c5 = self.res_net_50(inputs)
        
        m5 = c5
        p5 = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False)(m5)
        p5 = self.batch_norm(p5)
        p5 = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(p5)

        cv_c4 = Conv2D(filters=1024, kernel_size=1, strides=2, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False)(c4)
        cv_c4 = self.batch_norm(cv_c4)
        cv_c4 = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(cv_c4)
        m4 = keras.layers.Add()([m5, cv_c4])
        p4 = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid',
                    kernel_initializer='glorot_uniform', use_bias=False)(m4)
        p4 = self.batch_norm(p4)
        p4 = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(p4)

        cv_c3 = Conv2D(filters=1024, kernel_size=1, strides=2, padding='valid',
                       kernel_initializer='glorot_uniform', use_bias=False)(c3)
        cv_c3 = self.batch_norm(cv_c3)
        cv_c3 = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(cv_c4)
        m3 = keras.layers.Add()([m4, cv_c3])
        p3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid',
                    kernel_initializer='glorot_uniform', use_bias=False)(m3)
        p3 = self.batch_norm(p3)
        p3 = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(p3)


        return keras.layers.Add()([p3, p4, p5])


    def res_net_50(self, inputs):
        #1
        inputs = Conv2D(filters=32, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        #2
        inputs = MaxPooling2D(pool_size=(2, 2), strides=2, padding='SAME') (inputs)
        #11
        inputs = self.conv_II(inputs)
        #23
        inputs = self.conv_III(inputs)
        c3 = inputs
        #41
        inputs = self.conv_IV(inputs)
        c4 = inputs
        #50
        inputs = self.conv_V(inputs)
        c5 = inputs
        return c3, c4, c5

    def batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        return keras.layers.BatchNormalization (axis=3, momentum=_BATCH_NORM_DECAY,
                                    epsilon=_BATCH_NORM_EPSILON, scale=True)(inputs)

    def conv_II(self, inputs):
        shortcut = inputs
        for _ in range(3):
            #1
            inputs = Conv2D(filters=32, kernel_size=1, strides=1,  padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            #2
            inputs = Conv2D(filters=32, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            #3
            inputs = Conv2D(filters=128, kernel_size=1, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            shortcut = Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        return inputs

    def conv_III(self, inputs):
        shortcut = inputs
        for i in range(1, 2+1):
            #1
            inputs = Conv2D(filters=64, kernel_size=1, strides=self.get_strides(i),
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            #2
            inputs = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            #3
            inputs = Conv2D(filters=128, kernel_size=1, strides=1, padding='SAME',
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            # #adjust shape with convolution
            shortcut = Conv2D(filters=64, kernel_size=1, strides=self.get_strides(i), padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)
            shortcut = Conv2D(filters=128, kernel_size=3, strides=1,
                               padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        shortcut = inputs
        for _ in range(2):
            # 1
            inputs = Conv2D(filters=64, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 2
            inputs = Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 3
            inputs = Conv2D(filters=128, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        return inputs

    def conv_IV(self, inputs):
        shortcut = inputs
        for i in range(1, 2 + 1):
            # 1
            inputs = Conv2D(filters=128, kernel_size=1, strides=self.get_strides(i),
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 2
            inputs = Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 3
            inputs = Conv2D(filters=512, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            # adjust shape with convolution
            shortcut = Conv2D(filters=128, kernel_size=1, strides=self.get_strides(i), padding='SAME',
                              kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)
            shortcut = Conv2D(filters=512, kernel_size=3, strides=1,
                              padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        shortcut = inputs
        for _ in range(4):
            # 1
            inputs = Conv2D(filters=256, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 2
            inputs = Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 3
            inputs = Conv2D(filters=512, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        return inputs

    def conv_V(self, inputs):
        shortcut = inputs
        for i in range(1, 2 + 1):
            # 1
            inputs = Conv2D(filters=512, kernel_size=1, strides=self.get_strides(i),
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 2
            inputs = Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 3
            inputs = Conv2D(filters=1024, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            # adjust shape with convolution
            shortcut = Conv2D(filters=512, kernel_size=1, strides=self.get_strides(i), padding='SAME',
                              kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)
            shortcut = Conv2D(filters=1024, kernel_size=3, strides=1,
                              padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(shortcut)
            shortcut = self.batch_norm(shortcut)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        shortcut = inputs
        for _ in range(1):
            # 1
            inputs = Conv2D(filters=512, kernel_size=1, strides=1,
                            padding='SAME', kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 2
            inputs = Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            # 3
            inputs = Conv2D(filters=1024, kernel_size=1, strides=1, padding='SAME',
                            kernel_initializer='glorot_uniform', use_bias=False)(inputs)
            inputs = self.batch_norm(inputs)

            inputs = keras.layers.Add()([inputs, shortcut])
            inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
            shortcut = inputs

        return inputs

    def get_strides(self, iteration):
        if iteration%2 == 1:
            return 1
        else:
            return 2