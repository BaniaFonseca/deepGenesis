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
_MODEL_SIZE = (HEIGHT, WIDTH)

class Yolo():

    def __init__(self):
        pass

    def model(self):
        X_input = keras.layers.Input((HEIGHT, WIDTH, CHANELS))
        X =  X_input

        X = self.darknet53(X)
        X = self.yolo_convolution_block(X, filters=512)

        #Dense layer
        X = Flatten()(X)

        X = Dense(CLASS_NUMBER, activation='softmax',
                       kernel_initializer='glorot_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='Yolo_v3')
        return model

    def batch_norm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        return keras.layers.BatchNormalization (axis=3, momentum=_BATCH_NORM_DECAY,
                                    epsilon=_BATCH_NORM_EPSILON, scale=True)(inputs)

    def darknet53_residual_block(self, inputs, filters, strides=1):
        """Creates a residual block for Darknet."""

        shortcut = inputs

        #1
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=filters, kernel_size=1, strides=strides)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU) (inputs)

        #2
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                filters=2*filters, kernel_size=3, strides=strides)
        inputs = self.batch_norm(inputs)

        #Add
        inputs = keras.layers.Add() ([inputs, shortcut])
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        return inputs

    def fixed_padding(self, inputs, kernel_size):
        """ResNet implementation of fixed padding.
        Pads the input along the spatial dimensions independently of input size.
        Args:
            inputs: Tensor input to be padded.
            kernel_size: The kernel to be used in the conv2d or max_pool2d.
            data_format: The input format.
        Returns:
            A tensor with the same format as the input.
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs =  tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])

        return inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides=1):
        """Strided 2-D convolution with explicit padding."""
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
                        kernel_initializer='glorot_uniform', use_bias=False)(inputs)

        return inputs

    def darknet53(self, inputs):
        """Creates Darknet53 model for feature extraction."""
        # 1
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                            filters=32, kernel_size=3)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        # 2
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                            filters=64, kernel_size=3, strides=2)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        # 4
        inputs = self.darknet53_residual_block(inputs, filters=32)
        # 5
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=128, kernel_size=3, strides=2)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 9
        for _ in range(2):
            inputs = self.darknet53_residual_block(inputs, filters=64)
        # 10
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                        filters=256, kernel_size=3, strides=2)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 26
        for _ in range(8):
            inputs = self.darknet53_residual_block(inputs, filters=128)

        # 27
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=512, kernel_size=3, strides=2)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 43
        for _ in range(8):
            inputs = self.darknet53_residual_block(inputs, filters=256)

        # 44
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=1024, kernel_size=3, strides=2)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 52
        for _ in range(4):
            inputs = self.darknet53_residual_block(inputs, filters=512)

        return inputs

    def yolo_convolution_block(self, inputs, filters):
        """Creates convolution operations layer used after Darknet."""
        # 1
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                        filters=filters, kernel_size=1)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 2
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                        filters=2*filters, kernel_size=3)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 3
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=filters, kernel_size=1)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 4
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=2 * filters, kernel_size=3)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        # 5
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=filters, kernel_size=1)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)
        # 6
        inputs = self.conv2d_fixed_padding(inputs=inputs,
                                           filters=2 * filters, kernel_size=3)
        inputs = self.batch_norm(inputs)
        inputs = keras.layers.advanced_activations.LeakyReLU(alpha=_LEAKY_RELU)(inputs)

        return inputs