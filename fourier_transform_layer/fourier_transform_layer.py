import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu


# TODO: get_config implementation
class FTL(Layer):
    def __init__(self, activation=None, initializer='he_normal', inverse=False, phase_training=False,
                 use_imaginary=True, **kwargs):
        super(FTL, self).__init__(**kwargs)
        # activation - what activation to pull from keras; available for now: None, relu, softmax, sigmoid, tanh, selu;
        # recommended - None, relu or selu
        assert not (inverse is True and phase_training is True), 'Cannot phase train and inverse at the same time.'
        assert not (inverse is True and use_imaginary is False), 'Cannot inverse FFT without imaginary part.'
        assert not (phase_training is True and use_imaginary is False), 'Cannot phase train without imaginary part.'
        self.kernel = None
        self.kernel_imag = None
        self._activation = None
        if activation == 'relu':
            self._activation = relu
        elif activation == 'softmax':
            self._activation = softmax
        elif activation == 'sigmoid':
            self._activation = sigmoid
        elif activation == 'tanh':
            self._activation = tanh
        elif activation == 'selu':
            self._activation = selu
        self._initializer = initializer
        self._flag_phase_training = phase_training
        self._flag_inverse = inverse
        self._flag_use_imaginary = use_imaginary

    def build(self, input_shape):
        if self._flag_use_imaginary:
            self.kernel = self.add_weight(name='kernel',
                                          shape=tuple((2, *input_shape[1:])),
                                          initializer=self._initializer,
                                          trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel',
                                          shape=tuple((1, *input_shape[1:])),
                                          initializer=self._initializer,
                                          trainable=True)


    @tf.autograph.experimental.do_not_convert
    def call(self, input_tensor, **kwargs):
        # ifft for 2-tuple input
        if type(input_tensor) is tuple:
            x = tf.dtypes.complex(input_tensor[0] * self.kernel, input_tensor[1] * self.kernel_imag)
            if self._flag_inverse:
                x = tf.signal.ifft3d(tf.cast(x, tf.complex64))
            x = tf.math.abs(x)
            if self._activation is not None:
                return self._activation(x)
            return x

        x = tf.signal.fft3d(tf.cast(input_tensor, tf.complex64))
        real = tf.math.real(x)
        real = tf.multiply(real, self.kernel[0])

        if not self._flag_use_imaginary:
            if self._activation is not None:
                return self._activation(real)
            return real

        imag = tf.math.imag(x)
        imag = tf.multiply(imag, self.kernel[1])

        if self._flag_phase_training:
            if self._activation is not None:
                return self._activation(real), self._activation(imag)
            return real, imag

        x = tf.cast(tf.dtypes.complex(real, imag), tf.complex64)
        if self._flag_inverse:
            x = tf.signal.ifft3d(x)
        x = tf.math.abs(x)
        if self._activation is not None:
            return self._activation(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.phase_training:
            return input_shape, input_shape
        return input_shape


if __name__ == '__main__':
    ftl = FTL(name='test_ftl')
    print(ftl.name)
    print(ftl.get_config())