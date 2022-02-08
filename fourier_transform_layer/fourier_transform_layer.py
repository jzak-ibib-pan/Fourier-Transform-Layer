import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu


class FTL(Layer):
    def __init__(self, activation = None, initializer = 'he_normal', **kwargs):
        # activation - what activation to pull from keras; available for now: None, relu, softmax, sigmoid, tanh, selu;
        # recommended - None, relu or selu
        # whether to inverse the FFT or not
        inverse = False
        if 'inverse' in kwargs.keys():
            inverse = kwargs['inverse']
        # whether to train next layers (i.e. Convolutional) on returned phase
        phase_training = False
        if 'phase_training' in kwargs.keys():
            phase_training = kwargs['phase_training']
        use_imaginary = True
        if 'use_imaginary' in kwargs.keys():
            use_imaginary = kwargs['use_imaginary']
        assert not (inverse is True and phase_training is True), 'You cannot phase train and inverse at the same time.'
        assert not (inverse is True and use_imaginary is False), 'Cannot inverse FFT without imaginary part.'
        self.kernel = None
        self.kernel_imag = None
        self._activ = None
        if activation == 'relu':
            self._activ = relu
        elif activation == 'softmax':
            self._activ = softmax
        elif activation == 'sigmoid':
            self._activ = sigmoid
        elif activation == 'tanh':
            self._activ = tanh
        elif activation == 'selu':
            self._activ = selu
        self._initializer = initializer
        self._phase_training = phase_training
        self._inverse = inverse
        self._use_imaginary = use_imaginary
        super(FTL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=tuple(input_shape[1:]),
                                      initializer=self.initializer,
                                      trainable=True)
        if self._use_imaginary:
            self.kernel_imag = self.add_weight(name='kernel_imag',
                                               shape=tuple(input_shape[1:]),
                                               initializer=self.initializer,
                                               trainable=True)
        # Be sure to call this at the end
        super(FTL, self).build(input_shape)

    @tf.autograph.experimental.do_not_convert
    def call(self, input_tensor, **kwargs):
        # ifft for 2-tuple input
        if type(input_tensor) is tuple:
            x = tf.dtypes.complex(input_tensor[0] * self.kernel, input_tensor[1] * self.kernel_imag)
            if self.inverse:
                x = tf.signal.ifft3d(tf.cast(x, tf.complex64))
            x = tf.math.abs(x)
            if self.activ is not None:
                return self.activ(x)
            return x

        x = tf.signal.fft3d(tf.cast(input_tensor, tf.complex64))
        real = tf.math.real(x)
        # TODO: stronger connection between arguments 'inverse' and 'use_imaginary'
        imag = tf.math.imag(x)

        real = tf.multiply(real, self.kernel)
        if self._use_imaginary:
            imag = tf.multiply(imag, self.kernel_imag)

        x = tf.cast(tf.dtypes.complex(real, imag), tf.complex64)
        if self._phase_training:
            if self._activ is not None:
                return self._activ(tf.math.real(x)), tf.math.imag(x)
            return tf.math.real(x), tf.math.imag(x)
        if self._inverse:
            x = tf.signal.ifft3d(x)
        x = tf.math.abs(x)
        if self._activ is not None:
            return self.activ(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.phase_training:
            return input_shape, input_shape
        return input_shape


if __name__ == '__main__':
    print(0)