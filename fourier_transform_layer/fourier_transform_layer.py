import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu

# TODO: find source of nan loss and eliminate
# TODO: get_config implementation
class FTL(Layer):
    def __init__(self, activation=None, kernel_initializer='he_normal', use_imaginary=True, inverse=False,
                 use_bias=False, bias_initializer='zeros', normalize_to_image_shape=False,
                 phase_training=False, **kwargs):
        super(FTL, self).__init__(**kwargs)
        # activation - what activation to pull from keras; available for now: None, relu, softmax, sigmoid, tanh, selu;
        # recommended - None, relu or selu
        assert not (inverse is True and phase_training is True), 'Cannot phase train and inverse at the same time.'
        assert not (inverse is True and use_imaginary is False), 'Cannot inverse FFT without imaginary part.'
        assert not (phase_training is True and use_imaginary is False), 'Cannot phase train without imaginary part.'
        self._kernel_shape_0 = {True: 2,
                                False: 1,
                                }
        self._kernel = None
        self._bias = None
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
        self._kernel_initializer = kernel_initializer
        self._flag_phase_training = phase_training
        self._flag_inverse = inverse
        self._flag_use_imaginary = use_imaginary
        self._flag_normalize = normalize_to_image_shape
        self._bias_initializer = 'zeros'
        self._flag_use_bias = use_bias
        self._bias_initializer = bias_initializer

    def build(self, input_shape):
        self._kernel = self.add_weight(name='kernel',
                                      shape=(self._kernel_shape_0[self._flag_use_imaginary], *input_shape[1:]),
                                      initializer=self._kernel_initializer,
                                      trainable=True)
        if self._flag_use_bias:
            self._bias = self.add_weight(name='bias',
                                        shape=(self._kernel_shape_0[self._flag_use_imaginary], *input_shape[1:]),
                                        initializer=self._bias_initializer,
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
        if self._flag_normalize:
            shapes = tf.shape(input_tensor)[1:]
            x = tf.divide(x, tf.cast((shapes[0] * shapes[1]), tf.complex64))

        real = tf.math.real(x)
        real = tf.multiply(real, self._kernel[0])
        if self._flag_use_bias:
            real = tf.add(real, self._bias[0])

        if not self._flag_use_imaginary:
            if self._activation is not None:
                return self._activation(real)
            return real

        imag = tf.math.imag(x)
        imag = tf.multiply(imag, self._kernel[1])
        if self._flag_use_bias:
            imag = tf.add(imag, self._bias[1])

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

class FTL_super_resolution(FTL):
    def __init__(self, activation=None, kernel_initializer='he_normal', use_imaginary=True, inverse=False,
                 use_bias=False, bias_initializer='zeros', normalize_to_image_shape=False,
                 phase_training=False, sampling_nominator=2, direction='up',
                 **kwargs):
        super(FTL_super_resolution, self).__init__(activation, kernel_initializer, use_imaginary, inverse,
                                                   use_bias, bias_initializer, normalize_to_image_shape,
                                                   phase_training, **kwargs)



if __name__ == '__main__':
    ftl = FTL(name='test_ftl')
    print(ftl.name)
    print(ftl.get_config())