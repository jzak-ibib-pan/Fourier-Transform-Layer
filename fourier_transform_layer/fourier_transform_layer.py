import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu
from utils.sampling import DIRECTIONS, sampling_calculation

# TODO: remove concatenate requirement
# TODO: find source of nan loss and eliminate - nan appears with SGD; disappears with Adam or run_eagerly in compile
# TODO: get_config implementation
class FTL(Layer):
    def __init__(self, activation=None, kernel_initializer='he_normal', use_imaginary=True, inverse=False,
                 use_bias=False, bias_initializer='zeros', calculate_abs=True, normalize_to_image_shape=False,
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
        self._flag_calculate_abs = calculate_abs

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
        # TODO: rework to work with new version
        # if type(input_tensor) is tuple:
        #     x = tf.dtypes.complex(input_tensor[0] * self.kernel, input_tensor[1] * self.kernel_imag)
        #     if self._flag_inverse:
        #         x = tf.signal.ifft3d(tf.cast(x, tf.complex64))
        #     x = tf.math.abs(x)
        #     if self._activation is not None:
        #         return self._activation(x)
        #     return x

        real, imag = self._perform_fft(input_tensor, self._flag_normalize)
        return self._call_process_split_fft(real, imag)

    def compute_output_shape(self, input_shape):
        if self._flag_phase_training or self._flag_calculate_abs:
            return input_shape, input_shape
        return input_shape

    def _call_process_split_fft(self, real, imag):
        _real = tf.multiply(real, self._kernel[0])
        if self._flag_use_bias:
            _real = tf.add(_real, self._bias[0])

        if not self._flag_use_imaginary:
            if self._activation is not None:
                return self._activation(_real)
            return _real

        _imag = tf.multiply(imag, self._kernel[1])
        if self._flag_use_bias:
            _imag = tf.add(_imag, self._bias[1])

        if self._flag_phase_training:
            if self._activation is not None:
                return self._activation(_real), self._activation(_imag)
            return _real, _imag

        x = tf.cast(tf.dtypes.complex(_real, _imag), tf.complex64)
        if self._flag_inverse:
            x = tf.signal.ifft3d(x)

        if self._flag_calculate_abs:
            x = tf.math.abs(x)
            if self._activation is not None:
                return self._activation(x)
            return x

        # returning only real would work the same as use_imaginary = False
        result_real, result_imag = tf.math.real(x), tf.math.imag(x)
        if self._activation is not None:
            return self._activation(result_real), self._activation(result_imag)
        return result_real, result_imag

    @staticmethod
    def _perform_fft(input_tensor, normalize=False):
        x = tf.signal.fft3d(tf.cast(input_tensor, tf.complex64))
        if normalize:
            shapes = tf.shape(input_tensor)[1:]
            x = tf.divide(x, tf.cast((shapes[0] * shapes[1]), tf.complex64))
        return tf.math.real(x), tf.math.imag(x)


class FTLSuperResolution(FTL):
    # TODO: make sure can call predict without shape [1, X, X, C] - on skądś bierze original input shape[1] jako target_shape[0]
    def __init__(self, activation=None, kernel_initializer='he_normal', sampling_nominator=2, direction='up',
                 use_bias=False, bias_initializer='zeros', normalize_to_image_shape=False,
                 **kwargs):
        super(FTLSuperResolution, self).__init__(activation=activation,
                                                   kernel_initializer=kernel_initializer,
                                                   # required for superresolution
                                                   use_imaginary=True,
                                                   # required for superresolution
                                                   inverse=True,
                                                   use_bias=use_bias,
                                                   bias_initializer=bias_initializer,
                                                   normalize_to_image_shape=normalize_to_image_shape,
                                                   # required for superresolution
                                                   phase_training=False,
                                                   **kwargs)
        self._nominator = sampling_nominator
        # in this case direction must be specified
        self._sampling_direction = DIRECTIONS[direction]
        self._direction = direction
        self._target_shape = ()

    def build(self, input_shape):
        _target_shape = self._calculate_target_shape(input_shape[1:3], self._nominator, self._sampling_direction)
        self._target_shape = (*_target_shape, input_shape[-1])
        # build omits first shape, thus -1
        super(FTLSuperResolution, self).build((-1, *self._target_shape))

    def call(self, input_tensor, **kwargs):
        real, imag = self._perform_fft(input_tensor, self._flag_normalize)
        _real = self._pad_or_extract(real, self._target_shape, self._direction)
        if self._flag_use_imaginary:
            _imag = self._pad_or_extract(imag, self._target_shape, self._direction)
        return self._call_process_split_fft(_real, _imag)

    def compute_output_shape(self, input_shape):
        return self._target_shape

    @staticmethod
    def _pad_or_extract(x, target_shape, direction):
        if direction == 'down':
            # just extract important fft fragment
            return x[:, :target_shape[0], :target_shape[1], :]
        shapes = x.shape
        # not using fftshift, thus 0
        # shapes contain batch size, thus shifted by one
        pads = tf.constant([[0, 0], [0, target_shape[0] - shapes[1]], [0, target_shape[1] - shapes[2]], [0, 0]])
        replace_value = 1e-6
        core = tf.multiply(tf.ones_like(x), -replace_value)
        result = tf.pad(x, pads, 'CONSTANT')
        replace_tensor = tf.multiply(tf.ones_like(result), replace_value)
        core_padded = tf.pad(core, pads, 'CONSTANT')
        core_padded = tf.add(core_padded, replace_tensor)
        # padding to increase ifft image size
        # TODO: replace 0s with 1e-6
        return tf.add(result, core_padded)

    @staticmethod
    def _calculate_target_shape(value, nominator=2, direction='*'):
        return sampling_calculation(value, nominator, direction)
# TODO: colorization using FTL
# class FTLColorization(FTL):
if __name__ == '__main__':
    ftl = FTL(name='test_ftl')
    print(ftl.name)
    print(ftl.get_config())