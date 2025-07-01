import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu, gelu
from utils.sampling import DIRECTIONS, sampling_calculation
from numpy import arange

# SOLVED: remove concatenate requirement
# SOLVED: add fft_already parameter to split the trainings
# TODO: find source of nan loss and eliminate - nan appears with SGD; disappears with Adam or run_eagerly in compile
# TODO: source of nan loss may be inverse=False and calculate_abs=True
# TODO: get_config implementation
# TODO: move bias initializer to kwargs, depending on use_bias - to consider
# TODO: train on abs as a possibility
class FTL(Layer):
    def __init__(
            self,
            activation=None,
            kernel_initializer='he_normal',
            kernel_regularizer=None,
            train_imaginary=True,
            inverse=False,
            use_bias=False,
            bias_initializer='zeros',
            bias_regularizer=None,
            calculate_abs=True,
            normalize_to_image_shape=False,
            already_fft=False,
            use_sine=False,
            **kwargs
        ):
        super(FTL, self).__init__(**kwargs)
        self._noof_channels = 1
        # activation - what activation to pull from keras; available for now: None, relu, softmax, sigmoid, tanh, selu;
        # recommended - None, relu or selu
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
        elif activation == 'gelu':
            self._activation = gelu
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._flag_inverse = inverse
        self._flag_train_imaginary = train_imaginary
        self._flag_normalize = normalize_to_image_shape
        self._bias_initializer = 'zeros'
        self._flag_use_bias = use_bias
        self._bias_initializer = bias_initializer
        self._bias_regularizer = bias_regularizer
        self._flag_calculate_abs = calculate_abs
        self._flag_already_fft = already_fft
        self._flag_use_sine = use_sine

    def build(self, input_shape):
        # could also be -3:
        #kernel_shape = input_shape[1:] if not self._flag_already_fft else [*input_shape[1:-1], 1]
        self._noof_channels = input_shape[-1] // (1 + self._flag_already_fft)
        kernel_shape = input_shape[1:-1]
        self._kernel = self.add_weight(
            name='kernel',
            shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape),
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            trainable=True
        )
        if self._flag_use_sine:
            # sines
            self._amplitude = self.add_weight(
                name="amplitude",
                shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape, 1),
                initializer="he_normal",
                regularizer=self._kernel_regularizer,
                trainable=True
            )
            self._period = self.add_weight(
                name="period",
                shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape, 1),
                initializer="he_normal",
                regularizer=self._kernel_regularizer,
                trainable=True
            )
            self._phase_shift = self.add_weight(
                name="phase_shift",
                shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape, 1),
                initializer="he_normal",
                regularizer=self._kernel_regularizer,
                trainable=True
            )
            self._vertical_shift = self.add_weight(
                name="vertical_shift",
                shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape, 1),
                initializer="he_normal",
                regularizer=self._kernel_regularizer,
                trainable=True
            )
        if self._flag_use_bias:
            self._bias = self.add_weight(
                name='bias',
                shape=(self._kernel_shape_0[self._flag_train_imaginary] * self._noof_channels, *kernel_shape),
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                trainable=True
            )

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
        result = []
        sh = input_tensor.shape
        #
        # also make sure there are at least two tensors to split
        if self._flag_already_fft and sh[-1] >= 2:
            # extract real and imaginary parts per channel (that's how they will be returned)
            real_imag_pairs = tf.split(input_tensor, num_or_size_splits=self._noof_channels, axis=-1)
            for it, pair in enumerate(real_imag_pairs):
                real = pair[:, :, :, 0:1]
                imag = pair[:, :, :, 1:]
                id_start = it * (1 + self._flag_train_imaginary)
                id_end = id_start + self._flag_train_imaginary + 1
                if self._flag_use_bias:
                    _result = self._call_process_split_fft(
                        real,
                        imag,
                        kernel=self._kernel[id_start : id_end, :, :, tf.newaxis],
                        bias=self._bias[id_start : id_end, :, :, tf.newaxis],
                        )
                else:
                    _result = self._call_process_split_fft(
                        real,
                        imag,
                        kernel=self._kernel[id_start : id_end, :, :, tf.newaxis],
                        )
                result.append(_result)
        else:
            channels = tf.split(input_tensor, num_or_size_splits=self._noof_channels, axis=-1)
            for it, channel in enumerate(channels):
                real, imag = self._perform_fft(channel, self._flag_normalize)
                # real_by_channel.append(real)
                # imag_by_channel.append(imag)
                id_start = it * (1 + self._flag_train_imaginary)
                id_end = id_start + self._flag_train_imaginary + 1
                if self._flag_use_bias:
                    _result = self._call_process_split_fft(
                        real,
                        imag,
                        kernel=self._kernel[id_start : id_end, :, :, tf.newaxis],
                        bias=self._bias[id_start : id_end, :, :, tf.newaxis],
                        )
                else:
                    _result = self._call_process_split_fft(
                        real,
                        imag,
                        kernel=self._kernel[id_start : id_end, :, :, tf.newaxis],
                        )
                result.append(_result)
        return tf.concat(result, axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self._flag_already_fft:
            output_shape = [*input_shape[:-1], self._noof_channels]
        if not self._flag_calculate_abs:
            return [*input_shape[:-1], input_shape[:-1] * self._noof_channels * 2]
        return output_shape

    def _call_process_split_fft(
            self,
            real,
            imag,
            kernel=None,
            bias=None,
        ):
        # tf.print("0: before")
        # tf.print(tf.math.reduce_any(tf.math.is_nan(kernel[0])))
        _real = tf.math.multiply_no_nan(real, kernel[0])
        if self._flag_use_sine:
            # get sines
            sines_real = tf.math.multiply_no_nan(
                self._amplitude[0],
                tf.math.sin(
                    tf.math.multiply_no_nan(
                        tf.math.divide_no_nan(2 * 3.141592, self._period[0]),
                        _real
                    ) + self._phase_shift[0]
                )) + self._vertical_shift[0]
            _real = tf.math.multiply_no_nan(real, sines_real)
        # tf.print("0: after")
        # tf.print(tf.math.reduce_any(tf.math.is_nan(kernel[0])))
        #_real = tf.add(real, self._kernel[0])
        if self._flag_use_bias:
            _real = tf.add(_real, bias[0])

        if self._flag_train_imaginary:
            _imag = tf.math.multiply_no_nan(imag, kernel[1])
            if self._flag_use_sine:
                sines_imag = tf.math.multiply_no_nan(
                self._amplitude[1],
                tf.math.sin(
                    tf.math.multiply_no_nan(
                        tf.math.divide_no_nan(2 * 3.141592, self._period[1]),
                        _imag
                    ) + self._phase_shift[1]
                )) + self._vertical_shift[1]
                _imag = tf.math.multiply_no_nan(imag, sines_imag)
            # tf.print("1: before")
            # tf.print(tf.math.reduce_any(tf.math.is_nan(kernel[1])))
            # tf.print("1: after")
            # tf.print(tf.math.reduce_any(tf.math.is_nan(kernel[1])))
            #_imag = tf.add(imag, self._kernel[1])
            if self._flag_use_bias:
                _imag = tf.add(_imag, bias[1])
            x = tf.cast(tf.dtypes.complex(_real, _imag), tf.complex64)
        else:
            # use original imaginary part
            #x = tf.cast(tf.dtypes.complex(_real, imag), tf.complex64)
            # usage of imag caused "None values not supported" error
            x = tf.cast(tf.dtypes.complex(_real, tf.zeros_like(real)), tf.complex64)

        if self._flag_inverse:
            x = tf.signal.ifft2d(x)

        if self._flag_calculate_abs:
            _re, _im = tf.math.real(x), tf.math.imag(x)
            negatives = tf.math.logical_or(_re < 0, _im < 0)
            x = tf.math.abs(x)
            # TODO: to separate method (?)
            if self._activation is not None:
                return self._activation(x)
            elif self._activation in ["tanh"]:
                # keep the original signs for tanh
                return tf.where(
                    negatives,
                    -x,
                    x
                    )
            return x
            #return tf.math.multiply_no_nan(x, sines)
        # returning only real would work the same as use_imaginary = False
        result_real, result_imag = tf.math.real(x), tf.math.imag(x)
        if self._activation is not None:
            return tf.concat([self._activation(result_real), self._activation(result_imag)], axis=-1)
        return tf.concat([result_real, result_imag], axis=-1)
        # return tf.concat([
        #     tf.math.multiply_no_nan(result_real, sines),
        #     tf.math.multiply_no_nan(result_imag, sines)
        # ], axis=-1)

    @staticmethod
    def _perform_fft(input_tensor, normalize=False):
        # TODO: compare 1x fft2d and 1x fft3d on 1 channel, then 3x fft2d and 1x fft3d on 3 channels
        x = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))
        if normalize:
            shapes = tf.shape(input_tensor)[1:]
            real_x = tf.divide(tf.math.real(x), tf.cast(shapes[0] * shapes[1], tf.float32)) * 2
            imag_x = tf.divide(tf.math.imag(x), tf.cast(shapes[0] * shapes[1], tf.float32)) * 2
            return real_x, imag_x
        return tf.math.real(x), tf.math.imag(x)


class FTLReshape(FTL):
    # TODO: make sure can call predict without shape [1, X, X, C] - on skądś bierze original input shape[1] jako target_shape[0]
    def __init__(
            self,
            sampling_nominator: int=2,
            direction: str='up',
            **kwargs
        ):
        super(FTLReshape, self).__init__(**kwargs)
        self._nominator = sampling_nominator
        # in this case direction must be specified
        self._sampling_direction = DIRECTIONS[direction]
        self._direction = direction
        self._target_shape = ()

    def build(self, input_shape):
        _target_shape = self._calculate_target_shape(input_shape[1:3], self._nominator, self._sampling_direction)
        self._target_shape = (*_target_shape, input_shape[-1])
        # build omits first shape, thus -1
        super(FTLReshape, self).build((-1, *self._target_shape))

    def call(self, input_tensor, **kwargs):
        real, imag = self._perform_fft(input_tensor, self._flag_normalize)
        _real = self._pad_or_extract(real, self._target_shape, self._direction)
        _imag = None
        if self._flag_train_imaginary:
            _imag = self._pad_or_extract(imag, self._target_shape, self._direction)
        return self._call_process_split_fft(
            _real,
            _imag,
            kernel=self._kernel,
            bias=self._bias
        )

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


class FTLSuperResolution(FTLReshape):
    # TODO: make sure can call predict without shape [1, X, X, C] - on skądś bierze original input shape[1] jako target_shape[0]
    def __init__(self, **kwargs):
        super(FTLSuperResolution, self).__init__(
            train_imaginary=True,
            # required for superresolution
            inverse=True,
            # possibly required for superresolution
            calculate_abs=True,
            **kwargs
            )

# TODO: colorization using FTL
# class FTLColorization(FTL):

if __name__ == '__main__':
    ftl = FTL(name='test_ftl')
    print(ftl.name)
    print(ftl.get_config())