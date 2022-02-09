from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input
import tensorflow.keras.applications as apps
from numpy import squeeze, ones, expand_dims, pad
# Otherwise FTL cannot be called
from fourier_transform_layer.fourier_transform_layer import FTL


# Generic builder
class ModelBuilder:
    def __init__(self, **kwargs):
        self._params_build = {'model_type': '',
                             'input_shape': (8, 8, 1),
                             'noof_classes': 1,
                             # 'weights': '',
                             }
        self._params_compile = {'optimizer': '',
                               'loss': '',
                               }
        self._PARAMS = {'build': self._params_build,
                        'compile': self._params_compile,
                        }
        self._LENGTH = 0
        self._update_length(self._calculate_lengths(self._params_build))
        self._update_length(self._calculate_lengths(self._params_compile))
        # Fourier weights reveal whether imag is used or not
        self._SUMMARIES = {'fourier': True,
                           'default': False,
                           }
        self.model = []

    def build_model(self, model_type, input_shape, noof_classes, **kwargs):
        self._params_build = self._update_params(self._params_build, model_type=model_type, input_shape=input_shape,
                                                 noof_classes=noof_classes, **kwargs)
        self._update_length(self._calculate_lengths(self._params_build))

    def compile_model(self, optimizer, loss, **kwargs):
        self._params_compile = self._update_params(self._params_compile, optimizer=optimizer, loss=loss)
        if 'metrics' in kwargs.keys():
            self._params_compile.update({'metrics': kwargs['metrics']})
            self._update_length(self._calculate_lengths(self._params_compile))
            self.model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])
            return
        self.model.compile(optimizer=optimizer, loss=loss)

    def save_model_info(self, filename, notes='', filepath='', extension='', **kwargs):
        assert type(notes) == str, 'Notes must be a string.'
        if 'fourier' in self._params_build['model_type']:
            summary = self._SUMMARIES['fourier']
        else:
            summary = self._SUMMARIES['default']
        if 'summary' in kwargs.keys():
            summary = kwargs['summary']
        filename_expanded = self._expand_filename(filename, filepath)
        format_used = extension
        if len(format_used) < 1:
            format_used = '.txt'
        if '.' not in format_used:
            format_used = '.' + format_used
        with open(join(filepath, filename_expanded + format_used), 'w') as fil:
            for action in ['build', 'compile']:
                fil.write(self._prepare_text(action))
            fil.write(notes + '\n')
            if summary:
                fil.write('Weights summary:\n')
                for weight in self.model.get_weights():
                    fil.write(f'\t{weight.shape}\n')
                with redirect_stdout(fil):
                    self.model.summary()
        return filename_expanded

    # method for text cleanup
    def _prepare_text(self, what='build'):
        text_build = f'{what.capitalize()} parameters\n'
        for key, value in zip(self._PARAMS[what].keys(), self._PARAMS[what].values()):
            text_build += f'\t{key:{self._LENGTH}} - ' \
                              f'{str(value).rjust(self._LENGTH)}\n'
        return text_build

    def _update_length(self, new_candidate):
        self._LENGTH = max([self._LENGTH, new_candidate])

    @staticmethod
    def _expand_filename(filename, filepath=''):
        # List OF
        loof_files = [f for f in listdir(filepath) if filename in f]
        it = str(len(loof_files))
        while len(it) < 3:
            it = '0' + it
        date = dt.now().strftime('%Y-%m-%d_%H_%M_%S')
        filename_expanded = f'{filename}_{it}_{date}'
        return filename_expanded

    # a method to change the values of parameter holders
    @staticmethod
    def _update_params(parameters, **kwargs):
        for key in kwargs.keys():
            if key in parameters.keys():
                parameters[key] = kwargs[key]
            else:
                parameters.update({key: kwargs[key]})
        return parameters

    @staticmethod
    def _calculate_lengths(params):
        length_keys = max([len(str(f)) for f in params.keys()])
        length_vals = max([len(str(f)) for f in params.values()])
        return max([length_keys, length_vals])


# Standard CNNs for classification
class CNNBuilder(ModelBuilder):
    def __init__(self, model_type='mobilenet', input_shape=(32, 32, 3), noof_classes=1, **kwargs):
        super(CNNBuilder, self).__init__()
        self.model = self.build_model(model_type, input_shape, noof_classes, **kwargs)

    def build_model(self, model_type, input_shape, noof_classes, weights=None, **kwargs):
        super(CNNBuilder, self).build_model(model_type, input_shape, noof_classes, weights=None, **kwargs)
        model_type_low = model_type.lower()
        if 'mobilenet' in model_type_low:
            if '2' not in model_type_low:
                # load Mobilenet
                backbone = apps.mobilenet.MobileNet(input_shape, weights=weights, include_top=False)
            else:
                # load Mobilenetv2
                backbone = apps.mobilenet_v2.MobileNetV2(input_shape, weights=weights, include_top=False)
                # update BatchNormalization momentum - otherwise MobilenetV2 does not work
                for layer in backbone.layers:
                    if type(layer) != type(BatchNormalization):
                        continue
                    layer.momentum=0.9
        elif 'vgg' in model_type_low:
            if '16' in model_type_low:
                backbone = apps.vgg16.VGG16(input_shape, weights=weights, include_top=False)
            elif '19' in model_type_low:
                backbone = apps.vgg19.VGG19(input_shape, weights=weights, include_top=False)
        elif 'resnet' in model_type_low:
            if '50' in model_type_low:
                backbone = apps.resnet_v2.ResNet50V2(input_shape, weights=weights, include_top=False)
            elif '101' in model_type_low:
                backbone = apps.resnet_v2.ResNet101V2(input_shape, weights=weights, include_top=False)
        architecture = backbone.output
        # Classify
        flat = Flatten()(architecture)
        act = 'softmax'
        if noof_classes == 1:
            act = 'sigmoid'
        out = Dense(noof_classes, activation=act)(flat)
        return Model(inputs=[backbone.input], outputs=[out])


# Fourier Model for classification
class FourierBuilder(ModelBuilder):
    def __init__(self, model_type='fourier', input_shape=(32, 32, 1), noof_classes=1, **kwargs):
        super(FourierBuilder, self).__init__()
        self.model = self.build_model(model_type, input_shape, noof_classes, **kwargs)
        self._DIRECTIONS = {'up': '*',
                            'down': '//',
                            }

    def build_model(self, model_type, input_shape, noof_classes, **kwargs):
        super(FourierBuilder, self).build_model(model_type, input_shape, noof_classes, **kwargs)
        ftl_activation = 'relu'
        if 'ftl_activation' in kwargs.keys():
            ftl_activation = kwargs['ftl_activation']
        ftl_initializer = 'ones'
        if 'ftl_initializer' in kwargs.keys():
            ftl_initializer = kwargs['ftl_initializer']
        use_imag = True
        if 'use_imag' in kwargs.keys():
            use_imag = kwargs['use_imag']
        model_type_low = model_type.lower()
        inp = Input(input_shape)
        arch = FTL(activation=ftl_activation, initializer=ftl_initializer, inverse='inverse' in model_type_low,
                   use_imaginary=use_imag)(inp)
        flat = Flatten()(arch)
        if noof_classes == 1:
            act = 'sigmoid'
        else:
            act = 'softmax'
        out = Dense(noof_classes, activation=act, kernel_initializer='ones')(flat)
        return Model(inp, out)

    def sample_model(self, **kwargs):
        # TODO: finding FTL in the model
        # right now assume that FTL is the first layer and imaginary is used
        shape = self._params_build['input_shape'][:2]
        if 'direction' in kwargs.keys() and 'nominator' in kwargs.keys():
            shape_new = self._operation(shape, parameter=kwargs['nominator'],
                                        sign=self._DIRECTIONS[kwargs['direction']])
        if 'shape' in kwargs.keys():
            shape_new = kwargs['shape']
        weights = squeeze(self.model.get_weights()[:2])
        if 'weights' in kwargs.keys():
            weights = squeeze(kwargs['weights'])
        replace_value = 1e-5
        if 'replace_value' in kwargs.keys():
            replace_value = kwargs['replace_value']
        # bo real i imag
        replace_weights = ones((2, shape_new[0], shape_new[1], 1)) * replace_value
        for rep in range(2):
            # działa wyciąganie nawet fragmentu fft
            if shape_new[0] < shape[0]:
                replace_weights[rep] = expand_dims(weights[rep, :shape_new[0], :shape_new[1]], axis=-1)
            else:
                pads = [[0, shape_new[0]//2], [0, shape_new[1]//2]]
                replace_weights[rep] = expand_dims(pad(weights[rep, :, :], pad_width=pads, mode='constant',
                                                       constant_values=replace_value),
                                                   axis=-1)
        params_sampled = self._params_build
        params_sampled['input_shape'] = (*shape_new, shape[2])
        model_sampled = self.build_model(**self._params_build)
        head = self.model.get_weights()[2:]
        size_new = shape_new[0] * shape_new[1]
        if shape_new[0] < shape[0]:
            head[0] = head[0][:size_new, :]
        else:
            pads = [[0, size_new - shape[0] * shape[1]], [0, 0]]
            head[0] = pad(head[0], pad_width=pads, mode='constant', constant_values=replace_value)
        new_weights = [*replace_weights, *head]
        model_sampled.set_weights(new_weights)
        return model_sampled

    @staticmethod
    def _operation(value, parameter=2, sign='div'):
        assert sign in ['divide', 'div', '//', 'multiply', 'mult', '*']
        if sign in ['divide', 'div', '//']:
            return value[:2] // parameter
        elif sign in ['multiply', 'mult', '*']:
            return value[:2] * parameter


if __name__ == '__main__':
    builder = FourierBuilder('fourier', ftl_activation='relu', use_imag=True)
    builder.compile_model('adam' , 'mse')
    builder.sample_model(shape=(64, 64))
    builder.save_model_info(filename='test', notes='Testing saving method', filepath='../test', extension='.txt')