from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input
import tensorflow.keras.applications as apps
# Otherwise FTL cannot be called
from fourier_transform_layer.fourier_transform_layer import FTL


# Generic builder
class ModelBuilder:
    def __init__(self, **kwargs):
        self._params_build = {'model_type': '',
                             'input_shape': '',
                             'noof_classes': '',
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
                with redirect_stdout(fil):
                    self.model.summary()
                    for weight in self.model.get_weights():
                        print(weight.shape)
        return filename_expanded

    # method for text cleanup
    def _prepare_text(self, what='build'):
        text_build = f'{what.capitalize()} parameters\n'
        for key, value in zip(self._PARAMS[what].keys(), self._PARAMS[what].values()):
            text_build += f'\t{key:{self._LENGTH}}- ' \
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

    def build_model(self, model_type, input_shape, noof_classes, **kwargs):
        super(FourierBuilder, self).build_model(model_type, input_shape, noof_classes, **kwargs)
        ftl_activation = 'relu'
        if 'ftl_activation' in kwargs.keys():
            ftl_activation = kwargs['ftl_activation']
        ftl_initializer = 'ones'
        if 'ftl_initializer' in kwargs.keys():
            ftl_initializer = kwargs['ftl_initializer']
        model_type_low = model_type.lower()
        inp = Input(input_shape)
        arch = FTL(activation=ftl_activation, inverse='inverse' in model_type_low, initializer=ftl_initializer)(inp)
        flat = Flatten()(arch)
        if noof_classes == 1:
            act = 'sigmoid'
        else:
            act = 'softmax'
        out = Dense(noof_classes, activation=act, kernel_initializer='ones')(flat)
        return Model(inp, out)


if __name__ == '__main__':
    builder = FourierBuilder('fourier_inverse', ftl_activation='relu')
    builder.compile_model('adam' , 'mse')
    builder.save_model_info(filename='test', notes='Testing saving method', filepath='../test', extension='.txt')
    builder = CNNBuilder('mobilenet')
    builder.compile_model('adam' , 'mse')
    builder.save_model_info(filename='test', notes='Testing saving method', filepath='../test', extension='.txt')