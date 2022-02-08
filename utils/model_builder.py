from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
import tensorflow.keras.applications as apps


# Generic builder
class ModelBuilder:
    def __init__(self, **kwargs):
        self._params_build = {'model_type': '',
                             'input_shape': '',
                             'noof_classes': '',
                             'weights': '',
                             }
        self._params_compile = {'optimizer': '',
                               'loss': '',
                               }
        self._PARAMS = {'build': self._params_build,
                        'compile': self._params_compile,
                        }
        length_max_build = max([len(f) for f in self._params_build.keys()])
        length_max_compile = max([len(f) for f in self._params_compile.keys()])
        self._LENGTH = max([length_max_build, length_max_compile])
        self.model = []

    def build_model(self, model_type, input_shape, noof_classes, weights=None):
        self._params_build = self._update_params(self._params_build, model_type=model_type, input_shape=input_shape,
                                                 noof_classes=noof_classes, weights=weights)

    def compile_model(self, optimizer, loss, **kwargs):
        self._params_compile = self._update_params(self._params_compile, optimizer=optimizer, loss=loss)
        if 'metrics' in kwargs.keys():
            self._params_compile.update({'metrics': kwargs['metrics']})
            self.model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])
            return
        self.model.compile(optimizer=optimizer, loss=loss)

    def save_model_info(self, filename, extension='', filepath='', summary=False):
        filename_expanded = self._expand_filename(filename, filepath)
        format_used = extension
        if len(format_used) < 1:
            format_used = '.txt'
        if '.' not in format_used:
            format_used = '.' + format_used
        with open(join(filepath, filename_expanded + format_used), 'w') as fil:
            for action in ['build', 'compile']:
                fil.write(self._prepare_text(action))
            if summary:
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
        for key in parameters.keys():
            parameters[key] = kwargs[key]
        return parameters


# Standard CNNs for classification
class CNNBuilder(ModelBuilder):
    def __init__(self, model_type='mobilenet', input_shape=(32, 32, 1), noof_classes=1):
        super(CNNBuilder, self).__init__()
        self.model = self.build_model(model_type, input_shape, noof_classes)

    def build_model(self, model_type, input_shape, noof_classes, weights=None):
        super(CNNBuilder, self).build_model(model_type, input_shape, noof_classes, weights)
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


if __name__ == '__main__':
    builder = CNNBuilder()
    builder.compile_model('adam' , 'mse')
    builder.save_model_info('test', '.txt', '../test')