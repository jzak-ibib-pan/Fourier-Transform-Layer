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
        self.model = []

    def save_model_info(self, filename, extension='', filepath=''):
            filename_expanded = self._expand_filename(filename, filepath)
            format_used = extension
            if len(format_used) < 1:
                format_used = '.txt'
            if '.' not in format_used:
                format_used = '.' + format_used
            with open(join(filepath, filename_expanded + format_used), 'w') as fil:

                with redirect_stdout(fil):
                    self.model.summary()
            return filename_expanded

    @staticmethod
    def build(input_shape, noof_classes=1):
        return []

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


# Standard CNNs for classification
class CNNBuilder(ModelBuilder):
    def __init__(self, model_type='mnist', input_shape=(32, 32, 1), noof_classes=1):
        super(CNNBuilder, self).__init__()
        self.model = self.build(model_type, input_shape, noof_classes)

    def compile(self, optimizer, loss, **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])

    @staticmethod
    def build(model_type, input_shape, noof_classes, weights=None):
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
    builder.save_model_info('test', '.txt', '../test')