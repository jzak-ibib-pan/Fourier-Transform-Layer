from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.mobilenet import MobileNet


class ModelBuilder():
    def __init__(self):
        self.model = self.build((32, 32, 1))

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


class CNNBuilder(ModelBuilder):
    def __init__(self):
        super(CNNBuilder, self).__init__()

    @staticmethod
    def build(input_shape, noof_classes=1):
        backbone = MobileNet(input_shape, weights=None, include_top=False)
        architecture = backbone.output
        arch = backbone.output
        # Classify
        flat = Flatten()(arch)
        out = Dense(noof_classes, activation=act)(flat)
        return Model(inputs=[backbone.input], outputs=[out])


if __name__ == '__main__':
    builder = CNNBuilder()
    builder.save_model_info('test', '.txt', '../test')