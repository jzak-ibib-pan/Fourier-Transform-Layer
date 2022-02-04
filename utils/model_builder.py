from tensorflow.keras.models import Model
from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt


class ModelBuilder(Model):
    def __init__(self):
        super(ModelBuilder, self).__init__(self)

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

    def save_model_info(self, filename, extension='', filepath=''):
        filename_expanded = self._expand_filename(filename, filepath)
        format_used = extension
        if len(format_used) < 1:
            format_used = '.txt'
        if '.' not in format_used:
            format_used = '.' + format_used
        with open(join(filepath, filename_expanded + format_used), 'w') as fil:
            with redirect_stdout(fil):
                print('Over 9000!')
                # self.summary()
        return filename_expanded

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


if __name__ == '__main__':
    model = ModelBuilder()
    model.save_model_info('test', '.txt', '../test')