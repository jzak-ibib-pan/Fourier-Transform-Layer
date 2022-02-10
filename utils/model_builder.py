from contextlib import redirect_stdout
from os import listdir
from os.path import join
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input
import tensorflow.keras.applications as apps
from numpy import squeeze, ones, expand_dims, pad
from sklearn.utils import shuffle
# Otherwise FTL cannot be called
from fourier_transform_layer.fourier_transform_layer import FTL
from utils.callbacks import TimeHistory, EarlyStopOnBaseline


# Generic builder
class ModelBuilder:
    # TODO: private method for kwarg determination
    def __init__(self, **kwargs):
        self._params_build = {'model_type': '',
                             'input_shape': (8, 8, 1),
                             'noof_classes': 1,
                             # 'weights': '',
                             }
        self._params_compile = {'optimizer': '',
                               'loss': '',
                               }
        self._params_train = {'epochs': 10,
                              }
        self._PARAMS = {'build': self._params_build,
                        'compile': self._params_compile,
                        'train': self._params_train,
                        }
        self._LENGTH = 0
        self._update_all_lengths()
        # Fourier weights reveal whether imag is used or not
        self._SUMMARIES = {'fourier': True,
                           'default': False,
                           }
        self.model = []
        self.history = []

    def build_model(self, model_type, input_shape, noof_classes, **kwargs):
        self._params_build = self._update_params(self._params_build, model_type=model_type, input_shape=input_shape,
                                                 noof_classes=noof_classes, **kwargs)
        self._update_length(self._calculate_lengths(self._params_build))

    def compile_model(self, optimizer, loss, **kwargs):
        self._params_compile = self._update_params(self._params_compile, optimizer=optimizer, loss=loss)
        if 'metrics' in kwargs.keys():
            self._update_params(self._params_compile, metrics=kwargs['metrics'])
            self._update_length(self._calculate_lengths(self._params_compile))
            self.model.compile(optimizer=optimizer, loss=loss, metrics=kwargs['metrics'])
            return
        self.model.compile(optimizer=optimizer, loss=loss)

    def train_model(self, epochs, **kwargs):
        self.history = self._train_model(epochs, **kwargs)

    def _train_model(self, epochs, **kwargs):
        assert 'generator' in kwargs.keys() or sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2, \
        'Must provide either generator or full dataset.'
        # callbacks
        callbacks = []
        flag_time = False
        flag_stop = False
        if 'call_time' in kwargs.keys() and kwargs['call_time']:
            callback_time = TimeHistory()
            callbacks.append(callback_time)
            flag_time = True
        if 'call_stop' in kwargs.keys() and kwargs['call_stop']:
            # metric and monitor names must be the same
            callback_stop = EarlyStopOnBaseline(**kwargs['call_stop_kwargs'])
            callbacks.append(callback_stop)
            self._update_params(self._params_train, early_stop=kwargs['call_stop_kwargs'])
            flag_stop = True
        # full set or generator
        flag_full_set = False
        if sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2:
            x_train = kwargs['x_data']
            y_train = kwargs['y_data']
            split = 0
            if 'validation' in kwargs.keys():
                split = kwargs['validation']
            self._update_params(self._params_train, dataset_size=x_train.shape[0], validation_split=split)
            flag_full_set = True
        if 'generator' in kwargs.keys():
            data_gen = kwargs['generator']
            self._update_params(self._params_train, dataset='generator')
            if 'validation' in kwargs.keys():
                validation_data = kwargs['validation']
                self._update_params(self._params_train, validation_size=validation_data.shape[0])
        # other train params
        batch = 8
        if any([f in ['batch', 'batch_size'] for f in kwargs.keys()]):
            batch = kwargs['batch']
        self._update_params(self._params_train, batch_size=batch)

        verbosity = 1
        if 'verbose' in kwargs.keys():
            verbosity = kwargs['verbose']

        hist = []
        tims = []

        if not flag_full_set:
            hist.append(self.model.fit(data_gen, epochs=epochs, batch_size=batch, shuffle=False, verbose=verbosity,
                                       validation_data=validation_data, callbacks=callbacks).history)
            if flag_time:
                # time callback will always be before stop callback, thus 0
                return self._merge_history_and_times(hist, callbacks[0].times)
            return hist

        # this way ensures that every model will receive the same data
        stop = False
        epoch = 0
        while not stop:
            x_train, y_train = shuffle(x_train, y_train, random_state=epoch)
            hist.append(self.model.fit(x_train, y_train, epochs=1, batch_size=batch, shuffle=False, verbose=verbosity,
                                       validation_split=split, callbacks=callbacks).history)
            epoch += 1
            stop = epoch >= epochs
            if flag_stop:
                stop = stop or callbacks[1].stopped_training
            if not flag_time:
                continue
            tims.append(callbacks[0].times[0])
        if flag_time:
            return self._merge_history_and_times(hist, tims)
        return hist

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
            for action in ['build', 'compile', 'train']:
                fil.write(self._prepare_parameter_text(action))
            fil.write(notes + '\n')
            fil.write('Evaluation: \n')
            fil.write(self._prepare_history_text(self.history))
            if summary:
                fil.write('Weights summary:\n')
                # layers[1:] - Input has no weights
                for layer_got, weight_got in zip(self.model.layers[1:], self.model.get_weights()):
                    fil.write(f'\t{layer_got.name:{self._LENGTH}} - {str(weight_got.shape).rjust(self._LENGTH)}\n')
                with redirect_stdout(fil):
                    self.model.summary()
        return filename_expanded

    # method for text cleanup
    def _prepare_parameter_text(self, what='build'):
        text_build = f'{what.capitalize()} parameters\n'
        for key, value in zip(self._PARAMS[what].keys(), self._PARAMS[what].values()):
            if key == 'weights' and value is not None:
                text_build += f'\t{key:{self._LENGTH}} - \n'
                continue
            text_build += f'\t{key:{self._LENGTH}} - ' \
                              f'{str(value).rjust(self._LENGTH)}\n'
        return text_build

    def _update_all_lengths(self):
        self._update_length(self._calculate_lengths(self._params_build))
        self._update_length(self._calculate_lengths(self._params_compile))
        self._update_length(self._calculate_lengths(self._params_train))

    def _update_length(self, new_candidate):
        self._LENGTH = max([self._LENGTH, new_candidate])

    # a method to change the values of parameter holders
    def _update_params(self, parameters, **kwargs):
        result = parameters
        for key in kwargs.keys():
            # list of different params
            if type(kwargs[key]) is list:
                for it, key_interior in enumerate(kwargs[key]):
                    to_update = self._check_for_name(key_interior)
                    key_str = str(it)
                    while len(key_str) < 2:
                        key_str = '0' + key_str
                    result.update({f'{key}_{key_str}': to_update})
                continue
            if type(kwargs[key]) is dict:
                for key_interior in kwargs[key].keys():
                    to_update = self._check_for_name(key_interior)
                    result.update({f'{key}-{to_update}': kwargs[key][key_interior]})
                continue
            to_update = self._check_for_name(kwargs[key])
            if key in parameters.keys():
                result[key] = to_update
            else:
                result.update({key: to_update})
        return result

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

    @staticmethod
    def _check_for_name(checked_property):
        if hasattr(checked_property, 'name'):
            return checked_property.name
        return checked_property

    @staticmethod
    def _calculate_lengths(params):
        # protection from weights impact on length of text
        length_keys = max([len(str(f)) for f in params.keys() if len(str(f)) < 100])
        length_vals = max([len(str(f)) for f in params.values() if len(str(f)) < 100])
        return max([length_keys, length_vals])

    @staticmethod
    def _merge_history_and_times(history, times):
        assert len(history) == len(times), 'History and times are not the same length.'
        history_end = history
        for it, time in enumerate(times):
            history_end[it].update({'time': time})
        return history_end

    @staticmethod
    def _prepare_history_text(history):
        text_result = f'Training history: \n'
        for epoch in range(len(history)):
            epoch_str = str(epoch)
            while len(epoch_str) < len(str(len(history))):
                epoch_str = '0' + epoch_str
            # do not expect more than 10k training epochs
            text_result += f'\tEpoch {epoch_str:{5}} -- '
            for key, value in zip(history[epoch].keys(), history[epoch].values()):
                value_used = value
                if type(value) is list:
                    value_used = value[0]
                text_result += f'{key} {round(value_used, 6):{9}} || '
            text_result += '\n'
        return text_result

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
        ftl_initializer = 'he_normal'
        if 'ftl_initializer' in kwargs.keys():
            ftl_initializer = kwargs['ftl_initializer']
        use_imag = True
        if 'use_imag' in kwargs.keys():
            use_imag = kwargs['use_imag']
        head_initializer = 'he_normal'
        if 'head_initializer' in kwargs.keys():
            head_initializer = kwargs['head_initializer']
        if noof_classes == 1:
            head_activation = 'sigmoid'
        else:
            head_activation = 'softmax'
        if 'head_activation' in kwargs.keys():
            head_activation = kwargs['head_activation']
        model_type_low = model_type.lower()
        inp = Input(input_shape)
        arch = FTL(activation=ftl_activation, initializer=ftl_initializer, inverse='inverse' in model_type_low,
                   use_imaginary=use_imag)(inp)
        flat = Flatten()(arch)
        out = Dense(noof_classes, activation=head_activation, kernel_initializer=head_initializer)(flat)
        if 'weights' in kwargs.keys():
            model = Model(inp, out)
            model.set_weights(kwargs['weights'])
            return model
        return Model(inp, out)

    def sample_model(self, **kwargs):
        # SOLVED: finding FTL in the model
        shape = self._params_build['input_shape']
        shape_new = self._params_build['input_shape']
        if 'direction' in kwargs.keys() and 'nominator' in kwargs.keys():
            shape_new = self._operation(shape[:2], parameter=kwargs['nominator'],
                                        sign=self._DIRECTIONS[kwargs['direction']])
        if 'shape' in kwargs.keys():
            shape_new = kwargs['shape']
        params_sampled = self._params_build
        params_sampled['input_shape'] = (*shape_new, shape[2])
        # find the ftl layer
        ftl_index = 0
        while 'ftl' not in self.model.layers[ftl_index].name:
            ftl_index += 1
        # inpu does not have any weights
        ftl_index -= 1
        weights = self.model.get_weights()
        if 'weights' in kwargs.keys():
            weights = kwargs['weights']
        weights_ftl = expand_dims(squeeze(weights[ftl_index]), axis=0)
        noof_weights = weights_ftl.shape[0]
        replace_value = 1e-5
        if 'replace_value' in kwargs.keys():
            replace_value = kwargs['replace_value']
        weights_replace = ones((noof_weights, shape_new[0], shape_new[1], 1)) * replace_value
        for rep in range(noof_weights):
            # działa wyciąganie nawet fragmentu fft
            if shape_new[0] < shape[0]:
                weights_replace[rep] = expand_dims(weights_ftl[rep, :shape_new[0], :shape_new[1]], axis=-1)
            else:
                pads = [[0, shape_new[0]//2], [0, shape_new[1]//2]]
                weights_replace[rep] = expand_dims(pad(weights_ftl[rep, :, :], pad_width=pads, mode='constant',
                                                       constant_values=replace_value),
                                                   axis=-1)
        head = weights[-2:]
        size_new = shape_new[0] * shape_new[1]
        if shape_new[0] < shape[0]:
            head[0] = head[0][:size_new, :]
        else:
            pads = [[0, size_new - shape[0] * shape[1]], [0, 0]]
            head[0] = pad(head[0], pad_width=pads, mode='constant', constant_values=replace_value)
        return FourierBuilder(**params_sampled, weights=[weights_replace, *head])

    @staticmethod
    def _operation(value, parameter=2, sign='div'):
        assert sign in ['divide', 'div', '//', 'multiply', 'mult', '*']
        if sign in ['divide', 'div', '//']:
            return value[:2] // parameter
        elif sign in ['multiply', 'mult', '*']:
            return value[:2] * parameter


if __name__ == '__main__':
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = expand_dims(x_train / 255, axis=-1)[:3000]
    y_train = to_categorical(y_train, 10)[:3000]
    builder = FourierBuilder('fourier', input_shape=(28, 28, 1), noof_classes=10)
    builder.compile_model('adam', 'categorical_crossentropy', metrics=[CategoricalAccuracy(),
                                                                       TopKCategoricalAccuracy(k=5, name='top-5')])
    builder.train_model(100, x_data=x_train, y_data=y_train, call_stop=True, call_time=True, batch=64,
                        call_stop_kwargs={'baseline': 0.80,
                                          'monitor': 'categorical_accuracy',
                                          'patience': 2,
                                          })
    builder.save_model_info('test', 'Testing training pipeline', '../test')