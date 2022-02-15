from contextlib import redirect_stdout
from os import listdir, mkdir
from os.path import join, isdir
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, Conv2D
import tensorflow.keras.applications as apps
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, TopKCategoricalAccuracy
from numpy import squeeze, ones, pad
from sklearn.utils import shuffle
# Otherwise FTL cannot be called
from fourier_transform_layer.fourier_transform_layer import FTL
from utils.callbacks import TimeHistory, EarlyStopOnBaseline


# Generic builder
class ModelBuilder:
    # TODO: allowed kwargs
    def __init__(self, **kwargs):
        filepath = '../temp'
        if 'filepath' in kwargs.keys():
            filepath = kwargs['filepath']
        filename = 'dummy'
        if 'filename' in kwargs.keys():
            filename = kwargs['filename']
        if not isdir(filepath):
            mkdir(filepath)
        self._filename_original = filename
        self._filename = self._expand_filename(filename, filepath)
        self._filepath = filepath
        defaults = {'build' : {'model_type': 'any',
                               'input_shape': (8, 8, 1),
                               'noof_classes': -1,
                               'weights': None,
                               'freeze': 0,
                               },
                    'compile': {'optimizer': 'adam',
                                'loss': 'mse',
                                },
                    'train': {'epochs': 10,
                              'batch': 8,
                              'call_time': True,
                              'call_stop': True,
                              'call_stop_kwargs': {'baseline': 0.80,
                                                   'monitor': 'val_categorical_accuracy',
                                                   'patience': 2,
                                                   },
                              'call_checkpoint': True,
                              'call_checkpoint_kwargs': {'filepath': f'{filepath}/checkpoints/{self._filename}' +
                                                                     '_{epoch:03d}_.hdf5',
                                                         'monitor': 'val_categorical_accuracy',
                                                         'mode': 'auto',
                                                         'save_freq': 'epoch',
                                                         'save_weights_only': True,
                                                         'save_best_only': True,
                                                         },
                              'save_memory': True,
                              },
                    }
        if 'defaults' in kwargs.keys():
            defaults = self._update_build_defaults(defaults, kwargs['defaults'])
        self._METRICS = ['loss', 'acc', 'accuracy', 'categorical_accuracy', 'top-1', 'top-5',
                         'val_loss', 'val_acc', 'val_accuracy', 'val_categorical_accuracy', 'val_top-1', 'val_top-5']
        self._checkpoint_suffixes = self._make_suffixes(self._METRICS, length=1)
        self._checkfile_epoch_position = 0
        self._checkfile_temp_name = ''
        self._arguments = {}
        # noof_classes > 1
        self._ACTIVATIONS = {False: 'sigmoid',
                             True: 'softmax',
                             }
        for action in ['build', 'compile', 'train']:
            self._arguments.update({action: self._verify_arguments(defaults[action], **kwargs)})
        self._length = 0
        # Fourier weights reveal whether imag is used or not
        self._SUMMARIES = {'fourier': True,
                           'default': False,
                           }
        self._history = []
        self._evaluation = []
        # build the model
        self._model = self.build_model(**self._arguments['build'])

    def _build_model(self, **kwargs):
        return Model()

    # SOLVED: make freeze more generic
    @staticmethod
    def _freeze_model(model, freeze):
        result = model.copy()
        for layer in result.layers[1 : freeze + 1]:
            layer.trainable = False
        return result

    def _compile_model(self, optimizer, loss, **kwargs):
        self._model.compile(optimizer=optimizer, loss=loss, **kwargs)

    def _train_model(self, epochs, **kwargs):
        assert 'generator' in kwargs.keys() or sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2, \
        'Must provide either generator or full dataset.'
        # full set or generator
        flag_full_set = False
        flag_save_memory = False
        if 'save_memory' in kwargs.keys():
            flag_save_memory = kwargs['save_memory']
        split = 0
        if sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2:
            x_train = kwargs['x_data']
            y_train = kwargs['y_data']
            if 'validation_split' in kwargs.keys():
                split = kwargs['validation_split']
            self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                            dataset_size=x_train.shape[0], validation_split=split)
            # self._arguments['train']['call_checkpoint_kwargs']['save_best_only'] = False

            flag_full_set = True
        if 'generator' in kwargs.keys():
            data_gen = kwargs['generator']
            self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                            dataset='generator')
            if 'validation' in kwargs.keys():
                split = 1
                validation_data = kwargs['validation']
                self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                                validation_size=validation_data.shape[0])
        # callbacks
        callbacks = []
        flag_time = False
        flag_stop = False
        flag_checkpoint = False
        flag_checkpoint_best = True
        if 'call_time' in kwargs.keys() and kwargs['call_time']:
            callback_time = TimeHistory()
            callbacks.append(callback_time)
            flag_time = True
        if 'call_stop' in kwargs.keys() and kwargs['call_stop']:
            # metric and monitor names must be the same
            callback_stop = EarlyStopOnBaseline(**kwargs['call_stop_kwargs'])
            callbacks.append(callback_stop)
            self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                            call_stop_kwargs=callback_stop.get_kwargs())
            flag_stop = True
        if 'call_checkpoint' in kwargs.keys() and kwargs['call_checkpoint']:
            if not isdir(f'{self._filepath}/checkpoints'):
                mkdir(f'{self._filepath}/checkpoints')
            # prioritize validation metrics
            _v = any(['validation' in v for v in kwargs.keys()])
            filepath = self._manage_checkpoint_filepath(validation=_v)
            self._arguments['train']['call_checkpoint_kwargs']['filepath'] = filepath
            callback_checkpoint = ModelCheckpoint(**kwargs['call_checkpoint_kwargs'])
            callbacks.append(callback_checkpoint)
            flag_checkpoint = True
            flag_checkpoint_best = self._arguments['train']['call_checkpoint_kwargs']['save_best_only']

        # other train arguments
        batch = 8
        if any([f in ['batch', 'batch_size'] for f in kwargs.keys()]):
            batch = kwargs['batch']

        verbosity = 1
        if 'verbose' in kwargs.keys():
            verbosity = kwargs['verbose']

        hist = []
        tims = []

        if not flag_full_set:
            hist.append(self._model.fit(data_gen, epochs=epochs, batch_size=batch, shuffle=False, verbose=verbosity,
                                        validation_data=validation_data, callbacks=callbacks).history)
            if flag_time:
                # time callback will always be before stop callback if flag time is True, thus 0
                return self._merge_history_and_times(hist, callbacks[0].times)
            return hist

        # this way ensures that every model will receive the same data
        stop = False
        epoch = 0
        while not stop:
            if flag_checkpoint:
                callback_checkpoint.filepath = self._manage_checkpoint_filepath(epoch=epoch)
            x_train, y_train = shuffle(x_train, y_train, random_state=epoch)
            hist.append(self._model.fit(x_train, y_train, epochs=1, batch_size=batch, shuffle=False, verbose=verbosity,
                                       validation_split=split, callbacks=callbacks).history)
            epoch += 1
            stop = epoch >= epochs
            if flag_stop:
                call_index = [hasattr(call, 'stopped_training') for call in callbacks].index(True)
                stop = stop or callbacks[call_index].stopped_training
            # the flag makes sure times exist
            if flag_time:
                tims.append(callbacks[0].times[0])
            if flag_checkpoint and flag_checkpoint_best:
                callback_checkpoint.best = hist[-1][callback_checkpoint.monitor]
        self._model.save_weights(filepath=f'{self._filepath}/checkpoints/{self._filename}_finished.hdf5',
                                 overwrite=True)
        if flag_time:
            return self._merge_history_and_times(hist, tims)
        return hist

    def _evaluate_model(self, **kwargs):
        return self._model.evaluate(x=kwargs['x_data'], y=kwargs['y_data'], return_dict=True, verbose=2)

    # previous version caused not setting the weights, thus causing unexpected results for sampling
    def build_model(self, **kwargs):
        self._arguments['build'] = self._update_arguments(self._arguments['build'], **kwargs)
        model = self._build_model(**self._arguments['build'])
        # set the weights
        # this way ensures no key error
        if 'weights' in kwargs.keys() and kwargs['weights'] is not None:
            model.set_weights(kwargs['weights'])
        # freeze the model
        if 'freeze' in kwargs.keys() and kwargs['freeze'] != 0:
            model = self._freeze_model(model, kwargs['freeze'])
        return model


    def build_model_from_info(self):
        return self._build_model(**self._arguments['build'])

    def compile_model(self, optimizer, loss, **kwargs):
        metrics = []
        if 'metrics' in kwargs.keys():
            for metric in kwargs['metrics']:
                if metric == 'accuracy':
                    metrics.append(Accuracy())
                if metric == 'categorical_accuracy':
                    metrics.append(CategoricalAccuracy())
                if 'top' in metric:
                    metrics.append(TopKCategoricalAccuracy(k=5, name='top-5'))
        self._arguments['compile'] = self._update_arguments(self._arguments['compile'],
                                                          optimizer=optimizer, loss=loss, metrics=metrics)
        self._compile_model(**self._arguments['compile'])

    def compile_model_from_info(self):
        self._compile_model(**self._arguments['compile'])

    def train_model(self, epochs, **kwargs):
        self._arguments['train'] = self._update_arguments(self._arguments['train'], epochs=epochs, **kwargs)
        self._history = self._train_model(**self._arguments['train'])

    def evaluate_model(self, **kwargs):
        self._evaluation = self._evaluate_model(**kwargs)

    def prepare_model_from_info(self):
        self._model = self.build_model_from_info()
        self.compile_model_from_info()
        return self._model

    @staticmethod
    def _verify_arguments(arguments, **kwargs):
        result = arguments.copy()
        for key in result.keys():
            if key not in kwargs.keys():
                continue
            result[key] = kwargs[key]
        return result

    @staticmethod
    def _update_arguments(arguments, **kwargs):
        result = arguments.copy()
        for key in kwargs.keys():
            if type(kwargs[key]) is dict:
                for key_interior in kwargs[key].keys():
                    result[key].update({key_interior: kwargs[key][key_interior]})
                continue
            # just making sure - should never occur - helpful in kwargs extraction
            if key not in result.keys():
                result.update({key: kwargs[key]})
                continue
            result[key] = kwargs[key]
        return result

    @staticmethod
    def _update_build_defaults(arguments, defaults):
        result = arguments.copy()
        for key in defaults.keys():
            result['build'].update({key: defaults[key]})
        return result

    def _manage_checkpoint_filepath(self, **kwargs):
        _flag_save_memory = self._arguments['train']['save_memory']
        filepath_checkpoint = self._arguments['train']['call_checkpoint_kwargs']['filepath'][:-5]
        if 'epoch' in kwargs.keys():
            epoch = kwargs['epoch']
            if epoch == 0:
                # find where the epoch info is kept in the filename
                self._checkfile_epoch_position = filepath_checkpoint.split('_').index('{epoch:03d}')
                return filepath_checkpoint + '.hdf5'
            splits = filepath_checkpoint.split('_')
            splits[self._checkfile_epoch_position] = f'{epoch:03d}'
            if not _flag_save_memory or epoch % 10 == 0:
                filepath_checkpoint = '_'.join(splits)
        if 'validation' not in kwargs.keys() or _flag_save_memory:
            return filepath_checkpoint + '.hdf5'
        monitor = self._arguments['train']['call_checkpoint_kwargs']['monitor']
        if not kwargs['validation']:
            if 'loss' not in filepath_checkpoint:
                filepath_checkpoint += self._checkpoint_suffixes['loss'] + '{loss:.3f}_'
            # ensure validation data exists
            assert 'val' not in monitor, f'Val_{monitor} will be unavailable - no validation data.'
        else:
            if 'val' not in monitor:
                monitor = 'val_' + monitor
                self._arguments['train']['call_checkpoint_kwargs']['monitor'] = monitor
            if 'loss' not in filepath_checkpoint:
                filepath_checkpoint += self._checkpoint_suffixes['val_loss'] + '{val_loss:.3f}_'
        if monitor not in filepath_checkpoint:
            filepath_checkpoint += self._checkpoint_suffixes[monitor] + '{' + monitor + ':.3f}'
        return filepath_checkpoint + '.hdf5'

    # Text manipulation methods
    def save_model_info(self, notes='', extension='', **kwargs):
        assert type(notes) == str, 'Notes must be a string.'
        self._update_all_lengths()
        if 'fourier' in self._arguments['build']['model_type']:
            summary = self._SUMMARIES['fourier']
        else:
            summary = self._SUMMARIES['default']
        if 'summary' in kwargs.keys():
            summary = kwargs['summary']
        format_used = extension
        if len(format_used) < 1:
            format_used = '.txt'
        if '.' not in format_used:
            format_used = '.' + format_used
        with open(join(self._filepath, self._filename + format_used), 'w') as fil:
            for action in ['build', 'compile', 'train']:
                fil.write(f'{action.capitalize()} arguments\n')
                fil.write(self._prepare_argument_text(self._arguments[action]))
            fil.write(notes + '\n')
            # the prepare method accepts list
            if len(self._evaluation) > 0:
                suffixes = self._make_suffixes(metrics=[key for key in self._evaluation.keys()], length=-1, sign='_')
                eva_text = self._prepare_metrics_text([self._evaluation], suffixes)
                fil.write(f'Evaluation: \n{eva_text}')
            if len(self._history) > 0:
                suffixes = self._make_suffixes(metrics=[key for key in self._history[0].keys()], length=-1, sign='_')
                hist_text = self._prepare_metrics_text(self._history, suffixes)
                fil.write(f'Training history: \n{hist_text}')
            # SOLVED: move summary to different file - went back on the idea
            if summary:
                # SOLVED: layer, weights saving to method
                # layers[1:] - Input has no weights
                if 'layers' in self._arguments['build']:
                    fil.write('Layers list:\n')
                    fil.write(self._layer_weight_summary(layers=self._model.layers[1:],
                                                         arguments=self._arguments['build']['layers'],
                                                         summary=summary))
                else:
                    fil.write('Weights summary:\n')
                    fil.write(self._layer_weight_summary(layers=self._model.layers[1:]))
                with redirect_stdout(fil):
                    self._model.summary()

    def _layer_weight_summary(self, layers, **kwargs):
        _arguments = [None for _ in layers]
        if 'arguments' in kwargs.keys():
            _arguments = kwargs['arguments']
        summary = False
        if 'summary' in kwargs.keys():
            summary = kwargs['summary']
        result = ''
        for layer_got, layer_args in zip(layers, _arguments):
            weight = layer_got.weights
            weight_text = ''
            if type(weight) is list:
                weight_text += '|'.join([str(w.shape) for w in weight])
            else:
                weight_text = str(layer_got.weights.shape)
            result += self._prepare_paired_text(layer_got.name, weight_text)
            if not layer_args:
                continue
            layer_args = {layer_got.name: list(layer_args.values())[0]}
            result += self._prepare_argument_text(layer_args, summary) + self._prepare_paired_text()
        return result

    def _calculate_lengths(self, arguments):
        # protection from weights impact on length of text
        length_keys = self._length_calculator(arguments.keys())
        length_vals = self._length_calculator(arguments.values())
        return max([length_keys, length_vals])

    def _update_all_lengths(self):
        for action in ['build', 'compile', 'train']:
            self._update_length(self._calculate_lengths(self._update_arguments_text(self._arguments[action])))

    def _update_length(self, new_candidate):
        self._length = max([self._length, new_candidate])

    # method for text cleanup
    def _prepare_argument_text(self, arguments=None, summary=False):
        text_build = ''
        walkover = self._update_arguments_text(arguments, summary)
        for key, value in zip(walkover.keys(), walkover.values()):
            text_build += self._prepare_paired_text(key, value)
        return text_build

    def _prepare_paired_text(self, *args):
        left = "#" * self._length
        right = "X" * self._length
        if len(args) > 0:
            left = str(args[0])
        if len(args) > 1:
            right = str(args[1])
        return f'\t{left:{self._length}} - {right.rjust(self._length)}\n'

    # a method to change the values of argument holders
    def _update_arguments_text(self, arguments, summary=False):
        result = {}
        for key in arguments.keys():
            # list of different arguments
            if key == 'layers' and not summary:
                result.update({key: [list(layer)[0] for layer in arguments[key]]})
                continue
            if type(arguments[key]) is list:
                if 'weights' in key:
                    result.update({f'{key}': 'Loaded'})
                    continue
                for it, key_interior in enumerate(arguments[key]):
                    to_update = self._check_for_name(key_interior)
                    result.update({f'{key}_{it:03d}': to_update})
                continue
            if type(arguments[key]) is dict:
                for key_interior in arguments[key].keys():
                    if key_interior in ['filename', 'filepath']:
                        continue
                    to_update = self._check_for_name(key_interior)
                    result.update({f'{key}-{to_update}': arguments[key][key_interior]})
                continue
            to_update = self._check_for_name(arguments[key])
            if key in ['x_data', 'y_data']:
                continue
            if key not in result.keys():
                result.update({key: to_update})
                continue
            result[key] = to_update
        return result

    def _make_suffixes(self, metrics, length=1, sign=''):
        result = {}
        _WIDTHS = {'loss': 4,
                   'acc': 3,
                   'top': 3,
                   'time': 4,
                   'default': 3,
                   }
        for metric in metrics:
            suffix = metric.split('_')
            _suffix = metric.split('-')
            if length > 0:
                length_used = length
            else:
                length_used = self._determine_text_width(metric, _WIDTHS)
            result.update({metric: sign.join([s[:length_used] for s in suffix])})
            if len(_suffix) <= 1:
                continue
            result[metric] = sign.join([s[:length_used] for s in suffix]) + _suffix[-1][0]
        return result

    def _prepare_metrics_text(self, history, suffixes=None):
        _MAX_TRAILS = {'loss': 6,
                       'acc': 4,
                       'top': 4,
                       'time': 6,
                       'default': 6,
                       }
        _MAX_WIDTHS = {'loss': 10,
                       'acc': 8,
                       'top': 8,
                       'time': 12,
                       'default': 9,
                       }
        text_result = ''
        text_result += 'epochs'.center(15) + ' -- '
        for key in history[0].keys():
            key_str = key
            if suffixes:
                key_str = suffixes[key]
            width = self._determine_text_width(key, _MAX_WIDTHS)
            text_result += str(key_str).center(max([len(key_str), width])) +' || '
        text_result += '\n'
        for epoch in range(len(history)):
            epoch_str = str(epoch)
            # may be possible to use {epoch:0xd}
            while len(epoch_str) < len(str(len(history))):
                epoch_str = '0' + epoch_str
            # do not expect more than 10k training epochs
            text_result += ('Epoch ' + epoch_str).center(15) +' -- '
            for key, value in zip(history[epoch].keys(), history[epoch].values()):
                key_str = key
                if suffixes:
                    key_str = suffixes[key]
                value_used = value
                if type(value) is list:
                    value_used = value[0]
                width = self._determine_text_width(key, _MAX_WIDTHS)
                trail = self._determine_text_width(key, _MAX_TRAILS)
                text_result += f'{value_used:{max([len(key_str), width])}.{trail}f} || '
            text_result += '\n'
        return text_result

    @staticmethod
    def _determine_text_width(metric, widths):
        try:
            index = [key in metric for key in widths.keys()].index(True)
            return widths[list(widths.keys())[index]]
        except ValueError:
            return widths['default']

    @staticmethod
    def _expand_filename(filename, filepath=''):
        # List OF
        loof_files = [f for f in listdir(filepath) if filename in f]
        it = len(loof_files)
        date = dt.now().strftime('%Y-%m-%d_%H_%M_%S')
        filename_expanded = f'{filename}_{date}_{it:03d}'
        return filename_expanded

    @staticmethod
    def _check_for_name(checked_property):
        if hasattr(checked_property, 'name'):
            return checked_property.name

        return checked_property

    @staticmethod
    def _length_calculator(loof_values):
        return max([len(str(f)) for f in loof_values if len(str(f)) < 100 and type(f) is not dict])

    @staticmethod
    def _merge_history_and_times(history, times):
        assert len(history) == len(times), 'History and times are not the same length.'
        history_end = history
        for it, time in enumerate(times):
            history_end[it].update({'time': time})
        return history_end

    # Properties
    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return self._history

    @property
    def evaluation(self):
        return self._evaluation


# Custom model builder - can build any model (including hybrid), based on layer information
class CustomBuilder(ModelBuilder):
    # TODO: default sampling initializations
    def __init__(self, layers, input_shape=(32, 32, 3), noof_classes=1, **kwargs):
                    # copied from keras: https://keras.io/api/layers/convolution_layers/convolution2d/
        defaults = {'conv2d': {'filters': 128,
                               'kernel_size': 3,
                               'strides': (1, 1),
                               'padding': "valid",
                               'data_format': None,
                               'dilation_rate': (1, 1),
                               'groups': 1,
                               'activation': None,
                               'use_bias': True,
                               'kernel_initializer': "glorot_uniform",
                               'bias_initializer': "zeros",
                               'kernel_regularizer': None,
                               'bias_regularizer': None,
                               'activity_regularizer': None,
                               'kernel_constraint': None,
                               'bias_constraint': None,
                               },
                    # copied from keras: https://keras.io/api/layers/core_layers/dense/
                    'dense': {'units': 1,
                              'activation': None,
                              'use_bias': True,
                              'kernel_initializer': "glorot_uniform",
                              'bias_initializer': "zeros",
                              'kernel_regularizer': None,
                              'bias_regularizer': None,
                              'activity_regularizer': None,
                              'kernel_constraint': None,
                              'bias_constraint': None,
                              },
                    'flatten': {},
                    'ftl': {'activation': None,
                            'kernel_initializer': 'he_normal',
                            'use_imaginary': True,
                            'inverse': False,
                            'use_bias': False,
                            'bias_initializer': 'zeros',
                            'normalize_to_image_shape': False,
                            'phase_training': False,
                            },
                    }
        self._SAMPLING_DIRECTIONS = {'up': '*',
                                     'down': '//',
                                     }
        # layers - a list of dicts
        _NAMES = list(defaults.keys())
        self._UNSAMPLED = [name for name in _NAMES if name not in ['ftl', 'dense']]
        self._REPLACE_VALUE = 1e-5
        # TODO: name checking
        # l = _NAMES[0] in layers[0].keys()
        # assert all(_NAMES in layer.keys() for layer in layers), \
        #     f'Unsupported name. Supported names: {_NAMES}.'
        _layers = []
        for layer in layers:
            for key, value in zip(layer.keys(), layer.values()):
                _layer = self._verify_arguments(defaults[key], **value)
                _layers.append({key : _layer})
        if 'model_type' not in kwargs.keys():
            kwargs.update({'model_type' : 'custom'})
        super(CustomBuilder, self).__init__(input_shape=input_shape,
                                            noof_classes=noof_classes,
                                            defaults={'layers': _layers},
                                            **kwargs)

    def _build_model(self, layers, input_shape, noof_classes, **kwargs):
        inp = Input(input_shape)
        arch, flat = self._return_layer(layers[0], inp)
        for layer in layers[1:-1]:
            arch, flat = self._return_layer(layer, arch)
        layer = layers[-1]
        # make as many units as no of classes
        if 'dense' not in layer.keys():
            arch, flat = self._return_layer(layer, arch)
            return Model(inp, arch)
        _layer = layer
        for key, values in zip(layer.keys(), layer.values()):
            values['units'] = self._arguments['build']['noof_classes']
            values['activation'] = self._ACTIVATIONS[noof_classes > 1]
            _layer.update({key: values})
        arch, flat = self._return_layer(_layer, arch)
        return Model(inp, arch)

    @staticmethod
    def _return_layer(layer, previous):
        arguments = list(layer.values())[0]
        if 'conv2d' in layer.keys():
            return Conv2D(**arguments)(previous), False
        if 'ftl'  in layer.keys():
            return FTL(**arguments)(previous), False
        if 'flatten' in layer.keys():
            return Flatten()(previous), True
        if 'dense' not in layer.keys():
            return None
        return Dense(**arguments)(previous), True

    def _sample_model(self, **kwargs):
        # SOLVED: finding FTL in the model
        arguments_sampled = self._arguments['build'].copy()
        shape = arguments_sampled['input_shape']
        shape_new = shape
        if 'direction' in kwargs.keys() and 'nominator' in kwargs.keys():
            shape_new = self._operation(shape[:2], nominator=kwargs['nominator'],
                                        sign=self._SAMPLING_DIRECTIONS[kwargs['direction']])
        if 'shape' in kwargs.keys():
            shape_new = kwargs['shape']
        arguments_sampled['input_shape'] = (*shape_new, shape[2])
        # final shape
        shape_new = arguments_sampled['input_shape']
        model_weights = self._model.get_weights()
        model_layers = self._model.layers
        replace_value = self._REPLACE_VALUE
        if 'replace_value' in kwargs.keys():
            replace_value = kwargs['replace_value']
        weights_result = []
        size_new = shape_new[0] * shape_new[1] * shape_new[2]
        if 'weights' in kwargs.keys():
            model_weights = kwargs['weights']
        # SOLVED: other layers
        # gather layers and weights
        gathered_weights = {}
        names = []
        for layer in model_layers:
            for rep in range(len(layer.weights)):
                names.append(layer.name)
        for it, name in enumerate(names):
            if name not in gathered_weights.keys():
                gathered_weights.update({name: [model_weights[it]]})
            else:
                gathered_weights[name].append(model_weights[it])
        passed_ftl = False
        for layer_name, weights in zip(gathered_weights.keys(), gathered_weights.values()):
            if 'ftl' in layer_name:
                # now its known that weights are FTL (1u2, X, X, C) and maybe bias (1u2, X, X, C)
                # additional extraction from list (thus [0])
                # includes bias
                for step in range(len(weights)):
                    weights_ftl = weights[step]
                    noof_weights = weights_ftl.shape[0]
                    weights_replace = ones((noof_weights, shape_new[0], shape_new[1], shape[2])) * replace_value
                    for rep in range(noof_weights):
                        for ch in range(shape[2]):
                            if shape_new[0] < shape[0]:
                                weights_replace[rep, :, :, ch] = weights_ftl[rep, :shape_new[0], :shape_new[1], ch]
                            else:
                                pads = [[0, int(shn - sh)] for shn, sh in zip(shape_new[:2], shape[:2])]
                                weights_replace[rep, :, :, ch] = pad(squeeze(weights_ftl[rep, :, :, ch]), pad_width=pads,
                                                                     mode='constant', constant_values=replace_value)
                    weights_result.append(weights_replace)
                passed_ftl = True
                continue
            if 'dense' in layer_name and passed_ftl:
                # 0 - kernel, 1 - bias
                if shape_new[0] < shape[0]:
                    weights_result.append(weights[0][:size_new, :])
                else:
                    pads = [[0, size_new - shape[0] * shape[1] * shape[2]], [0, 0]]
                    pd = pad(weights[0], pad_width=pads, mode='constant', constant_values=replace_value)
                    weights_result.append(pd)
                weights_result.append(weights[1])
                passed_ftl = False
                continue
            # other layers which should not be sampled
            weights_result.append(weights)
        arguments_sampled['weights'] = weights_result
        builder = CustomBuilder(filename=self._filename_original, filepath=self._filepath, **arguments_sampled)
        for action in ['compile', 'train']:
            builder._arguments[action] = self._arguments[action]
        if 'compile' in kwargs.keys() and kwargs['compile']:
            builder.compile_model(**self._arguments['compile'])
        return builder

    def sample_model(self, **kwargs):
        return self._sample_model(**kwargs)

    @staticmethod
    def _operation(value, nominator=2, sign='div'):
        assert sign in ['divide', 'div', '//', 'multiply', 'mult', '*']
        if sign in ['divide', 'div', '//']:
            return [v // nominator for v in value]
        elif sign in ['multiply', 'mult', '*']:
            return [v * nominator for v in value]


# Standard CNNs for classification
class CNNBuilder(ModelBuilder):
    def __init__(self, model_type='mobilenet', input_shape=(32, 32, 3), noof_classes=1, **kwargs):
        super(CNNBuilder, self).__init__(model_type=model_type,
                                         input_shape=input_shape,
                                         noof_classes=noof_classes, **kwargs)

    def _build_model(self, model_type, input_shape, noof_classes, weights=None, freeze=0, **kwargs):
        model_type_low = model_type.lower()
        if 'mobilenet' in model_type_low:
            if '2' not in model_type_low:
                # load Mobilenet
                backbone = apps.mobilenet.MobileNet(input_shape=input_shape, weights=weights, include_top=False)
            else:
                # load Mobilenetv2
                backbone = apps.mobilenet_v2.MobileNetV2(input_shape=input_shape, weights=weights, include_top=False)
        elif 'vgg' in model_type_low:
            if '16' in model_type_low:
                backbone = apps.vgg16.VGG16(input_shape=input_shape, weights=weights, include_top=False)
            elif '19' in model_type_low:
                backbone = apps.vgg19.VGG19(input_shape=input_shape, weights=weights, include_top=False)
        elif 'resnet' in model_type_low:
            if '50' in model_type_low:
                backbone = apps.resnet_v2.ResNet50V2(input_shape=input_shape, weights=weights, include_top=False)
            elif '101' in model_type_low:
                backbone = apps.resnet_v2.ResNet101V2(input_shape=input_shape, weights=weights, include_top=False)
        # update BatchNormalization momentum - otherwise several models (MobilenetV2, VGG16) do not work
        for layer in backbone.layers:
            if type(layer) != type(BatchNormalization):
                continue
            layer.momentum=0.9
        architecture = backbone.output
        # Classify
        flat = Flatten()(architecture)
        act = 'softmax'
        if noof_classes == 1:
            act = 'sigmoid'
        out = Dense(noof_classes, activation=act)(flat)
        return Model(inputs=[backbone.input], outputs=[out])


# Fourier Model for classification
class FourierBuilder(CustomBuilder):
    def __init__(self, model_type='fourier', input_shape=(32, 32, 1), noof_classes=1, **kwargs):
        # just to be safe
        layers = [{'ftl': {'use_imaginary': True,
                           'inverse': 'inverse' in model_type,
                           }},
                  {'flatten': {}},
                  {'dense': {}},
                  ]
        if 'layers' in kwargs.keys():
            layers = self._verify_arguments(layers, kwargs['layers'])
        super(FourierBuilder, self).__init__(model_type=model_type,
                                             input_shape=input_shape,
                                             noof_classes=noof_classes,
                                             layers=layers,
                                             **kwargs)


if __name__ == '__main__':
    print(FourierBuilder())