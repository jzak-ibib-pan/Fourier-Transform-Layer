from contextlib import redirect_stdout
from os import listdir, mkdir
from os.path import join, isdir
from datetime import datetime as dt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, Conv2D, Concatenate, concatenate, Conv2DTranspose
from tensorflow.keras.layers import Add, ZeroPadding2D, Activation, MaxPooling2D, AveragePooling2D, ReLU, DepthwiseConv2D, UpSampling2D, Dropout, Reshape
from tensorflow import image as tfimage
import tensorflow.keras.applications as apps
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, TopKCategoricalAccuracy, AUC, BinaryAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow import data as tfdata
from tensorflow.keras.optimizers import Adam, SGD
from numpy import squeeze, pad, array, argmax, expand_dims, ones_like, mod
from numpy import max as np_max
from numpy import resize as resize_array
from cv2 import resize as resize_image
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
# Otherwise FTL cannot be called
from fourier_transform_layer.fourier_transform_layer import FTL, FTLSuperResolution
from utils.callbacks import TimeHistory, EarlyStopOnBaseline
from utils.sampling import DIRECTIONS, sampling_calculation
from utils.losses import ssim
from inspect import isgenerator
from types import GeneratorType


def unet_modified(input_shape=(None, None, 1), categories=2, **kwargs):
    # updated merge to to concacenate to remove warnings
    # updated Model
    # increased number of filters of up8 and beyond
    # added BatchNorm after every upsampling (ups)
    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    up6 = BatchNormalization()(up6)
    ###########################################################################
    # JZ update?
    merge6 = concatenate([drop4, up6], axis=3)
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    ###########################################################################
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization()(up7)
    ###########################################################################
    # JZ update?
    merge7 = concatenate([conv3, up7], axis=3)
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    ###########################################################################
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    ###########################################################################
    # JZ update?
    # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(192, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(192, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    # JZ update?
    merge9 = concatenate([conv1, up9], axis=3)
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(96, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(24, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    if categories == 1:
        conv10 = Conv2D(categories, 1, activation='sigmoid', strides=(1, 1))(conv9)
    else:
        conv10 = Conv2D(categories, 1, activation='softmax', strides=(1, 1))(conv9)

    # conv10 = Conv2D(categories, 1, activation = 'sigmoid', strides=(1, 1))(conv9)
    model = Model(inputs, conv10)
    return model


# TODO: error messages
# SOLVED: parameters to kwargs
# Generic builder
class ModelBuilder:
    # TODO: allowed kwargs
    # TODO: dataset name - will be reported by the data_loader, along with other kwargs
    def __init__(self, **kwargs):
        # placeholder
        self._allowed_kwargs = self._get_allowed_kwargs()
        # placeholder
        # kwargs = self._check_allowed_kwargs(allowed, kwargs)
        filepath = ['../temp' if 'filepath' not in kwargs.keys() else kwargs['filepath']][0]
        filename = ['dummy' if 'filename' not in kwargs.keys() else kwargs['filename']][0]
        if not isdir(filepath):
            mkdir(filepath)
            mkdir(join(filepath, 'checkpoints'))
        self._filename_original = filename
        self._filename = self._expand_filename(filename, filepath)
        self._filepath = filepath
        # here update received defaults as input - not possible to streamline
        defaults = self._get_default_arguments(filepath, self._filename)
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
                           'custom': True,
                           'default': False,
                           }
        self._history = []
        self._evaluation = []
        # build the model
        self._model = self.build_model(**self._arguments['build'])

    # Default or allowed kwargs
    @staticmethod
    def _define_default_arguments(filepath, filename):
        defaults = {'build' : {'model_type': 'any',
                               'input_shape': (8, 8, 1),
                               'noof_classes': -1,
                               'weights': None,
                               'freeze': 0,
                               },
                    'compile': {'optimizer': 'adam',
                                'loss': 'mse',
                                'run_eagerly': False,
                                },
                    'train': {'epochs': 10,
                              'batch': 4,
                              'call_time': True,
                              'call_stop': True,
                              'call_stop_kwargs': {'baseline': 0.80,
                                                   'monitor': 'val_categorical_accuracy',
                                                   'patience': 2,
                                                   },
                              'call_checkpoint': True,
                              'call_checkpoint_kwargs': {'filepath': f'{filepath}/checkpoints/{filename}' +
                                                                     '_{epoch:03d}_.hdf5',
                                                         'monitor': 'val_categorical_accuracy',
                                                         'mode': 'auto',
                                                         'save_freq': 'epoch',
                                                         'save_weights_only': True,
                                                         'save_best_only': True,
                                                         },
                              'save_memory': True,
                              'save_final': True,
                              },
                    }
        return defaults

    def _get_default_arguments(self, filepath, filename):
        return self._define_default_arguments(filepath, filename)

    @staticmethod
    def _define_allowed_kwargs():
        allowed = {'build' : {'model_type': ['any'],
                              # 'input_shape': (8, 8, 1), TODO: assertion
                              # 'noof_classes': -1, TODO: assertion
                              # 'weights': None,
                              # 'freeze': 0, TODO: assertion
                               },
                    'compile': {'optimizer': ['adam', 'sgd'],
                                'loss': ['mse'],
                                # 'run_eagerly': False, TODO: assertion
                                },
                    # 'train': {'epochs': 10, TODO: assertion
                    #           'batch': 8, TODO: assertion
                    #           'call_time': True, TODO: assertion
                    #           'call_stop': True, TODO: assertion
                    #           'call_stop_kwargs': {'baseline': 0.80,
                    #                                'monitor': 'val_categorical_accuracy',
                    #                                'patience': 2,
                    #                                },
                    #           'call_checkpoint': True, TODO: assertion
                    #           'call_checkpoint_kwargs': {'filepath': f'{filepath}/checkpoints/{filename}' +
                    #                                                  '_{epoch:03d}_.hdf5',
                    #                                      'monitor': 'val_categorical_accuracy',
                    #                                      'mode': 'auto',
                    #                                      'save_freq': 'epoch',
                    #                                      'save_weights_only': True,
                    #                                      'save_best_only': True,
                    #                                      },
                    #           'save_memory': True, TODO: assertion
                    #           'save_final': True, TODO: assertion
                    #           },
                    }
        return allowed

    def _get_allowed_kwargs(self):
        return self._define_allowed_kwargs()

    # placeholder
    # TODO: allow also keras class imports (Layers, Losses, Optimizers, etc.)
    @staticmethod
    def _check_allowed_kwargs(allowed, checked):
        return None

    # Model preparation
    # SOLVED: make freeze more generic
    @staticmethod
    def _freeze_model(model, freeze):
        for layer in model.layers[1 : freeze + 1]:
            layer.trainable = False
        return model

    # previous version caused not setting the weights, thus causing unexpected results for sampling
    # wrapper to build with parameters and freeze
    def build_model(self, **kwargs):
        self._arguments['build'] = self._update_arguments(self._arguments['build'], **kwargs)
        model = self._build_model(**self._arguments['build'])
        # set the weights
        # this way ensures no key error
        if 'weights' in kwargs.keys() and kwargs['weights'] is not None and type(kwargs['weights']) is list:
            model.set_weights(kwargs['weights'])
        # freeze the model
        if 'freeze' in kwargs.keys() and kwargs['freeze'] != 0:
            model = self._freeze_model(model, kwargs['freeze'])
        return model

    # placeholder
    def build_model_from_info(self):
        return self._build_model(**self._arguments['build'])

    # building model
    def _build_model(self, **kwargs):
        return Model()

    # wrapper
    def compile_model(self, optimizer, loss, **kwargs):
        self._arguments['compile'] = self._update_arguments(self._arguments['compile'],
                                                            optimizer=optimizer, loss=loss, **kwargs)
        self._compile_model(**self._arguments['compile'])

    # placeholder
    def compile_model_from_info(self):
        self._compile_model(**self._arguments['compile'])

    def _get_optimizer(self, optimizer, lr=1e-3):
        if optimizer == "adam":
            return Adam(learning_rate=lr)
        if optimizer == "sgd":
            return SGD(learning_rate=lr)

    def _compile_model(self, optimizer, loss, **kwargs):
        _loss = loss if loss != 'ssim' else ssim
        metrics = []
        lr = 1e-3
        if 'learning_rate' in kwargs.keys():
            lr = kwargs['learning_rate']
            # inputting learning rate leads to later issues with compilation
            kwargs.pop('learning_rate')
        _optimizer = self._get_optimizer(optimizer, lr)
        if 'metrics' not in kwargs.keys():
            self._model.compile(optimizer=_optimizer, loss=_loss, **kwargs)
            return
        if any([type(metric) is not str for metric in kwargs['metrics']]):
            self._model.compile(optimizer=_optimizer, loss=_loss, **kwargs)
            return
        for metric in kwargs['metrics']:
            if metric == 'accuracy':
                metrics.append(Accuracy())
            if metric == "binary_accuracy":
                metrics.append(BinaryAccuracy())
            if metric == 'categorical_accuracy':
                metrics.append(CategoricalAccuracy())
            if metric == 'topk_categorical_accuracy':
                metrics.append(TopKCategoricalAccuracy(k=5, name='top-5'))
            if metric == 'mAUC':
                metrics.append(AUC(multi_label=True, name='mAUC', num_thresholds=1000))
            if metric == 'mAUC':
                metrics.append(AUC(multi_label=False, name='uAUC', num_thresholds=1000))
        kwargs['metrics'] = metrics
        self._arguments['compile'] = self._verify_arguments(self._arguments['compile'],
                                                            optimizer=_optimizer, loss=_loss, **kwargs)
        self._model.compile(optimizer=_optimizer, loss=_loss, **kwargs)
        return

    # wrapper
    def train_model(self, epochs, **kwargs):
        self._arguments['train'] = self._update_arguments(self._arguments['train'], epochs=epochs, **kwargs)
        self._history = self._train_model(**self._arguments['train'])

    def _train_model(self, epochs, **kwargs):
        assert 'generator' in kwargs.keys() or sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2, \
        'Must provide either generator or full dataset.'
        # full set or generator
        flag_full_set = False
        # TODO: save_memory implementation
        flag_save_memory = [False if 'save_memory' not in kwargs.keys() else kwargs['save_memory']][0]
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
            if type(data_gen) == tfdata.Dataset:
                steps = data_gen.cardinality().numpy()
            else:
                steps = 1000
            steps = [steps if 'steps' not in kwargs.keys() else kwargs['steps']][0]
            self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                              dataset='generator')
            validation_data = None
            validation_size = 0
            if 'validation' in kwargs.keys():
                # split = 1, because generator and validation are two different datasets
                split = 1
                validation_data = kwargs['validation']
                # TODO: make sure that validation_data has shape
                if type(validation_data) == tfdata.Dataset:
                    validation_size = validation_data.cardinality().numpy()
                elif type(validation_data) == array:
                    validation_size = validation_data.shape[0]
                validation_size = [validation_size if 'val_steps' not in kwargs.keys() else kwargs['val_steps']][0]
                self._arguments['train'] = self._update_arguments(self._arguments['train'],
                                                                  validation_size=validation_size)


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
        batch = [8 if 'batch' not in kwargs.keys() else kwargs['batch']][0]
        batch = [batch if 'batch_size' not in kwargs.keys() else kwargs['batch_size']][0]
        verbosity = [1 if 'verbose' not in kwargs.keys() else kwargs['verbose']][0]

        hist = []
        tims = []

        if not flag_full_set:
            # train on generator
            hist.append(self._model.fit(data_gen, epochs=epochs, batch_size=batch, steps_per_epoch=steps,
                                        shuffle=False, verbose=verbosity,
                                        validation_data=validation_data, validation_steps=validation_size,
                                        callbacks=callbacks).history)
            self.check_if_final_save(kwargs)
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
        self.check_if_final_save(kwargs)
        if flag_time:
            return self._merge_history_and_times(hist, tims)
        return hist

    def check_if_final_save(self, train_kwargs):
        if 'save_final' not in train_kwargs.keys():
            return False
        if not train_kwargs['save_final']:
            return False
        self._model.save_weights(filepath=f'{self._filepath}/checkpoints/{self._filename}_trained.hdf5',
                                 overwrite=True)
        return True

    # wrapper
    def evaluate_model(self, **kwargs):
        # REMEMBER: first load weights, then compile the model to keep the results consistent with after training
        self._evaluation = self._evaluate_model(**kwargs)

    def _evaluate_model(self, **kwargs):
        evaluation = {}
        if sum([f in ['x_data', 'y_data'] for f in kwargs.keys()]) == 2:
            if 'auc' in kwargs.keys() and kwargs['auc']:
                y_true = kwargs['y_data']
                y_pred = self._model.predict(x=kwargs['x_data'],
                                             batch_size=self._arguments['train']['batch'])
                y_pred = to_categorical(argmax(y_pred, axis=1),
                                        num_classes=self._arguments['build']['noof_classes'])
                mauc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro', multi_class='ova')
                wauc = roc_auc_score(y_true=y_true, y_score=y_pred, average='weighted', multi_class='ova')
            eva = self._model.evaluate(x=kwargs['x_data'], y=kwargs['y_data'],
                                       batch_size=self._arguments['train']['batch'],
                                       return_dict=True, verbose=1)
        # TODO: on generator implicit args and kwargs
        elif 'generator' in kwargs.keys():
            assert 'steps' in kwargs.keys(), 'Must provide no. of steps.'
            if 'auc' in kwargs.keys() and kwargs['auc']:
                y_true = kwargs['y_data']
                y_pred = to_categorical(self._model.predict(x=kwargs['generator'], steps=kwargs['steps'],
                                                            batch_size=self._arguments['train']['batch'],),
                                        num_classes=self._arguments['build']['noof_classes'])
                mauc = roc_auc_score(y_true=y_true, y_score=y_pred, average='macro', multi_class='ova')
                wauc = roc_auc_score(y_true=y_true, y_score=y_pred, average='weighted', multi_class='ova')
            eva = self._model.evaluate(x=kwargs['generator'], steps=kwargs['steps'],
                                       batch_size=self._arguments['train']['batch'],
                                       return_dict=True, verbose=1)
        for key in eva.keys():
            evaluation.update({key: eva[key]})
        if 'auc' in kwargs.keys() and kwargs['auc']:
            evaluation.update({'mAUCsci': mauc})
            evaluation.update({'wAUCsci': wauc})
        return evaluation

    # placeholder
    def prepare_model_from_info(self):
        self._model = self.build_model_from_info()
        self.compile_model_from_info()
        return self._model

    # Arguments (build, compile, train, evaluate) checking and setting
    # method copies key-value pairs from kwargs into arguments, only if arguments contains the key
    @staticmethod
    def _verify_arguments(arguments, **kwargs):
        result = arguments.copy()
        for key in result.keys():
            if key not in kwargs.keys():
                continue
            result[key] = kwargs[key]
        return result

    # method copies key-value pairs from kwargs into arguments, by updating arguments keys
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

    # build settings copied from children
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
        fname = [self._filename if "filename" not in kwargs.keys() else self._filename + kwargs["filename"]][0]
        assert type(notes) == str, 'Notes must be a string.'
        self._update_all_lengths()
        if 'fourier' in self._arguments['build']['model_type']:
            summary = self._SUMMARIES['fourier']
        else:
            summary = self._SUMMARIES['default']
        if 'summary' in kwargs.keys():
            summary = kwargs['summary']
        format_used = ['.txt' if len(extension) < 1 else extension][0]
        if '.' not in format_used:
            format_used = '.' + format_used
        with open(join(self._filepath, fname + format_used), 'w') as fil:
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
                print(self._history)
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
        # TODO: implement to work with Fourier or CNN Builders
        _arguments = [[None for _ in layers] if 'arguments' not in kwargs.keys() else kwargs['arguments']][0]
        summary = [False if 'summary' not in kwargs.keys() else kwargs['summary']]
        result = ''
        for layer_got, layer_args in zip(layers, _arguments):
            weight = layer_got.weights
            weight_text = ''
            if type(weight) is list:
                if len(weight) < 3:
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
        left = ["#" * self._length if len(args) <= 0 else str(args[0])][0]
        right = ["X" * self._length if len(args) <= 1 else str(args[1])][0]
        overwrite = max([len(left) - self._length, 0])
        return f'\t{left:{self._length}} - {right.rjust(self._length - overwrite)}\n'

    # a method to change the values of argument holders
    # TODO: generator objects as shorter strings
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
            if type(arguments[key]) is GeneratorType:
                result.update({key: 'built-in'})
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
            length_used = [length if length > 0 else self._determine_text_width(metric, _WIDTHS)][0]
            result.update({metric: sign.join([s[:length_used] for s in suffix])})
            if len(_suffix) <= 1:
                continue
            result[metric] = sign.join([s[:length_used] for s in suffix]) + _suffix[-1][0]
        return result

    def _prepare_metrics_text(self, history, suffixes=None):
        _history = history.copy()
        _MAX_TRAILS = {'loss': 6,
                       'acc': 4,
                       'top': 4,
                       'time': 6,
                       'default': 6,
                       }
        _MAX_WIDTHS = {'loss': 13,
                       'acc': 8,
                       'top': 8,
                       'time': 12,
                       'default': 9,
                       }
        text_result = ''
        text_result += 'epochs'.center(15) + ' -- '
        for key in _history[0].keys():
            key_str = [key if not suffixes else suffixes[key]][0]
            width = self._determine_text_width(key, _MAX_WIDTHS)
            text_result += str(key_str).center(max([len(key_str), width])) +' || '
        text_result += '\n'
        # changes for RTX in server, may be caused by newer TF
        #_epochs = [1 if type(_history["loss"]) is not list else len(_history["loss"])][0]
        _epochs = len(_history)
        for epoch in range(_epochs):
            # epoch_str = str(epoch)
            # # may be possible to use {epoch:0xd}
            # while len(epoch_str) < len(str(len(history))):
            #     epoch_str = '0' + epoch_str
            # do not expect more than 10k training epochs
            # text_result += ('Epoch ' + epoch_str).center(15) +' -- '
            text_result += f'Epoch {epoch:0{len(str(_epochs))}d}'.center(15) + ' -- '
            for key, value in zip(_history[epoch].keys(), _history[epoch].values()):
                key_str = [key if not suffixes else suffixes[key]][0]
                # crucial change
                value_used = [value if type(value) is not list else value[epoch]][0]
                width = max([len(key_str), self._determine_text_width(key, _MAX_WIDTHS)])
                trail = self._determine_text_width(key, _MAX_TRAILS)
                text_result += f'{value_used:{width}.{trail}f} || '
            text_result += '\n'
        return text_result

    @staticmethod
    def _determine_text_width(metric, widths):
        try:
            index = [key in metric for key in widths.keys()].index(True)
            return widths[list(widths.keys())[index]]
        except ValueError:
            return widths['default']

    # SOLVED: change the formatting to more than 3 - 4 or 5, for HPO
    # TODO: HPO doesn't increase the numbering now - check and fix
    @staticmethod
    def _expand_filename(filename, filepath=''):
        # List OF
        loof_files = [f for f in listdir(filepath) if filename in f]
        _stop = False
        it = 0
        # a list of all iterators at the end of files; required to sort from lowest to highest
        loof_iterators = sorted(set([(l.split('_')[-1]).split('.')[0] for l in loof_files]))
        while not _stop and it < len(loof_iterators):
            # extract only number string
            extracted = loof_iterators[it]
            compared = f'{it:04d}'
            it += 1
            if compared == extracted:
                continue
            it -= 1
            _stop = True
        date = dt.now().strftime('%Y-%m-%d_%H_%M_%S')
        filename_expanded = f'{filename}_{date}_{it:04d}'
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
        _history = history[0]
        # history after full training, instead of by 1 epoch
        reshaped_history = []
        for it in range(len(_history['loss'])):
            _hist = {}
            for key in _history.keys():
                _hist.update({key: _history[key][it]})
            reshaped_history.append(_hist)
        assert len(reshaped_history) == len(times), 'History and times are not the same length.'
        for it, time in enumerate(times):
            reshaped_history[it].update({'time': time})
        return reshaped_history

    # Properties
    @property
    def allowed_kwargs(self):
        return  self._allowed_kwargs

    @property
    def model(self):
        return self._model

    @property
    def history(self):
        return self._history

    @property
    def evaluation(self):
        return self._evaluation

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, fname):
        self._filename = fname

    @property
    def filepath(self):
        return self._filepath


# Standard CNNs for classification
class CNNBuilder(ModelBuilder):
    def __init__(self, model_type='mobilenet', input_shape=(32, 32, 3), noof_classes=1, **kwargs):
        super(CNNBuilder, self).__init__(model_type=model_type,
                                         input_shape=input_shape,
                                         noof_classes=noof_classes, **kwargs)

    @staticmethod
    def _define_allowed_kwargs():
        allowed = {'build' : {'model_type': ['mobilenet', 'mobilenet2', 'vgg16', 'vgg19', 'resnet50', 'resnet101',
                                             'unet'],
                              },
                   'compile': {'optimizer': ['adam', 'sgd'],
                               'loss': ['mse', 'categorical_crossentropy'],
                               },
                   }
        return allowed

    def _get_allowed_backbones(self):
        return self._get_allowed_kwargs()['build']['model_type']

    # TODO: removal
    @staticmethod
    def _get_backbone(model_type, input_shape, weights, **kwargs):
        model_type_low = model_type.lower()
        _backbone = None
        if 'mobilenet' in model_type_low:
            if '2' not in model_type_low:
                # load Mobilenet
                _backbone = apps.mobilenet.MobileNet
            else:
                # load Mobilenetv2
                _backbone = apps.mobilenet_v2.MobileNetV2
        elif 'vgg' in model_type_low:
            if '16' in model_type_low:
                # load VGG16
                _backbone = apps.vgg16.VGG16
            elif '19' in model_type_low:
                # load VGG19
                _backbone = apps.vgg19.VGG19
        elif 'resnet' in model_type_low:
            if '50' in model_type_low:
                # load resnet50
                _backbone = apps.resnet_v2.ResNet50V2
            elif '101' in model_type_low:
                # load resnet101
                _backbone = apps.resnet_v2.ResNet101V2
        elif 'densenet' in model_type_low:
            if '121' in model_type_low:
                _backbone = apps.densenet.DenseNet121
            elif '169' in model_type_low:
                _backbone = apps.densenet.DenseNet169
            elif '201' in model_type_low:
                _backbone = apps.densenet.DenseNet201
            else:
                print("Not implemented.")
                return None
        elif 'inception' in model_type_low:
            if 'v3' in model_type_low:
                _backbone = apps.inception_v3.InceptionV3
            elif 'v2' in model_type_low:
                _backbone = apps.inception_resnet_v2.InceptionResNetV2
        elif "xception" in model_type_low:
            _backbone = apps.xception.Xception
        elif "unet" in model_type_low:
            _backbone = unet_modified
        if not _backbone:
            return None
        backbone = _backbone(input_shape=input_shape, weights=weights, include_top=False)
        # update BatchNormalization momentum - otherwise several models (MobilenetV2, VGG16) do not work
        for layer in backbone.layers:
            if type(layer) != type(BatchNormalization):
                continue
            layer.momentum=0.9
        return backbone

    @staticmethod
    def _get_backbone_layers(model_type, input_shape, weights, **kwargs):
        model_type_low = model_type.lower()
        _backbone = None
        if 'mobilenet' in model_type_low:
            if '2' in model_type_low:
                # load Mobilenetv2
                _backbone = apps.mobilenet_v2.MobileNetV2
            else:
                # load Mobilenet
                _backbone = apps.mobilenet.MobileNet
        elif 'vgg' in model_type_low:
            if '16' in model_type_low:
                # load VGG16
                _backbone = apps.vgg16.VGG16
            elif '19' in model_type_low:
                # load VGG19
                _backbone = apps.vgg19.VGG19
        elif 'resnet' in model_type_low:
            if '50' in model_type_low:
                # load resnet50
                _backbone = apps.resnet_v2.ResNet50V2
            elif '101' in model_type_low:
                # load resnet101
                _backbone = apps.resnet_v2.ResNet101V2
        elif 'unet' in model_type_low:
            _backbone = unet_modified
        elif 'densenet' in model_type_low:
            if '121' in model_type_low:
                _backbone = apps.densenet.DenseNet121
            elif '169' in model_type_low:
                _backbone = apps.densenet.DenseNet169
            elif '201' in model_type_low:
                _backbone = apps.densenet.DenseNet201
            else:
                print("Not implemented.")
                return None
        elif 'inception' in model_type_low:
            if 'v3' in model_type_low:
                _backbone = apps.inception_v3.InceptionV3
            elif 'v2' in model_type_low:
                _backbone = apps.inception_resnet_v2.InceptionResNetV2
        elif "xception" in model_type_low:
            _backbone = apps.xception.Xception
        if not _backbone:
            return None
        backbone = _backbone(input_shape=input_shape, weights=weights, include_top=False)
        # update BatchNormalization momentum - otherwise several models (MobilenetV2, VGG16) do not work
        for layer in backbone.layers:
            if type(layer) != type(BatchNormalization):
                continue
            layer.momentum=0.9
        return backbone.layers

    def _build_model(self, model_type, input_shape, noof_classes, weights=None, freeze=0, **kwargs):
        # could be streamlined but would lower readability
        backbone = self._get_backbone(model_type, input_shape, weights)
        if not backbone:
            return None
        architecture = backbone.output
        if model_type in ["unet"]:
            return Model(inputs=[backbone.input], outputs=[backbone.output])
        # Classify
        flat = Flatten()(architecture)
        act = ['softmax' if noof_classes > 1 else 'sigmoid'][0]
        out = Dense(noof_classes, activation=act)(flat)
        return Model(inputs=[backbone.input], outputs=[out])


# custom model builder to make a Unet-like architecture from any built-in CNN
class CustomUnetBuilder(CNNBuilder):
    def __init__(self,
                 backbone:str,
                 input_shape=(32, 32, 3),
                 # at least background and foreground
                 noof_classes=2,
                 noof_ftl_heads=-1,
                 reversed_heads=False,
                 **kwargs
                 ):
        self._max_ftl_heads = 0
        self._SAMPLING_DIRECTIONS = DIRECTIONS
        # TODO: name checking
        # l = _NAMES[0] in layers[0].keys()
        # assert all(_NAMES in layer.keys() for layer in layers), \
        #     f'Unsupported name. Supported names: {_NAMES}.'
        kwargs.update({'model_type': backbone})
        self._allowed_backbones = CNNBuilder()._get_allowed_backbones()
        super(CustomUnetBuilder, self).__init__(input_shape=input_shape,
                                                # this noof_classes should not have an impact, it should be thrown out
                                                noof_classes=noof_classes,
                                                **kwargs)
        self._model, self._max_ftl_heads = self._add_ftl_heads(
            self._model,
            noof_ftl_heads,
            noof_classes,
            input_shape,
            reversed_heads
            )

    @property
    def max_ftl_heads(self):
        return self._max_ftl_heads

    @staticmethod
    def _define_allowed_kwargs():
        allowed = {'build' : {'model_type': ['mobilenet', 'mobilenet2',
                                             'densenet121', 'densenet169', 'densenet201',
                                             'vgg16', 'vgg19',
                                             # These are not OK right now
                                             #'inceptionv2', 'inceptionv3',
                                             'xception',
                                             'unet',
                                             ]
                              },
                   'compile': {'optimizer': ['adam', 'sgd'],
                               'loss': ['mse', 'categorical_crossentropy'],
                               },
                   }
        return allowed

    @staticmethod
    def _add_ftl_heads(
            arch,
            noof_heads,
            noof_classes,
            input_shape,
            reverse=False
        ):
        decs = []
        it = -1
        while it < len(arch.layers) - 1:
            it = it + 1
            layer = arch.layers[it]
            #print(f"{it}: {layer.output_shape}")
            # find shape mod 2
            if any([modu != 0 for modu in mod(layer.output_shape[1:2], 2)]):
                continue
            itx = it
            if itx >= len(arch.layers) or itx == 0:
                continue
            lay = arch.layers[itx]
            # find another layer which has different (smaller) output shape
            while lay.output_shape[1:2] == layer.output_shape[1:2] and itx < len(arch.layers) - 1:
                itx = itx + 1
                lay = arch.layers[itx]
            #if np.mod(lay.output_shape[1:2], input_shape[:-1]) != [0, 0]:
            lay = arch.layers[itx - 1]
            it = itx
            # make sure there is a multiplication
            if any([modu != 0 for modu in mod(input_shape[:-1], lay.output_shape[1:2])]):
                continue
            upsample = input_shape[0] // lay.output_shape[1]
            current_dec = Conv2D(filters=1, kernel_size=1, activation=None)(lay.output)
            """
            current_dec =  FTLSuperResolution(
                activation="relu",
                kernel_initializer="ones",
                #train_imaginary=True,
                #inverse=True,
                #calculate_abs=True,
                already_fft=False,
                sampling_nominator=upsample,
                direction="up"
            )(current_dec)
            """
            current_dec = tfimage.resize(current_dec, [input_shape[0], input_shape[1]])
            decs.append(current_dec)
        max_ftl_heads = len(decs)
        if 0 < noof_heads < len(decs):
            if reverse:
                decs = decs[:noof_heads]
            else:
                decs = decs[-noof_heads:]
        if len(decs) > 1:
            dec = concatenate(decs, axis=-1)
        else:
            dec = decs[0]
        dec = Conv2D(filters=noof_classes, kernel_size=1, activation="sigmoid", trainable=True)(dec)
        return Model(arch.input, dec), max_ftl_heads


# Custom model builder - can build any model (including hybrid), based on layer information
# SOLVED: add _insert and _replace_layer methods
# TODO: add layer numbering
class CustomBuilder(CNNBuilder):
    # TODO: default sampling initializations
    def __init__(self, layers, input_shape=(32, 32, 3), noof_classes=1, **kwargs):
        assert len(layers) > 1, 'CustomBuilder requires at least two layers. May cause problems with FTL layers, ' \
                                'mainly calling build twice.'
        # copied from keras: https://keras.io/api/layers/convolution_layers/convolution2d/
        defaults = self._define_default_layers()
        self._SAMPLING_DIRECTIONS = DIRECTIONS
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
                _layers.append({key: _layer})
        if 'model_type' not in kwargs.keys():
            kwargs.update({'model_type': 'custom'})
        self._allowed_backbones = CNNBuilder()._get_allowed_backbones()
        super(CustomBuilder, self).__init__(input_shape=input_shape,
                                            noof_classes=noof_classes,
                                            defaults={'layers': _layers},
                                            **kwargs)

    @staticmethod
    def _define_default_layers():
        # copied from keras: https://keras.io/api/layers/
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
                    'BatchNormalization': {},
                    # strides must be None to achieve halving
                    'avepooling': {'pool_size': 2,
                                   'strides': None,
                                   'padding': 'valid'},
                    'maxpooling': {'pool_size': 2,
                                   'strides': None,
                                   'padding': 'valid'},
                    'concatenate': {'axis': -1,
                                    },
                    'reshape': {'target_shape': (1, -1)
                                },
                    # https://keras.io/api/layers/convolution_layers/convolution2d_transpose/
                    'conv2dtranspose' : {'filters': 1,
                                         'kernel_size': 1,
                                         'strides': (1, 1),
                                         'padding': "valid",
                                         'output_padding': None,
                                         'data_format': None,
                                         'dilation_rate': (1, 1),
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
                    # custom layers
                    'ftl': {'activation': None,
                            'kernel_initializer': 'he_normal',
                            'train_imaginary': True,
                            'inverse': False,
                            'use_bias': False,
                            'bias_initializer': 'zeros',
                            'calculate_abs': True,
                            'normalize_to_image_shape': False,
                            },
                    'ftl_super_resolution': {'activation': None,
                                             'kernel_initializer': 'he_normal',
                                             'sampling_nominator': 2,
                                             'direction': 'up',
                                             'use_bias': False,
                                             'bias_initializer': 'zeros',
                                             'calculate_abs': True,
                                             'normalize_to_image_shape': False,
                                             },
                    }
        for back in CNNBuilder()._get_allowed_backbones():
            defaults.update({back: {'weights': None,
                                    'insert': None,
                                    'replace': None,
                                    # TODO: default indexes
                                    # mobilenet - 2
                                    'index': -1,
                                    # redundant, but just in case
                                    'include_top': False},
                             })
        return defaults

    # placeholder
    @staticmethod
    def _define_allowed_kwargs():
        allowed = {'build': {'model_type': ['custom'],
                             },
                   'compile': {'optimizer': ['adam'],
                               'loss': ['mse'],
                               },
                   'layers': {'ftl': {'activation': [None, 'relu', 'softmax', 'sigmoid', 'tanh', 'selu']},
                              },
                   }
        return allowed

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

    def _return_layer(self, layer_dict, previous):
        layer_name = list(layer_dict.keys())[0]
        arguments = list(layer_dict.values())[0]
        # self. would import 'custom' model name thus causing erroneous behaviour
        if layer_name in self._allowed_backbones:
            _layers = self._get_backbone_layers(model_type=layer_name, input_shape=previous.shape[1:],
                                                **arguments)
            _arch = previous
            for it, layer in enumerate(_layers):
                _flag_replace = False
                _layer_name = str(type(layer)).split('.')[-1][:-2]
                # omit InputLayer, already provided
                if 'Input' in _layer_name:
                    continue
                _model_layer_dict = {_layer_name: layer.get_config()}
                # make sure there is something to replace or insert
                for command in ['replace', 'insert']:
                    _command = [None if command not in arguments.keys() else command][0]
                    if _command is None:
                        continue
                    if type(arguments[_command]) is not list:
                        arguments[_command] = [arguments[_command]]
                    if None in arguments[_command]:
                        continue
                    for _layer_args in arguments[_command]:
                        _key = list(_layer_args.keys())[0]
                        # because index is not a parameters of layer building
                        if 'index' not in _layer_args[_key].keys() or it != _layer_args[_key]['index']:
                            continue
                        _args_no_index = _layer_args.copy()
                        _args_no_index[_key].pop('index')
                        if _command == 'replace' and type(layer) == Conv2D:
                            _arch = self._return_layer(_args_no_index, _arch)[0]
                            _flag_replace = True
                            continue
                        elif _command == 'insert':
                            _arch = self._return_layer(_args_no_index, _arch)[0]
                if _flag_replace:
                    continue
                _arch = self._return_layer(_model_layer_dict, _arch)[0]
            return _arch, False
            # return self._get_backbone(model_type=layer_name, input_shape=previous.shape[1:],
            #                           **arguments)(previous), False
        if layer_name in ['conv2d', 'Conv2D']:
            return Conv2D(**arguments)(previous), False
        if layer_name in ['conv2dtranspose', 'Conv2DTranspose']:
            return Conv2DTranspose(**arguments)(previous), False
        if layer_name in ['ftl']:
            if 'super_resolution' in layer_name:
                return FTLSuperResolution(**arguments)(previous), False
            return FTL(**arguments)(previous), False
        if layer_name in ['flatten', 'Flatten']:
            return Flatten()(previous), True
        if layer_name in ['concatenate', 'Concatenate']:
            return Concatenate(**arguments)(previous), False
        if layer_name in ['reshape', 'Reshape']:
            return Reshape(**arguments)(previous), False
        if layer_name in ['DepthwiseConv2D']:
            return DepthwiseConv2D(**arguments)(previous), False
        if layer_name in ['BatchNormalization']:
            return BatchNormalization(**arguments)(previous), False
        if layer_name in ['ZeroPadding2D']:
            return ZeroPadding2D(**arguments)(previous), False
        if layer_name in ['maxpooling', 'MaxPooling2D']:
            return MaxPooling2D(**arguments)(previous), False
        if layer_name in ['avepooling', 'AveragePooling2D']:
            return AveragePooling2D(**arguments)(previous), False
        if layer_name in ['Add']:
            return Add(**arguments)(previous), False
        if layer_name in ['Activation']:
            return Activation(**arguments)(previous), False
        if layer_name in ['ReLU']:
            return ReLU(**arguments)(previous), False
        if layer_name not in ['dense', 'Dense']:
            raise ValueError(f'{layer_name} not implemented yet.')
        return Dense(**arguments)(previous), True

    def _sample_model(self, **kwargs):
        # SOLVED: finding FTL in the model
        # TODO: adding Conv2d to layers list causes errors
        arguments_sampled = self._arguments['build'].copy()
        shape = arguments_sampled['input_shape']
        shape_new = shape
        # get sampling methods for dense and/or conv2d
        sampling_method = {'dense': ['pad' if 'dense_method' not in kwargs.keys() else kwargs['dense_method']][0],
                           'ftl': ['pad' if 'ftl_method' not in kwargs.keys() else kwargs['ftl_method']][0],
                           'conv': [None if 'conv_method' not in kwargs.keys() else kwargs['conv_method']][0],
                           }
        # make sure no incorrect methods are provided
        # pad - either cut (smaller) or pad (larger) images
        for method in list(sampling_method.values())[:-1]:
            assert method in ['pad', 'resize'], 'Incorrect sampling methods provided.'
        assert sampling_method['conv'] in [None, 'pad', 'resize'], 'Incorrect sampling methods provided.'
        if 'direction' in kwargs.keys() and 'nominator' in kwargs.keys():
            shape_new = self._operation(shape[:2], nominator=kwargs['nominator'],
                                        sign=self._SAMPLING_DIRECTIONS[kwargs['direction']])
        if 'shape' in kwargs.keys():
            shape_new = kwargs['shape']
        # TODO: see if it shouldn't be (*shape_new[:2], shape[2])
        arguments_sampled['input_shape'] = (*shape_new[:2], shape[2])
        # final shape
        shape_new = arguments_sampled['input_shape']
        model_weights = self._model.get_weights()
        model_layers = self._model.layers
        replace_value = [self._REPLACE_VALUE if 'replace_value' not in kwargs.keys() else kwargs['replace_value']][0]
        weights_result = []
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
        gathered_weights_new = {}
        nominator = np_max([_shn / _sh for _shn, _sh in zip(shape_new[:2], shape[:2])])
        # it = 0
        # while 'conv2d' not in list(gathered_weights.keys())[it]:
        #     it += 1
        # # because gathered weights contains 2 (usually) and model_weights is just a list with no names
        # print(it * 2 + 1)
        # this way ensures recalculation for all CLs
        for idx, weight in enumerate(model_weights):
            # some layers are not 4-dimensional
            if not sampling_method['conv'] or len(weight.shape) < 4:
                continue
            # assuming that the number of filters > 2
            if weight.shape[-1] < 2:
                continue
            _shape_conv = weight.shape
            _shape_conv_new = [*[int(_sh * nominator) for _sh in _shape_conv[:2]], *_shape_conv[2:]]
            idx_name = 0
            # TODO: protection from numbered layers
            while 'conv2d' not in list(arguments_sampled['layers'][idx_name].keys())[0]:
                idx_name += 1
            name = list(arguments_sampled['layers'][idx_name].keys())[0]
            arguments_sampled['layers'][idx_name][name]['kernel_size'] = _shape_conv_new[0]
        model_weights_new = CustomBuilder(**arguments_sampled).model.get_weights()
        model_layers_new = CustomBuilder(**arguments_sampled).model.layers
        names = []
        for layer in model_layers_new:
            for rep in range(len(layer.weights)):
                names.append(layer.name)
        for it, name in enumerate(names):
            if name not in gathered_weights_new.keys():
                gathered_weights_new.update({name: [model_weights_new[it]]})
            else:
                gathered_weights_new[name].append(model_weights_new[it])
        # () protect from unpacking error (expected 4, got 2)
        for layer_name, weights, weights_new in zip(gathered_weights.keys(), gathered_weights.values(), gathered_weights_new.values()):
            if 'ftl' in layer_name:
                # now its known that weights are FTL (1u2, X, X, C) and maybe bias (1u2, X, X, C)
                # additional extraction from list (thus [0])
                # includes bias
                for step in range(len(weights)):
                    weights_ftl = weights[step]
                    noof_weights = weights_ftl.shape[0]
                    weights_replace = ones_like(weights_new[step]) * replace_value
                    for rep in range(noof_weights):
                        for ch in range(shape[2]):
                            if sampling_method['ftl'] == 'resize':
                                _resized = resize_image(weights_ftl[rep, :, :, ch],
                                                       weights_new[step].shape[1:3])
                                if len(_resized.shape) == 2:
                                    _resized = expand_dims(_resized, axis=-1)
                                weights_replace[rep, :, :, ch] = _resized
                            elif shape_new[0] < shape[0]:
                                weights_replace[rep, :, :, ch] = weights_ftl[rep, :shape_new[0], :shape_new[1], ch]
                            else:
                                pads = [[0, int(shn - sh)] for shn, sh in zip(weights_new[step].shape[1:3],
                                                                              weights_ftl.shape[1:3])]
                                _padded = pad(squeeze(weights_ftl[rep, :, :, ch]), pad_width=pads,
                                              mode='constant', constant_values=replace_value)
                                if len(_padded.shape) == 2:
                                    _padded = expand_dims(_padded, axis=-1)
                                weights_replace[rep, :, :, ch] = _padded
                    weights_result.append(weights_replace)
                continue
            if 'conv' in layer_name:
                # TODO: extract target shape from arguments_sampled - will not work with several CLs
                # 0 - kernel
                it = 0
                if not sampling_method['conv']:
                    weights_result.extend(weights)
                    # other methods require changing layers list
                    continue
                elif sampling_method['conv'] == 'resize':
                    # because image accepts only 2 dimensions
                    weights_replace = resize_array(weights[it], _shape_conv_new)
                elif nominator < 1:
                    weights_replace = weights[it][:_shape_conv_new[0], :_shape_conv_new[1], :, :]
                else:
                    _pads = [[(_shn - _sh) // 2, (_shn - _sh) // 2 + ((_shn - _sh) // 2) % 2] for _shn, _sh in zip(_shape_conv_new[:2],
                                                                                                               _shape_conv[:2])]
                    pads = [*_pads, [0, 0], [0, 0]]
                    weights_replace = pad(weights[it], pad_width=pads,
                                          mode='constant', constant_values=replace_value)
                weights_result.append(weights_replace)
                weights_result.append(weights[1])
                continue
            # TODO: make sure passed_ftl is necessary
            if 'dense' in layer_name:
                # 0 - kernel, 1 - bias
                it = 0
                size_new = weights_new[it].shape[0]
                size_old = weights[it].shape[0]
                # None and cut are the same here - Dense must be resized
                if sampling_method['dense'] == 'resize':
                    weights_result.append(resize_array(weights[it], (size_new, weights[it].shape[1])))
                elif shape_new[0] < shape[0]:
                    weights_result.append(weights[it][:size_new, :])
                elif sampling_method['dense'] == 'pad':
                    pads = [[0, size_new - size_old], [0, 0]]
                    pd = pad(weights[it], pad_width=pads, mode='constant', constant_values=replace_value)
                    weights_result.append(pd)
                # add bias
                weights_result.append(weights[1])
                continue
            # other layers which should not be sampled (Conv2D, ...)
            if type(weights) is list:
                weights_result.extend(weights)
            else:
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

    def _sample_ftl_by_pool(self, **kwargs):
        # This method only requires FTL to change shape; pooling then returns the size to the basic one, on which the
        # model was trained. Hence dense and conv2d methods are not required.
        # SOLVED: finding FTL in the model
        # TODO: adding Conv2d to layers list causes errors
        arguments_sampled = self._arguments['build'].copy()

        shape = arguments_sampled['input_shape']
        shape_new = shape
        # get sampling methods for dense and/or conv2d
        sampling_method = ['pad' if 'ftl_method' not in kwargs.keys() else kwargs['ftl_method']][0]
        # make sure no incorrect methods are provided
        # pad - either cut (smaller) or pad (larger) images
        assert sampling_method in ['pad', 'resize'], 'Incorrect sampling methods provided.'
        # pad - either cut (smaller) or pad (larger) images
        if 'direction' in kwargs.keys() and 'nominator' in kwargs.keys():
            shape_new = self._operation(shape[:2], nominator=kwargs['nominator'],
                                        sign=self._SAMPLING_DIRECTIONS[kwargs['direction']])
        if 'shape' in kwargs.keys():
            shape_new = kwargs['shape']
        arguments_sampled['input_shape'] = (*shape_new[:2], shape[2])
        # final shape
        shape_new = arguments_sampled['input_shape']

        # SOLVED: calculating pooling size from shapes
        # SOLVED: finding pooling by "pooling"
        # change pooling size to keep the result of FTL + pooling the same shape
        it_pool = 0
        key_pool = 'avepooling'
        while 'pooling' not in str(list(arguments_sampled['layers'][it_pool].keys())[0]):
            it_pool += 1
        key_pool = list(arguments_sampled['layers'][it_pool].keys())[0]
        arguments_sampled['layers'][it_pool][key_pool].update({'pool_size': (shape_new[0] // shape[0], shape_new[1] // shape[1])})

        model_weights = self._model.get_weights()
        model_layers = self._model.layers
        replace_value = [self._REPLACE_VALUE if 'replace_value' not in kwargs.keys() else kwargs['replace_value']][0]
        weights_result = []
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
        gathered_weights_new = {}
        # it = 0
        # while 'conv2d' not in list(gathered_weights.keys())[it]:
        #     it += 1
        # # because gathered weights contains 2 (usually) and model_weights is just a list with no names
        # print(it * 2 + 1)
        model_weights_new = CustomBuilder(**arguments_sampled).model.get_weights()
        model_layers_new = CustomBuilder(**arguments_sampled).model.layers
        names = []
        for layer in model_layers_new:
            for rep in range(len(layer.weights)):
                names.append(layer.name)
        for it, name in enumerate(names):
            if name not in gathered_weights_new.keys():
                gathered_weights_new.update({name: [model_weights_new[it]]})
            else:
                gathered_weights_new[name].append(model_weights_new[it])
        # () protect from unpacking error (expected 4, got 2)
        for layer_name, weights, weights_new in zip(gathered_weights.keys(), gathered_weights.values(), gathered_weights_new.values()):
            if 'ftl' in layer_name:
                # now its known that weights are FTL (1u2, X, X, C) and maybe bias (1u2, X, X, C)
                # additional extraction from list (thus [0])
                # includes bias
                for step in range(len(weights)):
                    weights_ftl = weights[step]
                    noof_weights = weights_ftl.shape[0]
                    weights_replace = ones_like(weights_new[step]) * replace_value
                    for rep in range(noof_weights):
                        for ch in range(shape[2]):
                            if sampling_method == 'resize':
                                _resized = resize_image(weights_ftl[rep, :, :, ch],
                                                        weights_new[step].shape[1:3])
                                if len(_resized.shape) == 2:
                                    _resized = expand_dims(_resized, axis=-1)
                                weights_replace[rep, :, :, ch] = squeeze(_resized)
                            elif shape_new[0] < shape[0]:
                                weights_replace[rep, :, :, ch] = weights_ftl[rep, :shape_new[0], :shape_new[1], ch]
                            else:
                                pads = [[0, int(shn - sh)] for shn, sh in zip(weights_new[step].shape[1:3],
                                                                              weights_ftl.shape[1:3])]
                                _padded = pad(squeeze(weights_ftl[rep, :, :, ch]), pad_width=pads,
                                              mode='constant', constant_values=replace_value)
                                if len(_padded.shape) == 2:
                                    _padded = expand_dims(_padded, axis=-1)
                                weights_replace[rep, :, :, ch] = squeeze(_padded)
                    weights_result.append(weights_replace)
                continue
            # other layers which should not be sampled (Conv2D, ...)
            if type(weights) is list:
                weights_result.extend(weights)
            else:
                weights_result.append(weights)
        arguments_sampled['weights'] = weights_result
        builder = CustomBuilder(filename=self._filename_original, filepath=self._filepath, **arguments_sampled)
        for action in ['compile', 'train']:
            builder._arguments[action] = self._arguments[action]
        if 'compile' in kwargs.keys() and kwargs['compile']:
            builder.compile_model(**self._arguments['compile'])
        return builder

    def sample_ftl_by_pool(self, **kwargs):
        return self._sample_ftl_by_pool(**kwargs)

    @staticmethod
    def _operation(value, nominator=2, sign='div'):
        return sampling_calculation(value, nominator, sign)


# Fourier Model
class FourierBuilder(CustomBuilder):
    def __init__(self, model_type='fourier', input_shape=(32, 32, 1), noof_classes=1, **kwargs):
        # just to be safe
        layers = self.__define_default_layers(model_type)
        # impossible to streamline in current implementation
        if 'layers' in kwargs.keys():
            layers = self._verify_arguments(layers, layers=kwargs['layers'])
        super(FourierBuilder, self).__init__(model_type=model_type,
                                             input_shape=input_shape,
                                             noof_classes=noof_classes,
                                             layers=layers,
                                             **kwargs)

    @staticmethod
    def __define_default_layers(model_type):
        structure = [{'ftl': {'train_imaginary': True,
                              'inverse': 'inverse' in model_type,
                              }},
                     {'flatten': {}},
                     {'dense': {}},
                     ]
        return structure


if __name__ == '__main__':
    print(CustomUnetBuilder(
        backbone="mobilenet",
        noof_ftl_heads=1,
        input_shape=(256, 256, 3)
        ).model.summary())