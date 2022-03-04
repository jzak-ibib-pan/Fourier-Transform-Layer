import numpy as np
from sklearn.utils import shuffle
from scipy import ndimage
from cv2 import resize, imread, cvtColor, COLOR_RGB2GRAY
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import join, isfile
from warnings import warn


class DataLoader:
    # TODO: add augmentation methods
    # TODO: saving data processing to .npy
    # split is redundant here, since keras will split the data during training on a whole dataset
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), **kwargs):
        assert all([sh >= 32 for sh in out_shape[:2]]), 'Must provide shapes larger than (32, 32).'
        self._VARIANCES = [1e-1, 1e-2, 1e-3, 1e-4]
        self._ROTATIONS = np.arange(181)
        self._flags = {'shift' : {},
                       'noise': {},
                       'rotation': {},
                       }
        self._flag_shift = False
        if 'shift' in kwargs.keys() and kwargs['shift']:
            assert type(kwargs['shift']) is int, 'Shift must be a boolean.'
            self._flag_shift = True
            self._flags['shift'].update({'threshold': 0, 'value': kwargs['shift']})
        self._flag_noise = False
        if 'noise' in kwargs.keys() and kwargs['noise']:
            assert kwargs['noise'] in self._VARIANCES, \
                f'Wrong variance value. Input one of the following {self._VARIANCES}.'
            self._flag_noise = True
            var = kwargs['noise']
            mean = 0
            sigma = var ** 0.5
            self._flags['noise'].update({'threshold': 0.5, 'mean': mean, 'sigma': sigma})
        self._flag_rotation = False
        if 'rotation' in kwargs.keys() and kwargs['rotation']:
            assert kwargs['rotation'] in self._ROTATIONS, \
                f'Wrong rotation value. Input one of the following {self._ROTATIONS}.'
            self._flag_rotation = True
            self._rot_angle = kwargs['rotation']
            self._flags['rotation'].update({'threshold': 0.5, 'angle': self._rot_angle})
        self._data_shape = out_shape[:2]
        self._channels = 1
        if len(out_shape) >= 3:
            self._channels = out_shape[-1]
        self.dataset = dataset_name
        self._x_train, self._y_train, self._x_test, self._y_test = self.load_data()
        _targets = None
        if 'targets' in kwargs.keys():
            _targets = kwargs['targets']
        if _targets is not None:
            self._x_train, self._y_train = self._select_data_by_target(self._x_train, self._y_train, _targets)
            self._x_test, self._y_test = self._select_data_by_target(self._x_test, self._y_test, _targets)

    def _load_data(self):
        x_train, y_train, x_test, y_test = (0, 0, 0, 0)
        noof_classes = 1
        if self.dataset.lower() == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            noof_classes = 10
        if self.dataset.lower() == 'fmnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
            noof_classes = 10
        if self.dataset.lower() == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            noof_classes = 10
        if self.dataset.lower() == 'cifar100':
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            noof_classes = 100
        y_train = to_categorical(y_train, noof_classes)
        y_test = to_categorical(y_test, noof_classes)
        x_train = self._preprocess_data(x_train, augmentation=False)
        x_test = self._preprocess_data(x_test, augmentation=False)
        return x_train, y_train, x_test, y_test

    def load_data(self):
        return self._load_data()

    def _preprocess_data(self, data, augmentation=True):
        result = data.copy()
        if augmentation:
            result = self._augment_data(result)
        # np.pad if necessary
        result = self._pad_data_to_32(result)
        # resize if necessary
        result = self._resize_data(result, self._data_shape)
        # expand dimentions if necessary
        result = self._expand_dims(result, channels=self._channels)
        # 255 - written this way to keep the same writing style
        if type(result) == np.uint8 and np.max(result) == 2**8 - 1:
            result = result / (2**8 - 1)
        elif type(result) == np.uint16 and np.max(result) == 2**16 - 1:
            result = result / (2**16 - 1)
        elif np.max(result) > 1:
            result = result / (2**8 - 1)
        return result

    @staticmethod
    def _resize_data(data, new_shape):
        _data = data.copy()
        # single image
        if len(_data.shape) == 2:
            _data = np.expand_dims(_data, axis=0)
        # collection of images
        elif len(_data.shape) == 3 and _data.shape[2] not in [1, 3]:
            _data = np.expand_dims(_data, axis=-1)
        # do not perform resize if unnecessary
        if _data.shape[1:] == new_shape:
            return data
        # iterate over every image
        result = np.zeros((_data.shape[0], *new_shape))
        # SOLVED: solve index out of bounds for axis - caused by iterating over data, not _data
        for it, image in enumerate(_data):
            result[it] = resize(image, new_shape)
        # resize removes trailing (1) shapes anyway
        return np.squeeze(result)

    # np.pad to at least 32x32
    @staticmethod
    def _pad_data_to_32(data):
        index_shape = 1
        if len(data) < 4:
            index_shape = 0
        if all([sh >= 32 for sh in data.shape[index_shape : index_shape + 2]]):
            return data
        np.pads = [(32 - sh) // 2 for sh in data.shape[index_shape : index_shape + 2]]
        return np.pad(data, [[0, 0], [np.pads[0], np.pads[0]], [np.pads[1], np.pads[1]]])

    @staticmethod
    def _expand_dims(data, channels=1):
        if len(data.shape) < 4:
            return np.repeat(np.expand_dims(data, axis=-1), repeats=channels, axis=-1)
        return data

    @staticmethod
    def _select_data_by_target(data_x, data_y, targets):
        # TODO: make sure it works on cifar10
        assert type(targets) is list, 'Must provide a list of targets.'
        assert len(targets) > 0, 'Must provide at least one target.'
        assert all([type(t) is int for t in targets]), 'Must provide a list of ints.'
        y_chosen = [False for _ in data_y]
        for target_01 in targets:
            y_chosen = np.logical_or(y_chosen, data_y == target_01)
            if len(targets) <= 1:
                continue
            for target_02 in targets:
                if target_01 == target_02:
                    continue
                assert target_01 != target_02, 'The same targets provided.'
        x = data_x[y_chosen]
        # otherwise may end up with non-consecutive values
        y_cat = data_y[y_chosen]
        for it, target in enumerate(sorted(targets)):
            y_cat[y_cat == target] = it
        y = to_categorical(y_cat)
        return x, y

    @staticmethod
    def _process_seed(shuffle_seed):
        if shuffle_seed is not None and shuffle_seed >= 0:
            return shuffle_seed
        return np.random.randint(2**31)

    # Augmentation methods
    # TODO: rotation in range
    def _augment_data(self, data):
        _data = data.copy()
        if self._flags['shift'] and np.random.rand() > self._flags['shift']['threshold']:
            _data = self._augment_shift(_data, self._flags['shift']['value'])
        if self._flags['rotation'] and np.random.rand() > self._flags['rotation']['threshold']:
            _data = self._augment_rotate(_data, self._flags['rotation']['angle'])
        if self._flags['noise'] and np.random.rand() > self._flags['noise']['threshold']:
            _data = self._augment_noise(_data, self._flags['noise'])
        return np.squeeze(_data)

    # placeholder
    @staticmethod
    def _augment_shift(data, shift_value):
        # SOLVED: implementation
        sx, sy = data.shape[:2]
        shift_shape = [sh + 2 * shift_value for sh in [sx, sy]]
        shift_x, shift_y = np.random.randint(shift_value, size=2)
        sign = {True: 1,
                False: -1,
                }
        randx, randy = (np.random.rand() > 0.5, np.random.rand() > 0.5)
        x_range = [shift_shape[0] // 2 - sx // 2 + sign[randx] * shift_x,
                   shift_shape[0] // 2 + sx // 2 + sign[randx] * shift_x]
        y_range = [shift_shape[1] // 2 - sy // 2 + sign[randy] * shift_y,
                   shift_shape[1] // 2 + sy // 2 + sign[randy] * shift_y]
        if len(data.shape) == 3:
            _data = np.zeros((*shift_shape, data.shape[2]))
            _data[x_range[0] : x_range[-1], y_range[0] : y_range[-1], :] = data
            return _data
        _data = np.zeros(shift_shape)
        _data[x_range[0] : x_range[-1], y_range[0] : y_range[-1]] = data
        return _data

    @staticmethod
    def _augment_rotate(data, angle):
        _data = data.copy()
        shape = [sh // 2 for sh in _data.shape[:2]]
        xy = ndimage.rotate(_data, angle)
        shx, shy = [sh // 2 for sh in xy.shape[:2]]
        # this returned half an image
        return xy[shx - shape[0] : shx + shape[0], shy - shape[1] : shy + shape[1]]

    @staticmethod
    def _augment_noise(data, flag):
        _data = data.copy()
        r, c, ch = _data.shape
        gausss = np.random.normal(flag['mean'], flag['sigma'], (r, c, ch))
        gausss = gausss.reshape(r, c, ch)
        _data = _data + gausss
        return _data


class DatasetLoader(DataLoader):
    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def train_data(self):
        return self.x_train, self.y_train

    @property
    def test_data(self):
        return self.x_test, self.y_test

    @property
    def full_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test


# a class for generating data when targets are separate variables
class DatasetGenerator(DataLoader):
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), batch=4, split=0, shuffle_seed=None, **kwargs):
        assert dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100'], 'Other datasets not supported.'
        super(DatasetGenerator, self).__init__(dataset_name=dataset_name,
                                               out_shape=out_shape,
                                               **kwargs)
        self._batch = batch
        self._seed = self._process_seed(shuffle_seed)
        self._flag_validation = split > 0
        # 1. prepare data list to be shuffled - changes with dataset - already loaded from DataLoader
        # 1a. (optional) split the list between train and val data
        self._x_train, self._y_train, self._x_val, self._y_val = self._split_data(self._x_train,
                                                                                self._y_train,
                                                                                split=split)

    def _generator(self, validation=False):
        x_data, y_data = self._x_train, self._y_train
        if validation:
            x_data, y_data = self._x_val, self._y_val
        # 2. actually shuffle and load the data
        x_data, y_data = shuffle(x_data, y_data, random_state=self._seed)
        index_data = 0
        while True:
            _X = np.zeros((self._batch, *x_data.shape[1:]))
            _Y = np.zeros((self._batch, *y_data.shape[1:]))
            # this will "eat" the end of dataset without loading, but shuffling should smooth the errors
            if index_data + self._batch >= x_data.shape[0]:
                x_data, y_data = shuffle(x_data, y_data, random_state=self._seed)
                index_data = 0
                continue
            for rep in range(self._batch):
                _X[rep] = self._preprocess_data(x_data[index_data])
                _Y[rep] = y_data[index_data]
                index_data += 1
            yield _X, _Y

    @staticmethod
    def _split_data(x_data, y_data, split=0.1):
        if split == 0:
            return x_data, y_data, [], []
        cutoff = (1 - split) * x_data.shape[0]
        return x_data[:cutoff], y_data[:cutoff], x_data[cutoff:], y_data[cutoff:]

    @property
    def generator(self):
        # implicit, just to make no mistakes
        return self._generator(validation=False)

    @property
    def validation_generator(self):
        assert self._flag_validation, 'Must have validation data to generate it.'
        # implicit, just to make no mistakes
        return self._generator(validation=True)


# Generic class
class DataGenerator(DataLoader):
    # TODO: MotherlistGenerator as a child
    def __init__(self, out_shape=(32, 32, 1), batch=4, shuffle_seed=None, **kwargs):
        self._batch = batch
        self._seed = self._process_seed(shuffle_seed)
        self._out_shape = out_shape
        self._noof_classes = 0

    @staticmethod
    def _generate_data(out_shape):
        return np.zeros(out_shape), 0

    def _generator(self):
        _X = np.zeros((self._batch, *self._out_shape))
        _Y = np.zeros((self._batch,))
        for rep in range(self._batch):
            _X[rep], _Y[rep] = self._generate_data(self._out_shape)
            _X[rep] = self._preprocess_data(_X[rep])
        yield _X, to_categorical(_Y, self._noof_classes)

    @property
    def generator(self):
        return self._generator()


class FringeGenerator(DataGenerator):
    # TODO: updated to DataLoader flags
    def __init__(self, out_shape=(32, 32, 1), batch=4, shuffle_seed=None, **kwargs):
        super(FringeGenerator, self).__init__(out_shape, batch, shuffle_seed)
        self._noof_classes = 2
        self._VARIANCES = [1e-1, 1e-2, 1e-3, 1e-4]
        self._ROTATIONS = [25, 45, 135, 170]
        self._WARNING_FLAGS = 'Both noise and rotation are used. Not recommended for experiments.'
        if self._flag_noise and self._flag_rotation:
            warn(self._WARNING_FLAGS)
        self._flag_test = False
        if 'test' in kwargs.keys() and kwargs['test']:
            assert kwargs['test'] in [0, 1], 'Test must be one of targets (0, 1).'
            self._test_class = kwargs['test']
            self._flag_test = True
        self._flag_discrete = False
        if 'discrete' in kwargs.keys() and kwargs['discrete']:
            assert type(kwargs['discrete']) is bool, 'Discrete must be a boolean.'
            self._flag_discrete = kwargs['discrete']

    # generate data must be a function
    def _generate_data(self, *args):
        rand = np.random.randint(1, 3)
        if self._flag_test:
            rand = 1 + self._test_class
        # no shifting fringes
        shift = 0
        if self._flag_shift:
            # added shifting fringes
            shift = np.pi * np.random.randn() / 2
            # shift = np.pi * np.random.choice([0, 0.25, 0.5], 1)[0]
        x = np.linspace(shift, shift + 2 * np.pi, self._out_shape[0])
        # 2 - two fringe tops
        multiplier = 3
        if self._flag_discrete:
            multiplier = 8
        y = 32.0 + (31.0 * np.sin(x * multiplier))
        y = np.uint8(y)

        if rand % 2 == 0:
            xy = np.tile(y, (self._out_shape[1], 1))  # pionowe prążki
            if self._flag_rotation:
                # 5 zamiast 2 - więcej prążków po obrocie
                x = np.linspace(shift, shift + 5 * np.pi, self._out_shape[0] * 2)
                y = 32.0 + (31.0 * np.sin(x * 2))
                y = np.uint8(y)
                xy = np.tile(y, (self._out_shape[1] * 2, 1))  # pionowe prążki
                xy = self._augment_rotate(xy, self._rot_angle)
            target = 1
        else:
            xy = np.tile(y, (self._out_shape[1], 1))  # poziome prążki
            xy = np.transpose(xy)
            target = 0

        xy = xy / (self._out_shape[0] - 1)
        # próba pozbycia się błędów
        xy = xy - np.min(xy)
        xy = xy / np.max(xy)

        if self._flag_discrete:
            xy[xy >= 0.5] = 1
            xy[xy < 1] = 0

        # dalsze próby
        x_t = xy / 100
        x_t = np.expand_dims(x_t, axis=-1)

        return x_t, target


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    generator = DatasetGenerator(shift=5).generator
    X, Y = next(generator)
    print(Y)
    plt.imshow(np.squeeze(X[0]))
    plt.show()