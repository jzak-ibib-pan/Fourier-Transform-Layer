import numpy as np
from sklearn.utils import shuffle
from scipy import ndimage
from cv2 import resize, imread, cvtColor, COLOR_RGB2GRAY
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory
from os import listdir
from os.path import join, isfile
from warnings import warn


# TODO: for imread remember to switch channels to keep RGB2GRAY working as expected
# TODO: add sampling somewhere
class DataLoader:
    # SOLVED: add augmentation methods
    # TODO: threshold reading
    # TODO: saving data processing to .npy
    # split is redundant here, since keras will split the data during training on a whole dataset
    # TODO: resize or pad to required shape
    def __init__(self, out_shape=(32, 32, 1), **kwargs):
        assert all([sh >= 1 for sh in out_shape[:2]]), 'Must provide shapes larger than (1, 1).'
        self._data_shape = out_shape[:2]
        self._channels = [1 if len(out_shape) < 3 else out_shape[-1]][0]
        self.dataset = str(None)
        self._noof_classes = 0
        # kwargs
        self._VARIANCES = [1e-1, 1e-2, 1e-3, 1e-4]
        self._ROTATIONS = np.arange(181)
        self._FLIPS = ['up', 'down', 'ud', 'left', 'right', 'lr']
        self._aug_flags = {'shift' : {},
                           'noise': {},
                           'rotation': {},
                           'flip': {},
                           }
        self._aug_flags = self._determine_augmentations(**kwargs)
        self._flag_empty_aug = all([_flag == {} for _flag in self._aug_flags.values()])

    def _load_data(self):
        x_train, y_train, x_test, y_test = (0, 0, 0, 0)
        return x_train, y_train, x_test, y_test

    def load_data(self):
        return self._load_data()

    def _determine_augmentations(self, **kwargs):
        _names = {'shift': 'value',
                  'rotation': 'angle',
                  'flip': 'direction',
                  }
        result_aug_flags = self._aug_flags.copy()
        for key in result_aug_flags.keys():
            if key not in kwargs.keys():
                continue
            if not kwargs[key]:
                continue
            augmentation = kwargs[key]
            if type(augmentation) is not list:
                augmentation = [augmentation]
            if not self._assert_augmentations(key, augmentation[0]):
                continue
            threshold = self._determine_threshold(augmentation)
            if key == 'noise':
                # TODO: mean as input
                var = augmentation[0]
                mean = 0
                sigma = var ** 0.5
                result_aug_flags[key].update({'threshold': threshold,
                                              'mean': mean, 'sigma': sigma})
                continue
            result_aug_flags[key].update({'threshold': threshold,
                                          _names[key]: augmentation[0]})
        return result_aug_flags

    def _assert_augmentations(self, key, augmentation_value):
        if key == 'shift':
            assert type(augmentation_value) is int, 'Shift must be an integer.'
            return True
        if key == 'noise':
            assert augmentation_value in self._VARIANCES, \
                f'Wrong variance value. Input one of the following {self._VARIANCES}.'
            return True
        if key == 'rotation':
            assert augmentation_value in self._ROTATIONS, \
                f'Wrong rotation value. Input one of the following {self._ROTATIONS}.'
            return True
        if key == 'flip':
            assert augmentation_value in self._FLIPS, f'Wrong flip value. Input one of the following {self._FLIPS}.'
            return True
        # default
        return True

    @staticmethod
    def _determine_threshold(augmentation):
        return [0.5 if len(augmentation) == 1 else augmentation[1]][0]

    # augment variable to prevent test data augmentation
    def _preprocess_data(self, data, augment=True):
        # SOLVED: merge with private __expand and sort order of actions
        _data = self.__expand_dims_for_eumeration(data)
        result = np.zeros((_data.shape[0], *self._data_shape, self._channels))
        # should split padding and resizing, but probably won't be using many small images 4 pixel border is
        # acceptable
        # np.pad if necessary
        # result = self._pad_data_to_32(result)
        # iterative methods
        for it, _point in enumerate(_data):
            # SOLVED: add grayscale as first preprocessing step
            _point = self._convert_to_grayscale(_point, self._channels)
            # resize if necessary
            _point = self._resize_data(_point, self._data_shape)
            # TODO: default augmentation methods
            if augment and not self._flag_empty_aug:
                _point = self._augment_data(_point)
            # any of the three methods (convert, resize, augment) remove the trailing dimensions
            _point = [_point if len(_point.shape) > 2 else np.expand_dims(_point, axis=-1)][0]
            result[it] = _point
        # expand dimentions if necessary
        result = self._expand_dims(result, channels=self._channels)
        # 255 - written this way to keep the same writing style
        # SOLVED: remove the second condition; may be unmet
        if type(result) == np.uint8:
            #  and np.max(result) == 2**8 - 1
            result = result / (2**8 - 1)
        elif type(result) == np.uint16:
            #  and np.max(result) == 2**16 - 1
            result = result / (2**16 - 1)
        elif np.max(result) > 1:
            result = result / (2**8 - 1)
        # TODO: float16 support
        return np.float32(result)

    @staticmethod
    def __expand_dims_for_eumeration(data):
        # single image with no channels
        if len(data.shape) == 2:
            return np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            # collection of images
            if data.shape[2] not in [1, 3]:
                return np.expand_dims(data, axis=-1)
            # single image with channels
            return np.expand_dims(data, axis=0)
        return data

    # np.pad to at least 32x32
    @staticmethod
    def _pad_data_to_32(data):
        # made sure its at least [1, X, Y, C] in preprocessing method
        _data = data.copy()
        if all([sh >= 32 for sh in _data.shape[1:3]]):
            return data
        pads = [(32 - sh) // 2 for sh in _data.shape[1:3]]
        # assumption - always has first channel
        if len(_data.shape) == 3:
            # no channels
            return np.pad(_data, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])
        return np.pad(_data, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]], [0, 0]])

    @staticmethod
    def _convert_to_grayscale(datapoint, channels=1):
        if len(datapoint.shape) == 2:
            # no channels
            return datapoint
        # must check last (3rd) dimension
        if datapoint.shape[-1] == 1 or datapoint.shape[-1] <= channels:
            return datapoint
        # it is known that the cvtColor removes trailing 1s and result expects trailing 1
        return cvtColor(datapoint, COLOR_RGB2GRAY)

    @staticmethod
    def _resize_data(datapoint, new_shape):
        # do not perform resize if unnecessary
        if datapoint.shape[:2] == new_shape:
            return datapoint
        # it is known that the resize removes trailing 1s and result expects trailing 1
        # TODO: may cause errors by switching x and y shapes - keep in mind
        return resize(datapoint, new_shape)

    @staticmethod
    def _expand_dims(data, channels=1):
        if data.shape[-1] == channels:
            return data
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
    # SOLVED: rotation in range
    def _augment_data(self, datapoint):
        # datapoint - single image
        _point = datapoint.copy()
        if self._determine_if_augment(self._aug_flags, 'flip'):
            _point = self._augment_flip(_point, self._aug_flags['flip']['direction'])
        if self._determine_if_augment(self._aug_flags, 'shift'):
            _point = self._augment_shift(_point, self._aug_flags['shift']['value'])
        if self._determine_if_augment(self._aug_flags, 'rotation'):
            _point = self._augment_rotate(_point, self._aug_flags['rotation']['angle'])
        if self._determine_if_augment(self._aug_flags, 'noise'):
            _point = self._augment_noise(_point, self._aug_flags['noise'])
        return np.squeeze(_point)

    @staticmethod
    def _determine_if_augment(flag, method):
        if not flag[method]:
            return False
        if np.random.rand() < flag[method]['threshold']:
            return False
        return True

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
        if len(data.shape) <= 2:
            _data = np.zeros(shift_shape)
            _data[x_range[0] : x_range[-1], y_range[0] : y_range[-1]] = data
        else:
            _data = np.zeros((*shift_shape, data.shape[2]))
            _data[x_range[0] : x_range[-1], y_range[0] : y_range[-1], :] = data
        x_return = [shift_shape[0] // 2 - sx // 2,
                    shift_shape[0] // 2 + sx // 2]
        y_return = [shift_shape[1] // 2 - sy // 2,
                    shift_shape[1] // 2 + sy // 2]
        return _data[x_return[0] : x_return[-1], y_return[0] : y_return[-1]]

    @staticmethod
    def _augment_rotate(data, angle):
        _data = data.copy()
        shape = [sh // 2 for sh in _data.shape[:2]]
        # make sure angle is not 0
        _angle = np.random.randint(angle - 1) + 1
        xy = ndimage.rotate(_data, _angle)
        shx, shy = [sh // 2 for sh in xy.shape[:2]]
        # this returned half an image
        return xy[shx - shape[0] : shx + shape[0], shy - shape[1] : shy + shape[1]]

    @staticmethod
    def _augment_noise(data, flag):
        _data = data.copy()
        # otherwise summing may cause errors
        _data = [_data if len(_data.shape) > 2 else np.expand_dims(_data, axis=-1)][0]
        r, c, ch = _data.shape
        gausss = np.random.normal(flag['mean'], flag['sigma'], (r, c, ch))
        gausss = gausss.reshape(r, c, ch)
        _data = _data + gausss
        if np.min(_data) < 0:
            _data = _data - np.min(_data)
        return _data

    @staticmethod
    def _augment_flip(data, direction):
        _data = data.copy()
        if direction == 'ud':
            return np.flipud(_data)
        elif direction == 'lr':
            return np.fliplr(_data)
        # just to make sure the method returns
        return _data

    @property
    def noof_classes(self):
        return self._noof_classes


# TODO: out_shape as None (?)
class DatasetLoader(DataLoader):
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), **kwargs):
        assert dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100'], 'Other datasets not supported.'
        super(DatasetLoader, self).__init__(out_shape=out_shape, **kwargs)
        self.dataset = dataset_name
        self._x_train, self._y_train, self._x_test, self._y_test = self.load_data()
        _targets = [None if 'targets' not in kwargs.keys() else kwargs['targets']][0]
        if _targets is not None:
            self._x_train, self._y_train = self._select_data_by_target(self.x_train, self.y_train, _targets)
            self._x_test, self._y_test = self._select_data_by_target(self.x_test, self.y_test, _targets)

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
        x_train = self._preprocess_data(x_train)
        # implicit to avoid bugs
        x_test = self._preprocess_data(x_test, augment=False)
        self._noof_classes = noof_classes
        return x_train, y_train, x_test, y_test

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


# SOLVED: add test dataset generator or return whole subset - by DatasetFlower
# SOLVED: add augmentation control to Generator classes
# a class for generating data when targets are separate variables
class DatasetGenerator(DatasetLoader):
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), batch=4, split=0, shuffle_seed=None, **kwargs):
        # default - no augmentation
        super(DatasetGenerator, self).__init__(dataset_name=dataset_name,
                                               out_shape=out_shape,
                                               **kwargs)
        self._batch = batch
        self._seed = self._process_seed(shuffle_seed)
        self._flag_validation = split > 0
        # 1. prepare data list to be shuffled - changes with dataset - already loaded from DataLoader
        # 1a. (optional) split the list between train and val data
        # SOLVED: flower as another option; split by dataset_name or path - make sure validation generator will work -
        # separate Class
        self._x_train, self._y_train, self._x_val, self._y_val = self._split_data(self.x_train,
                                                                                  self.y_train,
                                                                                  split=split)

    def _generator(self, validation=False, augment=True):
        x_data, y_data = [[self._x_train, self._y_train] if not validation else [self._x_val, self._y_val]][0]
        # 2. actually shuffle and load the data
        x_data, y_data = shuffle(x_data, y_data, random_state=self._seed)
        index_data = 0
        while True:
            _X = np.zeros((self._batch, *x_data.shape[1:3], self._channels))
            _Y = np.zeros((self._batch, *y_data.shape[1:]))
            # this will "eat" the end of dataset without loading, but shuffling should smooth the errors
            if index_data + self._batch >= x_data.shape[0]:
                x_data, y_data = shuffle(x_data, y_data, random_state=self._seed)
                index_data = 0
                continue
            for rep in range(self._batch):
                _X[rep] = self._preprocess_data(x_data[index_data], augment)
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
        # TODO: augment=False (?)
        return self._generator(validation=True)


# Generic class
class DataGenerator(DataLoader):
    # TODO: MotherlistGenerator as a child
    def __init__(self, out_shape=(32, 32, 1), batch=4, shuffle_seed=None, **kwargs):
        super(DataGenerator, self).__init__(out_shape=out_shape, **kwargs)
        self._batch = batch
        self._seed = self._process_seed(shuffle_seed)
        self._out_shape = out_shape
        self._noof_classes = 0

    @staticmethod
    def _generate_data(out_shape):
        return np.zeros(out_shape), 0

    # TODO: while and yield here
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


# Class for data flowing from directory
class DatasetFlower(DataGenerator):
    def __init__(self, path, split=0, **kwargs):
        super(DatasetFlower, self).__init__(**kwargs)
        self._path = path
        self._targets = [None if 'targets' not in kwargs.keys() else kwargs['targets']][0]
        self._dataset = None
        self._dataset = self._prepare_flower(split)
        self._len = self._dataset.cardinality().numpy()
        self._dataset_val = None
        self._len_val = None
        if split > 0:
            self._dataset_val = self._prepare_flower(split, validation=True)
            self._len_val = self._dataset_val.cardinality().numpy()

    def _prepare_flower(self, split=0, validation=False):
        _subset = None
        _split = None
        if split != 0:
            _subset = 'training'
            _split = split
            if validation:
                _subset = 'validation'
        return image_dataset_from_directory(self._path,
                                            labels="inferred",
                                            label_mode="categorical",
                                            class_names=self._targets,
                                            color_mode=["rgb" if self._channels > 1 else 'grayscale'][0],
                                            batch_size=self._batch,
                                            image_size=self._data_shape,
                                            shuffle=True,
                                            seed=self._seed,
                                            # SOLVED: check validation split is working - must use subset as either
                                            # "training" or "validation"
                                            validation_split=_split,
                                            subset=_subset,
                                            interpolation="bicubic",
                                            follow_links=False)

    @staticmethod
    def _generate_data(dataset):
        for data in dataset.as_numpy_iterator():
            yield data

    def _reset_dataset(self, validation=False):
        if validation:
            return self._generate_data(self._dataset_val)
        else:
            return self._generate_data(self._dataset)

    def _yield_data(self, generator, augment=True):
        X, Y = next(generator)
        # due to ValueError: assignment destination is read-only; quick workaround
        Xr = np.zeros_like(X)
        for rep in range(X.shape[0]):
            Xr[rep] = self._preprocess_data(X[rep], augment)
        return Xr, Y

    def _generator(self, validation=False, augment=True):
        _gen = self._reset_dataset(validation=validation)
        while True:
            try:
                yield self._yield_data(_gen, augment=augment)
            except StopIteration:
                # reset the generators
                _gen = self._reset_dataset(validation=validation)
                # must yield
                yield self._yield_data(_gen, augment=augment)

    @property
    def generator(self):
        return self._generator(validation=False)

    @property
    def validation_generator(self):
        assert self._dataset_val is not None, 'Must have validation data to generate it.'
        return self._generator(validation=True)

    @property
    def length(self):
        return self._len

    @property
    def validation_length(self):
        assert self._dataset_val is not None, 'Must have validation data to get its length.'
        return self._len_val


# Class for fringe images generation
class FringeGenerator(DataGenerator):
    def __init__(self, out_shape=(32, 32, 1), batch=4, shuffle_seed=None, **kwargs):
        # TODO: make VARIANCES and ROTATIONS relevant in init
        # no kwargs here - not expecting to perform additional augmentations
        # SOLVED: updated to DataLoader flags - unnecessary
        super(FringeGenerator, self).__init__(out_shape, batch, shuffle_seed)
        self._noof_classes = 2
        self._VARIANCES = [1e-1, 1e-2, 1e-3, 1e-4]
        self._ROTATIONS = [25, 45, 135, 170]
        self._WARNING_FLAGS = 'Both noise and rotation are used. Not recommended for experiments.'
        self._flag_shift = self._aug_flags['shift'] != {}
        self._flag_noise = self._aug_flags['noise'] != {}
        self._flag_rotation = self._aug_flags['rotation'] != {}
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
        rand_num = [np.random.randint(1, 3) if not self._flag_test else 1 + self._test_class][0]
        # no shifting fringes - shifting fringes
        shift = [0 if not self._flag_shift else np.pi * np.random.randn() / 2][0]
        x = np.linspace(shift, shift + 2 * np.pi, self._out_shape[0])
        # 2 - two fringe tops
        multiplier = [3 if not self._flag_discrete else 8][0]
        y = 32.0 + (31.0 * np.sin(x * multiplier))
        y = np.uint8(y)

        if rand_num % 2 == 0:
            # vertical fringes
            xy = np.tile(y, (self._out_shape[1], 1))
            if self._flag_rotation:
                # 5 instead of 2 - more fringes after rotating
                x = np.linspace(shift, shift + 5 * np.pi, self._out_shape[0] * 2)
                y = 32.0 + (31.0 * np.sin(x * 2))
                y = np.uint8(y)
                # vertical fringes
                xy = np.tile(y, (self._out_shape[1] * 2, 1))
                xy = self._augment_rotate(xy, self._aug_flags['rotation']['angle'])
            target = 1
        else:
            # horizontal fringes
            xy = np.tile(y, (self._out_shape[1], 1))
            xy = np.transpose(xy)
            target = 0

        xy = xy / (self._out_shape[0] - 1)
        # próba pozbycia się błędów
        xy = xy - np.min(xy)
        xy = xy / np.max(xy)

        if self._flag_discrete:
            xy[xy >= 0.5] = 1
            xy[xy < 1] = 0

        # further tries - normalization for
        x_t = xy / np.max(xy)
        x_t = np.expand_dims(x_t, axis=-1)

        return x_t, target

    def _generator(self):
        while True:
            X, Y = [], []
            for rep in range(self._batch):
                x, y = self._generate_data()
                X.append(x)
                Y.append(y)
            yield np.array(X), to_categorical(Y, 2)


# Class for loading images, according to motherlist
# TODO: validation generator
class MotherlistGenerator(DataGenerator):
    def __init__(self, path_motherlist, dir_tiles, out_shape=(32, 32, 1), batch=4, split=0, shuffle_seed=None, **kwargs):
        super(MotherlistGenerator, self).__init__(out_shape=out_shape, **kwargs)
        self._path_images = join(path_motherlist, dir_tiles)
        self._path_mother = join(path_motherlist, 'motherlist.txt')

    @staticmethod
    def _extract_motherlist_info(line):
        data_image, data_target = line.split(';')
        data_image = data_image.split(':')[1].strip()
        data_target = data_target.split(':')[1].strip()
        return data_image + '.tif', float(data_target)

    def _generator(self):
        with open(self._path_mother, 'r') as file:
            _files = file.readlines()
        # prepare indeces for shuffling
        shuf = np.arange(len(_files))
        # shuffle
        shuffle(shuf)
        idx_shuffle = 0
        while True:
            _X = np.zeros((self._batch, *self._out_shape))
            _Y = np.zeros((self._batch,))
            _comparison = np.zeros(self._out_shape)
            rep = 0
            _target = 0
            # make sure every empty space is loaded with an image
            while any([np.array_equal(_x, _comparison) for _x in _X]) and rep < self._batch:
                # make sure the first image is class 1
                if rep == 0:
                    while _target == 0:
                        # get the filename
                        _filename, _target = self._extract_motherlist_info(_files[shuf[idx_shuffle]])
                        idx_shuffle += 1
                        if idx_shuffle >= len(_files):
                            shuffle(shuf)
                            idx_shuffle = 0
                else:
                    # get the filename
                    _filename, _target = self._extract_motherlist_info(_files[shuf[idx_shuffle]])
                    idx_shuffle += 1
                    if idx_shuffle >= len(_files):
                        shuffle(shuf)
                        idx_shuffle = 0
                # get only images with fully marked masks
                if 0 < _target < 0.5:
                    continue
                _X[rep] = self._preprocess_data(imread(join(self._path_images, _filename)))
                _Y[rep] = [0 if _target == 0 else 1][0]
                rep += 1
            yield _X, to_categorical(_Y, self._noof_classes)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    loader = MotherlistGenerator(path_motherlist='Y:/Slinianki/miazsz/tiles_256', dir_tiles='obrazy',
                                 noof_classes=2,
                                 out_shape=(256, 256, 3))
    generator = loader.generator
    X, Y = next(generator)
    print(Y)
    plt.imshow(np.squeeze(X[0]))
    plt.show()