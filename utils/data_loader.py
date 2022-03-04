from numpy import logical_or, zeros, expand_dims, pad, repeat, array, uint8, arange, float32, save, load, squeeze
from numpy.random import shuffle, seed, randint
from cv2 import resize, imread, cvtColor, COLOR_RGB2GRAY
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import join, isfile


class DataLoader:
    # TODO: add augmentation methods
    # TODO: saving data processing to .npy
    # split is redundant here, since keras will split the data during training on a whole dataset
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), **kwargs):
        assert all([sh >= 32 for sh in out_shape[:2]]), 'Must provide shapes larger than (32, 32).'
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
        x_train = self._preprocess_data(x_train)
        x_test = self._preprocess_data(x_test)
        return x_train, y_train, x_test, y_test

    def load_data(self):
        return self._load_data()

    def _preprocess_data(self, data):
        # resize if necessary
        result = self._resize_data(data, self._data_shape)
        # pad if necessary
        result = self._pad_data_to_32(result)
        # expand dimentions if necessary
        result = self._expand_dims(result, channels=self._channels)
        return result

    @staticmethod
    def _resize_data(data, new_shape):
        _data = data.copy()
        # single image
        if len(_data.shape) == 2:
            _data = expand_dims(_data, axis=0)
        # collection of images
        if len(_data.shape) == 3:
            _data = expand_dims(_data, axis=-1)
        # do not perform resize if unnecessary
        if _data.shape[1:] == new_shape:
            return data
        # iterate over every image
        result = zeros((_data.shape[0], *new_shape))
        for it, image in enumerate(data):
            result[it] = resize(image, new_shape)
        # resize removes trailing (1) shapes anyway
        return squeeze(result)

    # pad to at least 32x32
    @staticmethod
    def _pad_data_to_32(data):
        index_shape = 1
        if len(data) < 4:
            index_shape = 0
        if all([sh >= 32 for sh in data.shape[index_shape : index_shape + 2]]):
            return data
        pads = [(32 - sh) // 2 for sh in data.shape[index_shape : index_shape + 2]]
        return pad(data, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])

    @staticmethod
    def _expand_dims(data, channels=1):
        if len(data.shape) < 4:
            return repeat(expand_dims(data, axis=-1), repeats=channels, axis=-1)
        return data

    @staticmethod
    def _select_data_by_target(data_x, data_y, targets):
        # TODO: make sure it works on cifar10
        assert type(targets) is list, 'Must provide a list of targets.'
        assert len(targets) > 0, 'Must provide at least one target.'
        assert all([type(t) is int for t in targets]), 'Must provide a list of ints.'
        y_chosen = [False for _ in data_y]
        for target_01 in targets:
            y_chosen = logical_or(y_chosen, data_y == target_01)
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


class DatasetLoader(DataLoader):
    @property
    def x_train(self):
        return self._x_train / 255

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test / 255

    @property
    def y_test(self):
        return self._y_test

    @property
    def train_data(self):
        return self.x_train / 255, self.y_train

    @property
    def test_data(self):
        return self.x_test / 255, self.y_test

    @property
    def full_data(self):
        return self.x_train / 255, self.y_train, self.x_test / 255, self.y_test


class DataGenerator(DataLoader):
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), batch=4, split=0, shuffle_seed=None, **kwargs):
        if dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
            super(DataGenerator, self).__init__(dataset_name=dataset_name,
                                                out_shape=out_shape,
                                                **kwargs)
        self._batch = batch
        self._val_split = split
        if shuffle_seed is None or shuffle_seed < 0:
            seed(randint(2**31))
        elif shuffle_seed:
            seed(shuffle_seed)

    def _generator(self):
        # 1. prepare data list to be shuffled
        # 1a. (optional) split the list between train and val data
        # 1b. (optional) the same generator, only generates validation data
        # 2. actually shuffle and load the data
        while True:
            yield self._batch

    @property
    def generator(self):
        return self._generator()


if __name__ == '__main__':
    generator = DataGenerator().generator
    print(next(generator))