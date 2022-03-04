import numpy as np
from sklearn.utils import shuffle
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
        # np.pad if necessary
        result = self._pad_data_to_32(result)
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
        if len(_data.shape) == 3:
            _data = np.expand_dims(_data, axis=-1)
        # do not perform resize if unnecessary
        if _data.shape[1:] == new_shape:
            return data
        # iterate over every image
        result = np.zeros((_data.shape[0], *new_shape))
        for it, image in enumerate(data):
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


# a class for generating data when the classes are as separate variable
class DatasetGenerator(DataLoader):
    # TODO: MotherlistGenerator, DataGenerator as children
    def __init__(self, dataset_name='mnist', out_shape=(32, 32, 1), batch=4, split=0, shuffle_seed=None, **kwargs):
        assert dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100'], 'Other datasets not supported.'
        super(DatasetGenerator, self).__init__(dataset_name=dataset_name,
                                               out_shape=out_shape,
                                               **kwargs)
        self._batch = batch
        self._shuffle_seed = np.random.randint(2**31)
        if shuffle_seed is not None and shuffle_seed >= 0:
            self._shuffle_seed = shuffle_seed
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
        x_data, y_data = shuffle(x_data, y_data, random_state=self._shuffle_seed)
        index_data = 0
        while True:
            _X = np.zeros((self._batch, *x_data.shape[1:]))
            _Y = np.zeros((self._batch, *y_data.shape[1:]))
            # this will "eat" the end of dataset without loading, but shuffling should smooth the errors
            if index_data + self._batch >= x_data.shape[0]:
                x_data, y_data = shuffle(x_data, y_data, random_state=self._shuffle_seed)
                index_data = 0
                continue
            for rep in range(self._batch):
                _X[rep] = x_data[index_data]
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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    generator = DatasetGenerator().generator
    X, Y = next(generator)
    print(Y)
    plt.imshow(np.squeeze(X[0]))
    plt.show()