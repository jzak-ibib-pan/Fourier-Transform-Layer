from numpy import logical_or, zeros, expand_dims, pad, repeat, array, int8
from cv2 import resize, imread
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import join


def _select_images_by_target(data_x, data_y, targets):
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


def _resize_data(data, new_shape):
    result = zeros((data.shape[0], *new_shape, data.shape[-1]))
    for it, x in enumerate(data):
        if len(x.shape) < 3:
            x = expand_dims(x, axis=-1)
        if x.shape[-1] > 1:
            result[it] = resize(x, (new_shape[:2]))
            continue
        result[it] = expand_dims(resize(x, (new_shape[:2])), axis=-1)
    return result


def _load_celeb():
    filepath = join('Y://', 'super_resolution', 'CelebAMask-HQ', 'CelebA-HQ-img')
    loof_files = listdir(filepath)[:10]
    # each image is of this size
    result = zeros((len(loof_files), 1024, 1024, 3), dtype=int8)
    for it, filename in enumerate(loof_files):
        result[it] = imread(join(filepath, filename))
    return result


def prepare_data_for_sampling(dataset, targets=None, data_channels=1, new_shape=None):
    assert type(dataset) is str, 'Dataset must be a str.'
    assert dataset in ['mnist', 'fmnist', 'celeb'], f'{dataset.capitalize()} not implemented yet.'
    if dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if dataset.lower() == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if dataset.lower() == 'celeb':
        X = _load_celeb()
        # calculate train / test split
        test_split = 0.1
        cutoff = int((1 - test_split) * X.shape[0])
        if not new_shape:
            # return split data (train), (test)
            return (X[:cutoff], None), (X[cutoff:], None)
        # smaller dataset-> // 4
        x_tr = []
        for x in X:
            # extract part of image - for calculation sake
            x_tr.append(x[512 - new_shape[0] // 4 : 512 + new_shape[0] // 4,
                        512 - new_shape[0] // 4 : 512 + new_shape[0] // 4])
        x_tr = array(x_tr)

        # 'resized' dataset-> // 2
        x_re = []
        for x in X:
            # extract part of image - for calculation sake
            x_tr.append(x[512 - new_shape[0] // 2 : 512 + new_shape[0] // 2,
                        512 - new_shape[0] // 2 : 512 + new_shape[0] // 2])
        x_re = array(x_re)
        return (x_tr[:cutoff], None), (x_tr[cutoff:], None), (x_re[:cutoff], x_re[cutoff:])


    if x_train.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_train.shape[1:3]]
        x_train = pad(x_train, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])
    x_train = repeat(expand_dims(x_train / 255, axis=-1), repeats=data_channels, axis=-1)

    if x_test.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_test.shape[1:3]]
        x_test = pad(x_test, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])
    x_test = repeat(expand_dims(x_test / 255, axis=-1), repeats=data_channels, axis=-1)

    if targets:
        x_train, y_train = _select_images_by_target(x_train, y_train, targets)
        x_test, y_test = _select_images_by_target(x_test, y_test, targets)

    if len(x_train.shape) < 4:
        x_train = expand_dims(x_train, -1)
    if len(x_test.shape) < 4:
        x_test = expand_dims(x_test, -1)

    if new_shape is None or new_shape == x_test.shape[1:]:
        return (x_train, y_train), (x_test, y_test)

    # for sampling
    x_tr = _resize_data(x_train, new_shape)
    if len(x_tr.shape) < 4:
        x_tr = expand_dims(x_tr, 3)

    x_ts = _resize_data(x_test, new_shape)
    if len(x_ts.shape) < 4:
        x_ts = expand_dims(x_ts, 3)

    return (x_train, y_train), (x_test, y_test), (x_tr, x_ts)


if __name__ == '__main__':
    print(0)