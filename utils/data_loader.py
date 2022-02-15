from numpy import logical_or, min, max, zeros, expand_dims, pad, repeat
from cv2 import resize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def select_images_by_target(data_x, data_y, targets):
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


def prepare_data_for_sampling(targets, data_channels = 1, new_shape=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    if x_train.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_train.shape[1:3]]
        x_train = pad(x_train, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])
    x_train = repeat(expand_dims(x_train / 255, axis=-1), repeats=data_channels, axis=-1)

    if x_test.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_test.shape[1:3]]
        x_test = pad(x_test, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])
    x_test = repeat(expand_dims(x_test / 255, axis=-1), repeats=data_channels, axis=-1)

    x_train, y_train = select_images_by_target(x_train, y_train, targets)
    x_test, y_test = select_images_by_target(x_test, y_test, targets)

    if new_shape is None or new_shape == x_test.shape[1:]:
        return (x_train, y_train), (x_test, y_test), x_test

    x_tr = zeros((x_train.shape[0], *new_shape, data_channels))
    for it, x in enumerate(x_train):
        x_tr[it] = resize(x, new_shape[:2], data_channels)

    x_ts = zeros((x_test.shape[0], *new_shape, data_channels))
    for it, x in enumerate(x_test):
        x_ts[it] = resize(x, new_shape[:2], data_channels)

    if len(x_train.shape) < 4:
        x_train = expand_dims(x_train, 3)
    if len(x_test.shape) < 4:
        x_test = expand_dims(x_test, 3)
    if len(x_ts.shape) < 4:
        x_ts = expand_dims(x_ts, 3)
    return (x_train, y_train), (x_test, y_test), x_ts
