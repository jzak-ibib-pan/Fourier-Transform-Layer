from numpy import logical_or, min, max, zeros, expand_dims, pad, repeat
from cv2 import resize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def select_images_by_target(data_x, data_y, targets):
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


def prepare_data_for_sampling(classes=[0, 1], data_channels = 1, op=64):
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

    x_train, y_train = select_images_by_target(x_train, y_train, classes)
    x_test, y_test = select_images_by_target(x_test, y_test, classes)

    x_tr = zeros((x_train.shape[0], op, op))
    for it, x in enumerate(x_train):
        x_tr[it] = resize(x, (op, op))

    x_ts = zeros((x_test.shape[0], op, op))
    for it, x in enumerate(x_test):
        x_ts[it] = resize(x, (op, op))

    if len(x_train.shape) < 4:
        x_train = expand_dims(x_train, 3)
    if len(x_test.shape) < 4:
        x_test = expand_dims(x_test, 3)

    return (x_train, y_train), (x_test, y_test), x_ts