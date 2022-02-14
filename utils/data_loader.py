from numpy import logical_or, min, max, zeros, expand_dims, pad, repeat
from cv2 import resize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 2 classes only for now
def choose_images(data_x, data_y, classes=[0, 1]):
    assert classes[0] != classes[1], 'The same class provided.'
    y_chosen = logical_or(data_y == classes[0], data_y == classes[1])
    x = data_x[y_chosen]
    # categorical inaczej wywala
    y_cat = data_y[y_chosen]
    y_min = min(y_cat)
    y_max = max(y_cat)
    y_cat[y_cat == y_min] = 0
    y_cat[y_cat == y_max] = 1
    y = to_categorical(y_cat)
    return x, y


def choose_3_images(data_x, data_y, classes=[0, 1]):
    assert classes[0] != classes[1], 'The same class provided.'
    assert classes[0] != classes[2], 'The same class provided.'
    y_chosen = logical_or(logical_or(data_y == classes[0], data_y == classes[1]), data_y == classes[2])
    x = data_x[y_chosen]
    # categorical inaczej wywala
    y_cat = data_y[y_chosen]
    ordered = sorted(classes)
    y_cat[y_cat == ordered[0]] = 0
    y_cat[y_cat == ordered[1]] = 1
    y_cat[y_cat == ordered[2]] = 2
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

    # wyciągnij tylko 2 klasy
    if len(classes) == 2:
        x_train, y_train = choose_images(x_train, y_train, classes)
        x_test, y_test = choose_images(x_test, y_test, classes)
    else:
        x_train, y_train = choose_3_images(x_train, y_train, classes)
        x_test, y_test = choose_3_images(x_test, y_test, classes)

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