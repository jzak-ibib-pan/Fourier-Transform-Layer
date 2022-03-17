from numpy import logical_or, zeros, expand_dims, pad, repeat, array, uint8, arange, float32, save, load, squeeze
from numpy.random import shuffle
from cv2 import resize, imread, cvtColor, COLOR_RGB2GRAY
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from os import listdir
from os.path import join, isfile


def _select_images_by_target(data_x, data_y, targets):
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


def _grayscale_data(data):
    result = zeros((data.shape[:-1]))
    for it, x in enumerate(data):
        result[it] = cvtColor(x, COLOR_RGB2GRAY)
    return result


def _load_celeb(reset=False):
    filepath = join('Y://', 'super_resolution', 'CelebAMask-HQ', 'CelebA-HQ-img')
    if isfile(join(filepath, 'images.npy')) and not reset:
        result = load(join(filepath, 'images.npy'))
        return result
    loof_files = [l for l in listdir(filepath) if '.jpg' in l]
    new_shape = (130, 130)
    # each image is of this size
    result = zeros((len(loof_files), *new_shape, 1), dtype=uint8)
    # iterators to be randomized
    numbers = arange(len(loof_files))
    shuffle(numbers)
    # instead of enumerate
    for it, filename in zip(numbers, loof_files):
        # 0 - grayscale
        image = imread(join(filepath, filename), 0)
        image = resize(image, new_shape)
        # to work with values setting into results
        if len(image.shape) < 3:
            image = expand_dims(image, axis=-1)
        result[it] = image
    result = float32(result) / 255
    save(join(filepath, 'images.npy'), result)
    return result


def prepare_data_for_sampling(dataset, **kwargs):
    assert type(dataset) is str, 'Dataset must be a str.'
    assert dataset in ['mnist', 'fmnist', 'cifar10', 'celeb'], f'{dataset.capitalize()} not implemented yet.'
    if dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_max = 10
    if dataset.lower() == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        train_max = 10
    if dataset.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        train_max = 10
    new_shape = None
    if 'new_shape' in kwargs.keys():
        new_shape = kwargs['new_shape']
    if dataset.lower() == 'celeb':
        reset = False
        if 'reset' in kwargs.keys():
            reset = kwargs['reset']
        X = _load_celeb(reset=reset)
        # calculate train / test split
        test_split = 0.1
        cutoff = int((1 - test_split) * X.shape[0])
        if not new_shape:
            # return split data (train), (test)
            return (X[:cutoff], None), (X[cutoff:], None)
        # smaller dataset-> // 4
        x_tr = []
        shx, shy = [x // 2 for x in X.shape[1:3]]
        for x in X:
            # extract part of image - for calculations' sake
            x_tr.append(x[shx - new_shape[0] // 4 : shx + new_shape[0] // 4,
                        shy - new_shape[0] // 4 : shy + new_shape[0] // 4])
        x_tr = array(x_tr)
        # 'resized' dataset-> // 2
        x_re = []
        for x in X:
            # extract part of image - for calculations' sake
            x_re.append(x[shx - new_shape[0] // 2 : shx + new_shape[0] // 2,
                        shy - new_shape[0] // 2 : shy + new_shape[0] // 2])
        x_re = array(x_re)
        return (x_tr[:cutoff], None), (x_tr[cutoff:], None), (x_re[:cutoff], x_re[cutoff:])

    noof_channels = 1
    if 'noof_channels' in kwargs.keys():
        noof_channels = kwargs['noof_channels']

    targets = range(10)
    if 'targets' in kwargs.keys():
        targets = kwargs['targets']

    if x_train.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_train.shape[1:3]]
        x_train = pad(x_train, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])

    if x_test.shape[1] < 32:
        pads = [(32 - sh) // 2 for sh in x_test.shape[1:3]]
        x_test = pad(x_test, [[0, 0], [pads[0], pads[0]], [pads[1], pads[1]]])

    if len(targets) < train_max:
        x_train, y_train = _select_images_by_target(x_train, y_train, targets)
        x_test, y_test = _select_images_by_target(x_test, y_test, targets)
    else:
        # suppose that both subsets contain the same classes
        y_train = to_categorical(y_train, train_max)
        y_test = to_categorical(y_test, train_max)

    if len(x_train.shape) < 4:
        x_train = repeat(expand_dims(x_train / 255, axis=-1), repeats=noof_channels, axis=-1)
    if len(x_test.shape) < 4:
        x_test = repeat(expand_dims(x_test / 255, axis=-1), repeats=noof_channels, axis=-1)

    # supposedly convert to grayscale
    if x_train.shape[3] > noof_channels:
        x_train = _grayscale_data(x_train)

    # supposedly convert to grayscale
    if x_test.shape[3] > noof_channels:
        x_test = _grayscale_data(x_test)

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


def flower():
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    directory = join('Y://', 'FCNN', 'archive_natural_scenes', 'seg_train')
    dataset = image_dataset_from_directory(directory,
                                           labels="inferred",
                                           label_mode="categorical",
                                           class_names=None,
                                           color_mode="rgb",
                                           batch_size=4,
                                           image_size=(100, 100),
                                           shuffle=True,
                                           seed=None,
                                           # SOLVED: check validation split is working - must use subset as either
                                           # "training" or "validation"
                                           validation_split=0.1,
                                           subset=None,
                                           interpolation="bicubic",
                                           follow_links=False)
    # SOLVED: check for possible error in iteration - iteration is infinite
    b = 0
    while True:
        X, Y = next(dataset.as_numpy_iterator())
        print(X.shape)
        b += 4
        print(b)


if __name__ == '__main__':
    print(flower())