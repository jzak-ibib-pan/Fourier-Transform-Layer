from tensorflow.keras.datasets import mnist
from utils.data_loader import select_images_by_target


def test_on_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    for target_01 in range(10):
        # train
        noof_images = sum(y_train == target_01)
        x, y = select_images_by_target(x_train, y_train, [target_01])
        assert x.shape[0] == noof_images, 'X train not passed.'
        assert y.shape[0] == noof_images, 'Y train not passed.'
        # test
        noof_images = sum(y_test == target_01)
        x, y = select_images_by_target(x_test, y_test, [target_01])
        assert x.shape[0] == noof_images, 'X test not passed.'
        assert y.shape[0] == noof_images, 'Y test not passed.'
        for target_02 in range(10):
            if target_01 == target_02:
                continue
            # train
            noof_images = sum([y == target_01 or y == target_02 for y in y_train])
            x, y = select_images_by_target(x_train, y_train, [target_01, target_02])
            assert x.shape[0] == noof_images, 'X train not passed for 2 classes.'
            assert y.shape[0] == noof_images, 'Y train not passed for 2 classes.'
            # test
            noof_images = sum([y == target_01 or y == target_02 for y in y_test])
            x, y = select_images_by_target(x_test, y_test, [target_01, target_02])
            assert x.shape[0] == noof_images, 'X test not passed for 2 classes.'
            assert y.shape[0] == noof_images, 'Y test not passed for 2 classes.'
    return True


if __name__ == '__main__':
    print(test_on_mnist())