from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from numpy import asarray, expand_dims, pad
from utils import turn_gpu_off
from utils.builders import CNNBuilder
import ipykernel


def main():
    for modelname in ['mobilenetv2', 'mobilenet']:
        builder = CNNBuilder(model_type=modelname, input_shape=(32, 32, 1), noof_classes=10,
                             filename='mnist_cnns', filepath='results')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_tr = []
        for x in x_train:
            x_tr.append(pad(x, pad_width=[[2, 2], [2, 2]], mode='constant', constant_values=0))
        x_train = expand_dims(asarray(x_tr) / 255, axis=-1)
        y_train = to_categorical(y_train, 10)

        x_tr = []
        for x in x_test:
            x_tr.append(pad(x, pad_width=[[2, 2], [2, 2]], mode='constant', constant_values=0))
        x_test = expand_dims(asarray(x_tr) / 255, axis=-1)
        y_test = to_categorical(y_test, 10)

        builder.compile_model('adam', 'categorical_crossentropy', metrics=[CategoricalAccuracy(),
                                                                           TopKCategoricalAccuracy(k=5, name='top-5')])
        builder.train_model(100, x_data=x_train, y_data=y_train, batch=8, validation_split=0.2,
                            call_time=True, call_stop=True, call_checkpoint=True,
                            call_stop_kwargs={'baseline': 0.90,
                                              'monitor': 'val_categorical_accuracy',
                                              'patience': 3,
                                              },
                            call_checkpoint_kwargs={'monitor': 'val_categorical_accuracy',
                                                    }
                            )
        builder.evaluate_model(x_data=x_test, y_data=y_test)
        # prefer comparison between different models on the same dataset
        builder.save_model_info('First training pipeline')


if __name__ == '__main__':
    main()