from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy, CategoricalAccuracy, binary_accuracy, BinaryAccuracy, AUC
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from matplotlib import pyplot as plt
from utils.callbacks import EarlyStopOnBaseline
#from utils.turn_gpu_off import turn_gpu_off
#from utils.reset_session import session_reset
from build_model_3_class import build_encoder, build_projector, build_supervised_model, train_step


dataset = fashion_mnist


def choose_classes(
        xdata,
        ydata,
        classes
        ):
    X, Y = [], []
    for x, y in zip(xdata, ydata):
        if y not in classes:
            continue
        X.append(x)
        Y.append(classes.index(y))
    return X, Y


def preprocess_data(
        xdata,
        ydata,
        cat=False,
        ):
    X, Y = [], []
    for x, y in zip(xdata, ydata):
        X.append(x.astype(np.float32) / 255)
        #X.append(x.astype(np.float32))
        Y.append(y)
    if cat:
        Y = to_categorical(Y)
    else:
        Y = np.array(Y)
    X = np.array(X)
    if len(X.shape) <= 3:
        X = np.expand_dims(X, axis=-1)
    if X.shape[1] < 32:
        X = np.pad(
            X,
            [[0, 0], [2, 2], [2, 2], [0, 0]]
        )
    #X = np.repeat(X, repeats=3, axis=-1)
    return X, Y


def load_dataset(classes=(0, 1, 2), cat=False):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    X_train, Y_train = choose_classes(x_train, y_train, classes)
    X_train, Y_train = preprocess_data(X_train, Y_train, cat)
    X = list(X_train)
    Y = list(Y_train)
    for x, y in zip(X_train, Y_train):
        X.append(np.fliplr(x))
        X.append(np.flipud(x))
        Y.append(y)
        Y.append(y)
    X_test, Y_test = choose_classes(x_test, y_test, classes)
    X_test, Y_test = preprocess_data(X_test, Y_test, cat)
    return (X_train, Y_train), (X_test, Y_test)


def main():
    call_early_stop = EarlyStopOnBaseline(
        monitor="val_categorical_accuracy",
        baseline=0.5,
        min_delta=0.0001,
        patience=10,
        verbose=0,
        restore_best=True,
    )
    call_lr_reductor = ReduceLROnPlateau(
        monitor="val_categorical_accuracy",
        min_delta=0.0001,
        factor=0.6,
        patience=2,
        verbose=1,
        cooldown=2,
        min_lr=1e-10,
    )
    classes = list(np.arange(0, 10))
    batch = 32
    (X, Y), (Xtest, Ytest) = load_dataset(classes)
    encoder = build_encoder(
        input_shape=X.shape[1:],
    )
    projector = build_projector()
    optimizer = Adam(
        learning_rate=1e-3,
        #clipnorm=5.,
        #epsilon=0.1
        )
    for epoch in range(5):
        losses = []
        for step in range(X.shape[0] // batch):
            x, y = [], []
            it = 0
            while len(x) < batch:
                x.append(np.squeeze(X[it]))
                y.append(np.squeeze(Y[it]))
                it = (it + 1) % X.shape[0]
            loss = train_step(
                encoder,
                projector,
                np.array(x),
                np.array(y),
                optimizer
            )
            losses.append(loss)
        print(np.mean(losses))

    # supervised
    (X, Y), (Xtest, Ytest) = load_dataset(classes, cat=True)
    supervised_model = build_supervised_model(
        input_shape=X.shape[1:],
        encoder=encoder
        )
    supervised_model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[CategoricalAccuracy()]
        )
    supervised_model.fit(
        X,
        Y,
        epochs=200,
        #callbacks=[call_lr_reductor, call_early_stop],
    )
    supervised_model.evaluate(Xtest, Ytest)
    return loss


if __name__ == "__main__":
    print(main())