import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist as dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, CategoricalAccuracy, binary_accuracy, BinaryAccuracy, AUC
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.callbacks import EarlyStopOnBaseline
from vision_transformer_base import build_model


def main():
    (X, Y), (Xtest, Ytest) = dataset.load_data()
    _X = []
    for x in X:
        _X.append(np.pad(x, [[2, 2], [2, 2]]) / 255)
    X = np.array(_X)
    _X = []
    for x in Xtest:
        _X.append(np.pad(x, [[2, 2], [2, 2]]) / 255)
    Xtest = np.array(_X)
    model = build_model(
        input_shape=(32, 32, 1),
        patch_size=4,
        noof_classes=10
    )
    call_early_stop = EarlyStopOnBaseline(
            monitor="val_categorical_accuracy",
            baseline=0.3,
            min_delta=0.01,
            patience=4,
            verbose=0,
            restore_best=True,
            )
    call_lr_reductor = ReduceLROnPlateau(
            monitor="val_loss",
            min_delta=0.001,
            factor=0.1,
            patience=1,
            verbose=1,
            cooldown=1,
            min_lr=1e-10,
        )
    model.compile(Adam(learning_rate=9e-4), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    history = model.fit(
                      X, to_categorical(Y),
                      validation_split=0.2,
                      epochs=100,
                      batch_size=16,
                      # required for shuffling
                      shuffle=True,
                      verbose=1,
                      callbacks=[call_lr_reductor, call_early_stop],
                      )
    model.evaluate(Xtest, to_categorical(Ytest), batch_size=16)
    return True


if __name__ == "__main__":
    print(main())