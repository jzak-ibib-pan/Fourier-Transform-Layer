import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist as dataset
from tensorflow.keras.utils import to_categorical
from vision_transformer_origin import VisionTransformer


def main():
    (X, Y), (Xtest, Ytest) = dataset.load_data()
    Xtrain = []
    for x in X:
        Xtrain.append(np.pad(x, pad_width=[[2, 2], [2, 2]]))
    Xtrain = np.array(Xtrain)
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=4,
        num_classes=10,
        d_model=64,
        num_heads=4,
        mlp_dim=128,
        channels=3,
        dropout=0.1,
    )
    model.compile("adam", "categorical_crossentropy")
    model.fit(x=np.expand_dims(Xtrain, axis=-1), y=to_categorical(Y), batch_size=1, epochs=1)
    model.summary()
    return True


if __name__ == "__main__":
    print(main())