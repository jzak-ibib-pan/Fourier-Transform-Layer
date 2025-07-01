# following https://keras.io/examples/vision/supervised-contrastive-learning/
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.callbacks import EarlyStopOnBaseline
from build_model_3_class import build_encoder


num_classes = 10
input_shape = (32, 32, 3)

# Load the train and test data splits
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#x_train = x_train / 255.
#x_test = x_test / 255.
# Display shapes of train and test datasets
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

print("CIFAR-10")

data_augmentation = keras.Sequential(
    [
        #layers.experimental.preprocessing.Normalization(),
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomFlip("vertical"),
        #layers.experimental.preprocessing.RandomRotation(0.02),
    ]
)

# Setting the state of the normalization layer.
data_augmentation.layers[0].adapt(x_train)


def create_encoder():
    # resnet = keras.applications.ResNet50V2(
    #     include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    # )
    #outputs = resnet(augmented)

    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    fourier = build_encoder(input_shape)
    outputs = fourier(augmented)
    #outputs = augmented
    # outputs = resnet(augmented)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model


encoder = create_encoder()
encoder.summary()
input("Waiting for input")

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 256
num_epochs = 1000
dropout_rate = 0.5
temperature = 0.05

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()


call_early_stop = EarlyStopOnBaseline(
        monitor="val_loss",
        baseline=6.,
        min_delta=0.0001,
        patience=10,
        verbose=0,
        restore_best=True,
    )
call_lr_reductor = ReduceLROnPlateau(
        monitor="val_loss",
        min_delta=0.0001,
        factor=0.6,
        patience=2,
        verbose=1,
        cooldown=2,
        min_lr=1e-10,
    )
history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
                         validation_split=0.2, callbacks=[call_lr_reductor, call_early_stop])

accuracy_classifier = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy_classifier * 100, 2)}%")

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Nadam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()


call_early_stop = EarlyStopOnBaseline(
        monitor="val_loss",
        baseline=6.,
        min_delta=0.0001,
        patience=10,
        verbose=0,
        restore_best=True,
    )
call_lr_reductor = ReduceLROnPlateau(
        monitor="val_loss",
        min_delta=0.0001,
        factor=0.6,
        patience=2,
        verbose=1,
        cooldown=2,
        min_lr=1e-10,
    )
history = encoder_with_projection_head.fit(
    x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
    validation_split=0.2, callbacks=[call_lr_reductor, call_early_stop]
)

classifier = create_classifier(encoder, trainable=False)


call_early_stop = EarlyStopOnBaseline(
        monitor="val_loss",
        baseline=6.,
        min_delta=0.0001,
        patience=10,
        verbose=0,
        restore_best=True,
    )
call_lr_reductor = ReduceLROnPlateau(
        monitor="val_loss",
        min_delta=0.0001,
        factor=0.6,
        patience=2,
        verbose=1,
        cooldown=2,
        min_lr=1e-10,
    )
history = classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs,
                         validation_split=0.2, callbacks=[call_lr_reductor, call_early_stop])

accuracy = classifier.evaluate(x_test, y_test)[1]
print(f"Test accuracy: {round(accuracy * 100, 2)}%, as opposed to {round(accuracy_classifier * 100, 2)}%")
