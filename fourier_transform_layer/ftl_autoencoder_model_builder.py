# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import Conv2DTranspose, Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import regularizers
from fourier_transform_layer import FTL, FTLReshape


class FTLAutoencoder:
	@staticmethod
	def build(
			width,
			height,
			depth,
			filters=(32, 64),
			latentDim=16
		):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1
		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs
		# loop over the number of filters
		x = FTLReshape(
			activation=None,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l1_l2(1e-5),
			sampling_nominator=2,
			direction="down",
            train_imaginary=True,
            inverse=False,
			calculate_abs=False,
			normalize_to_image_shape=False,
            already_fft=False,
            use_bias=True,
            bias_initializer='zeros',
            bias_regularizer=regularizers.l1_l2(1e-5),
			)(x)
		x = LeakyReLU(alpha=0.2)(x)
		#x = BatchNormalization(axis=chanDim)(x)
		#x = Conv2D(filters=height * width * depth * 2, kernel_size=1)(x)
		volumeSize = K.int_shape(x)
		# reshape to flatten
		x = Reshape(target_shape=(height * width // (2 ** 2) * depth * 2,))(x)
		for f in filters:
			x = Conv1D(f, kernel_size=3, strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		# flatten the network and then construct our latent vector
		latent = Dense(latentDim)(x)
		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")
		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape(volumeSize[1:])(x)
		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv1DTranspose(f, kernel_size=3, strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)
		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		#x = Conv1DTranspose(depth * 2, kernel_size=1, padding="same")(x)
		#x = Reshape(target_shape=(height // 2, width // 2, depth * 2))(x)
		outputs = FTLReshape(
			activation="sigmoid",
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l1_l2(1e-5),
			sampling_nominator=2,
			direction="up",
            train_imaginary=True,
            inverse=True,
			calculate_abs=True,
			normalize_to_image_shape=False,
            already_fft=True,
            use_bias=True,
            bias_initializer='zeros',
            bias_regularizer=regularizers.l1_l2(1e-5),
			)(x)
		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")
		# our autoencoder is the encoder + decoder
		autoencoder = Model(
			inputs,
			decoder(encoder(inputs)),
			name="autoencoder"
			)
		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)


if __name__ == "__main__":
	(encoder, decoder, autoencoder) = FTLAutoencoder.build(
		width=128,
		height=128,
		depth=1,
		filters=(),
		latentDim=256
	)
	encoder.summary()
	decoder.summary()
	autoencoder.summary()
	autoencoder.compile(loss="mse", optimizer="adam")
	autoencoder.run_eagerly = True
	autoencoder.fit(np.ones((1, 128, 128, 1)), np.ones((1, 128, 128, 2)), epochs=1, steps_per_epoch=1)