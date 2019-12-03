from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import initializers, regularizers, constraints, activations
import pandas as pd
import matplotlib.pyplot as plt
from keras.engine import InputSpec, Layer
import keras.backend as K
from fom_ls import fom

import sys

import numpy as np
import os

class MinibatchDiscrimination(Layer):
	"""Concatenates to each sample information about how different the input
	features for that sample are from features of other samples in the same
	minibatch, as described in Salimans et. al. (2016). Useful for preventing
	GANs from collapsing to a single output. When using this layer, generated
	samples and reference samples should be in separate batches.
	# Example
	```python
		# apply a convolution 1d of length 3 to a sequence with 10 timesteps,
		# with 64 output filters
		model = Sequential()
		model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
		# now model.output_shape == (None, 10, 64)
		# flatten the output so it can be fed into a minibatch discrimination layer
		model.add(Flatten())
		# now model.output_shape == (None, 640)
		# add the minibatch discrimination layer
		model.add(MinibatchDiscrimination(5, 3))
		# now model.output_shape = (None, 645)
	```
	# Arguments
		nb_kernels: Number of discrimination kernels to use
			(dimensionality concatenated to output).
		kernel_dim: The dimensionality of the space where closeness of samples
			is calculated.
		init: name of initialization function for the weights of the layer
			(see [initializations](../initializations.md)),
			or alternatively, Theano function to use for weights initialization.
			This parameter is only relevant if you don't pass a `weights` argument.
		weights: list of numpy arrays to set as initial weights.
		W_regularizer: instance of [WeightRegularizer](../regularizers.md)
			(eg. L1 or L2 regularization), applied to the main weights matrix.
		activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
			applied to the network output.
		W_constraint: instance of the [constraints](../constraints.md) module
			(eg. maxnorm, nonneg), applied to the main weights matrix.
		input_dim: Number of channels/dimensions in the input.
			Either this argument or the keyword argument `input_shape`must be
			provided when using this layer as the first layer in a model.
	# Input shape
		2D tensor with shape: `(samples, input_dim)`.
	# Output shape
		2D tensor with shape: `(samples, input_dim + nb_kernels)`.
	# References
		- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
	"""

	def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
				 W_regularizer=None, activity_regularizer=None,
				 W_constraint=None, input_dim=None, **kwargs):
		self.init = initializers.get(init)
		self.nb_kernels = nb_kernels
		self.kernel_dim = kernel_dim
		self.input_dim = input_dim

		self.W_regularizer = regularizers.get(W_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)

		self.initial_weights = weights
		self.input_spec = [InputSpec(ndim=2)]

		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(MinibatchDiscrimination, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 2

		input_dim = input_shape[1]
		self.input_spec = [InputSpec(dtype=K.floatx(),
									 shape=(None, input_dim))]

		self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
			initializer=self.init,
			name='kernel',
			regularizer=self.W_regularizer,
			trainable=True,
			constraint=self.W_constraint)

		# Set built to true.
		super(MinibatchDiscrimination, self).build(input_shape)

	def call(self, x, mask=None):
		activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
		diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
		abs_diffs = K.sum(K.abs(diffs), axis=2)
		minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
		return K.concatenate([x, minibatch_features], 1)

	def compute_output_shape(self, input_shape):
		assert input_shape and len(input_shape) == 2
		return input_shape[0], input_shape[1]+self.nb_kernels

	def get_config(self):
		config = {'nb_kernels': self.nb_kernels,
				  'kernel_dim': self.kernel_dim,
				  'init': self.init.__name__,
				  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				  'input_dim': self.input_dim}
		base_config = super(MinibatchDiscrimination, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def load_real_samples():
	#load toy-model events
	filename = 'new_ttb.csv'
	ttb_df = pd.read_csv(filename, sep=' ', header=None)
	data = ttb_df.values
	data = data[:,0:26]			
	max = np.empty(26)
	for i in range(0,data.shape[1]):
		max[i] = np.max(np.abs(data[:,i]))
		if np.abs(max[i]) > 0: 
			data[:,i] = data[:,i]/max[i]
		else:
			pass
	image_size = data.shape[1]
	original_dim = image_size
	data = np.reshape(data, [-1, original_dim])
	data = data.astype('float32')
	return data, max

class LSGAN():
	def __init__(self):
		self.img_rows = 26
		self.img_cols = 1
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='mse',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# The generator takes noise as input and generated imgs
		z = Input(shape=(self.latent_dim,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator)
		# Trains generator to fool discriminator
		self.combined = Model(z, valid)
		# (!!!) Optimize w.r.t. MSE loss instead of crossentropy
		self.combined.compile(loss='mse', optimizer=optimizer)

	def build_generator(self):

		model = Sequential()

		model.add(Dense(512, input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(Reshape(self.img_shape))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Flatten(input_shape=self.img_shape))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(128))
		model.add(LeakyReLU(alpha=0.2))
		# (!!!) No softmax
		model.add(Dense(128))
		model.add(MinibatchDiscrimination(1,3))
		model.add(Dense(1))
		model.summary()

		img = Input(shape=self.img_shape)
		validity = model(img)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, sample_interval=50):

		# Load the dataset
		X_train, max = load_real_samples()

		# Rescale -1 to 1
		#X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		X_train = np.expand_dims(X_train, axis=3)

		# Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			# Sample noise as generator input
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

			# Generate a batch of new images
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.combined.train_on_batch(noise, valid)

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# If at save interval => save generated image samples
			# if epoch % sample_interval == 0:
			#    self.sample_images(epoch)
			if epoch % 1000 == 0:
				self.sample_events(epoch)
				self.generator.save('generator_%d.h5' % epoch)
				
	def sample_events(self, epoch):
		n_samples = 1200000
		noise = np.random.normal(0, 1, (n_samples, 100))
		gen_events = self.generator.predict(noise)
		gen_events = np.reshape(gen_events, (n_samples, 26))
		np.savetxt('ls_gan_checkpoint_%d.csv' % epoch, gen_events, delimiter=' ')
		tot, tot2d = fom('ls_gan_checkpoint_%d.csv' % epoch)
		one_d_hist.append(tot)
		two_d_hist.append(tot2d)
		os.remove('ls_gan_checkpoint_%d.csv' % epoch)
		
	def sample_images(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, self.latent_dim))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/mnist_%d.png" % epoch)
		plt.close()


if __name__ == '__main__':
	gan = LSGAN()
	one_d_hist = []
	two_d_hist = []
	gan.train(epochs=96000, batch_size=1024, sample_interval=200)
	n_samples = 1200000
	noise = np.random.normal(0, 1, (n_samples, 100))
	gen_events = gan.generator.predict(noise)
	gen_events = np.reshape(gen_events, (n_samples, 26))
	for i in range(0,26):
		gen_events[:,i] = gen_events[:,i]*max[i]
	np.savetxt('ls_gan_events_96ksteps.csv', gen_events, delimiter=' ') 
	np.save('1d_hist_scores_lsgan.csv', one_d_hist)
	np.save('2d_hist_scores_lsgan.csv', two_d_hist)