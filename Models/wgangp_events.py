from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from functools import partial
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import pandas as pd
import sys

import numpy as np

#======================================================================
class RandomWeightedAverage(_Merge):
	"""Provides a (random) weighted average between real and generated image samples"""
	#----------------------------------------------------------------------
	def _merge_function(self, inputs):
		# FD: should this be (hps['nn_smallest_unit']*2, 1, 1 ,1) now?
		alpha = K.random_uniform((32, 1, 1, 1))
		return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def load_real_samples():
	#load toy-model events
	filename = 'SYDNEY_ttbar_lepton_2.txt'
	twodecay_df = pd.read_csv(filename, sep=',', header=None)
	twodecay_df = twodecay_df.reindex(np.random.permutation(twodecay_df.index))
	data = twodecay_df.values
	data = data[:,0:20]			
	max = np.empty(20)
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

#======================================================================
class WGANGP():
	#----------------------------------------------------------------------
	def __init__(self):
		#if (hps['npx']%4):
		#    raise ValueError('WGAN: Width and height need to be divisible by 4.')
		self.img_rows = 20
		self.img_cols = 1
		self.img_shape = (self.img_rows,)
		self.latent_dim = 500

		# Following parameter and optimizer set as recommended in paper
		self.n_critic = 3
		opt = Adam(0.0001, 0.5)

		# Build the generator and critic
		self.generator = self.build_generator(units=512)
		self.critic = self.build_critic(units=512,
										alpha=0.2,
										dropout=0.1)

		#-------------------------------
		# Construct Computational Graph
		#       for the Critic
		#-------------------------------

		# Freeze generator's layers while training critic
		self.generator.trainable = False

		# Image input (real sample)
		real_img = Input(shape=(20,))

		# Noise input
		z_disc = Input(shape=(self.latent_dim,))
		# Generate image based of noise (fake sample)
		fake_img = self.generator(z_disc)

		# Discriminator determines validity of the real and fake images
		fake = self.critic(fake_img)
		valid = self.critic(real_img)

		# Construct weighted average between real and fake images
		interpolated_img = RandomWeightedAverage()([real_img, fake_img])
		# Determine validity of weighted sample
		validity_interpolated = self.critic(interpolated_img)

		# Use Python partial to provide loss function with additional
		# 'averaged_samples' argument
		partial_gp_loss = partial(self.gradient_penalty_loss,
						  averaged_samples=interpolated_img)
		partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

		self.critic_model = Model(inputs=[real_img, z_disc],
							outputs=[valid, fake, validity_interpolated])
		self.critic_model.compile(loss=[self.wasserstein_loss,
											  self.wasserstein_loss,
											  partial_gp_loss],
										optimizer=opt,
										loss_weights=[1, 1, 10])
		#-------------------------------
		# Construct Computational Graph
		#         for Generator
		#-------------------------------

		# For the generator we freeze the critic's layers
		self.critic.trainable = False
		self.generator.trainable = True

		# Sampled noise for input to generator
		z_gen = Input(shape=(self.latent_dim,))
		# Generate images based of noise
		img = self.generator(z_gen)
		# Discriminator determines validity
		valid = self.critic(img)
		# Defines generator model
		self.generator_model = Model(z_gen, valid)
		self.generator_model.compile(loss=self.wasserstein_loss, optimizer=opt)


	#----------------------------------------------------------------------
	def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
		"""
		Computes gradient penalty based on prediction and weighted real / fake samples
		"""
		gradients = K.gradients(y_pred, averaged_samples)[0]
		# compute the euclidean norm by squaring ...
		gradients_sqr = K.square(gradients)
		#   ... summing over the rows ...
		gradients_sqr_sum = K.sum(gradients_sqr,
								  axis=np.arange(1, len(gradients_sqr.shape)))
		#   ... and sqrt
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		# compute lambda * (1 - ||grad||)^2 still for each single sample
		gradient_penalty = K.square(1 - gradient_l2_norm)
		# return the mean as loss over all the batch samples
		return K.mean(gradient_penalty)


	#----------------------------------------------------------------------
	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	#----------------------------------------------------------------------
	def build_generator(self, units=128, momentum=0.8):
		model = Sequential()
		init = RandomNormal(stddev=0.02)
		model.add(Dense(units, kernel_initializer=init, input_dim = self.latent_dim)) 
		model.add(Activation("relu"))
		for i in range(4):
			model.add(Dense(units, kernel_initializer=init)) 
			model.add(Activation("relu"))
		model.add(Dense(20))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		img = model(noise)

		return Model(noise, img)

	#----------------------------------------------------------------------
	def build_critic(self, units=128, alpha=0.2, dropout=0.25):

		model = Sequential()
		init = RandomNormal(stddev=0.02)
		for i in range(5):
			model.add(Dense(units, kernel_initializer=init, input_shape=(20,)))
			model.add(LeakyReLU(alpha=alpha))
			model.add(Dropout(dropout))
		model.add(Dense(1))

		model.summary()

		img = Input(shape=(20,))
		validity = model(img)

		return Model(img, validity)

	#----------------------------------------------------------------------
	def train(self, epochs, batch_size=128):

		# # Load the dataset
		# (X_train, _), (_, _) = mnist.load_data()

		# # Rescale -1 to 1
		# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		# X_train = np.expand_dims(X_train, axis=3)
		X_train, max = load_real_samples()
		#X_train = np.expand_dims(X_train, axis=3)
		# Adversarial ground truths
		valid = -np.ones((batch_size, 1))
		fake =  np.ones((batch_size, 1))
		dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
		for epoch in range(epochs):

			for _ in range(self.n_critic):

				# ---------------------
				#  Train Discriminator
				# ---------------------

				# Select a random batch of images
				idx = np.random.randint(0, X_train.shape[0], batch_size)
				imgs = X_train[idx]
				# Sample generator input
				noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
				# Train the critic
				d_loss = self.critic_model.train_on_batch([imgs, noise],
																[valid, fake, dummy])

			# ---------------------
			#  Train Generator
			# ---------------------

			g_loss = self.generator_model.train_on_batch(noise, valid)

			# Plot the progress
			print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

	#----------------------------------------------------------------------
	def generate(self, nev):
		noise = np.random.normal(0, 1, (nev, self.latent_dim))
		return self.generator.predict(noise)

	#----------------------------------------------------------------------
	def load(self, folder):
		"""Load GAN from input folder"""
		# load the weights from input folder
		self.generator.load_weights('%s/generator.h5'%folder)
		self.critic.load_weights('%s/critic.h5'%folder)

	#----------------------------------------------------------------------
	def save(self, folder):
		"""Save the GAN weights to file."""
		self.generator.save_weights('%s/generator.h5'%folder)
		self.critic.save_weights('%s/critic.h5'%folder)

	#----------------------------------------------------------------------
	def description(self):
		descrip = 'WGAN-GP with width=%i, height=%i, latent_dim=%i'\
			% (self.img_rows, self.img_cols, self.latent_dim)
		return descrip
if __name__ == '__main__':
	gan = WGANGP()
	gan.train(epochs=32000, batch_size=1024)
	n_samples = 5000000
	noise = np.random.normal(0, 1, (n_samples, 500))
	gen_events = gan.generator.predict(noise)
	gen_events = np.reshape(gen_events, (n_samples, 20))
	data, max = load_real_samples()
	for i in range(20):
		gen_events[:,i] = gen_events[:,i]*max[i]
	np.savetxt('wgangp_events_96ksteps_500lat.csv', gen_events, delimiter=' ') 
