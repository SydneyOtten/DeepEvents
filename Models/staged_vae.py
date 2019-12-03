'''Large parts taken from the VAE MNIST example.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Add, Activation
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import History, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import objectives
from keras import initializers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#load toy-model events
filename = 'ttbar_20de-6_latent_code.csv'
latcode_df = pd.read_csv(filename, sep=' ', header=None)
data = latcode_df.values
data = data[:,0:20]		

trainsize = 100000
	
print(np.shape(data))
x_train = data[:trainsize]
x_test = data[100000:200000]
image_size = x_train.shape[1]
original_dim = image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# network parameters
input_shape = (original_dim, )
intermediate_dim = 128
encoder_dim = 128
batch_size = 1024
latent_dim = 20
epochs = 120
eluvar = np.sqrt(1.55/intermediate_dim)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x1 = Dense(encoder_dim, activation='elu', kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(inputs)
x2 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x1)
x2 = Activation('elu')(x2)
x3 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x2)
x3 = Activation('elu')(x3)
x4 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x3)
x4 = Activation('elu')(x4)
z_mean = Dense(latent_dim, name='z_mean')(x4)
z_log_var = Dense(latent_dim, name='z_log_var')(x4)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x1 = Dense(intermediate_dim, activation='elu', kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(latent_inputs)
x2 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x1)
x2 = Activation('elu')(x2)
x3 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x2)
x3 = Activation('elu')(x3)
x4 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x3)
x4 = Activation('elu')(x4)
outputs = Dense(original_dim, activation='linear')(x4)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='ttbar_vae')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	help_ = "Load h5 model trained weights"
	parser.add_argument("-w", "--weights", help=help_)
	help_ = "Use mse loss instead of binary cross entropy (default)"
	parser.add_argument("-m",
						"--mse",
						help=help_, action='store_true')
	args = parser.parse_args()
	models = (encoder, decoder)
	data = (x_test, x_test)
	
	def vae_loss(x, x_decoded_mean):
		mse_loss = objectives.mse(x, x_decoded_mean)
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		beta=0.001
		loss = K.mean(mse_loss + beta*kl_loss)
		return loss
	
	learnrate = 0.001
	iterations = 7
	lr_limit = 0.001/(2**iterations)
	history = History()
	k=0
	checkpointer = ModelCheckpoint(filepath='staged_vae_2_latent_code.hdf5', verbose=1, save_best_only=True)
	opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	vae.compile(optimizer=opt, loss=vae_loss)

	vae.summary()

	k=0
	if args.weights:
		vae.load_weights(args.weights)
	else:
		while learnrate > lr_limit:
			if k < 4:
				opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
			else:
				opt = SGD(lr=learnrate, decay=1e-6, momentum=0.9, nesterov=True)
				epochs=120
			vae.compile(loss=vae_loss, optimizer=opt, metrics=['mse'])
			vae.fit(x_train, x_train,
					epochs=epochs,
					batch_size=batch_size,
					validation_data=(x_test, x_test),
					callbacks = [checkpointer, history])
			vae.load_weights('staged_vae_2_latent_code.hdf5')
			learnrate /= 2
			k=k+1
			
		# train the autoencoder

		vae.save_weights('staged_vae_2_latent_code.h5')

lat_dim = 20
z_samples = np.empty([1200000,lat_dim])

l=0
for i in range(0,1200000):
	for j in range(0,lat_dim):
		z_samples[l,j] = np.random.normal(0, 1)
	l=l+1
new_latent_code = decoder.predict(z_samples)
np.savetxt('staged_vae_2_latent_code.csv', new_latent_code)
