'''The VAE code skeleton taken from the VAE MNIST repository.
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
filename = 'ttb.csv'
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
epochs = 240
eluvar = np.sqrt(1.55/intermediate_dim)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x1 = Dense(encoder_dim, activation='elu', kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(inputs)
x2 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x1)
x2 = Activation('elu')(x2)
x3 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x2)
sc1 = Add()([x1,x3])
x3 = Activation('elu')(sc1)
x4 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x3)
sc2 = Add()([x2,x4])
x4 = Activation('elu')(sc2)
x5 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x4)
sc3 = Add()([x3,x5])
x5 = Activation('elu')(sc3)
x6 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x5)
sc4 = Add()([x4,x6])
x6 = Activation('elu')(sc4)
z_mean = Dense(latent_dim, name='z_mean')(x6)
z_log_var = Dense(latent_dim, name='z_log_var')(x6)

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
sc1 = Add()([x1,x3])
x3 = Activation('elu')(sc1)
x4 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x3)
sc2 = Add()([x2,x4])
x4 = Activation('elu')(sc2)
x5 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x4)
sc3 = Add()([x3,x5])
x5 = Activation('elu')(sc3)
x6 = Dense(encoder_dim, kernel_initializer=initializers.random_normal(mean=0.0, stddev=eluvar))(x5)
sc4 = Add()([x4,x6])
x6 = Activation('elu')(sc4)
outputs = Dense(original_dim, activation='tanh')(x6)

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
		beta=10**(-6)
		loss = K.mean((1-beta)*mse_loss + beta*kl_loss)
		return loss
	
	learnrate = 0.001
	iterations = 7
	lr_limit = 0.001/(2**iterations)
	history = History()
	k=0
	checkpointer = ModelCheckpoint(filepath='ttbar_20d_e-6.hdf5', verbose=1, save_best_only=True)
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
			vae.load_weights('ttbar_20d_e-6.hdf5')
			learnrate /= 2
			k=k+1
			
		# train the autoencoder

		vae.save_weights('ttbar_20d_e-6.h5')

latent_mean = encoder.predict(x_train)[0]
latent_logvar = encoder.predict(x_train)[1]
latent_var = np.exp(latent_logvar)
latent_std = np.sqrt(latent_var)
np.savetxt('latent_mean_20d_e-6.csv', latent_mean)
np.savetxt('latent_std_20d_e-6.csv', latent_std)
filename = 'latent_mean_20d_e-6.csv'
means_df = pd.read_csv(filename, sep=' ', header=None)
mean = means_df.values
filename = 'latent_std_20d_e-6.csv'
stds_df = pd.read_csv(filename, sep=' ', header=None)
std = stds_df.values
lat_dim = 20
b = 'e-6'
z_samples = np.empty([1200000,lat_dim])
l=0

#sampling from the new prior with gamma=0.05

l=0
for i in range(0,1200000):
	for j in range(0,lat_dim):
		z_samples[l,j] = np.random.normal(mean[i%100000,j], 0.05+std[i%100000,j])
	l=l+1
new_events = decoder.predict(z_samples)
for i in range(0,new_events.shape[1]):
	new_events[:,i]=new_events[:,i]*max[i]

np.savetxt('B-VAE_events.csv', new_events)
