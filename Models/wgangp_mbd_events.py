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
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations
from tqdm import tqdm
import tensorflow as tf
import keras as keras

import numpy as np

class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


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
    filename = 'dataset_26d.csv'
    twodecay_df = pd.read_csv(filename, sep=' ', header=None)
    twodecay_df = twodecay_df.reindex(np.random.permutation(twodecay_df.index))
    data = twodecay_df.values
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

#======================================================================
class WGANGP():
    #----------------------------------------------------------------------
    def __init__(self):
        #if (hps['npx']%4):
        #    raise ValueError('WGAN: Width and height need to be divisible by 4.')
        self.img_rows = 26
        self.img_cols = 1
        self.img_shape = (self.img_rows,)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        opt = Adam(0.0001, beta_1=0, beta_2=0.999)

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
        real_img = Input(shape=(26,))

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
    def build_generator(self, units=128, momentum=0.8, alpha=0.2):
        model = Sequential()
        init = RandomNormal(stddev=0.02)
        model.add(Dense(units, kernel_initializer=init, input_dim = self.latent_dim)) 
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization())
        for i in range(4):
            model.add(Dense(units, kernel_initializer=init)) 
            model.add(LeakyReLU(alpha=alpha))
            model.add(BatchNormalization())
        model.add(Dense(26))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    #----------------------------------------------------------------------
    def build_critic(self, units=128, alpha=0.2, dropout=0.25):

        model = Sequential()
        init = RandomNormal(stddev=0.02)
        for i in range(3):
            model.add(Dense(units, kernel_initializer=init, input_shape=(26,), kernel_regularizer=regularizers.l2(0.001)))
            model.add(LeakyReLU(alpha=alpha))
            model.add(LayerNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(256, kernel_initializer=init, kernel_regularizer=regularizers.l2(0.001)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(LayerNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(128, kernel_initializer=init, kernel_regularizer=regularizers.l2(0.001)))
        model.add(LeakyReLU(alpha=alpha))
        model.add(LayerNormalization())
        model.add(Dropout(dropout))
        model.add(MinibatchDiscrimination(1,3))
        model.add(Dense(1))

        model.summary()

        img = Input(shape=(26,))
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
        for epoch in tqdm(range(epochs)):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.uniform(0, 1, (batch_size, self.latent_dim))
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
        noise = np.random.uniform(0, 1, (nev, self.latent_dim))
        return self.generator.predict(noise)

    #----------------------------------------------------------------------
    def load(self, folder):
        """Load GAN from input folder"""
        # load the weights from input folder
        self.generator.load_weights('%s/generator.h5'%folder)
        self.critic.load_weights('%s/critic.h5'%folder)

    #----------------------------------------------------------------------
    def save(self, filename):
        """Save the GAN weights to file."""
        self.generator.save_weights('networks/generator-%s.h5'%filename)
        self.critic.save_weights('networks/critic-%s.h5'%filename)

    #----------------------------------------------------------------------
    def description(self):
        descrip = 'WGAN-GP with width=%i, height=%i, latent_dim=%i'\
            % (self.img_rows, self.img_cols, self.latent_dim)
        return descrip
    
    
TOTAL_EPOCHS = 1850000/4
INTERMEDIATE_STEPS=20
if __name__ == '__main__':
    gan = WGANGP()
    for iteration in tqdm(range(INTERMEDIATE_STEPS)):
        gan.train(epochs=int(TOTAL_EPOCHS/INTERMEDIATE_STEPS), batch_size=256)
        gan.save(str(iteration))
        n_samples = 1200000
        noise = np.random.uniform(0, 1, (n_samples, 100))
        gen_events = gan.generator.predict(noise)
        gen_events = np.reshape(gen_events, (n_samples, 26))
        data, max = load_real_samples()
        for i in range(26):
            gen_events[:,i] = gen_events[:,i]*max[i]
        np.savetxt('wgangpmbd_events_32ksteps_100lat-'+str(iteration)+'.csv', gen_events, delimiter=' ') 
