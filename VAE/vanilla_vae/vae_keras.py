from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


tf.python.control_flow_ops = tf

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)
X_train = mnist.train.images
X_test, y_test = mnist.test.images, mnist.test.labels

m = 50
n_z = 2
n_epoch = 10


# Q(z|X) -- encoder
inputs = Input(shape=(784,))
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# P(X|z) -- decoder
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(784, activation='sigmoid')

h_p = decoder_hidden(z)
outputs = decoder_out(h_p)


def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl


# We now have 3 models: encoder, decoder, & VAE (encoder + decoder)
vae = Model(inputs, outputs)
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(X_train, X_train, batch_size=m, nb_epoch=n_epoch)

encoder = Model(inputs, mu)

d_in = Input(shape=(n_z,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)


# Latent space visualization
encoded = encoder.predict(X_test, batch_size=m)

plt.figure(figsize=(6, 6))
plt.scatter(encoded[:, 0], encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()


# Reconstruction visualization
idxs = np.random.randint(0, X_test.shape[0], size=m)
X_test_subset = X_test[idxs]
X_recons = vae.predict(X_test_subset, batch_size=m)

n = 10
plt.figure(figsize=(20, 4))

for i in range(1, n+1):
    # Original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test_subset[i].reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstruction
    ax = plt.subplot(2, n, i+n)
    plt.imshow(X_recons[i].reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# Generating new samples from latent space; P(X|z) visualization
z_sample = np.random.randn(n, n_z)
X_gen = decoder.predict(z_sample)

plt.figure(figsize=(20, 4))

for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_gen[i-1].reshape(28, 28), cmap='Greys_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
