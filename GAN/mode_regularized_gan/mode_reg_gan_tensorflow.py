import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128
lam1 = 1e-2
lam2 = 1e-2

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

E_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
E_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
E_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
E_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_E = [E_W1, E_W2, E_b1, E_b2]
theta_G = [G_W1, G_W2, G_b1, G_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def encoder(x):
    E_h1 = tf.nn.relu(tf.matmul(x, E_W1) + E_b1)
    out = tf.matmul(E_h1, E_W2) + E_b2
    return out


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_log_prob = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_log_prob)
    return D_prob


G_sample = generator(z)
G_sample_reg = generator(encoder(X))

D_real = discriminator(X)
D_fake = discriminator(G_sample)
D_reg = discriminator(G_sample_reg)

mse = tf.reduce_sum((X - G_sample_reg)**2, 1)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
E_loss = tf.reduce_mean(lam1 * mse + lam2 * D_reg)
G_loss = -tf.reduce_mean(tf.log(D_fake)) + E_loss

E_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
            .minimize(E_loss, var_list=theta_E))
D_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=1e-3)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    _, E_loss_curr = sess.run(
        [E_solver, E_loss],
        feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
    )

    if it % 1000 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; E_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr, E_loss_curr))

        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
