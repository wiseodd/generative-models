import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.ndimage.interpolation


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

mb_size = 32
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
z_dim = 10
h_dim = 128
eps = 1e-8
lr = 1e-3
d_steps = 3
lam1, lam2 = 1000, 1000


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


X1 = tf.placeholder(tf.float32, shape=[None, X_dim])
X2 = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G1_W1 = tf.Variable(xavier_init([X_dim + z_dim, h_dim]))
G1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G1_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G1_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G2_W1 = tf.Variable(xavier_init([X_dim + z_dim, h_dim]))
G2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G2_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G2_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def G1(X1, z):
    inputs = tf.concat([X1, z], 1)
    h = tf.nn.relu(tf.matmul(inputs, G1_W1) + G1_b1)
    return tf.nn.sigmoid(tf.matmul(h, G1_W2) + G1_b2)


def G2(X2, z):
    inputs = tf.concat([X2, z], 1)
    h = tf.nn.relu(tf.matmul(inputs, G2_W1) + G2_b1)
    return tf.nn.sigmoid(tf.matmul(h, G2_W2) + G2_b2)


D1_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D1_W2 = tf.Variable(xavier_init([h_dim, 1]))
D1_b2 = tf.Variable(tf.zeros(shape=[1]))

D2_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D2_W2 = tf.Variable(xavier_init([h_dim, 1]))
D2_b2 = tf.Variable(tf.zeros(shape=[1]))


def D1(X):
    h = tf.nn.relu(tf.matmul(X, D1_W1) + D1_b1)
    return tf.matmul(h, D1_W2) + D1_b2


def D2(X):
    h = tf.nn.relu(tf.matmul(X, D1_W1) + D1_b1)
    return tf.matmul(h, D2_W2) + D2_b2


theta_G1 = [G1_W1, G1_W2, G1_b2, G1_b2]
theta_G2 = [G2_W1, G2_b1, G2_W2, G2_b2]
theta_G = theta_G1 + theta_G2

theta_D1 = [D1_W1, D1_W2, D1_b1, D1_b2]
theta_D2 = [D2_W1, D2_b1, D2_W2, D2_b2]

# D
X1_sample = G2(X2, z)
X2_sample = G1(X1, z)

D1_real = D1(X2)
D1_fake = D1(X2_sample)

D2_real = D2(X1)
D2_fake = D2(X1_sample)

D1_G = D1(X1_sample)
D2_G = D2(X2_sample)

X1_recon = G2(X2_sample, z)
X2_recon = G1(X1_sample, z)
recon1 = tf.reduce_mean(tf.reduce_sum(tf.abs(X1 - X1_recon), 1))
recon2 = tf.reduce_mean(tf.reduce_sum(tf.abs(X2 - X2_recon), 1))

D1_loss = tf.reduce_mean(D1_fake) - tf.reduce_mean(D1_real)
D2_loss = tf.reduce_mean(D2_fake) - tf.reduce_mean(D2_real)
G_loss = -tf.reduce_mean(D1_G + D2_G) + lam1*recon1 + lam2*recon2

D1_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
             .minimize(D1_loss, var_list=theta_D1))
D2_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
             .minimize(D2_loss, var_list=theta_D2))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D1 + theta_D2]


sess = tf.Session()
sess.run(tf.global_variables_initializer())

X_train = mnist.train.images
half = int(X_train.shape[0] / 2)

# Real image
X_train1 = X_train[:half]
# Rotated image
X_train2 = X_train[half:].reshape(-1, 28, 28)
X_train2 = scipy.ndimage.interpolation.rotate(X_train2, 90, axes=(1, 2))
X_train2 = X_train2.reshape(-1, 28*28)

# Cleanup
del X_train


def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(d_steps):
        X1_mb, X2_mb = sample_X(X_train1, mb_size), sample_X(X_train2, mb_size)
        z_mb = sample_z(mb_size, z_dim)

        _, _, D1_loss_curr, D2_loss_curr, _ = sess.run(
            [D1_solver, D2_solver, D1_loss, D2_loss, clip_D],
            feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
    )

    if it % 1000 == 0:
        sample1, sample2 = sess.run(
            [X1_sample, X2_sample],
            feed_dict={X1: X1_mb[:4], X2: X2_mb[:4], z: sample_z(4, z_dim)}
        )

        samples = np.vstack([X1_mb[:4], sample1, X2_mb[:4], sample2])

        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D1_loss_curr + D2_loss_curr, G_loss_curr))

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
