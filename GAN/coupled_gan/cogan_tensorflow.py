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

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G1_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G1_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G2_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G2_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def G(z):
    h = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G1 = tf.nn.sigmoid(tf.matmul(h, G1_W2) + G1_b2)
    G2 = tf.nn.sigmoid(tf.matmul(h, G2_W2) + G2_b2)
    return G1, G2


D1_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D1_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D2_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D2_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))


def D(X1, X2):
    h1 = tf.nn.relu(tf.matmul(X1, D1_W1) + D1_b1)
    h2 = tf.nn.relu(tf.matmul(X2, D2_W1) + D2_b1)
    D1_out = tf.nn.sigmoid(tf.matmul(h1, D_W2) + D_b2)
    D2_out = tf.nn.sigmoid(tf.matmul(h2, D_W2) + D_b2)
    return D1_out, D2_out


theta_G = [G1_W2, G2_W2, G1_b2, G2_b2]
theta_G_shared = [G_W1, G_b1]

theta_D = [D1_W1, D2_W1, D1_b1, D2_b1]
theta_D_shared = [D_W2, D_b2]

# Train D
G1_sample, G2_sample = G(z)
D1_real, D2_real = D(X1, X2)
D1_fake, D2_fake = D(G1_sample, G2_sample)

D1_loss = -tf.reduce_mean(tf.log(D1_real + eps) + tf.log(1. - D1_fake + eps))
D2_loss = -tf.reduce_mean(tf.log(D2_real + eps) + tf.log(1. - D2_fake + eps))
D_loss = D1_loss + D2_loss

# Train G
G1_loss = -tf.reduce_mean(tf.log(D1_fake + eps))
G2_loss = -tf.reduce_mean(tf.log(D2_fake + eps))
G_loss = G1_loss + G2_loss

# D optimizer
D_opt = tf.train.AdamOptimizer(learning_rate=lr)
# Compute the gradients for a list of variables.
D_gv = D_opt.compute_gradients(D_loss, theta_D)
D_shared_gv = D_opt.compute_gradients(D_loss, theta_D_shared)
# Average by halfing the shared gradients
D_shared_gv = [(0.5 * x[0], x[1]) for x in D_shared_gv]
# Update
D_solver = tf.group(
    D_opt.apply_gradients(D_gv), D_opt.apply_gradients(D_shared_gv)
)

# G optimizer
G_opt = tf.train.AdamOptimizer(learning_rate=lr)
# Compute the gradients for a list of variables.
G_gv = G_opt.compute_gradients(G_loss, theta_G)
G_shared_gv = G_opt.compute_gradients(G_loss, theta_G_shared)
# Average by halfing the shared gradients
G_shared_gv = [(0.5 * x[0], x[1]) for x in G_shared_gv]
# Update
G_solver = tf.group(
    G_opt.apply_gradients(G_gv), G_opt.apply_gradients(G_shared_gv)
)

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
    X1_mb, X2_mb = sample_X(X_train1, mb_size), sample_X(X_train2, mb_size)
    z_mb = sample_z(mb_size, z_dim)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss],
        feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={z: z_mb}
    )

    if it % 1000 == 0:
        sample1, sample2 = sess.run(
            [G1_sample, G2_sample], feed_dict={z: sample_z(8, z_dim)}
        )

        samples = np.vstack([sample1, sample2])

        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
