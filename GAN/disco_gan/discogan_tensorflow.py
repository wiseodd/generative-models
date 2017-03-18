import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.ndimage.interpolation


mb_size = 32
X_dim = 784
z_dim = 64
h_dim = 128
lr = 1e-3
d_steps = 3

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


def log(x):
    return tf.log(x + 1e-8)


X_A = tf.placeholder(tf.float32, shape=[None, X_dim])
X_B = tf.placeholder(tf.float32, shape=[None, X_dim])

D_A_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_A_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_A_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_A_b2 = tf.Variable(tf.zeros(shape=[1]))

D_B_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_B_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
D_B_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_B_b2 = tf.Variable(tf.zeros(shape=[1]))

G_AB_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
G_AB_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_AB_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_AB_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

G_BA_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
G_BA_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
G_BA_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_BA_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_D = [D_A_W1, D_A_W2, D_A_b1, D_A_b2,
           D_B_W1, D_B_W2, D_B_b1, D_B_b2]
theta_G = [G_AB_W1, G_AB_W2, G_AB_b1, G_AB_b2,
           G_BA_W1, G_BA_W2, G_BA_b1, G_BA_b2]


def D_A(X):
    h = tf.nn.relu(tf.matmul(X, D_A_W1) + D_A_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_A_W2) + D_A_b2)


def D_B(X):
    h = tf.nn.relu(tf.matmul(X, D_B_W1) + D_B_b1)
    return tf.nn.sigmoid(tf.matmul(h, D_B_W2) + D_B_b2)


def G_AB(X):
    h = tf.nn.relu(tf.matmul(X, G_AB_W1) + G_AB_b1)
    return tf.nn.sigmoid(tf.matmul(h, G_AB_W2) + G_AB_b2)


def G_BA(X):
    h = tf.nn.relu(tf.matmul(X, G_BA_W1) + G_BA_b1)
    return tf.nn.sigmoid(tf.matmul(h, G_BA_W2) + G_BA_b2)


# Discriminator A
X_BA = G_BA(X_B)
D_A_real = D_A(X_A)
D_A_fake = D_A(X_BA)

# Discriminator B
X_AB = G_AB(X_A)
D_B_real = D_B(X_B)
D_B_fake = D_B(X_AB)

# Generator AB
X_ABA = G_BA(X_AB)

# Generator BA
X_BAB = G_AB(X_BA)

# Discriminator loss
L_D_A = -tf.reduce_mean(log(D_A_real) + log(1 - D_A_fake))
L_D_B = -tf.reduce_mean(log(D_B_real) + log(1 - D_B_fake))

D_loss = L_D_A + L_D_B

# Generator loss
L_adv_B = -tf.reduce_mean(log(D_B_fake))
L_recon_A = tf.reduce_mean(tf.reduce_sum((X_A - X_ABA)**2, 1))
L_G_AB = L_adv_B + L_recon_A

L_adv_A = -tf.reduce_mean(log(D_A_fake))
L_recon_B = tf.reduce_mean(tf.reduce_sum((X_B - X_BAB)**2, 1))
L_G_BA = L_adv_A + L_recon_B

G_loss = L_G_AB + L_G_BA

# Solvers
solver = tf.train.AdamOptimizer(learning_rate=lr)
D_solver = solver.minimize(D_loss, var_list=theta_D)
G_solver = solver.minimize(G_loss, var_list=theta_G)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Gather training data from 2 domains
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


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    # Sample data from both domains
    X_A_mb = sample_X(X_train1, mb_size)
    X_B_mb = sample_X(X_train2, mb_size)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )

    if it % 1000 == 0:
        print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        input_A = sample_X(X_train1, size=4)
        input_B = sample_X(X_train2, size=4)

        samples_A = sess.run(X_BA, feed_dict={X_B: input_B})
        samples_B = sess.run(X_AB, feed_dict={X_A: input_A})

        # The resulting image sample would be in 4 rows:
        # row 1: real data from domain A, row 2 is its domain B translation
        # row 3: real data from domain B, row 4 is its domain A translation
        samples = np.vstack([input_A, samples_B, input_B, samples_A])

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
