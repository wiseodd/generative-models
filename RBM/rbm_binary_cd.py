import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


if not os.path.exists('out/'):
    os.makedirs('out/')

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]

mb_size = 16
h_dim = 36

W = np.random.randn(X_dim, h_dim) * 0.001
a = np.random.randn(h_dim) * 0.001
b = np.random.randn(X_dim) * 0.001


def sigm(x):
    return 1/(1 + np.exp(-x))


def infer(X):
    # mb_size x x_dim -> mb_size x h_dim
    return sigm(X @ W)


def generate(H):
    # mb_size x h_dim -> mb_size x x_dim
    return sigm(H @ W.T)


# Contrastive Divergence
# ----------------------
# Approximate the log partition gradient Gibbs sampling

alpha = 0.1
K = 10  # Num. of Gibbs sampling step

for t in range(1, 1001):
    X_mb = (mnist.train.next_batch(mb_size)[0] > 0.5).astype(np.float)
    g = 0
    g_a = 0
    g_b = 0

    for v in X_mb:
        # E[h|v,W]
        h = infer(v)

        # Gibbs sampling steps
        # --------------------
        v_prime = np.copy(v)

        for k in range(K):
            # h ~ p(h|v,W)
            h_prime = np.random.binomial(n=1, p=infer(v_prime))
            # v ~ p(v|h,W)
            v_prime = np.random.binomial(n=1, p=generate(h_prime))

        # E[h|v',W]
        h_prime = infer(v_prime)

        # Compute data gradient
        grad_w = np.outer(v, h) - np.outer(v_prime, h_prime)
        grad_a = h - h_prime
        grad_b = v - v_prime

        # Accumulate minibatch gradient
        g += grad_w
        g_a += grad_a
        g_b += grad_b

    # Monte carlo gradient
    g *= 1 / mb_size
    g_a *= 1 / mb_size
    g_b *= 1 / mb_size

    # Update to maximize
    W += alpha * g
    a += alpha * g_a
    b += alpha * g_b


# Visualization
# -------------

def plot(samples, size, name):
    size = int(size)
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(size, size), cmap='Greys_r')

    plt.savefig('out/{}.png'.format(name), bbox_inches='tight')
    plt.close(fig)


X = (mnist.test.next_batch(mb_size)[0] > 0.5).astype(np.float)

H = np.random.binomial(n=1, p=infer(X))
plot(H, np.sqrt(h_dim), 'H')

X_recon = (generate(H) > 0.5).astype(np.float)
plot(X_recon, np.sqrt(X_dim), 'V')
