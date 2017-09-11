import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


if not os.path.exists('out/'):
    os.makedirs('out/')

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]

mb_size = 16
h_dim = 36

W = np.random.randn(X_dim, h_dim) * 0.001


def sigm(x):
    return 1/(1 + np.exp(-x))


def infer(X):
    # mb_size x x_dim -> mb_size x h_dim
    return sigm(X @ W)


def generate(H):
    # mb_size x h_dim -> mb_size x x_dim
    return sigm(H @ W.T)


# Persistent Contrastive Divergence
# ---------------------------------
# Approximate the log partition gradient using single step Gibbs sampling

alpha = 0.1
K = 10  # Num. of MC iteration

# Initialize the markov chain
V_s = sigm(np.random.randn(mb_size, X_dim))
H_s = np.random.binomial(n=1, p=0.5, size=[mb_size, h_dim])

for t in range(1, 101):
    X_mb = (mnist.train.next_batch(mb_size)[0] > 0.5).astype(np.float)
    g = 0

    Mu = infer(X_mb)

    # Gibbs sampling step
    # -------------------
    for i, v_s in enumerate(V_s):
        for k in range(K):
            # h ~ p(h|v,W)
            h_prime = np.random.binomial(n=1, p=infer(v_s))
            # v ~ p(v|h,W)
            v_prime = np.random.binomial(n=1, p=generate(h_prime))

        # Replace with new sample
        V_s[i] = v_prime
        H_s[i] = h_prime

    # Compute gradient
    left = np.array([np.outer(X_mb[i], Mu[i]) for i in range(mb_size)])
    right = np.array([np.outer(V_s[i], H_s[i]) for i in range(mb_size)])
    g = np.mean(left, axis=0) - np.mean(right, axis=0)

    # Update
    W += (alpha/t) * g


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

X_recon = np.random.binomial(n=1, p=generate(H))
plot(X_recon, np.sqrt(X_dim), 'V')
