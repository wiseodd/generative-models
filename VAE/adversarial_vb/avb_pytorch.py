import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
eps_dim = 4
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


def log(x):
    return torch.log(x + 1e-8)


# Encoder: q(z|x,eps)
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim + eps_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)

# Decoder: p(x|z)
P = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

# Discriminator: T(X, z)
T = torch.nn.Sequential(
    torch.nn.Linear(X_dim + z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1)
)


def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    T.zero_grad()


def sample_X(size, include_y=False):
    X, y = mnist.train.next_batch(size)
    X = Variable(torch.from_numpy(X))

    if include_y:
        y = np.argmax(y, axis=1).astype(np.int)
        y = Variable(torch.from_numpy(y))
        return X, y

    return X


Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
T_solver = optim.Adam(T.parameters(), lr=lr)


for it in range(1000000):
    X = sample_X(mb_size)
    eps = Variable(torch.randn(mb_size, eps_dim))
    z = Variable(torch.randn(mb_size, z_dim))

    # Optimize VAE
    z_sample = Q(torch.cat([X, eps], 1))
    X_sample = P(z_sample)
    T_sample = T(torch.cat([X, z_sample], 1))

    disc = torch.mean(-T_sample)
    loglike = -nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size

    elbo = -(disc + loglike)

    elbo.backward()
    Q_solver.step()
    P_solver.step()
    reset_grad()

    # Discriminator T(X, z)
    z_sample = Q(torch.cat([X, eps], 1))
    T_q = nn.sigmoid(T(torch.cat([X, z_sample], 1)))
    T_prior = nn.sigmoid(T(torch.cat([X, z], 1)))

    T_loss = -torch.mean(log(T_q) + log(1. - T_prior))

    T_loss.backward()
    T_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; ELBO: {:.4}; T_loss: {:.4}'
              .format(it, -elbo.data[0], -T_loss.data[0]))

        samples = P(z).data.numpy()[:16]

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

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
