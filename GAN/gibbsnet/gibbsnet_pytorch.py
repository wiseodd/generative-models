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
from itertools import *


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 16
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 256
cnt = 0
lr = 1e-4
N = 10


def log(x):
    return torch.log(x + 1e-8)


# Inference net (Encoder) Q(z|X)
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)

# Generator net (Decoder) P(X|z)
P = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

D_ = torch.nn.Sequential(
    torch.nn.Linear(X_dim + z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)


def D(X, z):
    return D_(torch.cat([X, z], 1))


def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D_.zero_grad()


G_solver = optim.Adam(chain(Q.parameters(), P.parameters()), lr=lr)
D_solver = optim.Adam(D_.parameters(), lr=lr)


for it in range(1000000):
    # Sample data
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    # Discriminator
    z_hat = Q(X)

    # Do N step Gibbs sampling
    z = Variable(torch.randn(mb_size, z_dim))

    for _ in range(N):
        z_n = z.clone()
        X_hat = P(z_n)
        z = Q(X_hat)

    p_data = D(X, z_hat)
    p_model = D(X_hat, z_n)

    D_loss = -torch.mean(log(p_data) + log(1 - p_model))

    D_loss.backward(retain_graph=True)
    D_solver.step()
    G_solver.step()
    reset_grad()

    G_loss = -torch.mean(log(p_model) + log(1 - p_data))

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 100 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        z = Variable(torch.randn(mb_size, z_dim))

        for _ in range(N):
            z_n = z.clone()
            X_hat = P(z_n)
            z = Q(X_hat)

        samples = X_hat.data.numpy()[:16]

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

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
