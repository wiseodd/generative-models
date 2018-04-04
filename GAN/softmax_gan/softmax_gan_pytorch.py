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
mb_size = 16
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


def log(x):
    return torch.log(x + 1e-8)


G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)


D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1)
)


def reset_grad():
    G.zero_grad()
    D.zero_grad()


G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


D_target = 1./mb_size
G_target = 1./(mb_size*2)

for it in range(1000000):
    # Sample data
    z = Variable(torch.randn(mb_size, z_dim))
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    # Partition func.
    Z = torch.sum(torch.exp(-D_real)) + torch.sum(torch.exp(-D_fake))

    # Dicriminator
    D_loss = torch.sum(D_target * D_real) + log(Z)

    D_loss.backward(retain_graph=True)
    D_solver.step()
    reset_grad()

    # Generator
    G_loss = torch.sum(G_target * D_real) + torch.sum(G_target * D_fake) + log(Z)

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        samples = G(z).data.numpy()[:16]

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
