import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3
K = 100


def log(x):
    return torch.log(x + 1e-8)


G1_ = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)


D1_ = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

G2_ = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)


D2_ = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

nets = [G1_, D1_, G2_, D2_]


def reset_grad():
    for net in nets:
        net.zero_grad()


G1_solver = optim.Adam(G1_.parameters(), lr=lr)
D1_solver = optim.Adam(D1_.parameters(), lr=lr)
G2_solver = optim.Adam(G2_.parameters(), lr=lr)
D2_solver = optim.Adam(D2_.parameters(), lr=lr)

D1 = {'model': D1_, 'solver': D1_solver}
G1 = {'model': G1_, 'solver': G1_solver}
D2 = {'model': D2_, 'solver': D2_solver}
G2 = {'model': G2_, 'solver': G2_solver}

GAN_pairs = [(D1, G1), (D2, G2)]

for it in range(1000000):
    # Sample data
    z = Variable(torch.randn(mb_size, z_dim))
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    for D, G in GAN_pairs:
        # Discriminator
        G_sample = G['model'](z)
        D_real = D['model'](X)
        D_fake = D['model'](G_sample)

        D_loss = -torch.mean(log(D_real) + log(1 - D_fake))

        D_loss.backward()
        D['solver'].step()
        reset_grad()

        # Generator
        G_sample = G['model'](z)
        D_fake = D['model'](G_sample)

        G_loss = -torch.mean(log(D_fake))

        G_loss.backward()
        G['solver'].step()
        reset_grad()

    if it != 0 and it % K == 0:
        # Swap (D, G) pairs
        new_D1, new_D2 = GAN_pairs[1][0], GAN_pairs[0][0]
        GAN_pairs = [(new_D1, G1), (new_D2, G2)]

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        # Pick G randomly
        G_rand = random.choice([G1_, G2_])
        samples = G_rand(z).data.numpy()[:16]

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
