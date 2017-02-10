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
z_dim = 5
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


# Encoder
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim)
)

# Decoder
P = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

# Discriminator
D = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)


def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()


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
D_solver = optim.Adam(D.parameters(), lr=lr)


for it in range(1000000):
    X = sample_X(mb_size)

    """ Reconstruction phase """
    z_sample = Q(X)
    X_sample = P(z_sample)

    recon_loss = nn.binary_cross_entropy(X_sample, X)

    recon_loss.backward()
    P_solver.step()
    Q_solver.step()
    reset_grad()

    """ Regularization phase """
    # Discriminator
    z_real = Variable(torch.randn(mb_size, z_dim))
    z_fake = Q(X)

    D_real = D(z_real)
    D_fake = D(z_fake)

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator
    z_fake = Q(X)
    D_fake = D(z_fake)

    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    Q_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))

        samples = P(z_real).data.numpy()[:16]

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
