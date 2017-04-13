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
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
d_step = 3
lr = 5e-4
m = 5
n_iter = 1000
n_epoch = 1000
N = n_iter * mb_size  # N data per epoch


G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

# D is an autoencoder
D_ = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
)


# Energy is the MSE of autoencoder
def D(X):
    X_recon = D_(X)
    return torch.sum((X - X_recon)**2, 1)


def reset_grad():
    G.zero_grad()
    D_.zero_grad()


G_solver = optim.Adamax(G.parameters(), lr=lr)
D_solver = optim.Adamax(D_.parameters(), lr=lr)


# Pretrain discriminator
for it in range(2*n_iter):
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    loss = torch.mean(D(X))  # Minimize real samples energy

    loss.backward()
    D_solver.step()
    reset_grad()

    if it % 1000 == 0:
        print('Iter-{}; Pretrained D loss: {:.4}'.format(it, loss.data[0]))


# Initial margin, expected energy of real data
m = torch.mean(D(Variable(torch.from_numpy(mnist.train.images)))).data[0]
s_z_before = torch.from_numpy(np.array([np.inf], dtype='float32'))


# GAN training
for t in range(n_epoch):
    s_x, s_z = torch.zeros(1), torch.zeros(1)

    for it in range(n_iter):
        # Sample data
        z = Variable(torch.randn(mb_size, z_dim))
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        # Dicriminator
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss = torch.mean(D_real) + nn.relu(m - torch.mean(D_fake))

        D_loss.backward()
        D_solver.step()

        # Update real samples statistics
        s_x += torch.sum(D_real.data)

        reset_grad()

        # Generator
        z = Variable(torch.randn(mb_size, z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = torch.mean(D_fake)

        G_loss.backward()
        G_solver.step()

        # Update fake samples statistics
        s_z += torch.sum(D_fake.data)

        reset_grad()

    # Update margin
    if (((s_x[0] / N) < m) and (s_x[0] < s_z[0]) and (s_z_before[0] < s_z[0])):
        m = s_x[0] / N

    s_z_before = s_z

    # Convergence measure
    Ex = s_x[0] / N
    Ez = s_z[0] / N
    L = Ex + np.abs(Ex - Ez)

    # Visualize
    print('Epoch-{}; m = {:.4}; L = {:.4}'
          .format(t, m, L))

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
