import torch
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
z_dim = 16
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3
eps = 1e-8


G_ = torch.nn.Sequential(
    torch.nn.Linear(z_dim + y_dim, h_dim),
    torch.nn.PReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)


def G(z, c):
    inputs = torch.cat([z, c], 1)
    return G_(inputs)


D_shared = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.PReLU()
)

D_gan = torch.nn.Sequential(
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

D_aux = torch.nn.Sequential(
    torch.nn.Linear(h_dim, y_dim),
)


def D(X):
    h = D_shared(X)
    return D_gan(h), D_aux(h)


nets = [G_, D_shared, D_gan, D_aux]

G_params = G_.parameters()
D_params = (list(D_shared.parameters()) + list(D_gan.parameters()) +
            list(D_aux.parameters()))


def reset_grad():
    for net in nets:
        net.zero_grad()


G_solver = optim.Adam(G_params, lr=lr)
D_solver = optim.Adam(D_params, lr=lr)


for it in range(100000):
    # Sample data
    X, y = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    # c is one-hot
    c = Variable(torch.from_numpy(y.astype('float32')))
    # y_true is not one-hot (requirement from nn.cross_entropy)
    y_true = Variable(torch.from_numpy(y.argmax(axis=1).astype('int')))
    # z noise
    z = Variable(torch.randn(mb_size, z_dim))

    """ Discriminator """
    G_sample = G(z, c)
    D_real, C_real = D(X)
    D_fake, C_fake = D(G_sample)

    # GAN's D loss
    D_loss = torch.mean(torch.log(D_real + eps) + torch.log(1 - D_fake + eps))
    # Cross entropy aux loss
    C_loss = -nn.cross_entropy(C_real, y_true) - nn.cross_entropy(C_fake, y_true)

    # Maximize
    DC_loss = -(D_loss + C_loss)

    DC_loss.backward()
    D_solver.step()

    reset_grad()

    """ Generator """
    G_sample = G(z, c)
    D_fake, C_fake = D(G_sample)
    _, C_real = D(X)

    # GAN's G loss
    G_loss = torch.mean(torch.log(D_fake + eps))
    # Cross entropy aux loss
    C_loss = -nn.cross_entropy(C_real, y_true) - nn.cross_entropy(C_fake, y_true)

    # Maximize
    GC_loss = -(G_loss + C_loss)

    GC_loss.backward()
    G_solver.step()

    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([16, y_dim])
        c[range(16), idx] = 1
        c = Variable(torch.from_numpy(c.astype('float32')))

        z = Variable(torch.randn(16, z_dim))

        samples = G(z, c).data.numpy()

        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; Idx: {}'
              .format(it, -D_loss.data[0], -G_loss.data[0], idx))

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
