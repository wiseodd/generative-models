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
lr = 1e-3
m = 5
lam = 1e-3
k = 0
gamma = 0.5


G = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

D_ = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
)


# D is an autoencoder, approximating Gaussian
def D(X):
    X_recon = D_(X)
    # Use Laplace MLE as in the paper
    return torch.mean(torch.sum(torch.abs(X - X_recon), 1))


def reset_grad():
    G.zero_grad()
    D_.zero_grad()


G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D_.parameters(), lr=lr)


for it in range(1000000):
    # Sample data
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    # Dicriminator
    z_D = Variable(torch.randn(mb_size, z_dim))

    D_loss = D(X) - k * D(G(z_D))

    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator
    z_G = Variable(torch.randn(mb_size, z_dim))

    G_loss = D(G(z_G))

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Update k, the equlibrium
    k = k + lam * (gamma*D(X) - D(G(z_G)))
    k = k.item()  # k is variable, so unvariable it so that no gradient prop.

    # Print and plot every now and then
    if it % 1000 == 0:
        measure = D(X) + torch.abs(gamma*D(X) - D(G(z_G)))

        print('Iter-{}; Convergence measure: {:.4}'
              .format(it, measure.item()))

        samples = G(z_G).data.numpy()[:16]

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
