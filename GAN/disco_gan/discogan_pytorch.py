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
import scipy.ndimage.interpolation


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


def log(x):
    return torch.log(x + 1e-8)


def plot(samples):
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

    return fig


G_AB = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

G_BA = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim),
    torch.nn.Sigmoid()
)

D_A = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

D_B = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1),
    torch.nn.Sigmoid()
)

nets = [G_AB, G_BA, D_A, D_B]
G_params = list(G_AB.parameters()) + list(G_BA.parameters())
D_params = list(D_A.parameters()) + list(D_B.parameters())


def reset_grad():
    for net in nets:
        net.zero_grad()


G_solver = optim.Adam(G_params, lr=lr)
D_solver = optim.Adam(D_params, lr=lr)

if not os.path.exists('out/'):
    os.makedirs('out/')

# Gather training data: domain1 <- real MNIST img, domain2 <- rotated MNIST img
X_train = mnist.train.images
half = int(X_train.shape[0] / 2)
# Real image
X_train1 = X_train[:half]
# Rotated image
X_train2 = X_train[half:].reshape(-1, 28, 28)
X_train2 = scipy.ndimage.interpolation.rotate(X_train2, 90, axes=(1, 2))
X_train2 = X_train2.reshape(-1, 28*28)
# Cleanup
del X_train


def sample_x(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return Variable(torch.from_numpy(X[start_idx:start_idx+size]))


# Training
for it in range(1000000):
    # Sample data from both domains
    X_A = sample_x(X_train1, mb_size)
    X_B = sample_x(X_train2, mb_size)

    # Discriminator A
    X_BA = G_BA(X_B)
    D_A_real = D_A(X_A)
    D_A_fake = D_A(X_BA)

    L_D_A = -torch.mean(log(D_A_real) + log(1 - D_A_fake))

    # Discriminator B
    X_AB = G_AB(X_A)
    D_B_real = D_B(X_B)
    D_B_fake = D_B(X_AB)

    L_D_B = -torch.mean(log(D_B_real) + log(1 - D_B_fake))

    # Total discriminator loss
    D_loss = L_D_A + L_D_B

    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator AB
    X_AB = G_AB(X_A)
    D_B_fake = D_B(X_AB)
    X_ABA = G_BA(X_AB)

    L_adv_B = -torch.mean(log(D_B_fake))
    L_recon_A = torch.mean(torch.sum((X_A - X_ABA)**2, 1))
    L_G_AB = L_adv_B + L_recon_A

    # Generator BA
    X_BA = G_BA(X_B)
    D_A_fake = D_A(X_BA)
    X_BAB = G_AB(X_BA)

    L_adv_A = -torch.mean(log(D_A_fake))
    L_recon_B = torch.mean(torch.sum((X_B - X_BAB)**2, 1))
    L_G_BA = L_adv_A + L_recon_B

    # Total generator loss
    G_loss = L_G_AB + L_G_BA

    G_loss.backward()
    G_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0]))

        input_A = sample_x(X_train1, size=4)
        input_B = sample_x(X_train2, size=4)

        samples_A = G_BA(input_B).data.numpy()
        samples_B = G_AB(input_A).data.numpy()

        input_A = input_A.data.numpy()
        input_B = input_B.data.numpy()

        # The resulting image sample would be in 4 rows:
        # row 1: real data from domain A, row 2 is its domain B translation
        # row 3: real data from domain B, row 4 is its domain A translation
        samples = np.vstack([input_A, samples_B, input_B, samples_A])

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
