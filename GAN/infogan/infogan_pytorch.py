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
Z_dim = 16
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + 10, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def G(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Why = xavier_init(size=[h_dim, 1])
bhy = Variable(torch.zeros(1), requires_grad=True)


def D(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    return y


""" ====================== Q(c|X) ========================== """

Wqxh = xavier_init(size=[X_dim, h_dim])
bqxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whc = xavier_init(size=[h_dim, 10])
bhc = Variable(torch.zeros(10), requires_grad=True)


def Q(X):
    h = nn.relu(X @ Wqxh + bqxh.repeat(X.size(0), 1))
    c = nn.softmax(h @ Whc + bhc.repeat(h.size(0), 1))
    return c


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
Q_params = [Wqxh, bqxh, Whc, bhc]
params = G_params + D_params + Q_params


""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=1e-3)
D_solver = optim.Adam(D_params, lr=1e-3)
Q_solver = optim.Adam(G_params + Q_params, lr=1e-3)


def sample_c(size):
    c = np.random.multinomial(1, 10*[0.1], size=size)
    c = Variable(torch.from_numpy(c.astype('float32')))
    return c


for it in range(100000):
    # Sample data
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    z = Variable(torch.randn(mb_size, Z_dim))
    c = sample_c(mb_size)

    # Dicriminator forward-loss-backward-update
    G_sample = G(z, c)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss = -torch.mean(torch.log(D_real + 1e-8) + torch.log(1 - D_fake + 1e-8))

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Generator forward-loss-backward-update
    G_sample = G(z, c)
    D_fake = D(G_sample)

    G_loss = -torch.mean(torch.log(D_fake + 1e-8))

    G_loss.backward()
    G_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Q forward-loss-backward-update
    G_sample = G(z, c)
    Q_c_given_x = Q(G_sample)

    crossent_loss = torch.mean(-torch.sum(c * torch.log(Q_c_given_x + 1e-8), dim=1))
    mi_loss = crossent_loss

    mi_loss.backward()
    Q_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([mb_size, 10])
        c[range(mb_size), idx] = 1
        c = Variable(torch.from_numpy(c.astype('float32')))
        samples = G(z, c).data.numpy()[:16]

        print('Iter-{}; D_loss: {}; G_loss: {}; Idx: {}'
              .format(it, D_loss.data.numpy(), G_loss.data.numpy(), idx))

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
