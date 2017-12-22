# Generative Models
Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow.
Also present here are RBM and Helmholtz Machine.

## Note:
Generated samples will be stored in `GAN/{gan_model}/out` (or `VAE/{vae_model}/out`, etc) directory during training.

## What's in it?

#### Generative Adversarial Nets (GAN)
  1. [Vanilla GAN](https://arxiv.org/abs/1406.2661)
  2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
  3. [InfoGAN](https://arxiv.org/abs/1606.03657)
  4. [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
  5. [Mode Regularized GAN](https://arxiv.org/abs/1612.02136)
  6. [Coupled GAN](https://arxiv.org/abs/1606.07536)
  7. [Auxiliary Classifier GAN](https://arxiv.org/abs/1610.09585)
  8. [Least Squares GAN](https://arxiv.org/abs/1611.04076v2)
  9. [Boundary Seeking GAN](https://arxiv.org/abs/1702.08431)
  10. [Energy Based GAN](https://arxiv.org/abs/1609.03126)
  11. [f-GAN](https://arxiv.org/abs/1606.00709)
  12. [Generative Adversarial Parallelization](https://arxiv.org/abs/1612.04021)
  13. [DiscoGAN](https://arxiv.org/abs/1703.05192)
  14. [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782) & [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)
  15. [Boundary Equilibrium GAN](https://arxiv.org/abs/1703.10717)
  16. [Improved Training for Wasserstein GAN](https://arxiv.org/abs/1704.00028)
  17. [DualGAN](https://arxiv.org/abs/1704.02510)
  18. [MAGAN: Margin Adaptation for GAN](https://arxiv.org/abs/1704.03817)
  19. [Softmax GAN](https://arxiv.org/abs/1704.06191)
  20. [GibbsNet](https://papers.nips.cc/paper/7094-gibbsnet-iterative-adversarial-inference-for-deep-graphical-models.pdf)

#### Variational Autoencoder (VAE)
  1. [Vanilla VAE](https://arxiv.org/abs/1312.6114)
  2. [Conditional VAE](https://arxiv.org/abs/1406.5298)
  3. [Denoising VAE](https://arxiv.org/abs/1511.06406)
  4. [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644)
  5. [Adversarial Variational Bayes](https://arxiv.org/abs/1701.04722)

#### Restricted Boltzmann Machine (RBM)
  1. [Binary RBM with Contrastive Divergence](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)
  2. [Binary RBM with Persistent Contrastive Divergence](http://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)

#### Helmholtz Machine
  1. [Binary Helmholtz Machine with Wake-Sleep Algorithm](http://www.cs.toronto.edu/~fritz/absps/ws.pdf)

## Dependencies

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate generative-models`
4. Install [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
5. Install [Pytorch](https://github.com/pytorch/pytorch#installation)
