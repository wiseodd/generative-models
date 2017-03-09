# Generative Models
Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow.

**Note:** generated samples will be stored in `GAN/{gan_model}/out` or `VAE/{vae_model}/out` directory during training.

## What's in it?

1. Generative Adversarial Nets (GAN)
  1. [Vanilla GAN](https://arxiv.org/abs/1406.2661)
  2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
  3. [InfoGAN](https://arxiv.org/abs/1606.03657)
  4. [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
  5. [Mode Regularized GAN](https://arxiv.org/abs/1612.02136)
  6. [Coupled GAN](https://arxiv.org/abs/1606.07536)
  7. [Auxiliary Classifier GAN](https://arxiv.org/abs/1610.09585)
  8. [Least Squares GAN](https://arxiv.org/abs/1611.04076v2)
  9. [Boundary Seeking GAN](https://arxiv.org/abs/1702.08431)
2. Variational Autoencoder (VAE)
  1. [Vanilla VAE](https://arxiv.org/abs/1312.6114)
  2. [Conditional VAE](https://arxiv.org/abs/1406.5298)
  3. [Denoising VAE](https://arxiv.org/abs/1511.06406)
  4. [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644)
  5. [Adversarial Variational Bayes](https://arxiv.org/abs/1701.04722)

## Dependencies

1. Install miniconda <http://conda.pydata.org/miniconda.html>
2. Do `conda env create`
3. Enter the env `source activate generative-models`
4. Install [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
5. Install [Pytorch](https://github.com/pytorch/pytorch#installation)
