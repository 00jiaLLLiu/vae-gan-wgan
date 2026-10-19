# VAE, GAN & WGAN on MNIST and CIFAR-10

Generative models from scratch: Variational Autoencoders, GANs, and Wasserstein GANs (CSI 5340 Deep Learning, Homework 4).

## Background

Three ways to learn to *generate* images: **VAE** (encoder → latent distribution → decoder, trained on reconstruction + KL divergence), **GAN** (generator vs discriminator minimax game), and **WGAN** (Wasserstein/EMD distance with weight clipping for stabler training). This project implements all three on MNIST and CIFAR-10, varying network depth and latent dimension to see what actually matters. (Experiments were run on Colab with GPU/time limits.)

## The Assignment

- Implement VAE, GAN, and WGAN for MNIST and CIFAR-10.
- For each model/dataset, compare architectures: different numbers of layers and different latent dimensions.
- Track training/test loss curves (including KL and reconstruction terms for VAE, EMD for WGAN) and exhibit generated images.

## Implementation

- `VAE/mnist/vae_mnist.py`, `VAE/cifar/vae_cifar.py` — VAE (Adam, BCE reconstruction + KL loss)
- `GAN/mnist/GAN_mnist.py`, `GAN/cifar/GAN_cifar.py` — vanilla GAN
- `WGAN/mnist/WGAN_mnist*.py`, `WGAN/cifar/wgan_cifar.py` — WGAN with weight clipping (0.01)
- Every configuration logs losses to CSV and saves loss curves, KL/reconstruction splits, epoch-progression image grids, and final generated samples — about 130 files of code, logs, plots, and generated images across 3 model families × 2 datasets; `compare/plotVS.py` scripts regenerate the head-to-head figures.

## Key Findings

- **VAE/MNIST: latent dimension matters more than depth.** Shrinking latent dim 64→3 raises training loss substantially, while 4-layer vs 2-layer encoders barely differ. A too-small bottleneck simply can't carry the digit information — depth can't compensate for lost capacity. (KL loss behaves oppositely: it collapses toward 0 as training settles.)
- **VAE/CIFAR: depth can hurt.** Going 2→3 layers *raised* training loss throughout, while 256→512 latent dim changed almost nothing. On harder data, a bigger model just gives the optimizer more ways to get stuck.
- **WGAN/CIFAR: the deepest model was the least stable.** ly4_dim64 and ly4_dim128 show nearly identical EMD decline (0.1→~0), but the 5-layer model's EMD fell then climbed back up — under limited training budget, extra depth bought instability, not quality.
- **Across all three model families, the pattern is the same**: capacity knobs (depth, latent dim) have diminishing or negative returns past a point; the loss curves, not the parameter counts, tell you when to stop adding.

## How to Run

```bash
pip install -r requirements.txt
cd VAE/mnist && python vae_mnist.py
# MNIST/CIFAR-10 download automatically on first run
```

## Repository Structure

```
├── VAE/            # vae_mnist.py, vae_cifar.py + per-config results
│   ├── mnist/      # ly2_dim64, ly4_dim64 (+ epoch progressions, reconstructions)
│   └── cifar/      # ly2_dim256, ly2_dim512, ly3_dim512 + compare/
├── GAN/            # GAN_mnist.py, GAN_cifar.py + per-config results
│   ├── mnist/
│   └── cifar/
├── WGAN/           # WGAN_*.py (weight clipping 0.01) + per-config results
│   ├── mnist/
│   └── cifar/
└── report/
    └── Assignment4.pdf
```
