# Deep Generative Modelling

This repository contains learning-oriented PyTorch implementations of basic deep generative models. The code was written as part of self-study to understand the fundamentals of variational autoencoders and diffusion-based generative models.

## Implemented Models

### Variational Autoencoder (VAE)
- Encoder–decoder architecture
- Reparameterization trick
- Reconstruction loss with KL divergence regularization

### Denoising Diffusion Probabilistic Model (DDPM)
- Forward noising and reverse denoising processes
- Noise prediction objective using MSE loss
- Sampling via the standard DDPM reverse process

### Conditional DDPM with Classifier-Free Guidance (DDPM-CFG)
- Class-conditional diffusion model
- Classifier-free guidance implemented via random label dropping during training
- Adjustable guidance scale during sampling

## Repository Structure

Deep_Generative_Modelling/
├── models/
├── train/
├── sample/
├── checkpoints/
├── data/
├── requirements.txt
└── README.md

## Purpose

- To understand generative modeling concepts by implementing them from scratch
- To experiment with VAEs and diffusion models in PyTorch
- To serve as a base for further learning and extensions

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- matplotlib

Install dependencies using:
pip install -r requirements.txt

## Usage

Train a model using:
python train/train_ddpm.py

Sample from a trained model using:
python sample/sample_ddpm.py

Trained model checkpoints are expected in the checkpoints/ directory.
