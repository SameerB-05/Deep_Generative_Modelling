# Deep Generative Modelling

This repository contains learning-oriented PyTorch implementations of basic deep generative models. The code was written as part of self-study to understand the fundamentals of variational autoencoders, generative adversarial networks, and diffusion-based generative models.


## Academic Reference

The implementations in this repository were developed after completing the lecture series:

**Mathematical Foundations of Generative AI - [YouTube Playlist](https://youtube.com/playlist?list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu)**  
Instructor: Prof. Prathosh A P  
Offered by IIT Madras  
Division of Electrical, Electronics, and Computer Science (EECS), IISc Bangalore  

The course provided the theoretical foundations for variational inference, adversarial learning, and diffusion-based generative modeling.

## Implemented Models

### Variational Autoencoder (VAE)
- Encoder–decoder architecture  
- Reparameterization trick  
- Reconstruction loss with KL divergence regularization  

### Generative Adversarial Network (GAN)
- Convolutional GAN architecture for MNIST  
- Separate generator and discriminator networks trained adversarially  
- Binary cross-entropy objective implemented using `BCEWithLogitsLoss`  
- Alternating optimization of generator and discriminator  
- Sampling via latent noise vectors  

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
│   ├── vae/
│   ├── gan/
│   └── diffusion/
├── train/
├── eval/
├── sample/
├── checkpoints/
├── data/
├── legacy/
├── requirements.txt
└── README.md

## Purpose

- To understand generative modeling concepts by implementing them from scratch  
- To experiment with VAEs, GANs, and diffusion models in PyTorch  
- To analyze training dynamics (loss behavior, adversarial balance, sampling quality)  
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

Train models using:

python train/train_vae.py  
python train/train_gan.py  
python train/train_ddpm.py  

Sample from trained models using:

python eval/gan/sample_gan.py  
python sample/sample_ddpm.py  

Trained model checkpoints are expected in the `checkpoints/` directory.