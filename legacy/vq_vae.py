import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    
    def forward(self, z):
        
        # z: [B, D, H, W] --> [BHW, D]
        z_perm = z.permute(0,2,3,1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # distance to embeddings
        distances = (flat_z.pow(2).sum(1, keepdim=True)
                     - 2 * flat_z @ self.embeddings.weight.t()
                     + self.embeddings.weight.pow(2).sum(1))

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z_perm.shape)
        quantized = quantized.permute(0,3,1,2).contiguous()

        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss


# VQ-VAE model

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embedding_dim, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        x_hat = self.decoder(quantized)
        return x_hat, vq_loss


# training setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='../datasets', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

loss_curve = []
epochs = 15

for epoch in range(epochs):
    model.train()
    total_Loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        x_hat, vq_loss = model(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_curve.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# reconstruction

model.eval()
with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        x_hat, _ = model(x)
        x = x.cpu().numpy()
        x_hat = x_hat.cpu().numpy()
        break

plt.figure(figsize=(8,4))
for i in range(8):
    plt.subplot(2, 8, i+1)
    plt.imshow(x[i,0], cmap='gray')
    plt.axis('off')
    plt.subplot(2, 8, i+9)
    plt.imshow(x_hat[i,0], cmap='gray')
    plt.axis('off')

plt.suptitle('Top: Original Images, Bottom: Reconstructed Images')
plt.tight_layout()
plt.show()


# fit a GMM on latents

all_latents = []
model.eval()
with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        z_e = model.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous().view(-1, model.quantizer.embedding_dim)
        all_latents.append(z_e.cpu().numpy())

all_latents = torch.cat(all_latents, dim=0).numpy()

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=0)
gmm.fit(all_latents)

samples = gmm.sample(n_samples=16 * 7 * 7)[0]
samples = torch.tensor(samples, dtype=torch.float32).view(16, 7, 7, -1).to(device)

# flatten and compute nearest codebook vectors
flat = samples.view(-1, samples.shape[-1])
dist = (flat.pow(2).sum(1, keepdim=True)
        - 2 * flat @ model.quantizer.embeddings.weight.t()
        + model.quantizer.embeddings.weight.pow(2).sum(1))

encoding_indices = torch.argmin(dist, dim=1)

# get quantized vectors
quantized = model.quantizer.embeddings(encoding_indices)
quantized = quantized.view(16, 7, 7, -1).permute(0, 3, 1, 2).contiguous()

with torch.no_grad():
    generated = model.decoder(quantized)

# plot the results
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated[i][0].cpu(), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()