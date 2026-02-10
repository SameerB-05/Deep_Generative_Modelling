# reconstruction of input image = posterior sampling = z ~ q(z|x)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.vae.model import BetaVAE


# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 20
checkpoint_path = "checkpoints/beta_vae.pth"


# DATA
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="../dataset",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


# LOAD MODEL
model = BetaVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


# RECONSTRUCTION
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x.to(device)

    x_hat, _, _ = model(x)


# PLOT
plt.figure(figsize=(8, 4))

for i in range(8):
    # Original
    plt.subplot(2, 8, i + 1)
    plt.imshow(x[i][0].cpu(), cmap="gray")
    plt.axis("off")

    # Reconstruction
    plt.subplot(2, 8, i + 9)
    plt.imshow(x_hat[i][0].cpu(), cmap="gray")
    plt.axis("off")

plt.suptitle("Top: Original | Bottom: Reconstruction")
plt.tight_layout()
plt.show()
