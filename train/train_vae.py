import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

from models.vae.model import BetaVAE, beta_vae_loss



# DATA
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BetaVAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# CHECKPOINTING
PROJECT_ROOT = Path.cwd()          # GenerativeModelling
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "vae"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "beta_vae.pth"

best_loss = float("inf")


# TRAINING
num_epochs = 20
beta = 1.8

losses, recons, kls = [], [], []

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resumed from epoch {ckpt['epoch']} with loss {ckpt['loss']:.2f}")


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for x, _ in train_loader:
        x = x.to(device)

        x_hat, mu, logvar = model(x)
        loss, recon, kl = beta_vae_loss(x, x_hat, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    losses.append(total_loss / len(train_loader.dataset))
    recons.append(total_recon / len(train_loader.dataset))
    kls.append(total_kl / len(train_loader.dataset))

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {losses[-1]:.2f} | "
        f"Recon: {recons[-1]:.2f} | "
        f"KL: {kls[-1]:.2f}"
    )

    # SAVE CHECKPOINT (best model)
    if losses[-1] < best_loss:
        best_loss = losses[-1]

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "latent_dim": model.latent_dim,
                "beta": beta,
            },
            checkpoint_path
        )
        print(f"✓ Checkpoint saved at epoch {epoch+1}")




# PLOTS

plot_dir = Path("checkpoints/vae/")
plot_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(losses, label="Total Loss")
plt.plot(recons, label="Reconstruction Loss")
plt.plot(kls, label="KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Beta-VAE Training Loss")

plt.savefig(plot_dir / "training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

np.save(plot_dir / "losses.npy", losses)

# SAMPLING
def generate_images(model, num_images=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, model.latent_dim).to(device)
        samples = model.decode(z)
    return samples.cpu()


samples = generate_images(model)

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
    plt.axis("off")

plt.suptitle("Generated digits from Beta-VAE")
plt.tight_layout()
plt.show()
