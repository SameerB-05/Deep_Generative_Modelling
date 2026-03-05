import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from models.gan.model import Generator, Discriminator


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr_G = 0.0002
lr_D = 0.0001
z_dim = 100
num_epochs = 50
img_size = 28
img_channels = 1
checkpoint_dir = "checkpoints/gan"
os.makedirs(checkpoint_dir, exist_ok=True)


# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Models
generator = Generator(
    z_dim=z_dim,
    img_channels=img_channels,
    img_size=img_size
).to(device)

discriminator = Discriminator(
    img_channels=img_channels,
    img_size=img_size
).to(device)


# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))


g_losses = []
d_losses = []
best_g_loss = float("inf")

for epoch in range(num_epochs):

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    num_batches = 0

    for i, (imgs, _) in enumerate(train_loader):
        batch_size_curr = imgs.size(0)
        real_imgs = imgs.to(device)

        real_labels = torch.full((batch_size_curr, 1), 0.9, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()

        outputs_real = discriminator(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size_curr, z_dim, 1, 1, device=device)
        fake_imgs = generator(z)

        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        outputs_fake_for_G = discriminator(fake_imgs)
        g_loss = criterion(outputs_fake_for_G, real_labels)

        g_loss.backward()
        optimizer_G.step()

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        num_batches += 1

        if i % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Compute average losses
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches

    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

    # Save sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        utils.save_image(
            fake_imgs.data[:25],
            f"{checkpoint_dir}/epoch_{epoch+1}.png",
            nrow=5,
            normalize=True
        )

    # Save best model based on avg generator loss
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss

        torch.save({
            "epoch": epoch + 1,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "avg_g_loss": avg_g_loss,
            "avg_d_loss": avg_d_loss
        }, os.path.join(checkpoint_dir, "gan_mnist_best.pth"))

        print(f"Best model saved at epoch {epoch+1} "
              f"with Avg G Loss: {best_g_loss:.4f}")


# Save losses
np.save(
    os.path.join(checkpoint_dir, "losses.npy"),
    np.array([g_losses, d_losses])
)


# Plot training loss
plt.figure()
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Loss")
plt.savefig(os.path.join(checkpoint_dir, "training_loss.png"))
plt.close()


print("Training complete. Best model, losses, and plots saved.")