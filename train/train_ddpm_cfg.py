import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ddpm_cfg.model import DDPM_CFG, NUM_CLASSES

CKPT_DIR = "checkpoints/ddpm_cfg"
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_PATH = f"{CKPT_DIR}/ddpm_cfg_mnist_best.pth"
LOSS_NPY = f"{CKPT_DIR}/cfg_losses.npy"
LOSS_PNG = f"{CKPT_DIR}/cfg_training_loss.png"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        datasets.MNIST(
            "data", train=True, download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        ),
        batch_size=128, shuffle=True
    )

    model = DDPM_CFG().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses, best = [], float("inf")

    for epoch in range(10):
        total = 0.0
        model.train()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.randint(1, model.T, (x.size(0),), device=device)

            eps, noise = model(x, t, y)
            loss = F.mse_loss(eps, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()

        avg = total / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

        if avg < best:
            best = avg
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": best,
                    "model": model.state_dict(),
                },
                CKPT_PATH
            )
            print("✔ Saved best CFG checkpoint")

    np.save(LOSS_NPY, np.array(losses))

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CFG DDPM Training Loss")
    plt.grid(True)
    plt.savefig(LOSS_PNG)
    plt.close()


if __name__ == "__main__":
    train()
