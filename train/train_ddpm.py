import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ddpm.model import DDPM_eps

CHECKPOINT_DIR = "checkpoints/ddpm"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(CHECKPOINT_DIR, "ddpm_mnist_best.pth")
LOSS_NPY_PATH = os.path.join(CHECKPOINT_DIR, "losses.npy")
LOSS_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "training_loss.png")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        ),
        batch_size=128,
        shuffle=True
    )

    model = DDPM_eps().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    losses = []

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(1, model.T, (x.size(0),), device=device)

            eps_pred, noise = model(x, t)
            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "T": model.T,
                },
                CKPT_PATH
            )
            print("✔ Saved best checkpoint")

    # ---------- SAVE LOSSES ----------
    losses_np = np.array(losses)
    np.save(LOSS_NPY_PATH, losses_np)

    # ---------- SAVE PLOT ----------
    plt.figure()
    plt.plot(losses_np)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for DDPM (ε-prediction)")
    plt.grid(True)
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    print("✔ Saved losses.npy and training_loss.png")


if __name__ == "__main__":
    train()
