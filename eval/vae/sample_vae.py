# sampling from prior = z ~ N(0, I)
# training would have forced q(z|x) to be close to N(0, I)

import torch
import matplotlib.pyplot as plt

from models.vae.model import BetaVAE


# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 20
num_samples = 25
checkpoint_path = "checkpoints/vae/beta_vae.pth"

# LOAD MODEL
model = BetaVAE(latent_dim=latent_dim).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

# SAFETY CHECK
assert latent_dim == model.latent_dim, \
    "Latent dim mismatch: checkpoint and model config differ"

model.eval()

# SAMPLING
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim).to(device)
    samples = model.decode(z).cpu()

# PLOT
plt.figure(figsize=(5, 5))
for i in range(num_samples):
    plt.subplot(5, 5, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
    plt.axis("off")

plt.suptitle("Samples from Beta-VAE")
plt.tight_layout()
plt.show()
