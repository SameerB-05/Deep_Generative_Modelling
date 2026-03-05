import torch
from torchvision import utils
from models.gan.model import Generator
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints/gan"
os.makedirs(checkpoint_dir, exist_ok=True)

# Hyperparameters (must match training)
z_dim = 100
img_size = 28
img_channels = 1
num_samples = 25

# Load Generator
generator = Generator(
    z_dim=z_dim,
    img_channels=img_channels,
    img_size=img_size
).to(device)

checkpoint = torch.load(
    f"{checkpoint_dir}/gan_mnist_best.pth",
    map_location=device,
    weights_only=False
)

generator.load_state_dict(checkpoint["generator"])
generator.eval()

# Generate Samples
with torch.no_grad():
    # DCGAN requires 4D noise
    z = torch.randn(num_samples, z_dim, 1, 1, device=device)
    fake_imgs = generator(z)

grid_img = utils.make_grid(fake_imgs, nrow=5, normalize=True)

plt.imshow(grid_img.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()