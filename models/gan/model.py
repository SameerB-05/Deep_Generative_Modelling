import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, img_size=28, feature_g=64):
        super().__init__()

        self.net = nn.Sequential(
            # Input: (N, z_dim, 1, 1)

            nn.ConvTranspose2d(z_dim, feature_g * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=28, feature_d=64):
        super().__init__()

        self.net = nn.Sequential(
            # Input: (N, 1, 28, 28)

            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, 1, 7, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)