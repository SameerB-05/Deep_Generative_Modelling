import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES


# Diffusion buffers
def make_diffusion_buffers(T):
    beta = torch.linspace(1e-4, 0.01, T, dtype=torch.float32)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_acp": torch.sqrt(alpha_cumprod),
        "sqrt_omacp": torch.sqrt(1 - alpha_cumprod),
        "alpha_cumprod_prev": torch.cat(
            [torch.ones(1), alpha_cumprod[:-1]], dim=0
        )
    }


# UNet
class UNetTimeCond(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()

        def conv_block(in_ch, out_ch):
            groups = max(1, min(32, out_ch))
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(),
            )

        self.time_proj1 = nn.Linear(time_emb_dim, base_ch * 2)
        self.time_proj2 = nn.Linear(time_emb_dim, base_ch * 4)
        self.time_proj3 = nn.Linear(time_emb_dim, base_ch * 8)
        self.time_proj4 = nn.Linear(time_emb_dim, base_ch * 16)

        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, 1)

    def apply_film(self, h, emb, proj):
        g, b = proj(emb).chunk(2, dim=1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, x, emb):
        e1 = self.apply_film(self.enc1(x), emb, self.time_proj1)
        e2 = self.apply_film(self.enc2(self.pool(e1)), emb, self.time_proj2)
        e3 = self.apply_film(self.enc3(self.pool(e2)), emb, self.time_proj3)

        b = self.apply_film(self.bottleneck(self.pool(e3)), emb, self.time_proj4)

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.final(d1)


# Time embedding
def get_timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return F.pad(emb, (0, dim - emb.size(1)))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.mlp(x)


# CFG DDPM
class DDPM_CFG(nn.Module):
    def __init__(self, T=400, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.T = T
        self.time_emb_dim = time_emb_dim

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.unet = UNetTimeCond(base_ch, time_emb_dim)
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, time_emb_dim)

        buffers = make_diffusion_buffers(T)
        for k, v in buffers.items():
            self.register_buffer(k, v)

    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_acp[t][:, None, None, None] * x0 +
            self.sqrt_omacp[t][:, None, None, None] * noise
        )

    def predict_eps(self, x, t, y):
        t_emb = self.time_embedding(get_timestep_embedding(t, self.time_emb_dim))
        y_emb = self.label_emb(y)
        return self.unet(x, t_emb + y_emb)

    def forward(self, x0, t, y, p_uncond=0.1):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        drop = torch.rand_like(y.float()) < p_uncond
        y = y.clone()
        y[drop] = NULL_CLASS

        eps = self.predict_eps(x_t, t, y)
        return eps, noise
