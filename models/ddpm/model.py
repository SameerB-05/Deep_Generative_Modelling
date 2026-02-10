import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ SCHEDULE ------------------

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.01):
    beta = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    alpha_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=alpha.dtype), alpha_cumprod[:-1]], dim=0
    )

    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_cumprod": alpha_cumprod,
        "alpha_cumprod_prev": alpha_cumprod_prev,
        "sqrt_acp": torch.sqrt(alpha_cumprod),
        "sqrt_omacp": torch.sqrt(1.0 - alpha_cumprod),
    }


# ------------------ TIME EMBEDDING ------------------

def get_timestep_embedding(timesteps, embedding_dim):
    device = timesteps.device
    timesteps = timesteps.float().unsqueeze(1)
    half = embedding_dim // 2

    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device) / half
    )
    args = timesteps * freqs.unsqueeze(0)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------ UNET ------------------

class UNetTimeCond(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
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

    def apply_film(self, h, t_emb, proj):
        gamma, beta = proj(t_emb).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

    def forward(self, x, t_emb):
        e1 = self.apply_film(self.enc1(x), t_emb, self.time_proj1)
        e2 = self.apply_film(self.enc2(self.pool(e1)), t_emb, self.time_proj2)
        e3 = self.apply_film(self.enc3(self.pool(e2)), t_emb, self.time_proj3)

        b = self.apply_film(self.bottleneck(self.pool(e3)), t_emb, self.time_proj4)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# ------------------ DDPM ------------------

class DDPM_eps(nn.Module):
    def __init__(self, T=400):
        super().__init__()
        self.T = T
        self.time_emb_dim = 128

        self.time_embedding = TimeEmbedding(self.time_emb_dim)
        self.unet = UNetTimeCond(time_emb_dim=self.time_emb_dim)

        schedule = make_beta_schedule(T)
        for k, v in schedule.items():
            self.register_buffer(k, v)

    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_acp[t][:, None, None, None] * x0 +
            self.sqrt_omacp[t][:, None, None, None] * noise
        )

    def forward(self, x0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embedding(t_emb)

        eps_pred = self.unet(x_t, t_emb)
        return eps_pred, noise
