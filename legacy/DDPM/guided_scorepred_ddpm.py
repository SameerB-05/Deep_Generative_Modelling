import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES        # special token index for "unconditional"
T = 400                         # diffusion steps
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Diffusion schedule (buffers)
# -------------------------
beta = torch.linspace(1e-4, 0.01, T, dtype=torch.float32)
alpha = 1 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)
sqrt_acp = torch.sqrt(alpha_cumprod)
sqrt_omacp = torch.sqrt(1 - alpha_cumprod)
alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=alpha.dtype, device=alpha.device),
                                alpha_cumprod[:-1]], dim=0)


# -------------------------
# UNet (time-conditioned)
# -------------------------
class UNetTimeCond(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()

        def conv_block(in_ch, out_ch):
            def choose_groups(ch, max_groups=32):
                for g in range(min(max_groups, ch), 0, -1):
                    if ch % g == 0:
                        return g
                return 1
            groups = choose_groups(out_ch, max_groups=32)
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups=groups, num_channels=out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups=groups, num_channels=out_ch),
                nn.SiLU(),
            )

        # time/label projections (they expect an embedding of size `time_emb_dim`)
        self.time_proj1 = nn.Linear(time_emb_dim, base_ch * 2)
        self.time_proj2 = nn.Linear(time_emb_dim, base_ch * 4)
        self.time_proj3 = nn.Linear(time_emb_dim, base_ch * 8)
        self.time_proj4 = nn.Linear(time_emb_dim, base_ch * 16)

        # encoder
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        # decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # final output: single-channel epsilon prediction
        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def apply_film(self, h, t_emb, proj):
        """
        h: (B, C, H, W)
        t_emb: (B, D) -> proj -> (B, 2*C) -> split into (gamma, beta)
        """
        gamma, beta = proj(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return h * (1 + gamma) + beta

    def forward(self, x, t_emb):
        # encoder
        e1 = self.enc1(x)
        e1 = self.apply_film(e1, t_emb, self.time_proj1)

        e2 = self.enc2(self.pool(e1))
        e2 = self.apply_film(e2, t_emb, self.time_proj2)

        e3 = self.enc3(self.pool(e2))
        e3 = self.apply_film(e3, t_emb, self.time_proj3)

        # bottleneck
        b = self.bottleneck(self.pool(e3))
        b = self.apply_film(b, t_emb, self.time_proj4)

        # decoder (skip connections carry time info too)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# -------------------------
# Time embedding helper
# -------------------------
def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal embedding (same API as many diffusion implementations).
    timesteps: LongTensor with shape (B,) or (B,1) (will be cast to float inside).
    """
    device = timesteps.device
    timesteps = timesteps.float().unsqueeze(1)  # (B,1)
    half = embedding_dim // 2
    if half > 0:
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float, device=device) / float(half))
        args = timesteps * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    else:
        emb = torch.zeros((timesteps.shape[0], 0), device=device)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# small MLP to process time embeddings
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)


# -------------------------
# DDPM model with label embedding + predict helper
# -------------------------
class DDPM_eps(nn.Module):
    def __init__(self, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embedding = TimeEmbedding(self.time_emb_dim)
        self.unet = UNetTimeCond(base_ch=base_ch, time_emb_dim=self.time_emb_dim)

        # label embedding includes one extra token for NULL_CLASS (unconditional)
        self.label_emb = nn.Embedding(NUM_CLASSES + 1, self.time_emb_dim)

        # register buffers so they move with .to(device)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_acp', sqrt_acp)
        self.register_buffer('sqrt_omacp', sqrt_omacp)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)

    def q_sample(self, x0, t, noise=None):
        """Produce x_t given x0 and timesteps t.
        t: LongTensor of shape (B,) or scalar; we'll ensure it's on the buffer device.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        # ensure t on same device as buffers and is long
        t = t.to(self.sqrt_acp.device).long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # indexing with t (B,) -> produce (B,1,1,1) to broadcast to x0
        return (self.sqrt_acp[t].view(-1, 1, 1, 1) * x0
                + self.sqrt_omacp[t].view(-1, 1, 1, 1) * noise)

    def predict_eps(self, x, t_idx, y_idx):
        """
        Unified helper that produces eps prediction for (x, t_idx, y_idx).
        y_idx should be LongTensor of shape (B,) with values in 0..NUM_CLASSES
        or NUM_CLASSES (NULL_CLASS) for unconditional token.
        """
        # prepare timestep embedding
        t_idx = t_idx.to(next(self.parameters()).device).long()
        t_emb = get_timestep_embedding(t_idx, self.time_emb_dim).to(t_idx.device)  # (B, D)
        t_emb = self.time_embedding(t_emb)  # (B, D)

        # label embedding (sum)
        y_idx = y_idx.to(t_idx.device).long()
        lab = self.label_emb(y_idx)  # (B, D)

        t_emb = t_emb + lab  # fused conditioning (time + class)

        return self.unet(x, t_emb)

    def forward(self, x0, t, y, p_uncond=0.1):
        """
        Training forward.
        - x0: (B,1,H,W)
        - t: LongTensor (B,)
        - y: LongTensor (B,)
        - p_uncond: probability to replace y by NULL_CLASS (per-sample)
        Returns: eps_pred, x_t, noise
        """
        B = x0.shape[0]
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # per-sample label dropout: replace with NULL_CLASS token with prob p_uncond
        # create y_input with shape (B,)
        y_input = y.clone().to(x0.device)
        if p_uncond > 0:
            # draw per-sample uniform and set some to NULL_CLASS
            rand = torch.rand(B, device=x0.device)
            drop_mask = rand < p_uncond
            if drop_mask.any():
                y_input[drop_mask] = NULL_CLASS

        eps_pred = self.predict_eps(x_t, t, y_input)
        return eps_pred, x_t, noise


# -------------------------
# Sampling with classifier-free guidance
# -------------------------
@torch.no_grad()
def sample(model, shape, device, y=None, guidance_scale=3.0):
    """
    model: DDPM_eps (already .to(device))
    shape: (B, C, H, W)
    y: if not None: LongTensor of size B with class indices (0..NUM_CLASSES-1).
       If you want unconditional sampling, pass y=None.
    guidance_scale: s in eps = eps_uncond + s*(eps_cond - eps_uncond)
    """
    model.eval()
    x = torch.randn(shape, device=device)
    B = shape[0]
    T_local = model.alpha.shape[0]

    # prepare conditional labels
    if y is None:
        # for unconditional sampling we'll still query the NULL_CLASS token only
        y_cond = torch.full((B,), NULL_CLASS, device=device, dtype=torch.long)
    else:
        # user passed class indices -> ensure device
        y = y.to(device)
        # conditional labels for cond pass
        y_cond = y
    # unconditional labels for uncond pass: NULL_CLASS
    y_uncond = torch.full((B,), NULL_CLASS, device=device, dtype=torch.long)

    for t in reversed(range(1, T_local)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)

        # compute embeddings + predictions using model.predict_eps helper
        eps_uncond = model.predict_eps(x, t_idx, y_uncond)  # (B,1,H,W)
        if y is None:
            # if no conditional labels requested, cond==uncond
            eps_cond = eps_uncond
        else:
            eps_cond = model.predict_eps(x, t_idx, y_cond)

        # CFG combination
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # posterior calculations (same as before)
        acp_t = model.alpha_cumprod[t_idx].view(B, 1, 1, 1)
        sqrt_acp_t = model.sqrt_acp[t_idx].view(B, 1, 1, 1)
        sqrt_1m_acp_t = model.sqrt_omacp[t_idx].view(B, 1, 1, 1)

        # estimate x0
        x0_hat = (x - sqrt_1m_acp_t * eps_pred) / (sqrt_acp_t + 1e-8)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        alpha_t = model.alpha[t_idx].view(B, 1, 1, 1)
        acp_prev = model.alpha_cumprod_prev[t_idx].view(B, 1, 1, 1)

        coeff_x = torch.sqrt(alpha_t) * (1.0 - acp_prev) / (1.0 - acp_t)
        coeff_x0 = torch.sqrt(acp_prev) * (1.0 - alpha_t) / (1.0 - acp_t)

        mean = coeff_x * x + coeff_x0 * x0_hat

        if t > 1:
            noise = torch.randn_like(x)
            sigma = torch.sqrt((1.0 - acp_prev) / (1.0 - acp_t) * (1.0 - alpha_t))
            x = mean + sigma * noise
        else:
            x = mean

    return torch.clamp(x, -1.0, 1.0)


# -------------------------
# Main training script
# -------------------------
if __name__ == "__main__":
    # data transform -> [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    model = DDPM_eps(base_ch=64, time_emb_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    losses = []

    best_loss = float('inf')
    checkpoint_path = "ddpm_mnist_cfg_best.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            B = x.size(0)

            # sample timestep
            t = torch.randint(1, T, (B,), device=DEVICE)

            # forward: model performs per-sample label dropout internally
            eps_pred, x_t, noise = model(x, t, y, p_uncond=0.1)

            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

        # checkpoint best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "T": T,
            }, checkpoint_path)
            print(f"✔ Saved new best checkpoint at epoch {epoch+1}")

    # plot training loss
    plt.plot(losses)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss for DDPM (CFG ε-prediction)')
    plt.grid(True); plt.show()

    # -------------------------
    # Generate conditional samples (example)
    # -------------------------
    model.eval()

    # Create a batch of labels to sample: 16 examples with target classes 0..9 repeated
    y_sample = torch.tensor([i % NUM_CLASSES for i in range(16)], dtype=torch.long, device=DEVICE)

    samples = sample(model, (16, 1, 32, 32), DEVICE, y=y_sample, guidance_scale=3.0).cpu()

    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        img = (samples[i, 0] + 1) / 2   # [-1,1] -> [0,1]
        ax.imshow(img.clamp(0, 1).numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle('CFG-generated Samples from DDPM on MNIST (guidance_scale=3.0)')
    plt.tight_layout()
    plt.show()
