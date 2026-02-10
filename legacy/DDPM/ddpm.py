import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# diffusion schedule parameters
T = 400                                    # total diffusion steps
beta = torch.linspace(1e-4, 0.01, T, dtype=torch.float32)        # linear beta schedule
alpha = 1 - beta                            # alpha at each timestep
alpha_cumprod = torch.cumprod(alpha, dim=0) # cumulative product of alpha
sqrt_acp = torch.sqrt(alpha_cumprod)        # sqrt of alpha cumulative product
sqrt_omacp = torch.sqrt(1 - alpha_cumprod)  # sqrt of (1 - alpha cumulative product)
alpha_cumprod_prev = torch.cat(
    [torch.ones(1, dtype=alpha.dtype, device=alpha.device), alpha_cumprod[:-1]],
    dim=0
)



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


        # --- Time FiLM projections (γ, β) ---
        self.time_proj1 = nn.Linear(time_emb_dim, base_ch * 2)
        self.time_proj2 = nn.Linear(time_emb_dim, base_ch * 4)
        self.time_proj3 = nn.Linear(time_emb_dim, base_ch * 8)
        self.time_proj4 = nn.Linear(time_emb_dim, base_ch * 16)

        # --- Encoder ---
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        # --- Decoder ---
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # --- Output ε prediction ---
        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def apply_film(self, h, t_emb, proj):
        """
        h: feature map (B, C, H, W)
        t_emb: time embedding (B, D)
        proj: linear layer producing (γ, β)
        """
        gamma, beta = proj(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return h * (1 + gamma) + beta

    def forward(self, x, t_emb):
        # --- Encoder ---
        e1 = self.enc1(x)
        e1 = self.apply_film(e1, t_emb, self.time_proj1)

        e2 = self.enc2(self.pool(e1))
        e2 = self.apply_film(e2, t_emb, self.time_proj2)

        e3 = self.enc3(self.pool(e2))
        e3 = self.apply_film(e3, t_emb, self.time_proj3)

        # --- Bottleneck ---
        b = self.bottleneck(self.pool(e3))
        b = self.apply_film(b, t_emb, self.time_proj4)

        # --- Decoder ---
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)



def get_timestep_embedding(timesteps, embedding_dim):
    device = timesteps.device
    timesteps = timesteps.float().unsqueeze(1)  # (B,1)
    half = embedding_dim // 2
    if half > 0:
        # make arange float explicitly
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float, device=device) / float(half))
        args = timesteps * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    else:
        emb = torch.zeros((timesteps.shape[0], 0), device=device)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb




# Small MLP to project time embedding
# lets the model learn how to use time, instead of forcing it raw
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


class DDPM_eps(nn.Module):
    def __init__(self):

        super().__init__()
        self.time_emb_dim = 128
        self.time_embedding = TimeEmbedding(self.time_emb_dim)
        self.unet = UNetTimeCond(base_ch=64, time_emb_dim=self.time_emb_dim)


        # register buffers so they are moved to GPU with the model
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_acp', sqrt_acp)
        self.register_buffer('sqrt_omacp', sqrt_omacp)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)


    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        # x_t = sqrt_acp[t] * x0 + sqrt(1 - alpha_cumprod[t]) * noise
        return (self.sqrt_acp[t].view(-1, 1, 1, 1) * x0
                + self.sqrt_omacp[t].view(-1, 1, 1, 1) * noise)
    
    def forward(self, x0, t):
        
        # training: add noise to x0 at step t and predict ε

        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise) # add noise to x0 at step t

        # FiLM time embedding (Feature-wise Linear Modulation)
        B, _, H, W = x0.shape

        t_emb = get_timestep_embedding(t, self.time_emb_dim).to(x0.device)
        t_emb = self.time_embedding(t_emb)          # (B, D)

        eps_pred = self.unet(x_t, t_emb)  # predict ε

        return eps_pred, x_t, noise


@torch.no_grad()
def sample(model, shape, device):
    x = torch.randn(shape, device=device)
    B = shape[0]
    T = model.alpha.shape[0]
    for t in reversed(range(1, T)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        t_emb = get_timestep_embedding(t_idx, model.time_emb_dim).to(x.device)
        t_emb = model.time_embedding(t_emb)

        eps_pred = model.unet(x, t_emb)

        acp_t = model.alpha_cumprod[t_idx].view(B,1,1,1)
        sqrt_acp_t = model.sqrt_acp[t_idx].view(B,1,1,1)
        sqrt_1m_acp_t = model.sqrt_omacp[t_idx].view(B,1,1,1)

        x0_hat = (x - sqrt_1m_acp_t * eps_pred) / (sqrt_acp_t + 1e-8)  # small eps for safety
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        alpha_t = model.alpha[t_idx].view(B,1,1,1)
        acp_prev = model.alpha_cumprod_prev[t_idx].view(B,1,1,1)

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


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(datasets.MNIST('../data', train=True, download=True, 
                                    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])),
                            batch_size=128, shuffle=True, num_workers=4)

    model = DDPM_eps().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    losses = []

    best_loss = float('inf')
    checkpoint_path = "ddpm_mnist_best.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x, _ in loader:
            x = x.to(device)
            B = x.size(0)

            # sample timestep
            t = torch.randint(1, T, (B,), device=device)

            eps_pred, x_t, noise = model(x, t)

            loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
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
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss for DDPM (ε-prediction)')
    plt.grid(True); plt.show()


    # generate and display samples
    model.eval()
    samples = sample(model, (16, 1, 32, 32), device).cpu()
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axs.flatten()):
        img = (samples[i, 0] + 1) / 2   # [-1,1] → [0,1]
        ax.imshow(img.clamp(0,1).numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle('Generated Samples from DDPM (ε-prediction) on MNIST')
    plt.tight_layout()
    plt.show()
