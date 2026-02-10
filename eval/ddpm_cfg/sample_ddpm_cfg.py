import torch
import matplotlib.pyplot as plt
from models.ddpm_cfg.model import DDPM_CFG, NULL_CLASS

NUM_CLASSES = 10  # MNIST

# -------------------------
# Sampling with classifier-free guidance (CFG)
# -------------------------
@torch.no_grad()
def sample_cfg(model, shape, device, y=None, guidance_scale=3.0):
    """
    model: DDPM_CFG (already .to(device))
    shape: (B, C, H, W)
    y: LongTensor of size B with class indices (0..NUM_CLASSES-1), optional
    guidance_scale: s in eps = eps_uncond + s*(eps_cond - eps_uncond)
    """
    model.eval()
    x = torch.randn(shape, device=device)
    B = shape[0]
    T_local = model.alpha.shape[0]

    # conditional labels
    if y is None:
        y_cond = torch.full((B,), NULL_CLASS, device=device, dtype=torch.long)
    else:
        y = y.to(device)
        y_cond = y

    # unconditional labels
    y_uncond = torch.full((B,), NULL_CLASS, device=device, dtype=torch.long)

    for t in reversed(range(1, T_local)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)

        eps_uncond = model.predict_eps(x, t_idx, y_uncond)
        if y is None:
            eps_cond = eps_uncond
        else:
            eps_cond = model.predict_eps(x, t_idx, y_cond)

        # CFG
        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        acp_t = model.alpha_cumprod[t_idx].view(B, 1, 1, 1)
        sqrt_acp_t = model.sqrt_acp[t_idx].view(B, 1, 1, 1)
        sqrt_1m_acp_t = model.sqrt_omacp[t_idx].view(B, 1, 1, 1)

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
# Main script
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DDPM_CFG().to(device)
    ckpt = torch.load("checkpoints/ddpm_cfg/ddpm_cfg_mnist_best.pth",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Create a batch of labels: 16 examples, 0..9 repeated
    BATCH_SIZE = 16
    y_sample = torch.tensor([i % NUM_CLASSES for i in range(BATCH_SIZE)],
                            dtype=torch.long, device=device)

    samples = sample_cfg(model, (BATCH_SIZE, 1, 32, 32), device,
                         y=y_sample, guidance_scale=3.0).cpu()

    # Display in 4x4 grid
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        img = (samples[i, 0] + 1) / 2  # [-1,1] -> [0,1]
        ax.imshow(img.clamp(0, 1).numpy(), cmap="gray")
        ax.axis("off")

    plt.suptitle("CFG-generated Samples from DDPM_CFG (guidance_scale=3.0)")
    plt.tight_layout()
    plt.show()
