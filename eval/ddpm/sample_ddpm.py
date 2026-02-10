import torch
import matplotlib.pyplot as plt
from models.ddpm.model import DDPM_eps, get_timestep_embedding

@torch.no_grad()
def sample(model, shape, device):
    x = torch.randn(shape, device=device)
    B = shape[0]

    for t in reversed(range(1, model.T)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        t_emb = model.time_embedding(
            get_timestep_embedding(t_idx, model.time_emb_dim)
        )

        eps = model.unet(x, t_emb)

        acp = model.alpha_cumprod[t_idx][:, None, None, None]
        acp_prev = model.alpha_cumprod_prev[t_idx][:, None, None, None]
        alpha = model.alpha[t_idx][:, None, None, None]

        x0 = (x - torch.sqrt(1 - acp) * eps) / torch.sqrt(acp)
        x0 = x0.clamp(-1, 1)

        mean = (
            torch.sqrt(alpha) * (1 - acp_prev) / (1 - acp) * x +
            torch.sqrt(acp_prev) * (1 - alpha) / (1 - acp) * x0
        )

        if t > 1:
            noise = torch.randn_like(x)
            sigma = torch.sqrt((1 - acp_prev) / (1 - acp) * (1 - alpha))
            x = mean + sigma * noise
        else:
            x = mean

    return x.clamp(-1, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDPM_eps().to(device)
    ckpt = torch.load("checkpoints/ddpm/ddpm_mnist_best.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    samples = sample(model, (16, 1, 32, 32), device).cpu()

    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow((samples[i, 0] + 1) / 2, cmap="gray")
        ax.axis("off")
    plt.show()
