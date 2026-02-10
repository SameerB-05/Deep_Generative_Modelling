import torch
from models.ddpm_cfg.model import NULL_CLASS


@torch.no_grad()
def sample_cfg(model, shape, device, y=None, guidance_scale=3.0):
    model.eval()
    x = torch.randn(shape, device=device)
    B = shape[0]

    y_uncond = torch.full((B,), NULL_CLASS, device=device)
    y_cond = y if y is not None else y_uncond

    for t in reversed(range(1, model.T)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)

        eps_u = model.predict_eps(x, t_idx, y_uncond)
        eps_c = model.predict_eps(x, t_idx, y_cond)
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        acp_t = model.alpha_cumprod[t_idx][:, None, None, None]
        acp_prev = model.alpha_cumprod_prev[t_idx][:, None, None, None]

        x0 = (x - model.sqrt_omacp[t_idx][:, None, None, None] * eps) / \
             model.sqrt_acp[t_idx][:, None, None, None]
        x0 = x0.clamp(-1, 1)

        mean = (
            torch.sqrt(model.alpha[t_idx])[:, None, None, None] *
            (1 - acp_prev) / (1 - acp_t) * x +
            torch.sqrt(acp_prev) *
            (1 - model.alpha[t_idx])[:, None, None, None] /
            (1 - acp_t) * x0
        )

        if t > 1:
            noise = torch.randn_like(x)
            sigma = torch.sqrt((1 - acp_prev) / (1 - acp_t) * (1 - model.alpha[t_idx]))
            x = mean + sigma[:, None, None, None] * noise
        else:
            x = mean

    return x.clamp(-1, 1)
