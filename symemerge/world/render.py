import torch

def render_blobs(positions, log_sigmas, intensities, canvas_size, device=None):
    """Sum of 2D Gaussian blobs -> (B,1,H,W). Differentiable wrt all inputs."""
    B, K, _ = positions.shape
    H = W = canvas_size
    device = device or positions.device
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")          # (H,W)
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, H, W, 2) # (1,1,H,W,2) order (x,y)
    pos = positions.to(device).view(B, K, 1, 1, 2)
    sig = torch.exp(log_sigmas.to(device)).view(B, K, 1, 1)
    d2 = ((grid - pos) ** 2).sum(-1)                          # (B,K,H,W)
    blobs = torch.exp(-0.5 * d2 / (sig ** 2 + 1e-8))         # (B,K,H,W)
    canvas = (blobs * intensities.to(device).view(B, K, 1, 1)).sum(dim=1, keepdim=True)
    return canvas


def saturate(canvas, ceiling):
    """Soft per-pixel saturating nonlinearity = the note medium's capacity limit: ceiling*tanh(x/ceiling).
    Near-identity for x << ceiling, asymptotes to ceiling for x >> ceiling; smooth (stays emergent, no hard
    threshold/quantization). ceiling=None (or <=0) disables -> identity. Ramping the ceiling DOWN caps how much
    magnitude a single pixel can carry, forcing the agent to spread signal across MORE locations = discrete marks."""
    if ceiling is None or ceiling <= 0:
        return canvas
    return ceiling * torch.tanh(canvas / ceiling)


def render_content_blobs(positions, log_sigmas, intensities, colors, canvas_size, device=None):
    """Colored render_blobs: each object k carries a per-object colour vector colors (B,K,C) (its count-
    orthogonal IDENTITY, needed for cross-view content correspondence -- identical dots cannot support it).
    Channel c of the canvas = sum of blobs weighted by intensities*colors[...,c]; colors=ones((B,K,1))
    reduces exactly to render_blobs. Returns (B,C,H,W). Differentiable wrt all inputs."""
    B, K, _ = positions.shape
    H = W = canvas_size
    device = device or positions.device
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, H, W, 2)
    pos = positions.to(device).view(B, K, 1, 1, 2)
    sig = torch.exp(log_sigmas.to(device)).view(B, K, 1, 1)
    d2 = ((grid - pos) ** 2).sum(-1)
    blobs = torch.exp(-0.5 * d2 / (sig ** 2 + 1e-8))                       # (B,K,H,W)
    w = intensities.to(device).unsqueeze(-1) * colors.to(device)           # (B,K,C)
    canvas = torch.einsum("bkhw,bkc->bchw", blobs, w)
    return canvas
