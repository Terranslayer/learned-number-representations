# symemerge/world/shapes.py
"""Bench-side irregular-object generator (m51 spec 2026-07-16 §6; user rulings: objects are
randomly GENERATED irregular shapes, each visibly ONE object). One generator feeds BOTH the
scene sampler (scene4) and the CNN pretraining corpus. Bench-owned: no agent imports.

A canonical shape = binary mask on a MASK_RES grid, built as a UNION OF OVERLAPPING RANDOM
ELLIPSES chained along a random walk (each new center is placed inside the previous ellipse
=> the union is connected by construction, up to boundary-clamp rarities caught by the
check), accepted only if it passes the registered legibility checks: SINGLE connected
component (4-connectivity), COMPACTNESS area/bbox_area >= COMPACT_MIN (no stringy scatter),
and SQUARE-FILL area/max_dim^2 >= SQUARE_FILL_MIN (bounds the scaled footprint: the largest
budgeted object always fits its canvas, spec §5 placement v2.1). A seeded MASK POOL
amortizes generation; samplers draw pool indices independently of N, so shape identity
carries no count information."""
import math

import torch
import torch.nn.functional as F

MASK_RES = 33            # canonical odd resolution; bbox-cropped + rescaled at placement
COMPACT_MIN = 0.35       # area / bbox-area floor ("visibly one object", user ruling)
SQUARE_FILL_MIN = 0.42   # area / max(h,w)^2 floor (footprint bound, spec §5 v2.1)
N_ELL_LO, N_ELL_HI = 3, 6
MAX_TRIES = 256
POOL_SIZE = 4096
_POOL_CACHE = {}


def _components(mask):
    """List of 4-connected components (bool tensors) of a 0/1 mask (pure torch)."""
    remaining = mask > 0.5
    comps = []
    while bool(remaining.any()):
        idx = remaining.nonzero()[0]
        comp = torch.zeros_like(remaining)
        comp[idx[0], idx[1]] = True
        while True:
            grown = comp.clone()
            grown[1:, :] |= comp[:-1, :]
            grown[:-1, :] |= comp[1:, :]
            grown[:, 1:] |= comp[:, :-1]
            grown[:, :-1] |= comp[:, 1:]
            grown &= remaining
            if bool((grown == comp).all()):
                break
            comp = grown
        comps.append(comp)
        remaining &= ~comp
    return comps


def n_components(mask):
    return len(_components(mask))


def largest_component(mask):
    comps = _components(mask)
    if not comps:
        return mask
    return max(comps, key=lambda c: int(c.sum())).float()


def bbox_crop(mask):
    m = mask > 0.5
    ys, xs = m.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return mask
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def compactness(mask):
    m = bbox_crop(mask) > 0.5
    if m.numel() == 0:
        return 0.0
    return float(m.sum()) / float(m.shape[0] * m.shape[1])


def square_fill(mask):
    m = bbox_crop(mask) > 0.5
    if m.numel() == 0:
        return 0.0
    d = max(m.shape[0], m.shape[1])
    return float(m.sum()) / float(d * d)


def _ellipse(res, cy, cx, a, b, theta):
    yy, xx = torch.meshgrid(torch.arange(res, dtype=torch.float32),
                            torch.arange(res, dtype=torch.float32), indexing="ij")
    dy, dx = yy - cy, xx - cx
    ct, st = math.cos(theta), math.sin(theta)
    u = (dx * ct + dy * st) / a
    v = (-dx * st + dy * ct) / b
    return (u * u + v * v) <= 1.0


def gen_mask(rng):
    """One canonical irregular mask (variable bbox-cropped shape) float {0,1};
    deterministic in the generator state; retries until the legibility checks pass.
    Returned UNCROPPED at MASK_RES (pool stacking needs equal shapes); consumers crop."""
    res = MASK_RES
    for _ in range(MAX_TRIES):
        k = int(torch.randint(N_ELL_LO, N_ELL_HI + 1, (1,), generator=rng))
        mask = torch.zeros(res, res, dtype=torch.bool)
        cy = cx = res / 2.0
        prev_r = 0.0
        for j in range(k):
            a = float(torch.rand(1, generator=rng)) * (res * 0.18) + res * 0.12
            b = float(torch.rand(1, generator=rng)) * (res * 0.18) + res * 0.12
            theta = float(torch.rand(1, generator=rng)) * math.pi
            if j > 0:
                ang = float(torch.rand(1, generator=rng)) * 2 * math.pi
                d = float(torch.rand(1, generator=rng)) * prev_r * 0.6
                cy = min(max(cy + d * math.sin(ang), res * 0.25), res * 0.75)
                cx = min(max(cx + d * math.cos(ang), res * 0.25), res * 0.75)
            mask |= _ellipse(res, cy, cx, a, b, theta)
            prev_r = min(a, b)
        maskf = mask.float()
        if n_components(maskf) == 1 and compactness(maskf) >= COMPACT_MIN \
                and square_fill(maskf) >= SQUARE_FILL_MIN:
            return maskf
    raise RuntimeError("gen_mask: no legible mask in MAX_TRIES")


def mask_pool(seed=0, size=POOL_SIZE):
    """Seeded, cached pool of canonical masks (size, MASK_RES, MASK_RES)."""
    key = (seed, size)
    if key not in _POOL_CACHE:
        rng = torch.Generator().manual_seed(seed)
        _POOL_CACHE[key] = torch.stack([gen_mask(rng) for _ in range(size)])
    return _POOL_CACHE[key]


def scale_mask(mask, area_px):
    """Bbox-crop, then rescale to ~area_px on-pixels (bilinear + 0.5 threshold; largest
    component kept if thresholding splits it). Tight footprint => collision placement can
    nestle shapes into each other's bbox concavities."""
    mask = bbox_crop(mask)
    m_area = float(mask.sum())
    h, w = mask.shape
    sc = math.sqrt(max(area_px, 1.0) / m_area)
    nh, nw = max(3, int(round(h * sc))), max(3, int(round(w * sc)))
    scaled = F.interpolate(mask[None, None], size=(nh, nw), mode="bilinear",
                           align_corners=False)[0, 0]
    out = (scaled >= 0.5).float()
    if float(out.sum()) < 1:
        out = torch.zeros(3, 3)
        out[1, :] = 1.0
        out[:, 1] = 1.0
    elif n_components(out) != 1:
        out = bbox_crop(largest_component(out))
    return out


def stamp(canvas, mask, cy, cx, intensity):
    """Max-compose mask*intensity onto canvas (H,W) centered at (cy,cx); border-clipped.
    Returns the realized on-pixel count. (Corpus-side helper; scene4 places inline.)"""
    h, w = mask.shape
    H, W = canvas.shape
    y0, x0 = int(round(cy)) - h // 2, int(round(cx)) - w // 2
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(H, y0 + h), min(W, x0 + w)
    if ys1 <= ys0 or xs1 <= xs0:
        return 0
    sub = mask[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    region = canvas[ys0:ys1, xs0:xs1]
    canvas[ys0:ys1, xs0:xs1] = torch.maximum(region, sub * intensity)
    return int(sub.sum())
