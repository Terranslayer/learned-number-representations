# symemerge/world/scene4.py
"""m51 scene bench (spec 2026-07-16 §5 law v2 + placement v2.1 + canvas ruling 13,
user-ratified). N irregular objects (shapes.py pool) with TOTALS-decorrelated budgets:
per scene, both independent of N, draw an AREA budget A ~ LogUniform[A_LO, A_HI]·(side/64)²
and a BRIGHTNESS scale s ~ U[BRIGHT_LO, BRIGHT_HI]; per-object areas = MIN_AREA +
(A − N·MIN_AREA)·Dirichlet(ALPHA) (floor-guaranteed, sums to A exactly); per-object flat
intensity = s·jitter U[JIT_LO, JIT_HI] ∈ [0.10, 0.95] by construction (visible,
unsaturated — no clamps, no rejection). Canvas side is a per-BAND constant from
canvas_side(n_max); budgets scale with canvas AREA (ruling 13: extrapolation canvases GROW;
the conv eye scans any size; the NOTE canvas stays 64×64). PLACEMENT (v2.1): biggest-first
ACTUAL-MASK collision — whole footprint inside the canvas, 1px-dilated masks pairwise
disjoint (objects may nestle but never touch => exact CC count); bounding-circle min-sep is
geometrically infeasible under this budget law (its exclusion circles exceed the canvas on
the small-N giant-object tail). Candidate laws (mix42 / near-miss / legacy) are IMPORTED
from scene3 and re-rendered under this law with EVERYTHING redrawn per candidate.
Bench-owned: no agent imports. All TRIAL randomness from the caller's CPU generator (CRN);
the shape POOL is a fixed seeded asset by design (a shared texture atlas keyed by
pool_seed, identical across all conditions — not trial randomness; reviewer-confirmed
intent)."""
import math

import torch
import torch.nn.functional as F

from . import shapes
from .scene3 import choose_pair3, choose_pair42

GAME_SIDE = 96                  # spec §5 v2 (post-§11-review): ONE side for the WHOLE
                                # experiment (canvas_side(64)); per-band switching is
                                # RETIRED — frozen attention weights cannot recalibrate
                                # across cell counts on a band nothing ever trains on.
                                # canvas_side() remains the FUTURE-extension formula
                                # (raising N_max raises GAME_SIDE for the whole run).
A_LO, A_HI = 400.0, 1300.0      # area budget at side 64, px^2 (scaled by (side/64)^2)
BRIGHT_LO, BRIGHT_HI = 0.12, 0.80
JIT_LO, JIT_HI = 0.85, 1.18
MIN_AREA = 9.0
ALPHA = 2                        # Dirichlet concentration (integer: gamma = sum of ALPHA exps)
DILATE_PX = 1                    # inter-object gap: 1px-dilated footprints kept disjoint
CANVAS_BASE = 64
SIDES = (64, 96, 128, 192, 256)
OCCUPANCY = 0.35
PLACE_TRIES = 128
LAYOUT_RESTARTS = 8


def canvas_side(n_max):
    """Smallest registered side that is CLAMP-FREE for n_max objects: the scaled budget
    floor admits n_max·MIN_AREA with 1.1 headroom AND occupancy stays under the packing
    bound. Working band (<=40) -> 64; held-out 48/64 -> 96 (test-pinned)."""
    for s in SIDES:
        scale = (s / CANVAS_BASE) ** 2
        if A_LO * scale >= MIN_AREA * n_max * 1.1 and \
                MIN_AREA * n_max * 2.5 <= OCCUPANCY * s * s:
            return s
    raise ValueError(f"n_max={n_max} exceeds the largest registered canvas")


def _dirichlet(n, rng):
    """Dirichlet(ALPHA,...,ALPHA) via sums of ALPHA exponentials, CPU-generator-driven
    (torch.distributions takes no generator — CRN discipline requires one)."""
    g = -(torch.rand(ALPHA, n, generator=rng).clamp(min=1e-12).log()).sum(0)
    return g / g.sum()


def _dilate(m):
    """Dilate by DILATE_PX with the ring KEPT (output grows by DILATE_PX per side —
    truncating the ring at the bbox border would let another object sit flush there)."""
    k = 2 * DILATE_PX + 1
    p = F.pad(m[None, None].float(), (DILATE_PX,) * 4)
    return F.max_pool2d(p, k, stride=1, padding=DILATE_PX)[0, 0] > 0.5


def _place_all(canv_b, masks, intens, side, rng):
    """Biggest-first placement with actual-mask collision (spec §5 v2.1). The occupancy
    grid holds DILATED footprints of placed objects; a new RAW mask must be disjoint from
    it => every pair of objects is >= DILATE_PX apart (never merges under 4-connectivity).
    Returns True on success, False if any object exhausted PLACE_TRIES."""
    order = sorted(range(len(masks)), key=lambda k: -float(masks[k].sum()))
    occ = torch.zeros(side, side, dtype=torch.bool)
    for k in order:
        m = masks[k] > 0.5
        h, w = m.shape
        if h > side - 2 or w > side - 2:
            raise RuntimeError(f"footprint {h}x{w} exceeds canvas {side} "
                               f"(square-fill acceptance should forbid this)")
        placed = False
        for _ in range(PLACE_TRIES):
            y0 = int(torch.randint(1, side - h, (1,), generator=rng))
            x0 = int(torch.randint(1, side - w, (1,), generator=rng))
            if not bool((occ[y0:y0 + h, x0:x0 + w] & m).any()):
                occ[y0 - DILATE_PX:y0 + h + DILATE_PX,
                    x0 - DILATE_PX:x0 + w + DILATE_PX] |= _dilate(m)
                region = canv_b[y0:y0 + h, x0:x0 + w]
                canv_b[y0:y0 + h, x0:x0 + w] = torch.maximum(
                    region, masks[k] * float(intens[k]))
                placed = True
                break
        if not placed:
            return False
    return True


def sample_scene4(counts, rng, device, side=None, pool_seed=0, pool_size=shapes.POOL_SIZE,
                  return_truth=False):
    """counts (B,) long -> (B, side, side) float32 scenes under the m51 law. side=None =>
    canvas_side(counts.max()). return_truth: also a per-scene dict list (A, bright, areas,
    intens) for the REPORTED (not gated) per-object statistics."""
    counts = counts.long().cpu()
    if side is None:
        side = canvas_side(int(counts.max()))
    scale = (side / CANVAS_BASE) ** 2
    assert A_LO * scale >= MIN_AREA * int(counts.max()) * 1.1, \
        f"band N_max={int(counts.max())} infeasible at side {side} (use canvas_side)"
    pool = shapes.mask_pool(pool_seed, pool_size)
    B = counts.shape[0]
    canv = torch.zeros(B, side, side)
    truth = []
    log_lo, log_hi = math.log(A_LO * scale), math.log(A_HI * scale)
    for b in range(B):
        n = int(counts[b])
        A = math.exp(float(torch.rand(1, generator=rng)) * (log_hi - log_lo) + log_lo)
        bright = float(torch.rand(1, generator=rng)) * (BRIGHT_HI - BRIGHT_LO) + BRIGHT_LO
        shares = _dirichlet(n, rng)
        areas = MIN_AREA + (A - MIN_AREA * n) * shares
        jit = torch.rand(n, generator=rng) * (JIT_HI - JIT_LO) + JIT_LO
        intens = bright * jit
        idx = torch.randint(pool.shape[0], (n,), generator=rng)
        masks = [shapes.scale_mask(pool[int(i)], float(a)) for i, a in zip(idx, areas)]
        for attempt in range(LAYOUT_RESTARTS):
            canv[b].zero_()
            if _place_all(canv[b], masks, intens, side, rng):
                break
        else:
            raise RuntimeError(f"placement exhausted at N={n} side={side} "
                               f"(registered bands must never hit this)")
        if return_truth:
            truth.append(dict(A=A, bright=bright, areas=areas, intens=intens))
    canv = canv.to(device)
    return (canv, truth) if return_truth else canv


def scene_stats(canv):
    """Pixel-measured per-scene stats (the gate measures RENDERED REALITY): total ink,
    covered area (>0.05), peak."""
    flat = canv.flatten(1)
    return dict(ink=flat.sum(1), area=(flat > 0.05).float().sum(1), peak=flat.amax(1))


def sample_candidates4(counts, rng, device, near_miss=False, mix42_caps=None,
                       pool_seed=0, pool_size=shapes.POOL_SIZE, side=None):
    """counts (B,) -> (cands (B,3,side,side), tgt (B,), cand_ns (B,3)). Foil-N laws
    carried verbatim from scene3 (choose_pair42 under mix42_caps, else choose_pair3);
    per-lane order shuffle; every candidate is a FRESH full redraw under the m51 law
    (budgets, shapes, layout — same-N-different-look structural). side=None => the
    registered uniform GAME_SIDE (§5 v2; one side for the whole experiment)."""
    assert mix42_caps is None or not near_miss, "mix42 and near_miss are exclusive"
    counts = counts.long().cpu()
    B = counts.shape[0]
    trip = torch.empty(B, 3, dtype=torch.long)
    if mix42_caps is not None:
        caps = mix42_caps.long().cpu()
        assert caps.shape == counts.shape, "mix42_caps must align with counts"
        for b in range(B):
            n = int(counts[b])
            a, o = choose_pair42(n, rng, int(caps[b]))
            trip[b] = torch.tensor([n + a, n + o, n])
    else:
        for b in range(B):
            n = int(counts[b])
            a, o = choose_pair3(n, rng, near_miss=near_miss)
            trip[b] = torch.tensor([n + a, n + o, n])
    perm = torch.argsort(torch.rand(B, 3, generator=rng), dim=1)
    trip = torch.gather(trip, 1, perm)
    tgt = (perm == 2).float().argmax(1)
    if side is None:
        side = GAME_SIDE                     # the registered uniform game side (§5 v2)
    flat = sample_scene4(trip.reshape(-1), rng, device, side=side,
                         pool_seed=pool_seed, pool_size=pool_size)
    return flat.view(B, 3, side, side), tgt.long().to(device), trip.to(device)
