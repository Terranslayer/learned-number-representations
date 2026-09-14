# symemerge/world/scene3.py
"""M3 scene bench (spec §2.1-B): N varied blobs on a 64×64 pixel canvas, min-separation
enforced over the ENTIRE battery range (capacity printed, honored to N=64 at these frozen
constants), EQUAL-INTEGRAL blobs (±A_JITTER) so render-composition variance sits below the
Weber band — the ONLY quantity noise the eye's m̂ sees is its own multiplicative ε draw.
Scenes are NOT saturated (they are stimuli, not the incremental canvas medium; blob peaks
are ≤ 1 by construction). area_norm=True rescales total ink to a constant — the ANS-null /
global-magnitude-servo control (spec §2.6).

Candidate sampler (interface B, spec §2.2): counts {N+a, N+b, N} with (a,b) drawn per trial
from the VALID subset of OFFSET_PAIRS (all counts ≥ 1 and distinct — N=1 forces (+1,+2); no
re-roll loop, validity is precomputed), order shuffled so the target is neither a positional
nor a rank invariant.

Bench-owned: never imports symemerge.agent. All randomness from the caller's CPU generator.
"""
import torch

from .render import render_blobs

A_BLOB = 6.0          # target per-blob integral (ink units); Σscene ≈ A_BLOB·N
A_JITTER = 0.10       # ±10% per-blob integral jitter (composition CV ≤ 0.058/√N < W/2)
MIN_SEP = 0.08        # normalized center separation (5.1 px); capacity ≈ 77 ≥ 64
SIG_LO, SIG_HI = 0.018, 0.024   # blob σ, normalized (1.15–1.54 px). RANGE COUPLED TO THE
                      # EYE: peak = A_BLOB·jit/(2π(σ·64)²) ∈ [0.36, 0.79] — every blob's
                      # peak clears eye2d.THETA_BIN=0.30 with margin, so no blob is ever
                      # fovea/gist-invisible (test-pinned below). Widening σ past 0.0264
                      # drops dim blobs below the shared binarization threshold.
MARGIN_LO, MARGIN_HI = 0.08, 0.92
CANVAS = 64
A_NORM_TOTAL = 6.0 * 8          # area-normalized control: every scene sums to this
OFFSET_PAIRS = ((-2, -1), (-1, 1), (1, 2), (-2, 1), (-1, 2))
# m40 A2.2 (near-miss reweighting): SAME pair set, reweighted sampling only. Up-weights the two
# |offset|=1-only same-signed pairs (-2,-1)/(1,2) (near-miss hardening), down-weights the
# symmetric (-1,1) (its target is always the magnitude median), keeps the two mixed ±2 pairs
# moderate so target rank stays unpredictable (rank-shuffle defense: each rank ≥ 0.20 at every
# N≥3, test-pinned; N=1,2 are validity-forced degenerate exactly as under the legacy uniform
# draw). Index-aligned with OFFSET_PAIRS.
NEAR_MISS_WEIGHTS = (0.25, 0.15, 0.25, 0.175, 0.175)

# ---- m42 §2.2/§2.3 (spec 2026-07-14): the mix42 mixture-distractor law ----
MIX42_POOL_EXAM = (20, 24, 28, 32, 36, 40)   # exam-N pool (per-world-set pair draw)
MIX42_CAP_TAUGHT = 24                        # taught far-count cap (foil ceiling)
MIX42_CAP_EXAM = 40                          # exam far-count cap (held-out 48/64 moat)


def d0_far(n):
    """m42 §2.2: minimum far distance max(2, ceil(N/2)) — ≥ 2σ of the weber:0.25 kernel
    at every N, so every far pick earns ≤ exp(−2) (attained exactly at even N)."""
    return max(2, (n + 1) // 2)


def choose_pair42(n, rng, cap):
    """m42 §2.2 per-trial mixture draw → (near_delta, far_delta). near = ±1 (50/50; +1
    forced at N=1). far = s·d: side s 50/50 over the VALID sides, then d uniform-integer
    over that side's range [d0_far(n), side_max]; above side_max = cap−n, below = n−1
    (side invalid when max < d0 — N≤2 are above-only, exam N≥28 below-only under cap 40:
    the recorded edge degeneracies). The jitter in d is load-bearing (spec §10 S1): a
    FIXED d(N) makes the candidate triple a near-lossless code for N. Counts stay in
    [1, cap]; triple {n−?1, n±d, n} is distinct by |far|≥2≠1."""
    if n == 1:
        a = 1
    else:
        a = 1 if int(torch.randint(2, (1,), generator=rng)) == 1 else -1
    d0 = d0_far(n)
    sides = []
    if cap - n >= d0:
        sides.append(1)
    if n - 1 >= d0:
        sides.append(-1)
    assert sides, f"mix42: no valid far side at n={n} cap={cap}"
    s = sides[int(torch.randint(len(sides), (1,), generator=rng))] if len(sides) > 1 \
        else sides[0]
    dmax = (cap - n) if s > 0 else (n - 1)
    d = d0 + int(torch.randint(dmax - d0 + 1, (1,), generator=rng))
    return a, s * d


def render_capacity(min_sep=MIN_SEP):
    """Max N at which min-sep placement is honored (the scene.py packing formula)."""
    return int(0.55 * (MARGIN_HI - MARGIN_LO) ** 2 / (torch.pi * (min_sep / 2) ** 2))


def print_render_capacity(n_max=64, empirical_b=16, seed=0):
    """Formula capacity + an EMPIRICAL convergence check at battery-max density (the
    formula alone over-promises near jamming — the repair loop is the real limit)."""
    cap = render_capacity()
    for n in range(1, n_max + 1):
        if n > cap:
            print(f"RENDER_CAP N={n} EXCEEDS capacity {cap} -- battery must annotate",
                  flush=True)
    rng = torch.Generator().manual_seed(seed)
    pos = _sample_positions3(empirical_b, n_max, rng)
    resid = residual_violations(pos)
    print(f"RENDER_CAP capacity={cap} (min_sep={MIN_SEP}) battery_max={n_max} "
          f"honored={'YES' if cap >= n_max else 'NO'} "
          f"empirical_residuals@N={n_max}: {int(resid.sum())}/{empirical_b} scenes "
          f"(nonzero => battery annotates those N)", flush=True)
    return cap


def _sample_positions3(B, K, rng):
    """(B,K,2) centers in [MARGIN_LO,MARGIN_HI]^2 with pairwise distance ≥ MIN_SEP.
    scene.py's vectorised lower-triangular Gauss-Seidel repair, CPU-generator-driven."""
    lo, hi = MARGIN_LO, MARGIN_HI
    pos = torch.rand(B, K, 2, generator=rng) * (hi - lo) + lo
    if K < 2 or K > render_capacity():
        return pos
    tri = torch.tril(torch.ones(K, K, dtype=torch.bool), diagonal=-1).unsqueeze(0)
    for r in range(1024):                       # K=64 runs at ~83% of jamming capacity —
        d = torch.cdist(pos, pos)               # needs far more rounds than scene.py's
        bad = ((d < MIN_SEP) & tri).any(dim=2)  # K=20 regime; CPU tensors, checks cheap
        if not bool(bad.any()):
            break
        fresh = torch.rand(B, K, 2, generator=rng) * (hi - lo) + lo
        pos = torch.where(bad.unsqueeze(-1), fresh, pos)
    return pos


def residual_violations(pos, min_sep=MIN_SEP):
    """(B,) count of centers still violating min-sep — the honest-annotation hook: the
    battery must annotate any N whose renders report nonzero residuals (spec §2.1-B)."""
    K = pos.shape[1]
    if K < 2:
        return torch.zeros(pos.shape[0], dtype=torch.long)
    d = torch.cdist(pos, pos) + torch.eye(K).unsqueeze(0) * 10.0
    return ((d < min_sep).any(dim=2)).sum(dim=1)


def sample_scene3(counts, rng, device, area_norm=False):
    """counts (B,) long -> (B,64,64) float32. Equal-integral blobs: intensity_k =
    A_BLOB·(1+jitter)/(2π σ_px²) so each blob deposits ≈ A_BLOB ink; peak ≤ 1 at SIG_LO."""
    counts = counts.long().cpu()
    B = counts.shape[0]
    K = max(int(counts.max()), 1)
    pos = _sample_positions3(B, K, rng)
    sig = torch.rand(B, K, generator=rng) * (SIG_HI - SIG_LO) + SIG_LO
    jit = 1.0 + (torch.rand(B, K, generator=rng) * 2 - 1) * A_JITTER
    sig_px = sig * CANVAS
    inten = A_BLOB * jit / (2 * torch.pi * sig_px ** 2)
    present = (torch.arange(K).view(1, K) < counts.view(B, 1)).float()
    canvas = render_blobs(pos, sig.log(), inten * present, CANVAS, device="cpu")
    canvas = canvas.squeeze(1)
    if area_norm:
        sums = canvas.flatten(1).sum(1).clamp(min=1e-6)
        canvas = canvas * (A_NORM_TOTAL / sums).view(B, 1, 1)
    return canvas.to(device)


def valid_pairs_for(n):
    """Offset pairs legal at N=n: both counts ≥ 1 (offsets are nonzero so the three counts
    {n+a, n+b, n} are automatically distinct once ≥1-clamping is avoided by construction)."""
    return [(a, b) for (a, b) in OFFSET_PAIRS if n + a >= 1 and n + b >= 1]


def choose_pair3(n, rng, near_miss=False):
    """Per-lane offset-pair draw (m40 A2.2). near_miss=False: uniform over the valid subset —
    the legacy law, RNG-stream byte-identical to the historical inline randint (the in-flight
    m39 battery depends on it). near_miss=True: NEAR_MISS_WEIGHTS-weighted draw, renormalized
    over the valid subset (torch.multinomial normalizes internally)."""
    pairs = valid_pairs_for(n)
    if not near_miss:
        return pairs[int(torch.randint(len(pairs), (1,), generator=rng))]
    w = torch.tensor([NEAR_MISS_WEIGHTS[OFFSET_PAIRS.index(p)] for p in pairs])
    return pairs[int(torch.multinomial(w, 1, generator=rng))]


def sample_candidates3(counts, rng, device, area_norm=False, near_miss=False,
                       mix42_caps=None):
    """counts (B,) -> (cands (B,3,64,64), tgt (B,), cand_ns (B,3)). Fresh renders (never the
    scene layout); per-lane valid pair choice + per-lane order shuffle via the CPU generator.
    cand_ns (B,3) long on device = the SHUFFLED candidate-count triple (M3.3 ⑥ teacher's
    distance base; bench-side data, never an agent input). near_miss (m40 A2): weighted pair
    choice via choose_pair3 — the task-design lever; False = the pre-m40 uniform draw.
    mix42_caps (m42 §2.2): None = legacy/near_miss laws (code path and RNG stream
    byte-identical); else a (B,) long tensor of per-trial far-count caps (taught 24 /
    exam 40) switching the pair draw to choose_pair42. Mutually exclusive with near_miss."""
    assert mix42_caps is None or not near_miss, "mix42 and near_miss are exclusive"
    counts = counts.long().cpu()
    B = counts.shape[0]
    trip = torch.empty(B, 3, dtype=torch.long)
    if mix42_caps is not None:
        caps = mix42_caps.long().cpu()
        assert caps.shape == counts.shape, "mix42_caps must align with counts"
        for b in range(B):                   # host loop is bench-side, B ≤ a few hundred
            n = int(counts[b])
            a, o = choose_pair42(n, rng, int(caps[b]))
            trip[b] = torch.tensor([n + a, n + o, n])
    else:
        for b in range(B):                   # host loop is bench-side, B ≤ a few hundred
            n = int(counts[b])
            a, o = choose_pair3(n, rng, near_miss=near_miss)
            trip[b] = torch.tensor([n + a, n + o, n])
    perm = torch.argsort(torch.rand(B, 3, generator=rng), dim=1)
    trip = torch.gather(trip, 1, perm)
    tgt = (perm == 2).float().argmax(1)      # where the true-N slot landed
    flat = sample_scene3(trip.reshape(-1), rng, device, area_norm=area_norm)
    return flat.view(B, 3, CANVAS, CANVAS), tgt.long().to(device), trip.to(device)
