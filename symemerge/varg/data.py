# symemerge/varg/data.py
"""VAR-GROUNDING bench adapter (spec 2026-08-03 SS2-3).

The world (`symemerge/world/scene4.py`, `shapes.py`) is used AS-IS, BYTE-UNTOUCHED and
imported read-only (pin 1). This module adds the ONE thing the world does not expose:
per-object PLACEMENTS, so a nested edit pair can be cut out of ONE placement.

`sample_placements` is a faithful mirror of `scene4.sample_scene4`'s per-scene body and
`_place_all_rec` of `scene4._place_all` -- same law, same rng draw ORDER, same arithmetic.
It records (object -> y0, x0, mask, intensity) instead of compositing. The equivalence is
PINNED (`test_bench_equivalence`): compositing every recorded object reproduces
`sample_scene4` bit for bit under the same seed.

Why subsetting is exact (pin 2): `shapes.scale_mask` returns a BINARY {0,1} mask, and
placement keeps every pair of masks disjoint after a 1 px dilation -- no pixel belongs to
two objects. Max-compositing a subset therefore yields exactly the full scene's pixels on
those objects and zero elsewhere.
"""
import hashlib
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from ..world import scene4, shapes

# ---------------------------------------------------------------- registered dials (SS3)
BAND_TRAIN = (1, 15)          # result count trained normally
BAND_EXTRAP = (16, 20)        # zero training signal of any kind; evaluated only
DELTA_EDIT_MIN = 0            # delta = 0 ("copy the source") IS trained -- required by
                              # spec SS6's M6 null, which uses "+13 then +0" as the no-op
                              # control. An untrained delta=0 would make that null measure
                              # an out-of-distribution instruction instead of second-pass
                              # perturbation. Registered addition.
DELTA_EDIT_MAX = 8            # |delta| for edit episodes (registered; M6 uses +5)
P_ABSOLUTE = 0.5              # 50% absolute / 50% edit (SS3)
INK_RESCALE_LO, INK_RESCALE_HI = 0.6, 1.4      # SS3(b) ink-rescaled control
SIDE_CANDIDATES = (64, 96, 128)
PATCH = {64: 4, 96: 6, 128: 8}   # side / PATCH == 16 in all cases (SS2)
# 128 是 VARG-NA 单元新增 (spec 2026-08-05 SS7)。**继承常数重推, 两条后果必须记住:**
#   (a) 每格从 4x4 px 变成 8x8 px, 所以 GAP_PX=2 从 "半个格子" 变成 "1/4 格子" --
#       与边长 64 / gap 1 同比例, 那个条件下 T0 门是 0.9780 (不是 1.0000), 且 lambda_gap=5
#       是必需项 (lambda=0 的两个梯子实测 0.6075 / 0.5795)。
#   (b) MIN_AREA=9.0 是绝对值不随边长缩放, 所以最小物体 3x3 px 对 8x8 格子 = 线性比 0.375
#       (边长 64 上是 0.75)。161 个物体铺在 256 个格子上 -> 平均 0.63 个/格, 网格分不开。
#       这是 spec SS7.1 记的问题, 由 scripts/na_frontend_probe.py 实测。

# ---------------------------------------------------------------------------------------
# BRANCH-LOCAL BENCH AMENDMENT -- INTER-OBJECT GAP (user ruling 2026-08-03, Option 3).
#
# `symemerge/world/scene4.py` is STILL BYTE-UNTOUCHED (pin 1). This constant is applied
# only by this module's placement mirror; the mirror's bit-for-bit equivalence with
# `scene4.sample_scene4` is still pinned, at the WORLD's own gap (scene4.DILATE_PX = 1).
#
# WHY, measured. T0 run 1 FAILED at both canvas sides (side 64 exact 0.5845, side 96
# 0.4980, against a bar of 0.90 x CEIL_tau 1.0000). Diagnosis, all measured:
#   * error is 100% UNDER-count (over-count fraction 0.000); ink is preserved
#     (recon/bench 1.0046) while thresholded area GROWS 4% -- objects do not vanish, the
#     reconstruction fattens them until a 1 px gap is WELDED shut;
#   * split by the scene's minimum inter-object gap at fixed N (8..15): a scene with any
#     pair <=2 px apart scores 0.4544, a scene with none scores 1.0000 -- exactly;
#   * corr(number of tight pairs, under-count) +0.7047 vs corr(N, under-count) +0.5345;
#   * the spec's anticipated culprit, MIN_AREA, was REFUTED: at fixed N, accuracy does not
#     improve with larger objects (N=3: 0.971 / 0.971 / 0.818 across min-area tertiles).
#     The size-decile table looked like a size effect only because a budget independent of
#     N makes "small objects" and "many objects" the same statement;
#   * raising tau does not cut the weld (best 0.707 at tau 0.15, where the bench itself
#     has already fallen to 0.9295).
# DILATE_PX = 1 was inherited from the m51 DISCRIMINATION bench, where its only job was to
# keep connected components == true object count. It was never a generation constraint. It
# is now measured to bind the INSTRUMENT, not the model.
#
# MEASURED EFFECT of this amendment, with the gap-1-trained tokenizer, no retraining:
#   gap=1  placement failures 0/400  recon exact 0.5900 (train band 0.7000, N=20 0.250)
#   gap=2  placement failures 0/400  recon exact 0.9900 (train band 0.9933, N=20 0.950)
#   gap=3  placement failures 5/400  recon exact 1.0000 (train band 1.0000, N=20 1.000)
# gap=2 is taken: it clears the bar with the largest placement headroom.
#
# BINDING ON EVERY CITATION: this branch's world separates objects by >= 2 px, not the
# m51 bench's 1 px. Any comparison to a main-line number must state it.
GAP_PX = 2
LAYOUT_RESTARTS_AMENDED = 32   # only when GAP_PX != scene4.DILATE_PX (tighter packing)

# HARDENING of pin 8 beyond the spec's letter, registered here: the spec fences the RESULT
# count at 15, which would still let a -delta training pair SHOW the model a 23-object
# source image. Both endpoints of every training pair are capped at 15 instead. Strictly
# stronger than the registered firewall; asserted in the training loop.
TRAIN_ENDPOINT_MAX = BAND_TRAIN[1]


# ------------------------------------------------------------------------ the mask pool
def get_pool(seed=0, size=shapes.POOL_SIZE, cache_dir="outputs"):
    """The m51 shape pool, disk-cached. Content is IDENTICAL to `shapes.mask_pool(seed,
    size)` (a fixed seeded texture atlas); the cache only avoids paying ~1 min of
    generation in every DataLoader worker. The loaded pool is installed into
    `shapes._POOL_CACHE` so `scene4.sample_scene4` sees the very same tensor (required by
    the equivalence pin)."""
    key = (seed, size)
    if key in shapes._POOL_CACHE:
        return shapes._POOL_CACHE[key]
    path = os.path.join(cache_dir, f"varg_pool_{seed}_{size}.pt")
    if os.path.exists(path):
        pool = torch.load(path, map_location="cpu", weights_only=True)
        shapes._POOL_CACHE[key] = pool
        return pool
    pool = shapes.mask_pool(seed, size)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(pool, path)
    except OSError:
        pass
    return pool


def pool_hash(seed=0, size=shapes.POOL_SIZE, cache_dir="outputs"):
    p = get_pool(seed, size, cache_dir)
    return hashlib.sha1(p.numpy().tobytes()).hexdigest()[:16]


# ------------------------------------------------------- placement mirror (bit-equivalent)
def _dilate_by(m, gap):
    """`scene4._dilate` generalised to an arbitrary ring width; identical output at
    gap == scene4.DILATE_PX."""
    if gap == scene4.DILATE_PX:
        return scene4._dilate(m)
    k = 2 * gap + 1
    p = F.pad(m[None, None].float(), (gap,) * 4)
    return F.max_pool2d(p, k, stride=1, padding=gap)[0, 0] > 0.5


def _place_all_rec(rec, masks, intens, side, rng, gap):
    """Mirror of `scene4._place_all` that RECORDS placements instead of compositing, with
    the inter-object ring width as a parameter (GAP_PX amendment above).

    At `gap == scene4.DILATE_PX == 1` this is BIT-IDENTICAL to the original: the position
    draw `randint(gap, side - h - gap + 1)` reduces to `randint(1, side - h)`, the
    footprint bound to `h > side - 2`, and `_dilate_by` delegates to `scene4._dilate`. The
    canvas write in the original influences neither `occ` nor the generator, so dropping it
    is bit-safe. Pinned by `test_pin2_bench_equivalence_bit_identical`."""
    order = sorted(range(len(masks)), key=lambda k: -float(masks[k].sum()))
    occ = torch.zeros(side, side, dtype=torch.bool)
    for k in order:
        m = masks[k] > 0.5
        h, w = m.shape
        if h > side - 2 * gap or w > side - 2 * gap:
            raise RuntimeError(f"footprint {h}x{w} exceeds canvas {side} at gap {gap}")
        placed = False
        for _ in range(scene4.PLACE_TRIES):
            y0 = int(torch.randint(gap, side - h - gap + 1, (1,), generator=rng))
            x0 = int(torch.randint(gap, side - w - gap + 1, (1,), generator=rng))
            if not bool((occ[y0:y0 + h, x0:x0 + w] & m).any()):
                occ[y0 - gap:y0 + h + gap, x0 - gap:x0 + w + gap] |= _dilate_by(m, gap)
                rec.append((k, y0, x0))
                placed = True
                break
        if not placed:
            return False
    return True


def sample_placements(n, rng, side, pool_seed=0, pool_size=shapes.POOL_SIZE, gap=None):
    """One scene's PLACEMENT under the m51 law -> list of n dicts (in object order 0..n-1)
    {mask (h,w) float {0,1}, y0, x0, intens}. Mirrors `scene4.sample_scene4`'s per-scene
    body draw for draw. `gap` = inter-object ring width; None => the branch's amended
    GAP_PX. Pass `gap=scene4.DILATE_PX` for the bit-equivalence check."""
    assert n >= 1
    gap = GAP_PX if gap is None else gap
    restarts = (scene4.LAYOUT_RESTARTS if gap == scene4.DILATE_PX
                else LAYOUT_RESTARTS_AMENDED)
    scale = (side / scene4.CANVAS_BASE) ** 2
    assert scene4.A_LO * scale >= scene4.MIN_AREA * n * 1.1, \
        f"N={n} infeasible at side {side} (clamp-free certification, spec SS2)"
    pool = get_pool(pool_seed, pool_size)
    log_lo, log_hi = math.log(scene4.A_LO * scale), math.log(scene4.A_HI * scale)
    A = math.exp(float(torch.rand(1, generator=rng)) * (log_hi - log_lo) + log_lo)
    bright = float(torch.rand(1, generator=rng)) * \
        (scene4.BRIGHT_HI - scene4.BRIGHT_LO) + scene4.BRIGHT_LO
    shares = scene4._dirichlet(n, rng)
    areas = scene4.MIN_AREA + (A - scene4.MIN_AREA * n) * shares
    jit = torch.rand(n, generator=rng) * (scene4.JIT_HI - scene4.JIT_LO) + scene4.JIT_LO
    intens = bright * jit
    idx = torch.randint(pool.shape[0], (n,), generator=rng)
    masks = [shapes.scale_mask(pool[int(i)], float(a)) for i, a in zip(idx, areas)]
    for _ in range(restarts):
        rec = []
        if _place_all_rec(rec, masks, intens, side, rng, gap):
            break
    else:
        raise RuntimeError(f"placement exhausted at N={n} side={side} gap={gap}")
    out = [None] * n
    for k, y0, x0 in rec:
        out[k] = dict(mask=masks[k], y0=y0, x0=x0, intens=float(intens[k]))
    assert all(o is not None for o in out)
    return out


def composite(pl, keep, side):
    """Render the object subset `keep` (iterable of indices into `pl`) -> (side, side)."""
    canv = torch.zeros(side, side)
    for k in keep:
        p = pl[k]
        h, w = p["mask"].shape
        y0, x0 = p["y0"], p["x0"]
        region = canv[y0:y0 + h, x0:x0 + w]
        canv[y0:y0 + h, x0:x0 + w] = torch.maximum(region, p["mask"] * p["intens"])
    return canv


# ------------------------------------------------------------------------ episodes (SS3)
def _draw_edit_counts(rng, lo, hi, endpoint_max):
    """(n_src, n_tgt) for one edit episode: result count in [lo, hi], |delta| in
    [DELTA_EDIT_MIN, DELTA_EDIT_MAX], BOTH endpoints in [1, endpoint_max]."""
    n_tgt = int(torch.randint(lo, hi + 1, (1,), generator=rng))
    while True:
        d = int(torch.randint(DELTA_EDIT_MIN, DELTA_EDIT_MAX + 1, (1,), generator=rng))
        feas = []
        if n_tgt - d >= 1:
            feas.append(n_tgt - d)
        if n_tgt + d <= endpoint_max:
            feas.append(n_tgt + d)
        if feas:
            j = int(torch.randint(len(feas), (1,), generator=rng))
            return feas[j], n_tgt


def sample_episode(rng, side, band="train", kind=None, pool_seed=0,
                   endpoint_max=None, gap=None):
    """One episode -> dict of CPU tensors. `kind`: "abs" | "edit" | None (drawn).

    `gap` is passed explicitly (rather than read from the module global inside the worker)
    so that a run which varies GAP_PX -- cell A's A1 at gap 1 vs A2 at gap 2 -- does not
    depend on DataLoader workers inheriting a mutated global through fork."""
    lo, hi = BAND_TRAIN if band == "train" else BAND_EXTRAP
    if endpoint_max is None:
        endpoint_max = TRAIN_ENDPOINT_MAX if band == "train" else BAND_EXTRAP[1]
    if kind is None:
        kind = "abs" if float(torch.rand(1, generator=rng)) < P_ABSOLUTE else "edit"
    if kind == "abs":
        n_tgt = int(torch.randint(lo, hi + 1, (1,), generator=rng))
        pl = sample_placements(n_tgt, rng, side, pool_seed, gap=gap)
        tgt = composite(pl, range(n_tgt), side)
        src = torch.zeros(side, side)
        n_src = 0
    else:
        n_src, n_tgt = _draw_edit_counts(rng, lo, hi, endpoint_max)
        n_full = max(n_src, n_tgt)
        n_small = min(n_src, n_tgt)
        pl = sample_placements(n_full, rng, side, pool_seed, gap=gap)
        perm = torch.randperm(n_full, generator=rng).tolist()
        small = perm[:n_small]
        big = list(range(n_full))
        if n_src < n_tgt:
            src, tgt = composite(pl, small, side), composite(pl, big, side)
        else:
            src, tgt = composite(pl, big, side), composite(pl, small, side)
    return dict(src=src[None], tgt=tgt[None],
                delta=torch.tensor(float(n_tgt - n_src)),
                n_src=torch.tensor(n_src), n_tgt=torch.tensor(n_tgt),
                kind=torch.tensor(0 if kind == "abs" else 1))


class EpisodeStream(IterableDataset):
    """On-the-fly episode generator (SS3 data plumbing). Each DataLoader worker gets its
    own seeded CPU generator; the per-worker seeds are derived from `seed` and recorded."""

    def __init__(self, side, seed, band="train", pool_seed=0, gap=None):
        self.side, self.seed, self.band, self.pool_seed = side, seed, band, pool_seed
        self.gap = GAP_PX if gap is None else gap      # resolved in the PARENT process

    def worker_seed(self, wid):
        return self.seed * 9973 + wid * 101 + 7

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else int(info.id)
        rng = torch.Generator().manual_seed(self.worker_seed(wid))
        get_pool(self.pool_seed)                       # warm the cache once per worker
        while True:
            yield sample_episode(rng, self.side, self.band, pool_seed=self.pool_seed,
                                 gap=self.gap)


# ---------------------------------------------------------------- cached evaluation sets
def build_eval_cache(side, seed=1234, n_calib=2000, n_edit=2000, pool_seed=0, gap=None):
    """Pre-generated ONCE with a fixed seed (SS3). `calib`: n_calib bench scenes spread
    evenly over N = 1..20 (the tau-calibration / CEIL_tau / T0-reconstruction / M2-reference
    / M8-probe material). `edit`: n_edit nested train-band edit pairs.

    `gap` selects the inter-object ring width; the cache is material, so a run at a
    different gap MUST NOT reuse another gap's cache (see `cache_path`)."""
    rng = torch.Generator().manual_seed(seed)
    lo, hi = BAND_TRAIN[0], BAND_EXTRAP[1]
    per = n_calib // (hi - lo + 1)
    ns = torch.repeat_interleave(torch.arange(lo, hi + 1), per)
    scenes = torch.zeros(ns.numel(), side, side)
    for i, n in enumerate(ns.tolist()):
        pl = sample_placements(n, rng, side, pool_seed, gap=gap)
        scenes[i] = composite(pl, range(n), side)
    e_src = torch.zeros(n_edit, side, side)
    e_tgt = torch.zeros(n_edit, side, side)
    e_ns = torch.zeros(n_edit, dtype=torch.long)
    e_nt = torch.zeros(n_edit, dtype=torch.long)
    for i in range(n_edit):
        ep = sample_episode(rng, side, "train", kind="edit", pool_seed=pool_seed, gap=gap)
        e_src[i], e_tgt[i] = ep["src"][0], ep["tgt"][0]
        e_ns[i], e_nt[i] = ep["n_src"], ep["n_tgt"]
    cache = dict(side=torch.tensor(side), seed=torch.tensor(seed),
                 calib_scenes=scenes, calib_n=ns,
                 edit_src=e_src, edit_tgt=e_tgt, edit_n_src=e_ns, edit_n_tgt=e_nt)
    return cache


CACHE_KEYS = ("side", "seed", "calib_scenes", "calib_n",
              "edit_src", "edit_tgt", "edit_n_src", "edit_n_tgt")


def cache_hash(cache):
    """Order-fixed sha1 over the cache tensors (pin 9)."""
    h = hashlib.sha1()
    for k in CACHE_KEYS:
        h.update(k.encode())
        h.update(cache[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def cache_path(side, gap=None, out="outputs"):
    """The cache file for (side, gap). The gap is ALWAYS in the filename, including for the
    branch's own GAP_PX. A cache built at gap 2 is different material from one built at gap
    1, and cell A compares a gap-1 run against a gap-2 run: an unsuffixed legacy file whose
    build-time gap is not recorded in its own bytes is exactly the kind of silent mismatch
    that would make that comparison meaningless. Both runs therefore rebuild from the fixed
    seed, and the resulting hash is reported (the gap-2 hash is expected to reproduce the
    locked `284b8a06770581c7` at side 64 -- reported, not asserted, because the legacy file's
    build-time gap is not recorded on disk)."""
    g = GAP_PX if gap is None else gap
    return os.path.join(out, f"varg_cache_s{side}_gap{g}.pt")


# MEASURED 2026-08-04, not assumed: rebuilding the cache at gap 1 from the fixed seed
# reproduced `284b8a06770581c7`, so the session-87 locked caches were built at GAP 1 (they
# predate the 2026-08-03 gap ruling). The parent spec recorded the hash without recording
# which gap it was built at, which is exactly the silent-material-mismatch the gap-suffixed
# filenames now prevent.
LOCKED_CACHE_HASH = {(64, 1): "284b8a06770581c7", (96, 1): "3569dae8a03199f1"}


def load_or_build_cache(side, path, seed=1234, **kw):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    c = build_eval_cache(side, seed=seed, **kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(c, path)
    return c


def eval_set_ink_correlations(cache):
    """SS3(c): the measured ink<->count correlation of EVERY evaluation set, persisted next
    to its numbers. The ABSOLUTE set is where |rho| < 0.15 licenses M1 (pin 11). The EDIT
    set is the REGISTERED CONFOUND -- inside a nested pair total ink is proportional to
    object count, so a large |rho| here is expected and is exactly why M1 is measured on
    absolute episodes only and M5 always carries the ink-rescaled control."""
    ns = cache["calib_n"].float()
    ink = cache["calib_scenes"].flatten(1).sum(1)
    m = cache["calib_n"] <= BAND_TRAIN[1]
    out = dict(absolute_all=spearman(ink, ns),
               absolute_train_band=spearman(ink[m], ns[m]))
    if "edit_src" in cache:
        out["edit_source"] = spearman(cache["edit_src"].flatten(1).sum(1),
                                      cache["edit_n_src"].float())
        out["edit_target"] = spearman(cache["edit_tgt"].flatten(1).sum(1),
                                      cache["edit_n_tgt"].float())
    return out


def ink_rescale(src, rng):
    """SS3(b) control: multiply source intensities by u ~ U[0.6,1.4], clamped to the [0,1]
    pixel range. Returns (rescaled, u, clamped_fraction)."""
    u = torch.rand(src.shape[0], generator=rng) * \
        (INK_RESCALE_HI - INK_RESCALE_LO) + INK_RESCALE_LO
    out = src * u.reshape(-1, *([1] * (src.dim() - 1))).to(src.device)
    clamped = float((out > 1.0).float().mean())
    return out.clamp(max=1.0), u, clamped


def spearman(a, b):
    """Spearman rho between two 1-D tensors (average ranks)."""
    def rank(x):
        x = x.double()
        order = torch.argsort(x)
        r = torch.empty_like(x)
        r[order] = torch.arange(x.numel(), dtype=torch.float64)
        # average ties
        vals, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
        sums = torch.zeros(vals.numel(), dtype=torch.float64).index_add_(0, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a.flatten()), rank(b.flatten())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = (ra.norm() * rb.norm()).clamp(min=1e-12)
    return float((ra * rb).sum() / den)
