# symemerge/numcode/data.py
"""数据构造 (spec §4): 留出区间, Δ 先行抽样, 六任务族参数/标签, 候选集, GRPO 组包.

场景生成沿用 `symemerge/varg/nzpairs.py` 的 Render(N, Z),
即 m51 场景法则的显式因子化. 本模块只做抽样记账, 无可学参数.

留出区间 (params.md §5, 在 K_MAX=128 上定, 全课程防火墙): 内插空洞 {45..56} {83..94},
外推段 {113..128}. 任何损失/奖励/掩码画布/候选集都不含留出 N 的场景; 任务答案与 θ
的值域覆盖全域 (登记读法: 防火墙拦的是场景, 不是读头命名这些数的能力).

所有随机性来自调用方的 CPU generator (量具台架的 CRN 纪律); 场景种子从该 generator
派生后交给 nzpairs.draw_z.
"""
import torch

from ..varg import nzpairs as NZ
from . import geometry as GM

K = GM.K_MAX                     # 128
SCENE_SIDE = GM.SIDE             # 128: 与画布同边长, 共享主干单一输入几何 (params.md §3)
HOLE_A = (45, 56)                # 内插空洞 (含端点)
HOLE_B = (83, 94)
EXTRAP = (113, 128)              # 外推段 (含端点)

# Δ 分布 (spec §4.2 重度偏向 ±1,±2; 本分支登记值), 符号均匀
DELTA_ABS = (1, 2, 3, 4)
DELTA_P = (0.60, 0.25, 0.10, 0.05)

N_HARD, N_RAND = 2, 3            # spec §4.3: 硬负 1-2 取 2 (两轴交替), 随负 3-5 取 3
N_CAND = 1 + N_HARD + N_RAND     # 6

# 任务头类数 (params.md §6): 一次定死不随课程改形
TASK_NCLS = {"t1": 2, "t2": 2, "t3": K, "t4": 7, "t6": K + 3}
P_CHOICES = (3, 5, 7)
M_CHOICES = (1, 2, 3)


def band_of(n):
    assert 1 <= n <= K, n
    if HOLE_A[0] <= n <= HOLE_A[1] or HOLE_B[0] <= n <= HOLE_B[1]:
        return "hole"
    if EXTRAP[0] <= n <= EXTRAP[1]:
        return "extrap"
    return "train"


_TRAIN_CACHE = {}


def train_ns(k):
    """课程档 k 的训练域 (升序 tuple): {1..k} 里 band 为 train 的 N."""
    assert 1 <= k <= K, k
    if k not in _TRAIN_CACHE:
        _TRAIN_CACHE[k] = tuple(n for n in range(1, k + 1) if band_of(n) == "train")
    return _TRAIN_CACHE[k]


def hole_ns(k=K):
    return tuple(n for n in range(1, k + 1) if band_of(n) == "hole")


def extrap_ns():
    return tuple(range(EXTRAP[0], EXTRAP[1] + 1))


def _seed(rng):
    return int(torch.randint(0, 2 ** 60, (1,), generator=rng))


def _choice(rng, seq):
    return seq[int(torch.randint(0, len(seq), (1,), generator=rng))]


def sample_delta(rng):
    """有符号 Δ: |Δ| 按 DELTA_P, 符号均匀."""
    u = float(torch.rand(1, generator=rng))
    acc = 0.0
    mag = DELTA_ABS[-1]
    for a, p in zip(DELTA_ABS, DELTA_P):
        acc += p
        if u < acc:
            mag = a
            break
    sign = 1 if float(torch.rand(1, generator=rng)) < 0.5 else -1
    return sign * mag


def sample_n(rng, k):
    return _choice(rng, train_ns(k))


def sample_delta_in(rng, k, n, max_tries=64):
    """给定 n, 抽一个使 n+Δ 仍在训练域的 Δ (spec §4.2: 先 Δ 后 N 的配对语义)."""
    tns = set(train_ns(k))
    for _ in range(max_tries):
        d = sample_delta(rng)
        if (n + d) in tns:
            return d
    raise RuntimeError(f"no legal delta at n={n} k={k}")


def sample_theta(rng, k):
    """六任务参数 θ (组内冻结, spec §6.4). τ ~ U{1..k-1} 使任务1/3 在当前档非平凡."""
    assert k >= 2
    tau = 1 + int(torch.randint(0, k - 1, (1,), generator=rng))
    return dict(tau=tau, p=_choice(rng, P_CHOICES), m=_choice(rng, M_CHOICES))


def targets(n, theta):
    """六任务标签 -> 类索引 dict (t5 由候选集真值位给出, 不在此).
    t3 答案 max(n,τ) ∈ {1..K} -> 类 idx 值-1; t6 答案 n+m ∈ {2..K+3} -> idx 值-1."""
    assert 1 <= n <= K
    return dict(
        t1=int(n > theta["tau"]),
        t2=n % 2,
        t3=max(n, theta["tau"]) - 1,
        t4=n % theta["p"],
        t6=n + theta["m"] - 1,
    )


# ------------------------------------------------------------------ 场景与候选集
def render_scene(rng, n, budget_scale=1.0, max_tries=8):
    """独立 Z 的一张场景 -> (side, side) float32. 落位耗尽则换 Z 重试."""
    for _ in range(max_tries):
        z = NZ.draw_z(_seed(rng), SCENE_SIDE)
        try:
            return NZ.render_nz(z, n, budget_scale=budget_scale)["scene"]
        except RuntimeError:
            continue
    raise RuntimeError(f"render_scene exhausted at n={n}")


def sample_scene_batch(rng, k, B):
    """Phase 1 / 计数损失用: (B, side, side) 场景 + (B,) long 真值 N, N 均匀训练域."""
    ns = [sample_n(rng, k) for _ in range(B)]
    scenes = torch.stack([render_scene(rng, n) for n in ns])
    return scenes, torch.tensor(ns, dtype=torch.long)


def _sample_group_once(rng, k, n=None, with_scene=True):
    n = sample_n(rng, k) if n is None else int(n)
    zseed = _seed(rng)
    z = NZ.draw_z(zseed, SCENE_SIDE)
    # 查询场景 x1 = (N, Z); with_scene=False (预测子阶段池抽笔记的候选包) 只留 Z 给硬负, 不渲场景
    scene = NZ.render_nz(z, n)["scene"] if with_scene else None
    cands, kinds, ns = [], [], []
    zp = NZ.draw_z(_seed(rng), SCENE_SIDE)
    cands.append(NZ.render_nz(zp, n)["scene"])                 # 正确项 (N, Z') 原图
    kinds.append("pos")
    ns.append(n)
    d1 = sample_delta_in(rng, k, n)                            # 硬负 A: 面积配平
    cands.append(NZ.render_nz(z, n + d1)["scene"])             # 预算逐比特共享
    kinds.append("hard_area")
    ns.append(n + d1)
    d2 = sample_delta_in(rng, k, n)                            # 硬负 B: 周长一阶配平
    n2 = n + d2
    cands.append(NZ.render_nz(z, n2, budget_scale=n / n2)["scene"])
    kinds.append("hard_perim")
    ns.append(n2)
    for _ in range(N_RAND):                                    # 随负: 独立 (N'', Z'')
        nr = n
        while nr == n:
            nr = sample_n(rng, k)
        cands.append(render_scene(rng, nr))
        kinds.append("rand")
        ns.append(nr)
    perm = torch.randperm(N_CAND, generator=rng).tolist()      # 位置不许泄真值
    cands = [cands[i] for i in perm]
    kinds = [kinds[i] for i in perm]
    ns = [ns[i] for i in perm]
    theta = sample_theta(rng, k)
    return dict(n=n, scene=scene, theta=theta, targets=targets(n, theta),
                cands=torch.stack(cands), cand_ns=tuple(ns), cand_kinds=tuple(kinds),
                truth5=kinds.index("pos"), zseed=zseed)


def sample_group(rng, k, max_tries=8, n=None, with_scene=True):
    """一个 GRPO 组的静态部分 (spec §6.4): 共享场景 x1, 组内冻结的 θ 与候选集.
    G 条 rollout 由训练器在同一份上采样. 落位耗尽整包重抽. n 给定 = 指定数量的组
    (v4.1 M1 逐数量 CRN 池 / M6 首见池用; 数量必须在训练域内).
    with_scene=False: 只要 θ/候选集/标签 (预测子阶段池抽笔记的候选包, spec-v2 §4.2), scene=None."""
    if n is not None:
        assert n in train_ns(k), (n, k)
    for _ in range(max_tries):
        try:
            return _sample_group_once(rng, k, n, with_scene)
        except RuntimeError:
            continue
    raise RuntimeError(f"sample_group exhausted at k={k}")


# ------------------------------------------------------------------ 加法流 ([U] 2026-08-27: 输入两个场景 N 与 m 个物体, m = 1–3, 输出一张纸)
_ADD_CACHE = {}


def add_pairs(k):
    """课程档 k 的合法 (N, m) 对 (升序 tuple): m ∈ M_CHOICES, N / m / N+m 皆在训练域 (留出防火墙: 场景与标签都不碰留出 N)."""
    if k not in _ADD_CACHE:
        tns = set(train_ns(k))
        _ADD_CACHE[k] = tuple((n, m) for m in M_CHOICES for n in train_ns(k) if (n + m) in tns and m in tns)
    return _ADD_CACHE[k]


def add_sums(k):
    """合法 (N, m) 对的和 s = N+m 的集合 (升序 tuple; 评测集按 s 分层)."""
    return tuple(sorted({n + m for n, m in add_pairs(k)}))


def sample_add_group(rng, k, n_a=None, m=None, max_tries=8):
    """加法流一项: m ~ U(M_CHOICES), N ~ U{N ∈ 训练域: N+m ∈ 训练域}; 真值 s = N+m.
    候选包/θ/标签 = _sample_group_once(n=s, with_scene=False) 同构造器 (与池抽笔记的候选包同式: 正确项 (s, Z') / 硬负 (s±Δ, Z) / 随负);
    两张场景 scene = Render(N, Z1) 与 scene_m = Render(m, Z2), Z1/Z2 各自独立. 标签只由 s 定 (targets(s, θ)), N 与 m 只在元数据."""
    if m is None:
        m = _choice(rng, M_CHOICES)
    if n_a is None:
        tns = set(train_ns(k))
        ok = [n for n in train_ns(k) if (n + m) in tns and m in tns]     # 与 add_pairs 同条件 (k<3 时某些 m 无合法 N)
        n_a = _choice(rng, ok)
    n_a, m = int(n_a), int(m)
    assert (n_a, m) in add_pairs(k), (n_a, m, k)
    s = n_a + m
    for _ in range(max_tries):
        try:
            g = _sample_group_once(rng, k, s, with_scene=False)
            g["scene"] = render_scene(rng, n_a)
            g["scene_m"] = render_scene(rng, m)
            g["n_a"], g["m"] = n_a, m
            return g
        except RuntimeError:
            continue
    raise RuntimeError(f"sample_add_group exhausted at k={k} n_a={n_a} m={m}")
