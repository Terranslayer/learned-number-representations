# symemerge/varg/nzpairs.py
"""Render(N, Z) 因子化 + 正/硬负样本对 (session 107 探针; 用户 2026-08-11 规格:
正样本 (N,Z) vs (N,Z'), Z 独立重采样 -> 逼码丢弃 Z; 硬负样本 (N,Z) vs (N',Z),
Z 逐比特相同 -> 杀死面积/边缘捷径).

**量具, 无可学参数.** 只 import world 与 varg.data -- 测试台与被测物分离法则.
现行 m51 场景法则 (scene4.sample_scene4 的 data.py 逐比特镜像) 一个字节不动;
本模块只换"抽签的记账方式": 顺序流 -> 显式因子.

因子化 (Z 是四元组, 全部可跨 N 共享):
  A        面积预算标量, LogUniform[A_LO,A_HI]*(side/64)^2  -- 法则原样
  bright   亮度标量, U[BRIGHT_LO, BRIGHT_HI]                -- 法则原样
  obj_seed 逐物体流种子: 第 k 个物体的 (gamma 份额底子, 亮抖, 形状 id) 从
           seed(obj_seed, k) 的独立子流取 -- 前缀稳定: N 只决定读到第几个物体,
           增删物体不使其余物体的抽签移位
  pos_seed 逐物体落位尝试流种子, 同样按 (pos_seed, k, restart) 独立成流

Render(n, Z, budget_scale): shares = 归一化 g_1..g_n (边际 = Dirichlet(ALPHA..),
与 scene4._dirichlet 同法则); areas = MIN_AREA + (A*budget_scale - MIN_AREA*n)*shares
(公式照抄 -- 总目标面积恰好 = 预算, 与 N 无关, 这是 m51 反捷径机制的本体);
落位 = 最大先放 + 膨胀占位盘碰撞 (法则照抄), 尝试用连续 u -> floor(u*K)
(与 randint 同法则), 使同一 u 在掩码尺寸微变时映到近邻位置而非重抽.

"逐比特相同"的诚实边界: 逐物体分量的维度就是 N, 跨 N 的严格比特相同不可定义.
本模块的共享语义 = A, bright 逐比特相同 + 共同前缀物体的形状/亮抖/落位尝试流
逐比特相同; 必然不同的是实现面积 (同一预算摊 n 份 vs n' 份 -- 正是杀总面积捷径
的机制) 及由此的掩码缩放; 落位可能分叉, 来源有二: 掩码尺寸微变使同一 u 映到
近邻格, 以及 n+1 侧占位盘更挤使同一条尝试流走到更后面那次尝试 (审稿补记).
分叉率不作断言, 探针逐对实测 (pair_coupling).

单对之内面积与周长不可同时配平 (n 个同形物体总面积 A 时总周长 ~ sqrt(n*A)):
  match="area"  预算逐比特共享 -> 总面积配平, 总周长仍随 N 升 (合法的 N 线索)
  match="perim" 侧 B 预算乘 n/n' -> 总周长一阶配平, 总面积反向 (故意不一致试次)
负样本集内交替两轴, 任何单一模拟量线索都解不了整个集合.
已有做法 -> 为什么不用: 数量刺激的因子控制是发表过的标准做法 (DeWind et al. 2015,
J Vis, numerosity/size/spacing 正交轴; Gebuis & Reynvoet 2011, Behav Res Methods,
交替一致性控制); 不可直接复用, 因为被配平的是本分支钉死的 m51 不规则形状法则
(预算 Dirichlet 摊分 + 间隙约束), 不是那两家的圆点阵 -- 思想照搬, 实现本地.
"""
import math

import torch
import torch.nn.functional as F

from ..world import scene4, shapes
from . import data as D

ALPHA = scene4.ALPHA
_MOD = 2 ** 62


def draw_z(seed, side, pool_seed=0):
    """一份完整的干扰因子 Z. 抽签顺序镜像 sample_placements 的头两签 (A, bright);
    逐物体与落位部分以两个派生种子的形式存在 -- Z 因此是"无限流"的紧凑表示,
    任何 n 都从同一份 Z 读前 n 项."""
    rng = torch.Generator().manual_seed(int(seed) % _MOD)
    scale = (side / scene4.CANVAS_BASE) ** 2
    log_lo, log_hi = math.log(scene4.A_LO * scale), math.log(scene4.A_HI * scale)
    A = math.exp(float(torch.rand(1, generator=rng)) * (log_hi - log_lo) + log_lo)
    bright = float(torch.rand(1, generator=rng)) * \
        (scene4.BRIGHT_HI - scene4.BRIGHT_LO) + scene4.BRIGHT_LO
    obj_seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=rng))
    pos_seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=rng))
    return dict(A=A, bright=bright, obj_seed=obj_seed, pos_seed=pos_seed,
                side=side, pool_seed=pool_seed)


# 强混合乘数 (splitmix64 一族): 派生流的种子空间 2^62, 跨 Z 撞流是生日概率量级
# (百万流 ~ 1e-6), 而首版加法哈希 (k*1e9+7 / k*9973) 在 5 万次 draw_z 的探针里
# 期望撞出千次量级的偶然共享 (审稿 Minor#1, 2026-08-11) -- 统计上淹没在噪声下,
# 但没有理由留着它.
_PHI = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_RST = 0xD6E8FEB86659FD93


def _obj_rng(obj_seed, k):
    return torch.Generator().manual_seed(
        (obj_seed * _PHI + (k + 1) * _MIX) % _MOD)


def _pos_rng(pos_seed, k, restart):
    return torch.Generator().manual_seed(
        (pos_seed * _PHI + (k + 1) * _MIX + (restart + 1) * _RST) % _MOD)


def _obj_draws(z, n):
    """物体 0..n-1 的流读数 -> (u_gam (ALPHA,n), jit (n,), idx (n,) long).
    每个物体一个独立子流: 前缀稳定的机制本体."""
    pool = D.get_pool(z["pool_seed"])
    ug, jit, idx = [], [], []
    for k in range(n):
        rng = _obj_rng(z["obj_seed"], k)
        ug.append(torch.rand(ALPHA, generator=rng))
        jit.append(float(torch.rand(1, generator=rng))
                   * (scene4.JIT_HI - scene4.JIT_LO) + scene4.JIT_LO)
        idx.append(int(torch.randint(pool.shape[0], (1,), generator=rng)))
    return (torch.stack(ug, dim=1), torch.tensor(jit),
            torch.tensor(idx, dtype=torch.long))


def _place_streams(masks, side, gap, pos_seed, restart):
    """镜像 data._place_all_rec 的接受法则 (最大先放, 膨胀占位盘碰撞), 但每个物体的
    尝试来自自己的流. 连续 u -> gap + floor(u*(side-h-2*gap+1)): 支撑与
    randint(gap, side-h-gap+1) 相同, 且同一 u 在 h 微变时映到近邻位置."""
    order = sorted(range(len(masks)), key=lambda k: -float(masks[k].sum()))
    occ = torch.zeros(side, side, dtype=torch.bool)
    rec = []
    for k in order:
        m = masks[k] > 0.5
        h, w = m.shape
        if h > side - 2 * gap or w > side - 2 * gap:
            raise RuntimeError(f"footprint {h}x{w} exceeds canvas {side} at gap {gap}")
        rng_k = _pos_rng(pos_seed, k, restart)
        placed = False
        for _ in range(scene4.PLACE_TRIES):
            u = torch.rand(2, generator=rng_k)
            y0 = gap + int(float(u[0]) * (side - h - 2 * gap + 1))
            x0 = gap + int(float(u[1]) * (side - w - 2 * gap + 1))
            if not bool((occ[y0:y0 + h, x0:x0 + w] & m).any()):
                occ[y0 - gap:y0 + h + gap, x0 - gap:x0 + w + gap] |= \
                    D._dilate_by(m, gap)
                rec.append((k, y0, x0))
                placed = True
                break
        if not placed:
            return None
    return rec


def render_nz(z, n, gap=None, budget_scale=1.0):
    """Render(n, Z) -> dict(scene (side,side), placements, areas 目标面积 (n,),
    jit (n,), idx (n,), A_eff, restart). 公式逐行对应 data.sample_placements."""
    side = z["side"]
    assert n >= 1
    gap = D.GAP_PX if gap is None else gap
    scale = (side / scene4.CANVAS_BASE) ** 2
    assert scene4.A_LO * scale >= scene4.MIN_AREA * n * 1.1, \
        f"N={n} infeasible at side {side} (法则自己的可行门)"
    A_eff = z["A"] * float(budget_scale)
    assert A_eff >= scene4.MIN_AREA * n * 1.1, \
        f"N={n} infeasible at A_eff={A_eff:.0f} (budget_scale 把预算压穿了地板)"
    u_gam, jit, idx = _obj_draws(z, n)
    g = -(u_gam.clamp(min=1e-12).log()).sum(0)            # scene4._dirichlet 的列式
    shares = g / g.sum()
    areas = scene4.MIN_AREA + (A_eff - scene4.MIN_AREA * n) * shares
    intens = z["bright"] * jit
    pool = D.get_pool(z["pool_seed"])
    masks = [shapes.scale_mask(pool[int(i)], float(a)) for i, a in zip(idx, areas)]
    restarts = (scene4.LAYOUT_RESTARTS if gap == scene4.DILATE_PX
                else D.LAYOUT_RESTARTS_AMENDED)
    rec, used = None, 0
    for r in range(restarts):
        rec = _place_streams(masks, side, gap, z["pos_seed"], r)
        used = r
        if rec is not None:
            break
    if rec is None:
        raise RuntimeError(f"placement exhausted at N={n} side={side} gap={gap}")
    pl = [None] * n
    for k, y0, x0 in rec:
        pl[k] = dict(mask=masks[k], y0=y0, x0=x0, intens=float(intens[k]))
    return dict(scene=D.composite(pl, range(n), side), placements=pl, areas=areas,
                jit=jit, idx=idx, A_eff=A_eff, restart=used)


# ------------------------------------------------------------------------------ 配对
def pos_pair(seed, n, side, **kw):
    """正样本对: 同 N, Z 与 Z' 完全独立. 种子按 2*seed+{1,2} 派生 -- 各库用不相交的
    seed 段, 由调用方保证."""
    return (render_nz(draw_z(2 * seed + 1, side), n, **kw),
            render_nz(draw_z(2 * seed + 2, side), n, **kw))


def neg_pair(seed, n, n2, side, match="area", **kw):
    """硬负样本对: (n, Z) vs (n2, Z), Z 共享.
    match="area": 预算逐比特相同 -> 总面积配平; match="perim": 侧 B 预算乘 n/n2
    -> 总周长一阶配平, 总面积反向 (故意不一致试次)."""
    z = draw_z(2 * seed + 1, side)
    bs = 1.0 if match == "area" else float(n) / float(n2)
    return render_nz(z, n, **kw), render_nz(z, n2, budget_scale=bs, **kw)


# ------------------------------------------------------------------ 像素记分器 (自检臂)
def ink(scene):
    """总墨量 (像素值和). 面积配平负样本上它必须死, 周长配平上必须反向 -- 仪器自检."""
    return float(scene.sum())


def edge_px(scene, thr=0.05):
    """边界像素数: 前景像素中 3x3 邻域不全为前景的那些. 面积配平负样本上它是合法的
    N 线索 (总周长 ~ sqrt(N*A)), 周长配平上必须死 -- 仪器自检的另一半."""
    b = (scene > thr).float()[None, None]
    interior = 1.0 - F.max_pool2d(1.0 - b, 3, stride=1, padding=1)
    return float((b - interior).clamp(min=0).sum())


def pair_coupling(ra, rb):
    """共同前缀物体的落位耦合 -> (frac_same, med_linf): 中心逐比特同位的份额与
    中心 L-inf 距离中位数. 共享 Z 的对用它报告"落位有多共享"."""
    n = min(len(ra["placements"]), len(rb["placements"]))
    d = []
    for k in range(n):
        pa, pb = ra["placements"][k], rb["placements"][k]
        ca = (pa["y0"] + pa["mask"].shape[0] / 2, pa["x0"] + pa["mask"].shape[1] / 2)
        cb = (pb["y0"] + pb["mask"].shape[0] / 2, pb["x0"] + pb["mask"].shape[1] / 2)
        d.append(max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])))
    same = sum(1 for v in d if v < 0.75)
    med = sorted(d)[len(d) // 2] if d else 0.0
    return same / max(n, 1), med
