# symemerge/numcode/geometry.py
"""画布/图章几何 + 光栅化器 + 信道腐蚀 (spec §3; 取值核算 params.md §1/§2/§4).

量具, 无可学参数, 不 import 任何模型代码. 光栅化器不可导 (spec §1): 全部为 no-grad
张量算术, 梯度不经此处回流 (写者只由 GRPO 更新, spec §10 泄漏检查第 4 条).

动作空间 (spec §3.1 格索引制, 废弃 var 分支的位移制):
  aid = cell * S + stamp_type,  cell = gi * G + gj,  A_STOP = T * S, 词表 T*S+1 = 769.
  每格至多一枚章由解码端屏蔽保证; 本模块光栅化前对重复格 assert.

坐标约定: 格 (gi, gj) 的像素块 = [gi*P, gi*P+P) x [gj*P, gj*P+P); 章心名义落格心;
像素中心采样 + 0.5px 线性抗锯齿坡. 满腐蚀最坏离心距 + 0.5 <= P/2 + 0.5 (params.md §1
验算), 故邻格像素中心覆盖率恒为 0 -- 只写本格块不截任何墨.

腐蚀信道 (spec §3.4, 写与读之间, 单一强度旋钮 s ∈ [0,1] 乘满强度登记值):
  几何抖动 (位置/尺寸/旋转) 在光栅化时施加; 遮挡/模糊/像素噪声在 channel() 施加.
  JPEG 的替身是模糊+噪声 (params.md §4 登记).
"""
import math

import torch
import torch.nn.functional as F

SIDE = 128                       # 画布边长 px (params.md §1: 64 -> 128, 容量法则)
P = 8                            # 格距 px (var 分支列宽 = 眼睛潜格距, 照抄)
G = SIDE // P                    # 16 格/边
T = G * G                        # 256 格; T >= K_MAX (spec §3.2)
S = 3                            # 图章种类 (点/横杠/竖杠, 语义不继承)
STAMP_DOT, STAMP_DASH, STAMP_VBAR = 0, 1, 2
A_STOP = T * S                   # 停止动作 id
VOCAB = T * S + 1                # 769
K_MAX = 128                      # 课程顶档 (场景法则 128px 可行门 161, params.md §2)
K_STAGES = (4, 16, 64, 128)

# 名义形状参数 px (尺寸抖动前): 直径 ~0.5P (params.md §1)
DOT_R = 2.0                      # 点: 实心圆半径
DASH_HW, DASH_HH = 2.2, 0.7      # 杠: 4.4 x 1.4 (5px 在满腐蚀验算下越界, 已缩)
# 第三章型 = 竖杠 (横杠的转置, 同尺寸). 换型史: 3.5px 实心方 -> Phase 0 实测
# 方↔点章型比 2.2 (4px 尺度两种实心团形状差 L2~0.7, 与强度无关) -> 换空心环仍差
# 一口气 (环↔点 CNN token 空间 8.5 < 10, 加大孔径会顶到跨格上限并抬自身周长噪声)
# -> 竖杠: 与横杠差两条正交条带, 与点的差按对称性 = 已过门的点↔横杠对, 自身噪声
# 与横杠同量级 (三项 Phase 0 实测均 >= 10.2)

# 满强度腐蚀登记值 (params.md §4); 生效值 = 登记值 * s
JITTER_PX = 0.15 * P             # 格内位置抖动 ±1.2px, 均匀
SIZE_JIT = 0.15                  # 尺寸 ±15%, 均匀
ROT_DEG = 15.0                   # 旋转 ±15°, 均匀
OCCLUDE_MAX = 2                  # 遮挡 1..2 格 (发生概率 = s)
BLUR_SIG = 0.8                   # 高斯模糊 σ 上限 px
NOISE_SIG = 0.02                 # 加性像素噪声 σ


def l_max(k):
    """程序长度上限 ceil(1.5K)+8, 封顶 T (params.md §2; spec §3.2 要求 >= K)."""
    return min(int(math.ceil(1.5 * k)) + 8, T)


def aid_pack(cell, stamp):
    assert 0 <= cell < T and 0 <= stamp < S, (cell, stamp)
    return cell * S + stamp


def aid_unpack(aid):
    assert 0 <= aid < A_STOP, aid
    return aid // S, aid % S


def worst_reach(stamp):
    """满腐蚀最坏离格心距 px (params.md §1 验算式): 尺寸 +15%, 旋转 ±15°, 抖动 ±1.2.
    含 0.5px 抗锯齿坡的邻格零覆盖条件是 worst_reach <= P/2 (邻格最近像素中心距 4.5)."""
    s = 1.0 + SIZE_JIT
    c = math.cos(math.radians(ROT_DEG))
    n = math.sin(math.radians(ROT_DEG))
    if stamp == STAMP_DOT:
        r = DOT_R * s
    else:                                # 横杠/竖杠同尺寸, 最坏轴向可达相同
        r = (DASH_HW * c + DASH_HH * n) * s
    return r + JITTER_PX


def _dot_alpha(x, y, s):
    d = (x * x + y * y).sqrt()
    return (DOT_R * s - d + 0.5).clamp(0.0, 1.0)




def _box_alpha(x, y, s, hw, hh):
    ax = (hw * s - x.abs() + 0.5).clamp(0.0, 1.0)
    ay = (hh * s - y.abs() + 0.5).clamp(0.0, 1.0)
    return ax * ay


def stamp_alpha(types, jy, jx, scales, angles, pad=0):
    """(n,) 章型/抖动/尺寸/角度 -> (n, P+2pad, P+2pad) 覆盖率, 相对格心的像素中心网格.
    pad>0 只给测试用 (验证邻格零覆盖); 生产光栅化 pad=0."""
    n = types.shape[0]
    w = P + 2 * pad
    c = torch.arange(w, dtype=torch.float32) - (w - 1) / 2.0
    yy = c.view(1, w, 1).expand(n, w, w)
    xx = c.view(1, 1, w).expand(n, w, w)
    ys = yy - jy.view(n, 1, 1)
    xs = xx - jx.view(n, 1, 1)
    ca = torch.cos(angles).view(n, 1, 1)
    sa = torch.sin(angles).view(n, 1, 1)
    xr = xs * ca + ys * sa
    yr = -xs * sa + ys * ca
    sc = scales.view(n, 1, 1)
    out = torch.zeros(n, w, w)
    m = types == STAMP_DASH
    if bool(m.any()):
        out[m] = _box_alpha(xr[m], yr[m], sc[m], DASH_HW, DASH_HH)
    m = types == STAMP_VBAR
    if bool(m.any()):
        out[m] = _box_alpha(xr[m], yr[m], sc[m], DASH_HH, DASH_HW)
    m = types == STAMP_DOT
    if bool(m.any()):
        out[m] = _dot_alpha(xr[m], yr[m], sc[m])
    return out


def _flatten_progs(progs):
    """程序批 -> (b_idx, cells, types) 三条扁平 LongTensor. 逐画布 assert 无重复格,
    无 STOP (STOP 只终止解码, 不进入光栅化)."""
    bs, cs, ts = [], [], []
    for b, prog in enumerate(progs):
        seen = set()
        for aid in prog:
            aid = int(aid)
            assert aid != A_STOP, f"canvas {b}: STOP 不应出现在程序体里"
            cell, st = aid_unpack(aid)
            assert cell not in seen, f"canvas {b}: cell {cell} 重复落章 (动作空间应屏蔽)"
            seen.add(cell)
            bs.append(b)
            cs.append(cell)
            ts.append(st)
    if not bs:
        e = torch.zeros(0, dtype=torch.long)
        return e, e.clone(), e.clone()
    return (torch.tensor(bs, dtype=torch.long), torch.tensor(cs, dtype=torch.long),
            torch.tensor(ts, dtype=torch.long))


def raster(progs, strength=0.0, rng=None):
    """程序批 (list of list[aid]) -> (B, SIDE, SIDE) float32 画布, CPU.
    strength>0 时施加几何腐蚀 (位置/尺寸/旋转抖动), 需给 CPU generator."""
    B = len(progs)
    canv = torch.zeros(B, G, P, G, P)
    b, cells, types = _flatten_progs(progs)
    n = b.shape[0]
    if n:
        if strength > 0.0:
            assert rng is not None, "strength>0 需要 CPU generator"
            u = torch.rand(n, 4, generator=rng) * 2.0 - 1.0
            jy = u[:, 0] * JITTER_PX * strength
            jx = u[:, 1] * JITTER_PX * strength
            sc = 1.0 + u[:, 2] * SIZE_JIT * strength
            an = u[:, 3] * math.radians(ROT_DEG) * strength
        else:
            z = torch.zeros(n)
            jy, jx, an = z, z.clone(), z.clone()
            sc = torch.ones(n)
        patches = stamp_alpha(types, jy, jx, sc, an)
        canv[b, cells // G, :, cells % G, :] = patches
    # (B, gi, py, gj, px) 连续内存下 reshape 即为 (B, gi*P+py, gj*P+px)
    return canv.reshape(B, SIDE, SIDE)


def occlude(canvas, strength, rng):
    """遮挡: 概率 = strength, 命中则置空 U{1..OCCLUDE_MAX} 个随机格块. 就地修改."""
    B = canvas.shape[0]
    v = canvas.view(B, G, P, G, P)
    for bi in range(B):
        if float(torch.rand(1, generator=rng)) >= strength:
            continue
        k = int(torch.randint(1, OCCLUDE_MAX + 1, (1,), generator=rng))
        cells = torch.randperm(T, generator=rng)[:k]
        v[bi, cells // G, :, cells % G, :] = 0.0
    return canvas


def occlude_k(canvas, k, rng):
    """恒开遮挡 (params.md §9 裁定1): 每张画布恰置空 k 个随机格块. 就地修改.
    码级判据的信道本体 -- 任意两模板的可分性必须在丢 k 格后仍成立."""
    assert 0 < k <= T, k
    B = canvas.shape[0]
    v = canvas.view(B, G, P, G, P)
    for bi in range(B):
        cells = torch.randperm(T, generator=rng)[:k]
        v[bi, cells // G, :, cells % G, :] = 0.0
    return canvas


def blur(canvas, sigma):
    """高斯模糊, 零填充 same 卷积. sigma<=0 原样返回."""
    if sigma <= 0.0:
        return canvas
    r = max(1, int(math.ceil(2.0 * sigma)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-0.5 * (x / sigma) ** 2)
    k1 = k1 / k1.sum()
    c = canvas.unsqueeze(1)
    c = F.conv2d(c, k1.view(1, 1, 1, -1), padding=(0, r))
    c = F.conv2d(c, k1.view(1, 1, -1, 1), padding=(r, 0))
    return c.squeeze(1)


def channel(canvas, strength, rng, occ_k=0):
    """写与读之间的画布腐蚀 (几何抖动已在 raster 施加): 遮挡 -> 模糊 -> 噪声 -> clamp.
    strength=0 恒等. occ_k>0 时遮挡改恒开恰 k 格 (params.md §9 裁定1), 否则旧制概率遮挡."""
    if strength <= 0.0:
        return canvas
    assert rng is not None
    canvas = canvas.clone()
    if occ_k:
        canvas = occlude_k(canvas, occ_k, rng)
    else:
        canvas = occlude(canvas, strength, rng)
    sig = float(torch.rand(1, generator=rng)) * BLUR_SIG * strength
    canvas = blur(canvas, sig)
    canvas = canvas + torch.randn(canvas.shape, generator=rng) * NOISE_SIG * strength
    return canvas.clamp(0.0, 1.0)


def render_channel(progs, strength=0.0, rng=None, occ_k=0):
    """写者程序 -> 读者所见画布 (raster 几何腐蚀 + channel), 一步到位."""
    return channel(raster(progs, strength, rng), strength, rng, occ_k)


def cell_truth(progs):
    """程序批 -> (B, G, G) long 格位真值: 0 = 空, 1+t = 章型 t (掩码格位头的标签,
    spec §2.6; 腐蚀前内容, params.md §4 登记读法)."""
    B = len(progs)
    out = torch.zeros(B, G, G, dtype=torch.long)
    b, cells, types = _flatten_progs(progs)
    if b.shape[0]:
        out[b, cells // G, cells % G] = types + 1
    return out


def sample_uniform_prog(rng, lmax=None):
    """动作空间均匀随机程序 (掩码任务画布来源, spec §2.6: 不经过写者).
    长度 U{0..lmax}, 格不放回, 章型均匀. lmax 缺省 = l_max(K_MAX) = 200."""
    lmax = l_max(K_MAX) if lmax is None else lmax
    L = int(torch.randint(0, lmax + 1, (1,), generator=rng))
    cells = torch.randperm(T, generator=rng)[:L]
    types = torch.randint(0, S, (L,), generator=rng)
    return (cells * S + types).tolist()


def sample_illegal_prog(rng, l_max_legal, occ_k, margin=8, span=96):
    """结构性非法画布程序 (s113 改3, params §10d): 墨章数 M 严格超过「合法码字
    经信道后可能显示的章数上界」l_max_legal, 且自身再被 occ_k 遮挡后仍超过 --
    信道只删章不增章 (遮挡置空/模糊噪声不造章/抖动不出格, 见模块头验算), 故任何
    合法码字的腐蚀轨道都到不了这里, 与码演化成什么形状无关. 下界随 l_max 逐档重推."""
    lo = l_max_legal + occ_k + margin
    hi = min(T - margin, lo + span)
    assert lo < hi <= T, (lo, hi)   # l_max+occ_k 顶到画布容量时此构造失效, 需重设计
    M = int(torch.randint(lo, hi + 1, (1,), generator=rng))
    cells = torch.randperm(T, generator=rng)[:M]
    types = torch.randint(0, S, (M,), generator=rng)
    return (cells * S + types).tolist()


def illegal_capacity_ok(l_max_legal, occ_k, margin=8, span=96):
    """sample_illegal_prog 的可行性判定 (评审 Critical#1): K=128 档 l_max=200 时
    lo=256 > hi=248, 构造失效且 resume 后原地复现. 失效必须停表, 不降级构造
    (薄边界的非法性不可靠). 与采样器同式同参."""
    lo = l_max_legal + occ_k + margin
    return lo < min(T - margin, lo + span)
