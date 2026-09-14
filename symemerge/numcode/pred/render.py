# symemerge/numcode/pred/render.py
"""渲染与信道 (spec-v2 §4.3-4.4, §6.2): 逐格四类模板栈 T_j ∈ R^{4×8×8} + 硬前向/直通反传 +
逐像素可导腐蚀 + 占空比分层随机画布.

量具, 无可学参数, 不 import 模型. 几何常数与 alpha 公式逐字沿用 ../geometry.py
(同 jitter 下与 geometry.stamp_alpha 逐位同值, 测试钉住); 差别只是: 逐格 (而非逐章) 抽抖动、
在设备上算、并对 "空" 类给全零模板.

类别约定 (与掩码格位头同一套): 0 = 空, 1 = 点, 2 = 横杠, 3 = 竖杠 (= geometry 章型 + 1).

直通 (§4.4): x2 = x_hard + (x_soft − sg[x_soft]) -- 前向逐位等于硬渲染 (x − x 在浮点下精确为 0),
反向 ∂x2/∂p = ∂x_soft/∂p. (写成 soft + sg[hard − soft] 在浮点下不逐位, 故用等价的这一式.)

信道 (§4.4 末, 登记值 geometry §4): 遮挡 (格块乘 0) → 模糊 (可分离高斯卷积) → 像素噪声 → clamp,
全部逐像素可导; 随机数一律来自调用方 CPU generator (CRN 纪律), 由 ChannelDraw 一次抽定后可复用
(u_S 的两次前向共用同一份 draw, §7.4 硬要求 2).
"""
import dataclasses
import math

import torch
import torch.nn.functional as F

from .. import geometry as GM

NCLS = GM.S + 1          # 4
EMPTY = 0
P, G, T, SIDE = GM.P, GM.G, GM.T, GM.SIDE


@dataclasses.dataclass
class ChannelDraw:
    """一次信道实现: 逐格几何抖动 u (B,T,4) ∈ [-1,1], 遮挡保留掩码 keep (B,T) ∈ {0,1},
    模糊 σ (标量, 与 geometry.channel 同: 一次 draw 一个 σ), 像素噪声 noise (B,SIDE,SIDE)."""
    u: torch.Tensor
    keep: torch.Tensor
    sigma: float
    noise: torch.Tensor
    s: float

    @property
    def B(self):
        return self.u.shape[0]


def draw_channel(B, s, rng, dev, occ_k=0):
    """抽一份信道实现 (CPU generator -> 设备). s=0 时几何抖动/遮挡/模糊/噪声全为恒等."""
    u = (torch.rand(B, T, 4, generator=rng) * 2.0 - 1.0) if s > 0 else torch.zeros(B, T, 4)
    keep = torch.ones(B, T)
    sigma = 0.0
    noise = torch.zeros(B, SIDE, SIDE)
    if s > 0:
        if occ_k > 0:                       # 恒开遮挡恰 k 格 (params §9 裁定1)
            for b in range(B):
                cells = torch.randperm(T, generator=rng)[:occ_k]
                keep[b, cells] = 0.0
        else:                               # 旧制: 概率 s, 1..OCCLUDE_MAX 格
            for b in range(B):
                if float(torch.rand(1, generator=rng)) >= s:
                    continue
                k = int(torch.randint(1, GM.OCCLUDE_MAX + 1, (1,), generator=rng))
                cells = torch.randperm(T, generator=rng)[:k]
                keep[b, cells] = 0.0
        sigma = float(torch.rand(1, generator=rng)) * GM.BLUR_SIG * s
        noise = torch.randn(B, SIDE, SIDE, generator=rng) * GM.NOISE_SIG * s
    return ChannelDraw(u=u.to(dev), keep=keep.to(dev), sigma=sigma, noise=noise.to(dev), s=s)


def templates(u, s):
    """逐格四类模板栈 (B,T,4,P,P): 类 0 全零; 类 c≥1 = 章型 c−1 在该格抖动 (位置/尺寸/旋转 × s)
    下的覆盖率 patch. 公式 = geometry._dot_alpha/_box_alpha (像素中心网格, 0.5px 抗锯齿坡)."""
    B = u.shape[0]
    dev = u.device
    jy = u[..., 0] * GM.JITTER_PX * s                     # 结合顺序同 geometry.raster
    jx = u[..., 1] * GM.JITTER_PX * s
    sc = 1.0 + u[..., 2] * GM.SIZE_JIT * s
    an = u[..., 3] * math.radians(GM.ROT_DEG) * s
    c = torch.arange(P, dtype=torch.float32, device=dev) - (P - 1) / 2.0
    yy = c.view(1, 1, P, 1)
    xx = c.view(1, 1, 1, P)
    ys = yy - jy.unsqueeze(-1).unsqueeze(-1)              # (B,T,P,1) 广播
    xs = xx - jx.unsqueeze(-1).unsqueeze(-1)              # (B,T,1,P)
    ca = torch.cos(an).unsqueeze(-1).unsqueeze(-1)
    sa = torch.sin(an).unsqueeze(-1).unsqueeze(-1)
    xr = xs * ca + ys * sa                                # (B,T,P,P)
    yr = -xs * sa + ys * ca
    scv = sc.unsqueeze(-1).unsqueeze(-1)
    dot = (GM.DOT_R * scv - (xr * xr + yr * yr).sqrt() + 0.5).clamp(0.0, 1.0)
    dash = ((GM.DASH_HW * scv - xr.abs() + 0.5).clamp(0.0, 1.0)
            * (GM.DASH_HH * scv - yr.abs() + 0.5).clamp(0.0, 1.0))
    vbar = ((GM.DASH_HH * scv - xr.abs() + 0.5).clamp(0.0, 1.0)
            * (GM.DASH_HW * scv - yr.abs() + 0.5).clamp(0.0, 1.0))
    empty = torch.zeros_like(dot)
    return torch.stack([empty, dot, dash, vbar], dim=2)  # (B,T,4,P,P)


def assemble(patches):
    """(B,T,P,P) 逐格 patch -> (B,SIDE,SIDE) 画布 (格 j = gi*G+gj 的像素块 [gi*P, gi*P+P) × [gj*P, ...))."""
    B = patches.shape[0]
    return patches.view(B, G, G, P, P).permute(0, 1, 3, 2, 4).reshape(B, SIDE, SIDE)


def render_hard(k, tpl):
    """硬渲染: 类别图 k (B,T) long + 模板栈 -> (B,SIDE,SIDE). 无梯度路径 (gather 只沿类别)."""
    idx = k.view(k.shape[0], T, 1, 1, 1).expand(-1, -1, 1, P, P)
    return assemble(tpl.gather(2, idx).squeeze(2))


def render_soft(p, tpl):
    """软渲染: 逐格类别概率 p (B,T,4) 的模板期望 -> (B,SIDE,SIDE) (对 p 可导)."""
    return assemble(torch.einsum("btc,btcyx->btyx", p, tpl))


def render_ste(logits, tpl, k_hard=None):
    """§4.4 直通渲染. logits (B,T,4) -> dict(x2 前向=硬渲染/反向=软, hard, soft, p, k).
    k_hard 给定 = 硬类别冻结 (§7.4 硬要求 1: 扰动参数下的第二次前向沿用未扰动 argmax)."""
    p = F.softmax(logits, dim=-1)
    k = logits.argmax(-1) if k_hard is None else k_hard
    hard = render_hard(k, tpl)
    soft = render_soft(p, tpl)
    x2 = hard.detach() + (soft - soft.detach())
    return dict(x2=x2, hard=hard, soft=soft, p=p, k=k)


def render_classes(k, draw):
    """无梯度渲染 (池抽笔记 / 随机画布 / 评测): 类别图 + 信道实现 -> 硬画布 (腐蚀前)."""
    with torch.no_grad():
        return render_hard(k, templates(draw.u, draw.s))


def _blur(x, sigma):
    if sigma <= 0.0:
        return x
    r = max(1, int(math.ceil(2.0 * sigma)))
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k1 = torch.exp(-0.5 * (t / sigma) ** 2)
    k1 = k1 / k1.sum()
    c = x.unsqueeze(1)
    c = F.conv2d(c, k1.view(1, 1, 1, -1), padding=(0, r))
    c = F.conv2d(c, k1.view(1, 1, -1, 1), padding=(r, 0))
    return c.squeeze(1)


def keep_pixels(keep):
    """(B,T) 格保留掩码 -> (B,SIDE,SIDE) 像素掩码."""
    B = keep.shape[0]
    return keep.view(B, G, 1, G, 1).expand(B, G, P, G, P).reshape(B, SIDE, SIDE)


def channel(x, draw):
    """腐蚀信道 (逐像素可导): 遮挡 → 模糊 → 噪声 → clamp. draw.s<=0 恒等 (与 geometry.channel 同约定)."""
    if draw.s <= 0.0:
        return x
    x = x * keep_pixels(draw.keep)
    x = _blur(x, draw.sigma)
    x = x + draw.noise
    return x.clamp(0.0, 1.0)


def stamp_count(k):
    """章数 = #{j: k_j ≠ 空} (B,) (留白零代价, §4.3)."""
    return (k != EMPTY).sum(-1)


def random_canvas_classes(B, rng, dev=None):
    """占空比分层随机画布 (§6.2): ρ~U[0,1], 随机 ⌊256ρ⌋ 格非空, 非空格章型均匀 {1,2,3}.
    返回 (B,T) long (CPU generator 抽, 可选搬到 dev)."""
    rho = torch.rand(B, generator=rng)
    n = (rho * T).floor().long()                                  # 0..255
    order = torch.argsort(torch.rand(B, T, generator=rng), dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(T).unsqueeze(0).expand(B, T))
    on = rank < n.unsqueeze(1)
    types = torch.randint(1, NCLS, (B, T), generator=rng)
    k = torch.where(on, types, torch.zeros_like(types))
    return k.to(dev) if dev is not None else k


def random_canvas_like(k_src, rng):
    """等占空随机画布 (§7.5 地板): 每行章数与 k_src 相同, 格位与章型随机重抽. (B,T) long, 同设备."""
    B = k_src.shape[0]
    n = stamp_count(k_src).cpu()
    order = torch.argsort(torch.rand(B, T, generator=rng), dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(T).unsqueeze(0).expand(B, T))
    on = rank < n.unsqueeze(1)
    types = torch.randint(1, NCLS, (B, T), generator=rng)
    k = torch.where(on, types, torch.zeros_like(types))
    return k.to(k_src.device)


def classes_from_progs(progs):
    """geometry 程序批 (list[list[aid]]) -> (B,T) long 类别图 (测试/对表用)."""
    B = len(progs)
    k = torch.zeros(B, T, dtype=torch.long)
    for b, prog in enumerate(progs):
        for aid in prog:
            cell, st = GM.aid_unpack(int(aid))
            k[b, cell] = st + 1
    return k
