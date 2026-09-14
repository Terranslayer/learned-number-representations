# symemerge/numcode/losses.py
"""损失函数 (spec §5). 全部为精确梯度, 全部流入主干; 直读损失是量具 (权重 μ 取小).

L_sup = L_count + L_task + L_mask + μ L_read  (spec §5.5)
"""
import math

import torch
import torch.nn.functional as F

from . import geometry as GM
from .model import ABSTAIN

SQRT6 = math.sqrt(6.0)


def count_loss(logits, ns):
    """L_count (spec §5.1): 真值 N ∈ {1..K} -> 类 N-1."""
    return F.cross_entropy(logits, ns - 1)


def task_loss(task_logits, t5_scores, targets, truth5):
    """L_task (spec §5.2): 六项 CE 等权相加 / sqrt(6) (六项同源于一个 N, 等权相加
    等于给主干六倍学习率 -- spec 给的缩放理由). 读者要学全部六个任务, 此处无自适应
    权重 -- 那是奖励侧的事 (spec §6.1)."""
    parts = [F.cross_entropy(task_logits[t], targets[t])
             for t in ("t1", "t2", "t3", "t4", "t6")]
    parts.append(F.cross_entropy(t5_scores, truth5))
    return sum(parts) / SQRT6


def read_loss(logits, ns, c):
    """L_read (spec §5.3, Chow 判据软化): (1-a)·(-log p̃[N]) + a·c, a = p[⊥],
    p̃ 为 N 类上重归一化. c 以 nat 计 (c = log K' 的含义: 读不到击败 K' 选一就弃权);
    从高退火到中由训练器排程 (spec §8)."""
    logp = F.log_softmax(logits, dim=-1)
    a = logp[:, ABSTAIN].exp()
    log1m = torch.log1p(-a.clamp(max=1.0 - 1e-6))
    lpn = logp.gather(1, (ns - 1).unsqueeze(1)).squeeze(1)
    return ((1.0 - a) * (log1m - lpn) + a * c).mean()


def sample_mask_cells(B, rng, lo=0.15, hi=0.40):
    """(B, T) bool 遮蔽格集合, 逐样本比例 ~ U[lo, hi] (spec §5.4: 15-40% 随机),
    至少 1 格."""
    out = torch.zeros(B, GM.T, dtype=torch.bool)
    for b in range(B):
        r = lo + float(torch.rand(1, generator=rng)) * (hi - lo)
        k = max(1, int(round(r * GM.T)))
        out[b, torch.randperm(GM.T, generator=rng)[:k]] = True
    return out


def mask_loss(cell_logits, truth, mask_cells):
    """L_mask (spec §5.4): 只在被遮格上算 CE. cell_logits (B,T,S+1);
    truth (B,G,G) ∈ {0..S} (0=空); mask_cells (B,T) bool."""
    assert bool(mask_cells.any()), "空遮蔽集"
    t = truth.view(truth.shape[0], GM.T)
    return F.cross_entropy(cell_logits[mask_cells], t[mask_cells])
