# symemerge/numcode/grpo.py
"""GRPO (spec §6.3) + 奖励 (§6.1) + 自适应任务权重 (分辨力阶梯, §6.1).

组的定义 (spec §6.4): 一组 = G 条 rollout 共享逐像素相同的 x1、同一套 θ、同一候选集;
基线 = 组内均值, 优势只衡量程序质量. 策略梯度只进写者 (spec §2.7).

已有做法: GRPO 与 KL 的 k3 无偏估计按 Shao et al. 2024 (DeepSeekMath, arXiv:2402.03300)
与 Schulman 的 KL 估计笔记; 本模块是其最小实现 (不裁剪比率 -- 单步 on-policy 下比率恒 1).
"""
import torch

TASK_KEYS = ("t1", "t2", "t3", "t4", "t5", "t6")


def task_accs(task_logits, t5_scores, targets, truth5):
    """逐 rollout 六任务 0/1 命中 -> (B, 6) float, 列序 TASK_KEYS."""
    cols = []
    for t in ("t1", "t2", "t3", "t4"):
        cols.append((task_logits[t].argmax(-1) == targets[t]).float())
    cols.insert(4, (t5_scores.argmax(-1) == truth5).float())
    cols.append((task_logits["t6"].argmax(-1) == targets["t6"]).float())
    return torch.stack(cols, dim=1)


def reward(accs, weights, n_stamps, lam, lam0=0.0, probe_hits=None, alpha=0.0,
           c_align=None, kappa=0.0):
    """r = Σ w_t acc_t + α·r_probe − λ|stamps| − λ0·1[|stamps|=0] + κ·(Σ w_t acc_t)·c
    (spec §6.1 + params §10c 空白惩罚 + §10e 探针项 [U] s114 C1 + v4.1 §6.3 κ 项).
    κ 项 (v4.1 §6.3): 乘性调制, 不可改加性 -- 常数码使 c=1 满分, 唯一封死它的是左边
    acc 塌零; c 由影子读者算 (§6.4), 无新参数. κ=0 或 c=None ⇒ 与旧制逐位相同.
    λ0 (s112 [U]): 信息载体是章; 价格只加在章数投影上, 码就往投影为零的方向跑
    (p2d 零墨码字"空白=4"). 给零投影点定价后空白不再是免费角落 -- λ0 > λ 使
    一章严格优于零章; 组内中心化下只有"是否空白"的差分起作用.
    r_probe (s114 v2 [U]): 逐 rollout ∈ [0,1] = p̂(N|canvas) 概率 (m 抽均值,
    probe.ProbePair 供给) -- 像素可分性当步付酬, 不等在位读者 SGD 追平 (p3b
    点火差价 +0.005 需 4-5k 步/数量; M5 复核: 读者与像素探针同口径本持平
    0.786, 探针的价值在时效与部分酬, 不在视力). 无间距项: s114 计划显式
    否决 η·max(0, d_floor−dmin) 入奖励, dmin 保持体外仪表 (C6)."""
    n = n_stamps.float()
    task = accs @ weights
    r = task - lam * n - lam0 * (n == 0).float()
    if probe_hits is not None:
        r = r + alpha * probe_hits
    if c_align is not None and kappa != 0.0:
        r = r + kappa * task * c_align
    return r


def advantages(r, n_groups, g):
    """A_i = r_i − 组内均值 (spec §6.3). r (n_groups*g,) 按组连续排列."""
    rg = r.view(n_groups, g)
    return (rg - rg.mean(dim=1, keepdim=True)).view(-1)


def k3_kl(lp, lp_ref, keep):
    """KL(π‖π_ref) 的 k3 无偏估计, 逐 token: exp(Δ) − Δ − 1, Δ = logπ_ref − logπ;
    逐 rollout 对有效 token 取均值再对 rollout 取均值 (token 平均而非序列求和 --
    params §10 [U]: 序列求和下 L_max 让长程序天然吃更多 KL 惩罚, 是一条未登记的
    长度压力; 长度压力只许 λ 承担). 恒 ≥ 0."""
    d = lp_ref - lp
    per_tok = d.exp() - d - 1.0
    per_seq = (per_tok * keep).sum(-1) / keep.sum(-1).clamp(min=1)
    return per_seq.mean()


def grpo_loss(logp_sum, adv, kl, beta):
    """L_GRPO = −E[A · Σ log π] + β·KL (spec §6.3). logp_sum (B,), adv (B,) 已中心化."""
    return -(adv.detach() * logp_sum).mean() + beta * kl


class TaskWeights:
    """w_t ∝ 1 − acc̄_t, 滑动平均, 归一化和为 1 (spec §6.1 分辨力阶梯).
    已解决任务自动退出奖励, 压力移向未解决处 -- 自生成课程, 不手动分阶段.
    decay 是 [C] 登记值."""

    def __init__(self, decay=0.99, mul=None):
        self.decay = decay
        self.ema = torch.zeros(len(TASK_KEYS))
        # [U] 2026-08-23「将 t4 权重设置为 3 倍」: 逐任务损失倍率 (缺省 None = 不乘, weights() 逐位同旧); 只乘进损失权重,
        # 池成绩 σ 的 w 加权均值 (C4.5, 归一和 1) 走 weights_norm() 不受影响.
        self.mul = None
        if mul is not None:
            m = torch.as_tensor([float(x) for x in mul], dtype=torch.float32)
            assert m.shape == (len(TASK_KEYS),), f"task weight multiplier needs {len(TASK_KEYS)} entries, got {tuple(m.shape)}"
            assert bool((m > 0).all()), "task weight multiplier entries must be > 0"
            self.mul = m

    def update(self, accs):
        self.ema = self.decay * self.ema + (1.0 - self.decay) * accs.mean(0).cpu()

    def weights_norm(self, device=None):
        """自动权重 w_t ∝ 1 − acc̄_t, 归一和 1 (池成绩 σ 加权与倍率前的基准)."""
        w = (1.0 - self.ema).clamp(min=1e-3)
        w = w / w.sum()
        return w.to(device) if device is not None else w

    def weights(self, device=None):
        """损失用权重 = weights_norm × mul (mul 缺省 ⇒ 原张量原样返回, 逐位同旧)."""
        w = self.weights_norm(device)
        return w if self.mul is None else w * self.mul.to(w.device)
