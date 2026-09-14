# symemerge/numcode/model.py
"""模块架构 (spec §2): 共享主干 E + 计数头 + 六任务头 + 直读头(⊥) + 掩码格位头 + 写者 D1.

三条信号路径的实现约定 (spec §1/§2.7/§10):
  - 精确梯度: 计数/任务/直读/掩码损失 -> 主干 + 各头 (正常反传).
  - 策略梯度: GRPO -> 只更新 D1. **写者的交叉注意力输入必须由调用方 detach 后传入**
    (`E(x1).tokens.detach()` 或 no_grad 前向) -- stop_grad 精确落在 E 与 D1 之间.
  - 任务头的输入只有 h 与 φ(θ), 结构上不可能碰到场景侧张量 (spec §10 第 2 条).

合法动作屏蔽 (spec §3.1 每格至多一枚, 由动作空间保证): 已盖格子的全部章型置 -inf.
屏蔽同时作用于采样与教师强制的 logits -- 它是策略分布本身的一部分, GRPO 的 log π
必须在屏蔽后的分布上计算.

架构常数是 [C] 工程取值 (params.md §6), 不承载科学结论.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as GM

D_MODEL = 256
N_HEADS = 8
N_ENC = 6
N_DEC = 4
FF = 4 * D_MODEL
D_THETA = 64
MODAL_SCENE, MODAL_CANVAS = 0, 1
BOS = GM.VOCAB                   # 写者解码起始符 (词表外)
L_POS = GM.l_max(GM.K_MAX) + 1   # 201: 解码位置嵌入行数
K = GM.K_MAX
ABSTAIN = K                      # 直读头第 K+1 类 = ⊥


class Backbone(nn.Module):
    """共享主干 E (spec §2.1): CNN 下采样 8x (= 格距, 一 token 一格) -> 16x16 token
    -> +2D 位置嵌入 +模态标记 +CLS -> 双向 Transformer. 同一套参数处理两种模态."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        # k4/s2/p1 而非 k3: 三层复合后 token 中心 = 8j+3.5 = 格心, 相位精确对齐
        # (k3/s2/p1 的 token 中心落在 8j 格角, 单章归属在两 token 间摇摆 --
        # 对齐测试 test_cnn_token_cell_alignment 逮的就是它)
        ch = (32, 64, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, ch[0], 4, stride=2, padding=1), nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], ch[1], 4, stride=2, padding=1), nn.GroupNorm(8, ch[1]),
            nn.SiLU(),
            nn.Conv2d(ch[1], ch[2], 4, stride=2, padding=1), nn.GroupNorm(8, ch[2]),
            nn.SiLU(),
            nn.Conv2d(ch[2], d, 1),
        )
        self.pos = nn.Parameter(torch.randn(GM.T, d) * 0.02)
        self.modal = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)   # [MASK] 向量 (spec §2.6)
        layer = nn.TransformerEncoderLayer(
            d, N_HEADS, FF, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, N_ENC, norm=nn.LayerNorm(d),
                                         enable_nested_tensor=False)

    def forward(self, x, modality, mask_cells=None):
        """x (B, SIDE, SIDE) -> dict(cls (B,d), tokens (B,T,d)).
        mask_cells (B,T) bool: True 的格 token 在进 Transformer 前替换为 [MASK]
        (在 CNN 之后、位置/模态嵌入之前替换 -- 遮的是内容, 不遮位置)."""
        B = x.shape[0]
        f = self.cnn(x.unsqueeze(1))                       # (B, d, G, G)
        tok = f.flatten(2).transpose(1, 2)                 # (B, T, d)
        if mask_cells is not None:
            tok = torch.where(mask_cells.unsqueeze(-1), self.mask_tok.expand_as(tok),
                              tok)
        tok = tok + self.pos.unsqueeze(0)
        m = self.modal.weight[modality].view(1, 1, -1)
        tok = tok + m
        cls = self.cls.expand(B, 1, -1) + m
        h = self.enc(torch.cat([cls, tok], dim=1))
        return dict(cls=h[:, 0], tokens=h[:, 1:])


class Heads(nn.Module):
    """全部读出头. 任务头严格线性 (spec §2.4); 计数头/直读头深度按 spec §2.3/§2.5."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.count = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                   nn.Linear(4 * d, K))              # spec §2.3, 2 层上限
        self.emb_tau = nn.Embedding(K, D_THETA)      # τ ∈ {1..K-1} -> 行 τ-1 (t1/t3 共用)
        self.emb_p = nn.Embedding(len((3, 5, 7)), D_THETA)
        self.emb_m = nn.Embedding(len((1, 2, 3)), D_THETA)
        self.t1 = nn.Linear(d + D_THETA, 2)
        self.t2 = nn.Linear(d, 2)                    # 奇偶无 θ
        self.t3 = nn.Linear(d + D_THETA, K)
        self.t4 = nn.Linear(d + D_THETA, 7)
        self.t5 = nn.Linear(d, d, bias=False)        # 双线性: score = (W5 h) · h_k
        self.t6 = nn.Linear(d + D_THETA, K + 3)
        self.read = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, K + 1))           # spec §2.5, 含 ⊥
        self.cell = nn.Linear(d, GM.S + 1)           # 掩码格位头, 逐 token (spec §2.6)

    def task_logits(self, h, theta_idx):
        """h (B,d) 画布 CLS; theta_idx = dict(tau (B,), p (B,), m (B,)) 皆为表索引
        (τ-1, P_CHOICES 下标, M_CHOICES 下标). 输入只有 h 与 φ(θ) -- spec §10 第 2 条."""
        et = self.emb_tau(theta_idx["tau"])
        return {
            "t1": self.t1(torch.cat([h, et], -1)),
            "t2": self.t2(h),
            "t3": self.t3(torch.cat([h, et], -1)),
            "t4": self.t4(torch.cat([h, self.emb_p(theta_idx["p"])], -1)),
            "t6": self.t6(torch.cat([h, self.emb_m(theta_idx["m"])], -1)),
        }

    def t5_scores(self, h, h_cands):
        """h (B,d), h_cands (B,C,d) 候选 CLS (场景模态各自过主干) -> (B,C) 打分."""
        return torch.einsum("bd,bcd->bc", self.t5(h), h_cands)


def legal_bias(actions_in):
    """教师强制输入 (B,L) (含 BOS, 可含 STOP/PAD) -> (B,L,VOCAB) 加性偏置:
    步 l 预测时, 之前已盖格子的全部章型 -inf; STOP 恒合法. 与采样端同一分布."""
    B, L = actions_in.shape
    is_stamp = actions_in < GM.A_STOP
    cells = torch.where(is_stamp, actions_in // GM.S,
                        torch.zeros_like(actions_in))
    onehot = F.one_hot(cells, GM.T).float() * is_stamp.unsqueeze(-1)  # (B,L,T)
    # 输出位 l 预测的是下一个 token (actions_in[l+1] 位置的动作), 故 "之前已盖" =
    # 含本位输入在内的整个前缀 -- cumsum 不移位. (曾错移一位: 只漏紧邻上一步的格,
    # 装配级测试在 raster 的无重复格 assert 上逮住连续重复落章.)
    used = onehot.cumsum(1)
    bias = torch.zeros(B, L, GM.VOCAB, device=actions_in.device)
    bias[:, :, :GM.A_STOP] = torch.repeat_interleave(
        used, GM.S, dim=2).neg() * 1e9
    return bias


class Writer(nn.Module):
    """写者头 D1 (spec §2.2): 自回归 decoder, 因果自注意, 交叉注意场景全 token.
    无 teacher forcing 损失 -- forward 只为 GRPO 计算已采样程序的 log π."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(GM.VOCAB + 1, d)     # +1 = BOS
        self.pos = nn.Embedding(L_POS, d)
        layer = nn.TransformerDecoderLayer(
            d, N_HEADS, FF, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, N_DEC, norm=nn.LayerNorm(d))
        self.head = nn.Linear(d, GM.VOCAB)

    def logits(self, actions_in, scene_tokens):
        """actions_in (B,L) 教师强制输入 (首列 BOS) -> (B,L,VOCAB) 合法化 logits.
        scene_tokens 必须已 detach (spec §2.7); 此处 assert 把关."""
        assert not scene_tokens.requires_grad, \
            "scene_tokens 带梯度进入写者 -- stop_grad 必须落在 E 与 D1 之间 (spec §2.7)"
        B, L = actions_in.shape
        h = self.emb(actions_in) + self.pos.weight[:L].unsqueeze(0)
        cm = torch.triu(torch.full((L, L), float("-inf"),
                                   device=actions_in.device), diagonal=1)
        out = self.dec(h, scene_tokens, tgt_mask=cm)
        return self.head(out) + legal_bias(actions_in)

    @torch.no_grad()
    def rollout(self, scene_tokens, lmax, temps, rng=None, ban_mode=None,
                ban_pos=None):
        """采样 rollout -> (programs list[list[aid]], acts (B,lmax+1) 含 STOP padding,
        lens (B,) 含 STOP 的 token 数). temps (B,): 0 -> 贪心, >0 -> 温度采样
        (spec §6.5 组内混合温度). 屏蔽与教师强制同一分布.
        ban_mode (B,) bool (C9 forced 臂, [U] p3e 改形): True 行在随机位置
        u ~ U{0..lmax-1} 禁当步众数动作 (合法化 logits 的 argmax) -- 旧形式
        「首动作禁 STOP」对贪心程序已有墨的类是空操作. 若行在到达 u 前采样出
        STOP, 则该步改禁 STOP 提前消费 (退化策略下 STOP 即众数; 保证每条受迫
        臂恰被迫偏离一次, 产出正常质量短程序而非高温垃圾). 只改采样分布;
        logp_of 仍按真策略计 log π (受迫程序被正优势采纳时梯度推真策略跟进).
        ban_pos (B,) long 可选: 显式给定 u (测试用); None 则内部抽."""
        B = scene_tokens.shape[0]
        dev = scene_tokens.device
        acts = torch.full((B, lmax + 1), GM.A_STOP, dtype=torch.long, device=dev)
        seq = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        lens = torch.zeros(B, dtype=torch.long, device=dev)
        pend = None
        if ban_mode is not None and bool(ban_mode.any()):
            pend = ban_mode.clone()
            upos = (ban_pos if ban_pos is not None else
                    torch.randint(0, lmax, (B,), device=dev, generator=rng))
        for step in range(lmax + 1):
            lg = self.logits(seq, scene_tokens)[:, -1]         # (B, VOCAB)
            if pend is not None and step < lmax:
                hit = pend & alive & (upos == step)
                if bool(hit.any()):
                    lg = lg.clone()
                    lg[hit, lg[hit].argmax(-1)] = float("-inf")
                    pend = pend & ~hit
            if step == lmax:                                   # 长度上限: 只许 STOP
                choice = torch.full((B,), GM.A_STOP, dtype=torch.long, device=dev)
            else:
                greedy = lg.argmax(-1)
                # 贪心行 (temp=0) 的采样结果会被丢弃, 用温度 1 采样防止除零溢出
                t = torch.where(temps > 0, temps, torch.ones_like(temps)).unsqueeze(-1)
                probs = F.softmax(lg / t, dim=-1)
                sampled = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
                choice = torch.where(temps > 0, sampled, greedy)
                if pend is not None:                           # STOP 拦截提前消费
                    icp = pend & alive & (choice == GM.A_STOP)
                    if bool(icp.any()):
                        lg2 = lg[icp].clone()
                        lg2[:, GM.A_STOP] = float("-inf")
                        t2 = t[icp]
                        alt = torch.multinomial(F.softmax(lg2 / t2, dim=-1), 1,
                                                generator=rng).squeeze(-1)
                        choice = choice.clone()
                        choice[icp] = alt
                        pend = pend & ~icp
            acts[:, step] = torch.where(alive, choice, acts[:, step])
            lens = lens + alive.long()
            alive = alive & (choice != GM.A_STOP)
            seq = torch.cat([seq, acts[:, step:step + 1]], dim=1)
            if not bool(alive.any()):
                break
        programs = []
        for b in range(B):
            pb = acts[b, :lens[b]].tolist()
            programs.append(pb[:-1] if pb and pb[-1] == GM.A_STOP else pb)
        return programs, acts[:, :int(lens.max())], lens

    def logp_of(self, acts, lens, scene_tokens):
        """已采样序列的 Σ log π (spec §2.2: 因式分解只用于计算 log π).
        acts (B,L) 每行前 lens[b] 个 token 计入 (末位可为 STOP)."""
        B, L = acts.shape
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=acts.device)
        lg = self.logits(torch.cat([bos, acts[:, :-1]], dim=1), scene_tokens)
        lp = F.log_softmax(lg, dim=-1).gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        keep = torch.arange(L, device=acts.device).unsqueeze(0) < lens.unsqueeze(-1)
        return (lp * keep).sum(-1), lp, keep


class NumCodeModel(nn.Module):
    """整机: E + 头 + D1. 前向路径由训练器编排, 此处只聚合参数与常用组合."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.E = Backbone(d)
        self.heads = Heads(d)
        self.writer = Writer(d)

    def encode_scene(self, x):
        return self.E(x, MODAL_SCENE)

    def encode_canvas(self, x, mask_cells=None):
        return self.E(x, MODAL_CANVAS, mask_cells)


def theta_to_idx(theta, device=None):
    """data.sample_theta 的 θ 值 -> 嵌入表索引 (支持标量 dict 或已批量的 dict)."""
    import symemerge.numcode.data as D
    tau = theta["tau"]
    if not torch.is_tensor(tau):
        tau = torch.tensor([theta["tau"]], device=device)
        return dict(tau=tau - 1,
                    p=torch.tensor([D.P_CHOICES.index(theta["p"])], device=device),
                    m=torch.tensor([D.M_CHOICES.index(theta["m"])], device=device))
    return dict(tau=tau - 1, p=theta["p"], m=theta["m"])
