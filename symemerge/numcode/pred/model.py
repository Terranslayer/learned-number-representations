# symemerge/numcode/pred/model.py
"""器官 (spec-v2 §3-§4 + spec-v3 §3-§4 + spec-v3.1 C1/C5/C6): 共享主干 E (CNN + 6 层双向 Transformer)
+ 六任务头 (严格线性) + 直读头 (2 层 MLP, 128 类, 无 ⊥) + 掩码格位头 + 写头 D1 (逐 token 4 路线性,
冻结不更新) + 段嵌入 seg_cur/seg_org (v3.1 C1: 代码路径保留于 enc_seq, 本阶段不加载) + 成绩预测头 V̂
(6 路, 条件化 (h, φ(θ)); v3.1 C6.1 删独立 Ŵ 头) + 前向模型 g (v3.1 C6.2: ĥ_{k+1} = h_k + g(h_k),
Ŵ = V̂(ĥ), A25 无独立 Ŵ 参数) + 闸 (2 个标量 a,b; v3.1 C3 本阶段关闭, 器官保留).

v3 §3.1 删 flag token: v3 前向不再追加; **参数保留在 state_dict** 只为 A18 兼容回归路
(与 v2 训练器逐位比对需要 v2 装配). v3 序列装配走 tokenize + enc_seq, v2 装配 forward 原样保留
(诊断脚本 dchk/tprobe/calib 与 A9/A18 的参照物, 不动).

梯度路由 (v3 §5.5 + v3.1 C6.5): Θ_E + 各读出头 收 L_chain/L_mask/L_λ; V̂ 头与 g 只收 L_pred
(= Σ BCE(V̂, y_soft) + η_fwd·L_fwd; pred_to_backbone=0 时不进 Θ_E (A16), =1 时进 (A26)); 闸 (a,b)
只收 L_gate (A17); W_D1 不更新 (A1). 新参数一律注册在旧参数之后 (A18 兼容路的 clip_grad_norm_
求和序须与 v2 逐位同).
"""
import torch
import torch.nn as nn
import torch.nn.functional as _F

from .. import geometry as GM
from ..model import D_MODEL, D_THETA, FF, N_ENC, N_HEADS
from .render import NCLS

K = GM.K_MAX


class Backbone(nn.Module):
    """共享主干 E (§4.1): tokens = CNN(x) + PosEmb; seq = [CLS] ⊕ tokens ⊕ [flag]^{1[场景]}."""

    def __init__(self, d=D_MODEL):
        super().__init__()
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
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.flag = nn.Parameter(torch.randn(1, 1, d) * 0.02)      # §3.3 flag token (单个可学向量)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)         # [MASK] (掩码格位任务)
        layer = nn.TransformerEncoderLayer(
            d, N_HEADS, FF, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, N_ENC, norm=nn.LayerNorm(d),
                                         enable_nested_tensor=False)
        # ---- v3 (spec-v3 §3.2/§9.2): 段嵌入, 零初始化. 注册在旧参数之后 (A18 参数序).
        # v3.1 C1: 代码路径保留 (enc_seq seg 分支), 本阶段序列装配一律 seg=False 不加载.
        self.seg_cur = nn.Parameter(torch.zeros(d))
        self.seg_org = nn.Parameter(torch.zeros(d))

    def tokenize(self, x, mask_cells=None):
        """CNN + [MASK] 替换 + 位置嵌入 -> (B,T,d). 与 forward 的对应三行逐字同 (A18 兼容路的逐位基础)."""
        f = self.cnn(x.unsqueeze(1))                       # (B,d,G,G)
        tok = f.flatten(2).transpose(1, 2)                 # (B,T,d)
        if mask_cells is not None:
            tok = torch.where(mask_cells.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
        return tok + self.pos.unsqueeze(0)

    def enc_seq(self, cur_tok, org_tok=None, seg=True, use_flag=False):
        """序列装配: v3.1 C1 现行 = 单槽 [CLS] ⊕ cur_tok (seg=False, 长 257, 段嵌入不加载).
        seg=True 的两槽/段嵌入路径保留 (v3 §4.1 origin 拼接的代码路, 本阶段不走).
        seg=False + use_flag = A18 兼容路 (v2 装配: 场景追加 flag).
        返回 dict(cls, tokens=当前槽 256 个输出) — z_k 只取当前槽 (§4.1)."""
        B = cur_tok.shape[0]
        if seg:
            cur = cur_tok + (self.seg_cur if org_tok is not None else self.seg_org)
        else:
            assert org_tok is None, "兼容路 (seg=False) 无 origin 槽"
            cur = cur_tok
        parts = [self.cls.expand(B, 1, -1), cur]
        if org_tok is not None:
            parts.append(org_tok + self.seg_org)
        if use_flag:
            parts.append(self.flag.expand(B, 1, -1))
        h = self.enc(torch.cat(parts, dim=1))
        return dict(cls=h[:, 0], tokens=h[:, 1:1 + GM.T])

    def forward(self, x, is_scene, mask_cells=None):
        """x (B,SIDE,SIDE) -> dict(cls (B,d), tokens (B,T,d)). is_scene: 追加 flag token.
        mask_cells (B,T) bool: True 的格 token 在进 Transformer 前换 [MASK] (CNN 之后、位置嵌入之前)."""
        B = x.shape[0]
        f = self.cnn(x.unsqueeze(1))                       # (B,d,G,G)
        tok = f.flatten(2).transpose(1, 2)                 # (B,T,d)
        if mask_cells is not None:
            tok = torch.where(mask_cells.unsqueeze(-1), self.mask_tok.expand_as(tok), tok)
        tok = tok + self.pos.unsqueeze(0)
        parts = [self.cls.expand(B, 1, -1), tok]
        if is_scene:
            parts.append(self.flag.expand(B, 1, -1))
        h = self.enc(torch.cat(parts, dim=1))
        return dict(cls=h[:, 0], tokens=h[:, 1:1 + GM.T])


class Heads(nn.Module):
    """读出头 (§4.2): 六任务严格线性 (t5 双线性), 直读 2 层 MLP → K 类, 掩码格位逐 token 线性 → 4 类."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.emb_tau = nn.Embedding(K, D_THETA)      # τ ∈ {1..K-1} -> 行 τ-1 (t1/t3 共用)
        self.emb_p = nn.Embedding(3, D_THETA)
        self.emb_m = nn.Embedding(3, D_THETA)
        self.t1 = nn.Linear(d + D_THETA, 2)
        self.t2 = nn.Linear(d, 2)
        self.t3 = nn.Linear(d + D_THETA, K)
        self.t4 = nn.Linear(d + D_THETA, 7)
        self.t5 = nn.Linear(d, d, bias=False)        # score = (W5 h) · h_k
        self.t6 = nn.Linear(d + D_THETA, K + 3)
        self.read = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, K))
        self.cell = nn.Linear(d, NCLS)

    def task_logits(self, h, th):
        """h (B,d); th = dict(tau, p, m) 表索引. 输入只有 h 与 φ(θ) (A5)."""
        et = self.emb_tau(th["tau"])
        return {
            "t1": self.t1(torch.cat([h, et], -1)),
            "t2": self.t2(h),
            "t3": self.t3(torch.cat([h, et], -1)),
            "t4": self.t4(torch.cat([h, self.emb_p(th["p"])], -1)),
            "t6": self.t6(torch.cat([h, self.emb_m(th["m"])], -1)),
        }

    def t5_scores(self, h, h_cands):
        """h (B,d), h_cands (B,C,d) 候选 CLS (候选是场景, 带 flag 各自过主干) -> (B,C)."""
        return torch.einsum("bd,bcd->bc", self.t5(h), h_cands)


class WriteHead(nn.Module):
    """写头 D1 (§4.3): ℓ_j = W_D1 z_j ∈ R^4, 随机初始化后冻结 (requires_grad=False).

    cm=True ([U] 2026-08-21 共模扣除, 唯一的代码改动 = 5 个参数): ℓ_{j,c} = w_c·(z_j − α·z̄) + b_c, z̄ = mean_j z_j
    (沿 256 格); α 可学标量 (初值 1.0; 推回 0 即取回全局符号), b_c 4 个可学逐类偏置 (初值 0; 纸的默认类与稀疏度
    由此成为被训练的量); W_D1 仍冻结 (A1 不动). 旧冻结偏置 lin.bias 在 cm 路**不参与** (用户公式无此项; 参数留在
    state_dict 供 cm=False 路). cm=False 时**不注册** α/b_c: 旧检查点逐键兼容, A18 平价路零新参数. α/b_c 为常量
    初始化, 不耗 RNG ⇒ 同种子两式 W_D1 逐位同."""

    def __init__(self, d=D_MODEL, cm=False):
        super().__init__()
        self.lin = nn.Linear(d, NCLS)
        for p in self.lin.parameters():
            p.requires_grad_(False)
        self.cm = bool(cm)
        if self.cm:
            self.alpha = nn.Parameter(torch.ones(()))
            self.bias_c = nn.Parameter(torch.zeros(NCLS))

    def forward(self, z):
        if not self.cm:
            return self.lin(z)
        zc = z - self.alpha * z.mean(dim=1, keepdim=True)          # (B,T,d): 扣掉沿格共模 α·z̄
        return _F.linear(zc, self.lin.weight) + self.bias_c        # 冻结 W_D1, 可学 b_c (旧偏置不参与)


THETA_TASKS = ("t1", "t3", "t4", "t6")     # 有 θ 嵌入的任务 (t2 无参数, t5 候选抽法不可嵌入)


class PredHeads(nn.Module):
    """成绩预测头 V̂ (spec-v3 §4.4 + v3.1 C5/C6): 6 路 sigmoid, 条件化在 (h, φ(θ_t)); φ 用 Heads 的
    既有嵌入表 (值共享). v3.1 C6.1 删独立 Ŵ 头: Ŵ = V̂(ĥ), 无独立参数 (A25).
    h 的 detach 由调用方定 (pred_to_backbone 路由, 想象态 ĥ 须保 g 的梯度故不能在此一刀切);
    φ 的 detach 由 detach_emb 定 (ptb=0 时 sg, 防 L_pred 经共享表漏进任务头参数, A16).
    输出 logits, 概率 = sigmoid (损失用 with_logits 版)."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.V = nn.ModuleDict({t: nn.Linear(d + (D_THETA if t in THETA_TASKS else 0), 1)
                                for t in ("t1", "t2", "t3", "t4", "t5", "t6")})

    @staticmethod
    def _emb(heads, th):
        return dict(t1=heads.emb_tau(th["tau"]), t3=heads.emb_tau(th["tau"]),
                    t4=heads.emb_p(th["p"]), t6=heads.emb_m(th["m"]))

    def v_logits(self, heads, h, th, detach_emb=True):
        """V̂ logits (B,6). h 由调用方决定是否 detach (真实态 ptb=0 时 sg; 想象态 ĥ 恒不 sg)."""
        emb = self._emb(heads, th)
        if detach_emb:
            emb = {t: e.detach() for t, e in emb.items()}
        cols = []
        for t in ("t1", "t2", "t3", "t4", "t5", "t6"):
            inp = torch.cat([h, emb[t]], -1) if t in emb else h
            cols.append(self.V[t](inp))
        return torch.cat(cols, dim=1)                       # (B,6) logits


class FwdModel(nn.Module):
    """潜空间前向模型 g (v3.1 C6.2): ĥ_{k+1} = h_k + g(h_k), g = MLP(d → 64 → d), GELU,
    末层零初始化 (起点 ĥ_{k+1} = h_k ⇒ 初始 Δ̂ ≡ 0). 塌缩防线 (C6.6): L_fwd 目标端 sg (A20)
    + 本瓶颈 MLP + h 被 L_chain 钉住."""

    def __init__(self, d=D_MODEL, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(d, hidden)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, h):
        return h + self.l2(self.act(self.l1(h)))


class Gate(nn.Module):
    """闸 (spec-v3 §4.5 + v3.1 C3): 仅两个标量 (a,b). λ_k = σ(softplus(a)·sg[Δ̂_k] − b).
    v3.1 本阶段闸关闭 ((a,b) 不定, 器官保留); 闸位 = 场景链 k≥1 / 池链 k≥0 (C3, 由 step.pred_gate 编排).
    一致性读数 b/softplus(a) ≈ c_step (§4.5)."""

    def __init__(self, a0=0.5413248546129181, b0=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a0)))
        self.b = nn.Parameter(torch.tensor(float(b0)))

    def lam(self, dhat_sg):
        """dhat_sg (B,K) 已 detach 的 Δ̂ -> λ (B,K)."""
        return torch.sigmoid(_F.softplus(self.a) * dhat_sg - self.b)

    def slope(self):
        return float(_F.softplus(self.a.detach()))


class PredModel(nn.Module):
    """整机: E + 读出头 + D1 + V̂ + 前向模型 g + 闸. 前向路径由 step.py 编排.
    新模块注册在 D1 之后 (A18 参数序: 共享参数子序列与 v2 逐位同序; 新前缀 pred./fwd./gate.)."""

    def __init__(self, d=D_MODEL, gate_a0=0.5413248546129181, gate_b0=0.0, d_bottleneck=64, d1_cm=False):
        super().__init__()
        self.E = Backbone(d)
        self.heads = Heads(d)
        self.D1 = WriteHead(d, cm=d1_cm)
        self.pred = PredHeads(d)
        self.fwd = FwdModel(d, d_bottleneck)
        self.gate = Gate(gate_a0, gate_b0)

    def encode(self, x, is_scene, mask_cells=None):
        return self.E(x, is_scene, mask_cells)

    def trainable_params(self):
        """Θ (§5.4): 全部 requires_grad 参数; W_D1 (D1.lin) 不在其中 (cm 路的 α/b_c 在其中)."""
        return [p for p in self.parameters() if p.requires_grad]

    def d1_weights(self):
        """冻结写头权重 (W_D1 + 旧偏置; A1 逐位不变口径), 与 cm 无关."""
        return [p.detach().clone() for p in self.D1.lin.parameters()]
