# symemerge/numcode/align.py
"""对齐最小化 v4.1 (多任务协同分支执行件, [C] 起草): 度规 / 滚动留出 / 交叉项 u_S /
探针 (R 谱, τ, 𝒮, 曲率, c 通道) / 定标码 CAL1-4 / 建造门 GA1-GA5.

依赖的基线接口 (v4.1 §1 F1-F6) 在本模块只读: 主干 E 两路由共享 (model.Backbone),
任务头严格线性 (Heads.t1..t6 皆 nn.Linear, 本模块启动时断言 -- v4.1 §12.4),
stop_grad 落在 E 与 D1 之间 (Writer.logits 断言), AdamW 二阶矩可读.

度规约定 (v4.1 §2.2 公式 与 §3.2(a) 配方指数不一致, [C] 取公式为准并登记):
  F = diag(√v̂ + ε), v̂ = AdamW 二阶矩 (偏差校正), ε = AdamW 同值.
  ⟨a,b⟩_{F⁻¹} = Σ a_i b_i / (√v̂_i + ε)                        (§2.2 公式, 本模块 Metric.inner)
  白化 a ↦ a / √(√v̂+ε): 白化后欧氏内积 = ⟨·,·⟩_{F⁻¹}            (§3.2(a) 的「乘度规」按此指数落地)
  F⁻¹ g = g / (√v̂+ε) = AdamW 步方向                              (§10 δ 定法「与单步实际位移同量级」)
  ⇒ g⊤F⁻¹g = AdamW 单步对该损失的一阶下降量 / lr (τ 的白话「训 V 顺带覆盖 S 的比率」即此比).
草图 P (§3.2(a)) 在本分支参数量 (~5M) 下不必要: 全部内积精确计算 (P = 恒等, GA3 误差恒 0,
m 项作废) -- 少一个可调件, 判据不失真.

只有精确梯度与无梯度量; 不含任何可学习参数 (v4.1 §3.1 新增参数为零).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from . import geometry as GM
from . import grpo as GR
from .model import ABSTAIN

TASKS_LIN = ("t1", "t2", "t3", "t4", "t6")   # 线性任务头 (含 φ(θ) 拼接者)
BETA_CAP = 0.1        # §10: β·‖u_S‖_{F⁻¹} ≤ 0.1 (修正项不得主导主流), 逐步夹取
GA1_GLOBAL, GA1_GROUP = 0.7, 0.4   # §7.1 GA1 门: 全局 ρ_s ≥ 0.7 且无参数组 < 0.4
GA2_COS, GA2_REL = 0.95, 0.10      # §7.1 GA2 门: 至少一档 余弦 ≥ .95 且相对误差 ≤ .1
GA4_SNR = 2.0                       # §7.1 GA4 门: 单评信噪比 ≥ 2
LSTAR_FRAC = 0.5      # [C] L* = 0.5 × GA5 弃权地板 (§5.5「定在弃权地板之下」的实例化)
# ([U] 2026-08-17 正式跑改动 4: G_n 的 EMA 缓冲取消, Ḡ_{¬n} 每评从当前评测窗的 rollout 直接
#  重算; X4 缓冲时效钉子随之删除 -- 旧常数 REBUILD_V 不再存在.)


# ================================================================ 参数块
def _named(model, pred):
    return [p for n, p in model.named_parameters() if pred(n)]


def cv_params(model):
    """画布路由参数 Θ_cv = Θ_E ∪ 任务头 (含 θ 嵌入表) ∪ 直读头 (v4.1 §5.1 g^cv 的定义域)."""
    return _named(model, lambda n: n.startswith("E.") or n.startswith("heads.emb_")
                  or any(n.startswith(f"heads.t{i}.") for i in range(1, 7))
                  or n.startswith("heads.read."))


def sc_params(model):
    """场景路由参数 Θ_sc = Θ_E ∪ 计数头."""
    return _named(model, lambda n: n.startswith("E.") or n.startswith("heads.count."))


def bb_params(model):
    """主干 Θ_E (M1 分块报数用)."""
    return _named(model, lambda n: n.startswith("E."))


def head_params(model):
    """c 通道闭式块: t1..t6 的 weight/bias, 顺序固定 (t1.w, t1.b, t2.w, t2.b, t3.w, t3.b,
    t4.w, t4.b, t5.w, t6.w, t6.b)."""
    hs = model.heads
    out = []
    for t in ("t1", "t2", "t3", "t4", "t5", "t6"):
        lin = getattr(hs, t)
        assert isinstance(lin, nn.Linear), f"{t} 非严格线性 -- §6.2 闭式失效 (v4.1 §12.4)"
        out.append(lin.weight)
        if lin.bias is not None:
            out.append(lin.bias)
    return out


def param_groups_of(model):
    """GA1 分组: 主干各层 / 计数头 / 任务头 / 直读头 / 格位头. 返回 dict name -> [params]."""
    groups = {}
    for n, p in model.named_parameters():
        if n.startswith("writer."):
            continue
        if n.startswith("E.enc.layers."):
            key = "E.enc." + n.split(".")[3]
        elif n.startswith("E.enc.norm"):
            key = "E.enc.norm"
        elif n.startswith("E.cnn"):
            key = "E.cnn"
        elif n.startswith("E."):
            key = "E.embed"
        elif n.startswith("heads.count"):
            key = "count_head"
        elif n.startswith("heads.read"):
            key = "read_head"
        elif n.startswith("heads.cell"):
            key = "cell_head"
        else:
            key = "task_heads"
        groups.setdefault(key, []).append(p)
    return groups


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def grads_of(loss, params, retain=False, create=False):
    """autograd.grad -> 与 params 对齐的梯度列表 (未用者补零)."""
    gs = torch.autograd.grad(loss, params, retain_graph=retain, create_graph=create,
                             allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(gs, params)]


# ================================================================ 度规 (§2.2)
class Metric:
    """F⁻¹ 度规 (模块头约定). source: 'adam' (v̂ 自优化器状态) / 'efisher' (当场估经验
    Fisher 对角, GA1 不过时的降级一) / 'identity' (单位度规, 降级二 -- 全部报数标注).
    denom(params) 给出该参数列表的 √v̂+ε; inner/norm/cos/precond/whiten 皆接 den 列表."""

    def __init__(self, opt=None, source="adam", efisher=None):
        self.opt = opt
        self.source = source
        self.efisher = efisher          # dict id(param) -> E[g²] 张量 (efisher 用)
        self.eps = 1e-8
        if opt is not None:
            self.eps = float(opt.param_groups[0].get("eps", 1e-8))
        self._b2 = {}
        if opt is not None:
            for g in opt.param_groups:
                for q in g["params"]:
                    self._b2[id(q)] = g["betas"][1]

    def adam_vhat(self, p):
        st = self.opt.state.get(p) if self.opt is not None else None
        if not st or "exp_avg_sq" not in st:
            return None
        v = st["exp_avg_sq"]
        step = st.get("step", None)
        b2 = self._b2.get(id(p), 0.999)
        t = float(step) if step is not None else 1.0
        bc = 1.0 - b2 ** t if t > 0 else 1.0
        return v / bc

    def denom(self, params):
        """√v̂+ε 逐参数 (list). 无优化器状态的参数 (未走过一步) 退化为 1."""
        out = []
        for p in params:
            if self.source == "identity":
                out.append(torch.ones_like(p))
            elif self.source == "efisher":
                f = self.efisher.get(id(p)) if self.efisher else None
                out.append(f.sqrt() + self.eps if f is not None else torch.ones_like(p))
            else:
                v = self.adam_vhat(p)
                out.append(v.sqrt() + self.eps if v is not None else torch.ones_like(p))
        return out

    @staticmethod
    def inner(a, b, den):
        return sum(((x * y) / d).sum() for x, y, d in zip(a, b, den))

    @staticmethod
    def norm(a, den):
        return Metric.inner(a, a, den).clamp(min=0).sqrt()

    @staticmethod
    def cos(a, b, den):
        na, nb = Metric.norm(a, den), Metric.norm(b, den)
        if float(na) == 0.0 or float(nb) == 0.0:
            return torch.zeros(())
        return Metric.inner(a, b, den) / (na * nb)

    @staticmethod
    def precond(a, den):
        """F⁻¹a = a/(√v̂+ε) (AdamW 步方向)."""
        return [x / d for x, d in zip(a, den)]

    @staticmethod
    def whiten(a, den):
        """a/√(√v̂+ε): 白化后欧氏内积 = ⟨·,·⟩_{F⁻¹}."""
        return [x / d.sqrt() for x, d in zip(a, den)]

    @staticmethod
    def whiten_flat(a, den):
        return flat(Metric.whiten(a, den))


def spearman(x, y):
    """Spearman 秩相关 (平局按 argsort 序, 大样本下无碍)."""
    x = x.reshape(-1).double()
    y = y.reshape(-1).double()
    rx = torch.empty_like(x)
    ry = torch.empty_like(y)
    n = x.numel()
    ar = torch.arange(n, dtype=torch.float64, device=x.device)
    rx[torch.argsort(x)] = ar
    ry[torch.argsort(y)] = ar
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = rx.norm() * ry.norm()
    return float((rx * ry).sum() / d) if float(d) > 0 else 0.0


# ================================================================ 滚动留出 (§4.1)
class HoldoutSched:
    """S ⊂ 已引入数量: |S| = max(1, round(α·F)), 每 T_rotate 步换下一组 (固定循环序 =
    按种子置换, 每轮全覆盖后重置换); F = |train_ns(k)|. 扩张时新数量并入序 (on_expand).
    状态: order / ptr / S / rot_step / last_in[n] / cycle 种子."""

    def __init__(self, ns, frac, t_rotate, seed=0, step0=0):
        self.frac = float(frac)
        self.t_rotate = int(t_rotate)
        self.seed = int(seed)
        self.cycle = 0
        self.ns = sorted(int(n) for n in ns)
        self.order = self._perm()
        self.ptr = 0
        self.S = []
        self.rot_step = None
        self.last_in = {n: -1 for n in self.ns}
        self.enabled = self.frac > 0.0 and len(self.ns) > 1
        self.step0 = step0

    def _perm(self):
        g = torch.Generator().manual_seed(self.seed * 7919 + self.cycle)
        idx = torch.randperm(len(self.ns), generator=g).tolist()
        return [self.ns[i] for i in idx]

    def size(self):
        return max(1, int(round(self.frac * len(self.ns)))) if self.enabled else 0

    def _rotate(self, step):
        k = self.size()
        S = []
        while len(S) < k:
            if self.ptr >= len(self.order):
                self.cycle += 1
                self.order = self._perm()
                self.ptr = 0
            n = self.order[self.ptr]
            self.ptr += 1
            if n not in S:
                S.append(n)
        self.S = sorted(S)
        self.rot_step = step
        for n in self.S:
            self.last_in[n] = step

    def current(self, step):
        """本步的 S (集合). 跨过轮换边界时先轮换."""
        if not self.enabled:
            return set()
        if self.rot_step is None or step - self.rot_step >= self.t_rotate:
            self._rotate(step)
        return set(self.S)

    def on_expand(self, ns):
        """K 扩张: 新数量并入 (追加到循环序末端, 当轮即可入 S); 现 S 不变."""
        new = [int(n) for n in ns if int(n) not in self.ns]
        self.ns = sorted(self.ns + new)
        self.order = self.order + new
        for n in new:
            self.last_in[n] = -1
        self.enabled = self.frac > 0.0 and len(self.ns) > 1

    def retreat(self, keep_ns):
        """前沿退回 ([U] 2026-08-17 第四阶段 改动 1): 只保留 keep_ns 中的数量; 循环序按原序过滤
        (指针按过滤后位置折算), 现 S 若含撤下数量则下一次 current() 立即重轮换."""
        keep = {int(n) for n in keep_ns}
        old_order = self.order
        self.ns = sorted(n for n in self.ns if n in keep)
        self.order = [n for n in old_order if n in keep]
        self.ptr = min(sum(1 for n in old_order[:self.ptr] if n in keep), len(self.order))
        self.last_in = {n: v for n, v in self.last_in.items() if n in keep}
        if any(n not in keep for n in self.S):
            self.S = [n for n in self.S if n in keep]
            self.rot_step = None                   # 下次 current() 重轮换
        self.enabled = self.frac > 0.0 and len(self.ns) > 1

    def state(self):
        return dict(frac=self.frac, t_rotate=self.t_rotate, seed=self.seed, cycle=self.cycle,
                    ns=self.ns, order=self.order, ptr=self.ptr, S=self.S,
                    rot_step=self.rot_step, last_in=self.last_in)

    def load_state(self, d):
        for k in ("cycle", "ns", "order", "ptr", "S", "rot_step", "last_in"):
            if k in d:
                setattr(self, k, d[k])
        self.enabled = self.frac > 0.0 and len(self.ns) > 1


# ================================================================ 交叉项 u_S (§5.2)
def cross_term(params, den, g_V, g_S, recompute_gS, delta_mult=1.0, opt=None,
               delta_abs=None):
    """u_S = [g_S(Θ + δ·F⁻¹ĝ_V) − g_S(Θ)] / (δ·‖g_S‖_{F⁻¹}),  ĝ_V = g_V/‖g_V‖_{F⁻¹}.
    δ 定法 (§10): ‖δ·F⁻¹ĝ_V‖₂ = delta_mult × 上一步 AdamW 实际位移 ‖lr·m̂/(√v̂+ε)‖₂
    (优化器状态缺失时退化为 δ = delta_mult·lr); delta_abs 给定则直接用 (GA2 扫描).
    recompute_gS(): 在**当前参数值**上重算 S 的画布路由损失并返回其对 params 的梯度列表.
    参数在扰动后逐位复原 (copy_ 回保存副本). 返回 (u_S list, diag dict)."""
    nV = Metric.norm(g_V, den)
    nS = Metric.norm(g_S, den)
    diag = dict(gV=float(nV), gS=float(nS))
    if float(nV) == 0.0 or float(nS) == 0.0:
        return [torch.zeros_like(p) for p in params], dict(diag, delta=0.0, u=0.0,
                                                           degenerate=True)
    dirn = Metric.precond([g / nV for g in g_V], den)          # F⁻¹ĝ_V
    ndir = math.sqrt(sum(float((d * d).sum()) for d in dirn))
    delta = (float(delta_abs) if delta_abs is not None
             else delta_of(opt, params, delta_mult, ndir))
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, d in zip(params, dirn):
            p.add_(d, alpha=delta)
    try:
        g_S2 = recompute_gS()
    finally:
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)                                     # 逐位复原
    u = [(a - b) / (delta * nS) for a, b in zip(g_S2, g_S)]
    diag.update(delta=float(delta), u=float(Metric.norm(u, den)), degenerate=False)
    return u, diag


def last_step_disp(opt, params):
    """上一步 AdamW 实际位移 ‖lr·m̂/(√v̂+ε)‖₂ (自优化器状态; 缺状态返回 None)."""
    if opt is None:
        return None
    tot = 0.0
    lr = None
    seen = False
    for p in params:
        st = opt.state.get(p)
        if not st or "exp_avg" not in st:
            continue
        for g in opt.param_groups:
            if any(q is p for q in g["params"]):
                lr = g["lr"]
                b1, b2 = g["betas"]
                eps = g["eps"]
                break
        t = float(st["step"])
        m = st["exp_avg"] / (1 - b1 ** t)
        v = st["exp_avg_sq"] / (1 - b2 ** t)
        tot += float(((lr * m / (v.sqrt() + eps)) ** 2).sum())
        seen = True
    return math.sqrt(tot) if seen else None


def delta_of(opt, params, delta_mult, ndir):
    disp = last_step_disp(opt, params)
    if disp is None or disp == 0.0:
        lr = opt.param_groups[0]["lr"] if opt is not None else 3e-4
        return float(delta_mult * lr)
    return float(delta_mult * disp / max(ndir, 1e-30))


# ================================================================ 曲率 (GA2 / M7)
def _math_sdpa():
    """double backward 需要注意力走 math 核 (flash/efficient 核无二阶导)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:                       # 旧版 torch: 无此接口, 直接跑
        import contextlib
        return contextlib.nullcontext()


def hvp(loss_fn, params, vec):
    """真 HVP: H·vec = ∇_Θ ⟨∇L(Θ), vec⟩ (double backward). loss_fn() 须重建图."""
    with _math_sdpa():
        loss = loss_fn()
        g = grads_of(loss, params, retain=True, create=True)
        s = sum((a * v).sum() for a, v in zip(g, vec))
        return grads_of(s, params, retain=False, create=False)


def hutchinson_trace(loss_fn, params, gen=None):
    """tr(H) 单次 Hutchinson 估计: z Rademacher, zᵀHz. 返回 float."""
    z = []
    for p in params:
        r = torch.randint(0, 2, p.shape, generator=gen)          # CPU 生成 (CRN)
        z.append((r.float() * 2.0 - 1.0).to(p.device))
    hz = hvp(loss_fn, params, z)
    return float(sum((a * b).sum() for a, b in zip(hz, z)))


# ================================================================ M1 R 谱
def gram_R(vecs, demean=True):
    """vecs: (F, D) 已白化梯度 (行 = 数量). 去均值 → Gram → 本征谱 → 参与比
    R = (Σλ)²/Σλ². 返回 dict(R, eig(list 降序), evecs (F,F), gram)."""
    X = vecs
    if demean:
        X = X - X.mean(0, keepdim=True)
    G = X @ X.T
    G = 0.5 * (G + G.T)
    ev, U = torch.linalg.eigh(G.double())
    ev = ev.flip(0).clamp(min=0)
    U = U.flip(1)
    s1, s2 = float(ev.sum()), float((ev * ev).sum())
    R = (s1 * s1 / s2) if s2 > 0 else float("nan")
    return dict(R=R, eig=[float(e) for e in ev], evecs=U.float(), gram=G)


def alignment_A(eig, evecs, ell):
    """对齐量 A = F·Σ_k c_k²·λ̂_k, c_k² = ⟨u_k, ℓ̃⟩²/‖ℓ̃‖², ℓ̃ = 逐数量超额损失去均值.
    A≫1 残差落在梯度已强的方向 (学得动) / A≈1 无对齐 / A≪1 结构性地板."""
    l = torch.tensor(ell, dtype=torch.float32) if not torch.is_tensor(ell) else ell.float()
    l = l - l.mean()
    nl = float((l * l).sum())
    if nl == 0.0:
        return float("nan")
    lam = torch.tensor(eig, dtype=torch.float32)
    tot = float(lam.sum())
    if tot == 0.0:
        return float("nan")
    lam_hat = lam / tot
    U = evecs.to(l.device)
    c2 = ((U.T @ l) ** 2) / nl
    return float(len(ell) * (c2 * lam_hat).sum())


def R_truncation(vecs, ns):
    """R(f) 截断曲线: 按数量升序取前 f 个数量的子 Gram (f=2..F). 返回 list of (f, R)."""
    order = sorted(range(len(ns)), key=lambda i: ns[i])
    out = []
    for f in range(2, len(ns) + 1):
        idx = order[:f]
        out.append((f, gram_R(vecs[idx])["R"]))
    return out


# ================================================================ 画布路由损失 (分路)
def canvas_route(model, x2, th, tg, truth5, cands, ns, c_read, mu):
    """一次画布路由前向 (带梯度): x2 (B,side,side) → h → 任务头/直读头; 候选集 cands
    (nG, NCAND, side, side) 各自过主干 (场景模态), 每组 g 条 rollout 共享.
    返回 dict(h, tl, t5, read, lrow (B,) 逐行 L_task + μ·L_read, l_task, l_read, accs)."""
    B = x2.shape[0]
    nG = cands.shape[0]
    g = B // nG
    h = model.encode_canvas(x2)["cls"]
    tl = model.heads.task_logits(h, th)
    hc = model.encode_scene(cands.view(nG * D.N_CAND, *cands.shape[2:]))["cls"]
    hc = hc.view(nG, D.N_CAND, -1).repeat_interleave(g, 0)
    t5 = model.heads.t5_scores(h, hc)
    parts = [F.cross_entropy(tl[t], tg[t], reduction="none") for t in TASKS_LIN]
    parts.append(F.cross_entropy(t5, truth5, reduction="none"))
    task_row = sum(parts) / math.sqrt(6.0)
    rl = model.heads.read(h)
    logp = F.log_softmax(rl, dim=-1)
    a = logp[:, ABSTAIN].exp()
    log1m = torch.log1p(-a.clamp(max=1.0 - 1e-6))
    lpn = logp.gather(1, (ns - 1).unsqueeze(1)).squeeze(1)
    read_row = (1.0 - a) * (log1m - lpn) + a * c_read
    lrow = task_row + mu * read_row
    return dict(h=h, tl=tl, t5=t5, hc=hc, read=rl, lrow=lrow,
                task_row=task_row, read_row=read_row,
                l_task=task_row.mean(), l_read=read_row.mean(),
                accs=GR.task_accs(tl, t5, tg, truth5).detach())


def null_targets(ns, th, truth5, k, rng, dev):
    """X2 打乱数量标签: 每行换一个训练域内的错数 n' 并按该行 θ 重算五任务标签,
    t5 真值位换成一个错候选. 返回 (tg', truth5', ns')."""
    tns = list(D.train_ns(k))
    n_new, tg = [], {t: [] for t in TASKS_LIN}
    t5n = []
    for i in range(ns.shape[0]):
        n = int(ns[i])
        pool = [m for m in tns if m != n]
        m = pool[int(torch.randint(0, len(pool), (1,), generator=rng))]
        theta = dict(tau=int(th["tau"][i]) + 1, p=D.P_CHOICES[int(th["p"][i])],
                     m=D.M_CHOICES[int(th["m"][i])])
        tt = D.targets(m, theta)
        for t in TASKS_LIN:
            tg[t].append(tt[t])
        n_new.append(m)
        alt = [j for j in range(D.N_CAND) if j != int(truth5[i])]
        t5n.append(alt[int(torch.randint(0, len(alt), (1,), generator=rng))])
    return ({t: torch.tensor(v, device=dev) for t, v in tg.items()},
            torch.tensor(t5n, device=dev), torch.tensor(n_new, device=dev))


# ================================================================ c 通道 (§6)
class CChannel:
    """闭式任务头梯度 (§6.2, 严格线性 ⇒ 零额外反传) + G_n 窗均值 + c^(i) =
    cos_{F⁻¹}(G^(i), Ḡ_{¬n}) + 打乱零假设.
    G^(i) 分块: 线性任务 t: a_t = w_t (p_t − y_t) (C_t), x_t = [h; φ(θ_t)] 或 h (d_in),
      ∂/∂W_t = a_t ⊗ x_t, ∂/∂b_t = a_t;  检索 t5: score_k = (W₅h)·h_k ⇒
      ∂/∂W₅ = w₅ Δ ⊗ h, Δ = Σ_k (p_k − y_k) h_k (d).
    内积用度规 M² = 1/(√v̂+ε) 逐元素 (= ⟨·,·⟩_{F⁻¹}): ⟨a⊗x, Q⟩ = aᵀ Q x, Q = M²⊙Ḡ.
    G_n 口径 ([U] 2026-08-17 改动 4): **无 EMA 缓冲** -- 评测窗内逐数量累加 (acc/n_acc: 每步
    该数量全部 rollout 的 mean G^(i) 之和 / 步数), 每评 roll() 把窗均值整体换成 bar/n_bar,
    下一窗的 c 全用这份 bar (Ḡ_{¬n} = bar 在 dom\\{n} 已填行的均值); 无偏差校正、无 X4 重建.
    一致的码族切换 (全体同时换码系) 只经一评就整体换新, 特异漂移使 c 低.
    bar 存未白化的 raw G_n (度规每步现取), 形状 (K_MAX+1, C_t, d_in) 逐块."""

    def __init__(self, model, zeta, dev):
        self.zeta = float(zeta)               # 名义字段 (改动 4 后不参与计算, 保留供检查点兼容)
        self.dev = dev
        self.hp = head_params(model)          # 顺序: t1.w t1.b t2.w t2.b t3.w t3.b t4.w t4.b t5.w t6.w t6.b
        self.blocks = []                       # (name, w_idx, b_idx|None, shape)
        i = 0
        for t in ("t1", "t2", "t3", "t4", "t5", "t6"):
            lin = getattr(model.heads, t)
            wi = i
            i += 1
            bi = None
            if lin.bias is not None:
                bi = i
                i += 1
            self.blocks.append((t, wi, bi, tuple(lin.weight.shape)))
        self.acc, self.bar = {}, {}
        self.n_acc = torch.zeros(GM.K_MAX + 1, dtype=torch.long, device=dev)   # 窗内入册步数
        self.n_bar = torch.zeros(GM.K_MAX + 1, dtype=torch.long, device=dev)   # 上评窗均值来源步数
        for t, wi, bi, shp in self.blocks:
            self.acc[t + ".w"] = torch.zeros(GM.K_MAX + 1, *shp, device=dev)
            self.bar[t + ".w"] = torch.zeros(GM.K_MAX + 1, *shp, device=dev)
            if bi is not None:
                self.acc[t + ".b"] = torch.zeros(GM.K_MAX + 1, shp[0], device=dev)
                self.bar[t + ".b"] = torch.zeros(GM.K_MAX + 1, shp[0], device=dev)
        self.rolls = 0

    @property
    def cnt(self):
        """兼容读法: 已填 (有窗均值) 的逐数量计数 = n_bar."""
        return self.n_bar

    # -------------------------------------------------- 闭式部件
    @torch.no_grad()
    def parts(self, heads, h, tl, t5_scores, hc, tg, truth5, th, w):
        """一批 rollout 的闭式梯度因子 (来自影子读者的 h/p, v4.1 §6.4).
        返回 dict t -> (a (B,C_t), x (B,d_in)); t5 -> (Δ (B,d), h (B,d))."""
        out = {}
        et = heads.emb_tau(th["tau"])
        xs = {"t1": torch.cat([h, et], -1), "t2": h, "t3": torch.cat([h, et], -1),
              "t4": torch.cat([h, heads.emb_p(th["p"])], -1),
              "t6": torch.cat([h, heads.emb_m(th["m"])], -1)}
        for ti, t in enumerate(TASKS_LIN):
            p = tl[t].softmax(-1)
            y = F.one_hot(tg[t], p.shape[1]).float()
            wt = float(w[GR.TASK_KEYS.index(t)])
            out[t] = (wt * (p - y), xs[t])
        p5 = t5_scores.softmax(-1)
        y5 = F.one_hot(truth5, p5.shape[1]).float()
        delta = torch.einsum("bc,bcd->bd", p5 - y5, hc)
        out["t5"] = (float(w[4]) * delta, h)
        return out

    def _m2(self, den):
        """度规 M² = 1/(√v̂+ε) 逐块 (den 与 self.hp 对齐)."""
        m2 = {}
        for t, wi, bi, shp in self.blocks:
            m2[t + ".w"] = 1.0 / den[wi]
            if bi is not None:
                m2[t + ".b"] = 1.0 / den[bi]
        return m2

    def ready(self, ns):
        """就绪 = 训练域内已有窗均值的数量 ≥ 2 (Ḡ_{¬n} 可算; 新扩张数量本窗无 bar 不阻塞 --
        c^(i) 只需 Ḡ_{¬n}, 不需 n 自己的行). 首评前恒 False."""
        return sum(1 for n in ns if int(self.n_bar[int(n)]) > 0) >= 2

    def _hat(self, n):
        """上评窗均值 G_n (raw)."""
        return {k: v[n] for k, v in self.bar.items()}

    @torch.no_grad()
    def c_of(self, parts, ns, den, dom, tg=None):
        """逐 rollout c^(i) (B,). ns (B,) 各 rollout 数量; dom = 当前训练域数量列表
        (Ḡ_{¬n} 取 dom \\ {n} 中已填行的均值). 未就绪行 → 0."""
        B = ns.shape[0]
        m2 = self._m2(den)
        filled = [n for n in dom if int(self.n_bar[int(n)]) > 0]
        if len(filled) < 2:
            return torch.zeros(B, device=self.dev)
        S = {k: torch.zeros_like(v[0]) for k, v in self.bar.items()}
        for n in filled:
            hn = self._hat(int(n))
            for k in S:
                S[k] += hn[k]
        c = torch.zeros(B, device=self.dev)
        uniq = sorted(set(int(n) for n in ns.tolist()))
        for n in uniq:
            rows = (ns == n).nonzero().squeeze(1)
            others = [m for m in filled if m != n]
            if len(others) < 1:
                continue
            hn = self._hat(n) if int(self.n_bar[n]) > 0 else None
            Gbar = {}
            for k in S:
                Gbar[k] = (S[k] - (hn[k] if hn is not None else 0.0)) / len(others)
            Q = {k: m2[k] * Gbar[k] for k in S}
            nG2 = sum(float((Gbar[k] * Q[k]).sum()) for k in S)
            num = torch.zeros(rows.shape[0], device=self.dev)
            nrm2 = torch.zeros(rows.shape[0], device=self.dev)
            for t, wi, bi, shp in self.blocks:
                a, x = parts[t]
                a, x = a[rows], x[rows]
                num += torch.einsum("bc,cj,bj->b", a, Q[t + ".w"], x)
                nrm2 += torch.einsum("bc,cj,bj->b", a * a, m2[t + ".w"], x * x)
                if bi is not None:
                    num += a @ Q[t + ".b"]
                    nrm2 += (a * a) @ m2[t + ".b"]
            den_i = nrm2.clamp(min=0).sqrt() * math.sqrt(max(nG2, 0.0))
            c[rows] = torch.where(den_i > 0, num / den_i.clamp(min=1e-30),
                                  torch.zeros_like(num))
        return c

    # -------------------------------------------------- 投影率 c ([U] 2026-08-17 第四阶段 改动 2)
    PROJ_RCOND = 1e-6     # Gram 伪逆的相对截断 (= 正交基 QR 的秩判定; 报 rank)

    @torch.no_grad()
    def v_basis(self, parts, ns, den, isV):
        """V 侧子空间: 本步 n∈V 的 rollout 按数量取 mean G^(i) = Ḡ_m (raw, 每块 C_t×d_in / C_t),
        Gram[a,b] = ⟨Ḡ_a, Ḡ_b⟩_{F⁻¹} (M² = 1/(√v̂+ε) 逐元素), 伪逆 (相对截断 PROJ_RCOND) = span 的
        正交投影 (等价于草图空间 QR 后 B Bᵀ, 秩不足自动截断). **不中心化** (硬约束 a).
        返回 dict(ms (list of m), G (dict k -> (K, C, d)), ginv (K,K), rank, m2) 或 None (无 V 行)."""
        rows_V = isV.nonzero().squeeze(1)
        if rows_V.numel() == 0:
            return None
        m2 = self._m2(den)
        ms = sorted(set(int(n) for n in ns[rows_V].tolist()))
        G = {}
        for t, wi, bi, shp in self.blocks:
            a, x = parts[t]
            Gw, Gb = [], []
            for m in ms:
                r = ((ns == m) & isV).nonzero().squeeze(1)
                Gw.append(torch.einsum("bc,bj->cj", a[r], x[r]) / r.shape[0])
                if bi is not None:
                    Gb.append(a[r].mean(0))
            G[t + ".w"] = torch.stack(Gw)                       # (K, C, d)
            if bi is not None:
                G[t + ".b"] = torch.stack(Gb)                   # (K, C)
        K = len(ms)
        gram = torch.zeros(K, K, device=self.dev, dtype=torch.float64)
        for k, Gk in G.items():
            Wk = (Gk * m2[k]).reshape(K, -1).double()            # M²⊙Ḡ
            gram += Wk @ Gk.reshape(K, -1).double().T
        gram = 0.5 * (gram + gram.T)
        ev, U = torch.linalg.eigh(gram)
        keep = ev > self.PROJ_RCOND * float(ev.max().clamp(min=1e-30))
        inv = torch.where(keep, 1.0 / ev.clamp(min=1e-30), torch.zeros_like(ev))
        ginv = (U * inv) @ U.T
        return dict(ms=ms, G=G, ginv=ginv.float(), rank=int(keep.sum()), m2=m2, k=K)

    @torch.no_grad()
    def c_from_basis(self, basis, parts, rows):
        """给定 V 子空间, 对 rows 各 rollout 算投影率 c^(i) = ‖P_V ĝ^(i)‖_{F⁻¹}/‖ĝ^(i)‖_{F⁻¹}
        = sqrt(qᵀ Gram⁺ q / ‖ĝ‖²), q[m] = ⟨ĝ^(i), Ḡ_m⟩_{F⁻¹} (闭式块结构: aᵀ(M²⊙Ḡ_m)x). ∈ [0,1].
        返回 (B_rows,) 张量; basis=None ⇒ 全 0."""
        n = rows.shape[0]
        if basis is None or n == 0:
            return torch.zeros(n, device=self.dev)
        m2, G, K = basis["m2"], basis["G"], basis["k"]
        q = torch.zeros(n, K, device=self.dev)
        nrm2 = torch.zeros(n, device=self.dev)
        for t, wi, bi, shp in self.blocks:
            a, x = parts[t]
            a, x = a[rows], x[rows]
            Qw = G[t + ".w"] * m2[t + ".w"]                       # (K, C, d) = M²⊙Ḡ_m
            q += torch.einsum("bc,mcj,bj->bm", a, Qw, x)
            nrm2 += torch.einsum("bc,cj,bj->b", a * a, m2[t + ".w"], x * x)
            if bi is not None:
                q += a @ (G[t + ".b"] * m2[t + ".b"]).T
                nrm2 += (a * a) @ m2[t + ".b"]
        proj2 = torch.einsum("bm,mk,bk->b", q, basis["ginv"], q)
        c2 = proj2 / nrm2.clamp(min=1e-30)
        c = c2.clamp(min=0.0, max=1.0).sqrt()
        c = torch.where(nrm2 > 0, c, torch.zeros_like(c))
        return c

    @torch.no_grad()
    def c_proj(self, parts, ns, den, isS):
        """投影率 c 全批装配 ([U] 第四阶段 改动 2 三条硬约束): 只在 n∈S 的行上算 (V 行恒 0 ⇒ κ 项对
        V 组置零), 子空间 = 本步 V 行按数量的 mean G^(i) 张成 (不中心化). 返回 (c (B,), diag)."""
        B = ns.shape[0]
        c = torch.zeros(B, device=self.dev)
        rows_S = isS.nonzero().squeeze(1)
        basis = self.v_basis(parts, ns, den, ~isS)
        diag = dict(rank=(basis["rank"] if basis else 0), kV=(basis["k"] if basis else 0),
                    nS=int(rows_S.numel()))
        if basis is not None and rows_S.numel() > 0:
            c[rows_S] = self.c_from_basis(basis, parts, rows_S)
        return c, basis, diag

    @torch.no_grad()
    def update(self, parts, ns):
        """入册: 每个数量本步的 mean_i G^(i) (raw) 累加进本窗 acc (步计数 n_acc)."""
        uniq = sorted(set(int(n) for n in ns.tolist()))
        for n in uniq:
            rows = (ns == n).nonzero().squeeze(1)
            for t, wi, bi, shp in self.blocks:
                a, x = parts[t]
                a, x = a[rows], x[rows]
                Gw = torch.einsum("bc,bj->cj", a, x) / rows.shape[0]
                self.acc[t + ".w"][n].add_(Gw)
                if bi is not None:
                    self.acc[t + ".b"][n].add_(a.mean(0))
            self.n_acc[n] += 1

    @torch.no_grad()
    def roll(self):
        """每评 (改动 4): 本窗有入册的数量, 其 bar 整体换成本窗均值 (旧值弃); 本窗无入册的
        数量保留上一份 bar (不清零 -- 一个数量偶然缺席一窗不应使 Ḡ_{¬n} 少一行). 窗清零.
        返回本窗入册数量数."""
        filled = (self.n_acc > 0).nonzero().squeeze(1)
        for i in filled.tolist():
            k = float(self.n_acc[i])
            for name in self.acc:
                self.bar[name][i].copy_(self.acc[name][i] / k)
            self.n_bar[i] = self.n_acc[i]
        for v in self.acc.values():
            v.zero_()
        self.n_acc.zero_()
        self.rolls += 1
        return int(filled.numel())

    def state(self):
        return dict(bar={k: v.cpu() for k, v in self.bar.items()}, n_bar=self.n_bar.cpu(),
                    acc={k: v.cpu() for k, v in self.acc.items()}, n_acc=self.n_acc.cpu(),
                    zeta=self.zeta, rolls=self.rolls)

    @torch.no_grad()
    def load_state(self, d):
        if "bar" not in d:                     # 旧检查点 (EMA 缓冲制): 缓冲不迁移, 首评后重填
            return
        for k, v in d["bar"].items():
            if k in self.bar:
                self.bar[k].copy_(v.to(self.dev))
        self.n_bar.copy_(d["n_bar"].to(self.dev))
        for k, v in d.get("acc", {}).items():
            if k in self.acc:
                self.acc[k].copy_(v.to(self.dev))
        if "n_acc" in d:
            self.n_acc.copy_(d["n_acc"].to(self.dev))
        self.rolls = int(d.get("rolls", 0))


def head_grad_direct(model, heads_out, w, tg, truth5):
    """测试用: 逐 rollout Σ_t w_t CE_t 对 t1..t6 参数的 autograd 梯度 (与闭式对表).
    heads_out: canvas_route 输出 (需图). 返回 list[list of grads] 逐 rollout."""
    hp = head_params(model)
    B = heads_out["h"].shape[0]
    outs = []
    for i in range(B):
        loss = 0.0
        for ti, t in enumerate(TASKS_LIN):
            loss = loss + float(w[GR.TASK_KEYS.index(t)]) * F.cross_entropy(
                heads_out["tl"][t][i:i + 1], tg[t][i:i + 1])
        loss = loss + float(w[4]) * F.cross_entropy(heads_out["t5"][i:i + 1],
                                                    truth5[i:i + 1])
        outs.append([g.detach() for g in grads_of(loss, hp, retain=True)])
    return outs


# ================================================================ 定标码 (§7.2)
def cal_program(kind, n, rng, l_rand=None):
    """合成码表: 'const' 全数量同画布 (12 章固定程序, 由 rng 一次抽定后调用方复用) /
    'random' 每数量独立随机配置 / 'unary' n 枚点章按格序 / 'place' 基数-4 位值码
    (格 j 章型 = 第 j 位数字−1, 数字 0 = 空; 低位在前). 返回程序 (list[aid])."""
    if kind == "unary":
        assert n <= GM.T
        return [GM.aid_pack(c, GM.STAMP_DOT) for c in range(n)]
    if kind == "place":
        prog, j, v = [], 0, int(n)
        while v > 0:
            dgt = v % (GM.S + 1)
            if dgt > 0:
                prog.append(GM.aid_pack(j, dgt - 1))
            v //= (GM.S + 1)
            j += 1
        return prog
    if kind == "random":
        L = int(l_rand) if l_rand else int(torch.randint(1, 41, (1,), generator=rng))
        cells = torch.randperm(GM.T, generator=rng)[:L]
        types = torch.randint(0, GM.S, (L,), generator=rng)
        return (cells * GM.S + types).tolist()
    if kind == "const":
        cells = torch.randperm(GM.T, generator=rng)[:12]
        types = torch.randint(0, GM.S, (12,), generator=rng)
        return (cells * GM.S + types).tolist()
    raise ValueError(kind)


def cal_table(kind, ns, seed):
    """kind -> dict n -> program (const: 全 n 同程序)."""
    rng = torch.Generator().manual_seed(seed)
    if kind == "const":
        p = cal_program("const", 0, rng)
        return {n: list(p) for n in ns}
    return {n: cal_program(kind, n, rng) for n in ns}


# ================================================================ GA5 弃权地板
def blind_task_ce(k):
    """画布盲最优读者的六任务 CE (N ~ U(train_ns(k)), θ 按抽样分布精确枚举; t5 盲分布 =
    剔除两条硬负后在 4 项均匀 → log 4), 合成 L_task_blind = Σ/√6. 返回 dict."""
    ns = torch.tensor(D.train_ns(k), dtype=torch.float64)
    W = ns.shape[0]

    def H(p):
        p = p[p > 0]
        return float(-(p * p.log()).sum())

    ce = {}
    acc1, acc3 = 0.0, 0.0
    for tau in range(1, k):
        q = float((ns > tau).sum()) / W
        acc1 += H(torch.tensor([q, 1 - q], dtype=torch.float64))
        p3 = torch.full((GM.K_MAX,), 0.0, dtype=torch.float64)
        p3[tau - 1] += float((ns <= tau).sum()) / W
        for n in ns[ns > tau]:
            p3[int(n) - 1] += 1.0 / W
        acc3 += H(p3)
    ce["t1"] = acc1 / (k - 1)
    ce["t3"] = acc3 / (k - 1)
    ce["t2"] = H(torch.tensor([float((ns % 2 == 0).sum()) / W,
                               float((ns % 2 == 1).sum()) / W], dtype=torch.float64))
    a4 = 0.0
    for p in D.P_CHOICES:
        r = ns.long() % p
        a4 += H(torch.bincount(r, minlength=p).double() / W)
    ce["t4"] = a4 / len(D.P_CHOICES)
    ce["t5"] = math.log(D.N_CAND - D.N_HARD)
    ce["t6"] = math.log(W)
    ce["l_task_blind"] = sum(ce[t] for t in GR.TASK_KEYS) / math.sqrt(6.0)
    return ce


def abstain_floor(k, mu, c_read):
    """GA5 弃权地板 = L_task_blind + μ·c (直读全弃权 ⇒ 读损失恒 c). L* = LSTAR_FRAC × 地板."""
    ce = blind_task_ce(k)
    floor = ce["l_task_blind"] + mu * c_read
    return dict(ce=ce, floor=floor, l_star=LSTAR_FRAC * floor, c_read=c_read, mu=mu)


# ================================================================ GA1 经验 Fisher
def efisher_diag(model, params, loss_fns):
    """经验 Fisher 对角: 逐样本梯度平方均值. loss_fns: 可迭代的零参可调用, 每个返回
    单样本损失 (调用方切片到 batch=1). 返回与 params 对齐的 E[g²] 列表."""
    acc = [torch.zeros_like(p) for p in params]
    n = 0
    for fn in loss_fns:
        loss = fn()
        gs = grads_of(loss, params)
        for a, g in zip(acc, gs):
            a += g * g
        n += 1
    return [a / max(n, 1) for a in acc]


def ga1_report(model, opt, params, ef):
    """v̂ (Adam) 对经验 Fisher 对角的 Spearman: 全局 + 逐参数组. 返回 dict."""
    met = Metric(opt, "adam")
    vh = []
    for p in params:
        v = met.adam_vhat(p)
        vh.append(v if v is not None else torch.zeros_like(p))
    groups = param_groups_of(model)
    ids = {id(p): i for i, p in enumerate(params)}
    out = {"global": spearman(flat(vh), flat(ef))}
    for name, ps in groups.items():
        idx = [ids[id(p)] for p in ps if id(p) in ids]
        if not idx:
            continue
        out[name] = spearman(flat([vh[i] for i in idx]), flat([ef[i] for i in idx]))
    vals = [v for k, v in out.items() if k != "global"]
    out["pass"] = bool(out["global"] >= GA1_GLOBAL and all(v >= GA1_GROUP for v in vals))
    return out


def snr_from_cos(c):
    """两半批估计的余弦 → 单评 (全批) 信噪比: 半批 SNR = √(c/(1−c)), 全批 ×√2."""
    c = max(min(float(c), 0.999999), -0.999999)
    if c <= 0:
        return 0.0
    return math.sqrt(2.0 * c / (1.0 - c))
