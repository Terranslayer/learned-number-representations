# symemerge/numcode/pred/step.py
"""前向装配 + 损失 (spec-v2 §4-§5) + 更新规则 (§7): 场景路由 → 写 (D1, 冻结) → 直通渲染 → 信道 →
当步笔记读 (闭环 B 方案) · 池抽笔记读 · 随机画布掩码 · S/V 梯度分路 · 交叉项 u_S (硬类别冻结 + CRN)
· 读数 (𝒮 / τ / cos / ℓ_ex / 投影率 c / 地板).

三条信号路径 (§2): 精确梯度 (L_task, L_read, L_mask, L_λ) → Θ_E + 各读出头; 直通梯度 (经 D1 穿过) → Θ_E;
无梯度 (verifier 成绩、池记账). 不存在策略梯度.

度规与交叉项复用 ../align.py (Metric / cross_term / grads_of / null_targets, v4.1 §7 理论继承);
本模块只写物理落地.
"""
import contextlib
import math

import torch
import torch.nn.functional as F

from .. import align as AL
from .. import data as D
from .. import geometry as GM
from ..grpo import task_accs
from ..losses import sample_mask_cells
from . import render as R
from .data import LIN, TASKS

W_INDEX = {t: i for i, t in enumerate(TASKS)}


# ================================================================ 读出与逐图像损失
def lds_rows(model, h, hc, th, tg, truth5, ns, w, mu):
    """逐图像 L_ds (§5.1) = Σ_t w_t CE(ŷ_t, y_t) + μ CE(N̂, N). h (B,d) 该图像 CLS; hc (B,C,d) 候选 CLS.
    返回 dict(rows (B,), accs (B,6) 六任务命中, ce (B,6) 六任务未加权 CE (detach, TASKS 序), read_pred (B,) 直读 N̂, tl, t5, rl)."""
    tl = model.heads.task_logits(h, th)
    t5 = model.heads.t5_scores(h, hc)
    rows = 0.0
    ce = [None] * len(W_INDEX)
    for t in LIN:
        ce[W_INDEX[t]] = F.cross_entropy(tl[t], tg[t], reduction="none")
        rows = rows + w[W_INDEX[t]] * ce[W_INDEX[t]]
    ce[W_INDEX["t5"]] = F.cross_entropy(t5, truth5, reduction="none")
    rows = rows + w[W_INDEX["t5"]] * ce[W_INDEX["t5"]]
    rl = model.heads.read(h)
    read_row = F.cross_entropy(rl, ns - 1, reduction="none")
    rows = rows + mu * read_row
    return dict(rows=rows, read_row=read_row, accs=task_accs(tl, t5, tg, truth5).detach(),
                ce=torch.stack(ce, dim=-1).detach(),   # 逐任务未加权 CE (B,6): t4 权重实验的观测量, 与 w 无关
                read_pred=(rl.argmax(-1) + 1).detach(), tl=tl, t5=t5, rl=rl,
                marg=margins(tl, t5, tg, truth5))    # v3.1 C5.1 正确类边距 (detach; y_soft 标的)


def cell_perm(cfg, dev):
    """读者恢复时间实测 (§9.2 C_pool 行) 的「人为切换编码」: perm_seed>0 时对写头输出做固定格置换
    (logits[:, π]), 码字整体搬家而写头/读者权重不动; 0 = 无. 由 write_logits 统一施加 (训练/评测/码表同一处)."""
    ps = int(getattr(cfg, "perm_seed", 0) or 0)
    if ps <= 0:
        return None
    return torch.randperm(GM.T, generator=torch.Generator().manual_seed(ps)).to(dev)


def write_logits(model, tokens, cfg):
    """D1 写头 (§4.3): ℓ_j = W_D1 z_j; 可选固定格置换 (perm_seed, 只为恢复时间实测)."""
    lg = model.D1(tokens)
    perm = cell_perm(cfg, tokens.device)
    return lg if perm is None else lg[:, perm]


def encode_cands(model, cands):
    """候选集 (B,C,side,side) 各自过主干 (候选是场景 ⇒ 带 flag) -> (B,C,d)."""
    B, C = cands.shape[:2]
    hc = model.encode(cands.reshape(B * C, *cands.shape[2:]), is_scene=True)["cls"]
    return hc.view(B, C, -1)


def lam_cost(lam, p, k):
    """§5.2' ([U] 2026-08-20): L_λ = λ·Σ_{j:k_j≠空}(1−p_{j,空}) — 只对实际落章的格计费 (落章判定 = 硬类别
    argmax, 无梯度路径); 全空画布 ⇒ 恒 0 且零梯度 (旧口径在纸已全白后仍持续加深饱和, lam02 跑实测 6000 步
    过冲, 由此修复). checks.py A10 与 tests 直接调本函数 (与训练路径同一代码对象, 防插针漂移)."""
    ink = (k != R.EMPTY).float()
    return lam * ((1.0 - p[..., R.EMPTY]) * ink).sum(-1).mean()


def scene_side(model, sb, w, mu, cfg, draw, tpl, k_hard=None, field=False):
    """场景路由 + 写 + 当步闭环 (§4.5): x1 (flag) → h1 (场景 L_ds) 与 z → D1 → 直通渲染 → 信道 → x2 (无 flag)
    → h2 (笔记 L_ds, 标签 = 场景真值). 候选 CLS 一次编码, 场景查询与笔记查询共用.
    k_hard: 硬类别冻结 (§7.4; FD 旧路径的第二次前向 / 显式场在扰动参数处的求值).
    field=True: 显式场模式 (D-CHK C10; u_S 精确 vjp 用) -- 信道后画布 y 的值 = channel(hard) (直通前向), 微分经 soft;
    读路径的输入 x2 改为 y.detach() 叶子 (值钉在硬画布), 使 ∂L/∂x2 与 ∇_Θ L|_{x2 常量} 可分开取并保 create_graph 图;
    额外返回 y, x2_leaf. field=False 时前向与反向逐位同旧 (A9)."""
    enc = model.encode(sb["x1"], is_scene=True)
    hc = encode_cands(model, sb["cands"])
    sc = lds_rows(model, enc["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, mu)
    logits = write_logits(model, enc["tokens"], cfg)       # (B,T,4), 梯度穿过冻结的 W_D1
    r = R.render_ste(logits, tpl, k_hard)
    x2 = R.channel(r["x2"], draw)
    extra = {}
    if field:
        assert torch.equal(r["x2"].detach(), r["hard"]), "显式场: 直通前向画布须逐位 = 硬渲染 (A2)"
        extra = dict(y=x2, x2_leaf=x2.detach().requires_grad_(True))
        x2 = extra["x2_leaf"]
    enc2 = model.encode(x2, is_scene=False)
    S = lds_rows(model, enc2["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, mu)
    l_lam = lam_cost(cfg.lam, r["p"], r["k"])    # §5.2' 只对实际落章的格计费 ([U] 2026-08-20; 留白零代价不变)
    return dict(sc=sc, S=S, p=r["p"], k=r["k"], hard=r["hard"], x2=x2, l_lam=l_lam, hc=hc,
                h2=enc2["cls"], logits=logits, **extra)


def field_grads(out, params, halves=False):
    """显式场一阶值 (D-CHK C10): g_S = ∇_Θ L|_{x2 常量} + J_yᵀ ∂L/∂x2 (与直通训练路径的 g_S 同值, T-PROBE P1 相对差 ≤ 2.6e-7).
    gr / gx 带 create_graph 图 (供 exact_cross_term 取 Jᵀw); halves=True 另给双半批 g1/g2 (𝒮̂ 用, 图保留).
    out = scene_side(field=True) 的返回. 返回 dict(g_S, g1, g2, gx, gr)."""
    y, x2, rows = out["y"], out["x2_leaf"], out["S"]["rows"]
    L_S = rows.mean()
    gx = torch.autograd.grad(L_S, x2, create_graph=True)[0]
    gr = AL.grads_of(L_S, params, retain=True, create=True)
    gw = torch.autograd.grad((y * gx.detach()).sum(), params, create_graph=True, allow_unused=True)   # J_yᵀ gx₀ (带 Θ 图, HVP 式)
    gw = [b if b is not None else torch.zeros_like(p) for b, p in zip(gw, params)]
    g_S = [a.detach() + b.detach() for a, b in zip(gr, gw)]
    g1 = g2 = None
    B = rows.shape[0]
    if halves and B >= 2 and B % 2 == 0:
        h = B // 2
        gs = []
        for sl in (slice(0, h), slice(h, B)):
            Lh = rows[sl].mean()
            gxh = torch.autograd.grad(Lh, x2, retain_graph=True)[0]
            grh = AL.grads_of(Lh, params, retain=True)
            gwh = torch.autograd.grad((y * gxh).sum(), params, retain_graph=True, allow_unused=True)
            gs.append([a + (b if b is not None else 0.0) for a, b in zip(grh, gwh)])
        g1, g2 = gs
    return dict(g_S=g_S, g1=g1, g2=g2, gx=gx, gr=gr, gw=gw)


def exact_cross_term(params, den, g_V, g_S, fg, y):
    """精确 u_S = Jᵀw/‖g_S‖_{F⁻¹}, w = F⁻¹ĝ_V, J = 显式场 g(Θ) = gr(Θ) + J_y(Θ)ᵀ gx(Θ) 的雅可比. Jᵀw = ∇_Θ⟨g, w⟩ 分三项取:
      (1) ∇_Θ⟨gr, w⟩ (读项二阶, gr 带 create_graph);
      (2) ∇_Θ⟨J_yᵀ gx₀, w⟩|_{gx₀ 固定} = ∇_Θ⟨gw, w⟩, gw = ∇_Θ⟨y, gx₀⟩ 带 create_graph (HVP 式);
      (3) (∂gx/∂Θ)ᵀ (J_y w) = ∇_Θ⟨gx, (J_y w)₀⟩, J_y w 的**值**由双 vjp 前向模取后 detach.
    [C] 2026-08-18 f64 中心差分实测: 双 vjp 前向模给出的 J_y w 值正确, 但其 Θ 图在写项少 ~15% (12.58 对 14.85), 故 (2) 不用双 vjp 的图;
    三项合计对 [⟨g(Θ+εa),w⟩−⟨g(Θ−εa),w⟩]/2ε 相对差 ≤ 1e-8 (tests: test_exact_cross_term_is_true_transpose_f64).
    有限差分不用 ([U] 2026-08-18: 停用). 调用后图释放 (须在 g_V/g_aux 之后调用). 返回 (u_S list, diag)."""
    nV = AL.Metric.norm(g_V, den)
    nS = AL.Metric.norm(g_S, den)
    diag = dict(gV=float(nV), gS=float(nS))
    if float(nV) == 0.0 or float(nS) == 0.0:
        return [torch.zeros_like(p) for p in params], dict(diag, u=0.0, JTw=0.0, degenerate=True)
    dirn = AL.Metric.precond([g / nV for g in g_V], den)          # F⁻¹ĝ_V
    u0 = torch.zeros_like(y, requires_grad=True)
    Jyu = torch.autograd.grad((y * u0).sum(), params, create_graph=True, allow_unused=True)
    Jyu = [a if a is not None else torch.zeros_like(p) for a, p in zip(Jyu, params)]
    Jyw0 = torch.autograd.grad(sum((a * b).sum() for a, b in zip(Jyu, dirn)), u0)[0].detach()      # J_y w 的值
    Sw = (sum((a * b).sum() for a, b in zip(fg["gr"], dirn)) + sum((a * b).sum() for a, b in zip(fg["gw"], dirn))
          + (fg["gx"] * Jyw0).sum())
    JTw = [t.detach() for t in AL.grads_of(Sw, params, retain=False)]
    u = [t / nS for t in JTw]
    diag.update(u=float(AL.Metric.norm(u, den)), JTw=float(AL.Metric.norm(JTw, den)), degenerate=False)
    return u, diag


def pool_side(model, pb, w, mu):
    """池抽笔记读 (§6.1 第二流): pb.x (B,side,side) 已重渲+腐蚀的常量位图 (无 flag), 自带候选/θ/标签."""
    hc = encode_cands(model, pb["cands"])
    h = model.encode(pb["x"], is_scene=False)["cls"]
    return lds_rows(model, h, hc, pb["th"], pb["tg"], pb["truth5"], pb["ns"], w, mu)


def split_mask_keep(selected, rng, keep_frac):
    """选中格拆 replace/keep 桶 (MASK-KEEP, params §8): keep 桶比例 keep_frac (逐格独立抽)."""
    u = torch.rand(selected.shape, generator=rng)
    keep = selected & (u < keep_frac)
    return selected & ~keep, keep


def make_mask_batch(B, cfg, rng, dev):
    """随机画布 (§6.2 占空比分层) → 素养信道 (s 旋钮 + 概率遮挡, 不跟 occ_k: params §9 接线二) →
    遮蔽选格 (15-40%, 其中 keep_frac 保留). 不经 D1、不进池 (A6)."""
    k = R.random_canvas_classes(B, rng)
    draw = R.draw_channel(B, cfg.s, rng, dev, occ_k=0)
    with torch.no_grad():
        x = R.channel(R.render_classes(k.to(dev), draw), draw)
    sel = sample_mask_cells(B, rng)
    rep, keep = split_mask_keep(sel, rng, cfg.keep_frac)
    return dict(x=x, k=k.to(dev), sel=sel.to(dev), rep=rep.to(dev), keep=keep.to(dev))


def mask_side(model, mb):
    """L_mask (§4.2, 只在随机画布上): [MASK] 替换 rep 桶, 损失在全部选中格; 返回 (loss, 仪表)."""
    enc = model.encode(mb["x"], is_scene=False, mask_cells=mb["rep"])
    logits = model.heads.cell(enc["tokens"])                 # (B,T,4)
    sel = mb["sel"]
    loss = F.cross_entropy(logits[sel], mb["k"][sel])
    with torch.no_grad():
        pred = logits.argmax(-1)
        rep, keep = mb["rep"], mb["keep"]
        m_acc = float((pred[rep] == mb["k"][rep]).float().mean()) if bool(rep.any()) else float("nan")
        k_acc = float((pred[keep] == mb["k"][keep]).float().mean()) if bool(keep.any()) else float("nan")
    return loss, dict(mask_acc=m_acc, kept_acc=k_acc)


def render_pool_batch(pool, idx, kits, cfg, rng, dev, draw=None):
    """池条目 idx + 候选包 kits -> 池抽笔记 batch: 类别重渲 + 腐蚀重抽 (§6.3), 标签自池元数据.
    draw 给定 = v3.1 C7 池起链共同随机数 (x0 重渲与整链写步同一腐蚀实现, 调用方同份传 chain_side)."""
    from .data import build_batch
    k = pool.classes(idx).to(dev)
    if draw is None:
        draw = R.draw_channel(k.shape[0], cfg.s, rng, dev, cfg.occ_k)
    with torch.no_grad():
        x = R.channel(R.render_classes(k, draw), draw)
    pb = build_batch(kits, dev, with_scene=False)
    ns_pool = pool.labels(idx).to(dev)
    assert torch.equal(ns_pool, pb["ns"]), "kit 标记与池条目标记不一致"
    pb.update(x=x, k=k, idx=idx)
    return pb


def concat_batches(sbs):
    """若干场景批 (build_batch 输出) 沿 batch 维拼接 (闸量批用). 单批原样返回."""
    if len(sbs) == 1:
        return sbs[0]
    out = dict(B=sum(b["B"] for b in sbs))
    for k in ("x1", "ns", "truth5", "cands", "zseed"):
        out[k] = torch.cat([b[k] for b in sbs])
    out["th"] = {t: torch.cat([b["th"][t] for b in sbs]) for t in sbs[0]["th"]}
    out["tg"] = {t: torch.cat([b["tg"][t] for b in sbs]) for t in sbs[0]["tg"]}
    return out


# ================================================================ 读数
def readings(g_S, g_V, den, g1=None, g2=None):
    """§7.6 读数: 扰动 𝒮 (双半批无偏估 Σ g1 g2/(√v+ε), 及全批 g_Sᵀ F⁻¹ g_S), 覆盖率 τ, 余弦, 模长."""
    nS = float(AL.Metric.norm(g_S, den))
    nV = float(AL.Metric.norm(g_V, den))
    ip = float(AL.Metric.inner(g_S, g_V, den))
    out = dict(gS=nS, gV=nV, S_full=nS * nS,
               tau=(ip / (nS * nS)) if nS > 0 else 0.0,
               cos=(ip / (nS * nV)) if nS > 0 and nV > 0 else 0.0)
    if g1 is not None and g2 is not None:
        out["S_hat"] = float(AL.Metric.inner(g1, g2, den))
    return out


def excess_surprise(l_S, rows_pool, ns_pool):
    """ℓ_ex = L̄_ds|_S − median_N L̄_ds|_{V,笔记} (§7.6, 稠密每步可读). 池路缺席 -> None."""
    if rows_pool is None:
        return None
    r = rows_pool.detach()
    per_n = []
    for n in sorted(set(ns_pool.tolist())):
        per_n.append(float(r[ns_pool == n].mean()))
    return float(l_S) - float(torch.tensor(per_n).median())


# ================================================================ 训练一步 (§5.3 合成 + §7.3 更新)
def train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, beta=0.0, split=False,
               want_S=False, delta_mult=1.0, beta_frac=0.0):
    """一步: 前向三路 + L_mask + L_λ; split=False ⇒ 单次反传 (β=0 且不读数); split=True ⇒ g_S / g_V /
    g_aux 分开取 (读数); 施压时加交叉项 −β‖g_V‖u_S:
      · u_S 默认精确 vjp (cfg.u_mode="exact": 显式场 Jᵀw/‖g_S‖, 场景路径 math 注意力核以支持二阶反传; [U] 2026-08-18 有限差分停用),
        cfg.u_mode="fd" 只留给旧诊断脚本 (硬类别冻结 + 同一 ChannelDraw 的两次前向);
      · beta_frac=f>0 ⇒ 每步 β = f·β_ceil, β_ceil = BETA_CAP/‖u_S‖_{F⁻¹} (代码口径; 修正项 F⁻¹ 范数 = f·0.1·‖g_V‖), 优先于绝对 β;
        绝对 beta>0 ⇒ 夹取 β‖u_S‖ ≤ BETA_CAP. 返回仪表 dict."""
    B = sb["B"]
    press = beta > 0.0 or beta_frac > 0.0
    exact = press and getattr(cfg, "u_mode", "exact") == "exact"
    draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    with (AL._math_sdpa() if exact else contextlib.nullcontext()):        # 二阶反传只需场景路径走 math 核
        out = scene_side(model, sb, w, cfg.mu, cfg, draw, tpl, field=exact)
    L_sc = out["sc"]["rows"].mean()
    L_S = out["S"]["rows"].mean()
    po = pool_side(model, pb, w, cfg.mu) if pb is not None else None
    L_pool = po["rows"].mean() if po is not None else torch.zeros((), device=dev)
    L_V = L_sc + L_pool
    l_mask, mstat = mask_side(model, mb)
    L_aux = l_mask + out["l_lam"]
    total = float(L_S + L_V + L_aux)
    m = dict(L_sc=float(L_sc), L_S=float(L_S), L_pool=(float(L_pool) if po is not None else None),
             L_mask=float(l_mask), L_lam=float(out["l_lam"]), L=total,
             acc_sc=out["sc"]["accs"].mean(0).tolist(), acc_S=out["S"]["accs"].mean(0).tolist(),
             acc_pool=(po["accs"].mean(0).tolist() if po is not None else None),
             read_sc=float((out["sc"]["read_pred"] == sb["ns"]).float().mean()),
             read_S=float((out["S"]["read_pred"] == sb["ns"]).float().mean()),
             read_pool=(float((po["read_pred"] == pb["ns"]).float().mean()) if po is not None else None),
             stamps=float(R.stamp_count(out["k"]).float().mean()),
             p_empty=float(out["p"][..., R.EMPTY].mean()),
             ell_ex=excess_surprise(L_S, po["rows"] if po is not None else None,
                                    pb["ns"] if pb is not None else None), **mstat)
    opt.zero_grad(set_to_none=True)
    if not (split or press):
        (L_S + L_V + L_aux).backward()
    else:
        rows_S = out["S"]["rows"]
        g1 = g2 = None
        halves = want_S and B >= 2 and B % 2 == 0
        if exact:
            fg = field_grads(out, params, halves=halves)         # 显式场一阶 (gr/gx 保图供 Jᵀw)
            g_S, g1, g2 = fg["g_S"], fg["g1"], fg["g2"]
        elif halves:
            h = B // 2
            g1 = AL.grads_of(rows_S[:h].mean(), params, retain=True)
            g2 = AL.grads_of(rows_S[h:].mean(), params, retain=True)
            g_S = [0.5 * (a + b) for a, b in zip(g1, g2)]
        else:
            g_S = AL.grads_of(L_S, params, retain=True)
        g_V = AL.grads_of(L_V, params, retain=True)
        g_aux = AL.grads_of(L_aux, params, retain=exact)
        met = AL.Metric(opt, "adam")
        den = met.denom(params)
        m["al"] = readings(g_S, g_V, den, g1, g2)
        corr = None
        if press:
            if exact:
                u, d = exact_cross_term(params, den, g_V, g_S, fg, out["y"])
                m["al"].update(u=d["u"], JTw=d["JTw"], degenerate=float(bool(d.get("degenerate"))), u_mode="exact")
            else:
                k_frozen = out["k"]
                hard0 = out["hard"]

                def recompute():
                    o2 = scene_side(model, sb, w, cfg.mu, cfg, draw, tpl, k_hard=k_frozen)
                    assert torch.equal(o2["hard"], hard0), "A7: 两次前向的 x2_hard 不逐位相同"
                    return AL.grads_of(o2["S"]["rows"].mean(), params)

                u, d = AL.cross_term(params, den, g_V, g_S, recompute, delta_mult=delta_mult, opt=opt)
                m["al"].update(u=d["u"], delta=d["delta"], degenerate=float(bool(d.get("degenerate"))), u_mode="fd")
            beta_eff = 0.0
            if not d.get("degenerate"):
                nu = d["u"]
                beta_ceil = AL.BETA_CAP / max(nu, 1e-30)
                capped = 0
                if beta_frac > 0.0:                               # 固定剂量: β = f·β_ceil, 每步重算
                    beta_eff = beta_frac * beta_ceil
                else:
                    beta_eff = beta
                    if beta * nu > AL.BETA_CAP:
                        beta_eff = beta_ceil
                        capped = 1
                scale = beta_eff * d["gV"]
                corr = [x * (-scale) for x in u]
                m["al"].update(capped=capped, beta_ceil=beta_ceil, f=(beta_frac if beta_frac > 0.0 else beta_eff / beta_ceil),
                               corr_frac=float(scale * nu / max(d["gV"], 1e-30)))
            m["al"]["beta_eff"] = beta_eff
        with torch.no_grad():
            for i, p in enumerate(params):
                g = g_S[i] + g_V[i] + g_aux[i]
                if corr is not None:
                    g = g + corr[i]
                p.grad = g
    if cfg.clip > 0:
        m["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
    opt.step()
    m["accs_notes"] = torch.cat([out["S"]["accs"]] + ([po["accs"]] if po is not None else []))
    m["k"] = out["k"].detach()
    m["pool_accs"] = po["accs"].mean(1) if po is not None else None
    return m


# ================================================================ 投影率 c (§7.6, 每评)
def proj_rate(model, opt, params, sb, pb, cfg, w, dev, rng, k, max_groups=64):
    """c = ‖P_B ĝ_S‖/‖ĝ_S‖ ∈ [0,1], ĝ = 白化梯度 (⟨·,·⟩_{F⁻¹}), B = span{已消化梯度} = 本步 V 行按数量的
    均值梯度 (场景 ∪ 池抽笔记, 不中心化, 硬约束 1); 只在 S 上算 (硬约束 2); 零假设 = S 打乱标签重跑
    (硬约束 3, 每评重算). P = 恒等 (精确内积, 参数量 ~5M 下草图不必要, 同 align.py 模块头约定).
    子空间投影用 Gram 伪逆 (相对截断 1e-6 = QR 秩判定). 返回 dict(c, c_null, nV, rank)."""
    met = AL.Metric(opt, "adam")
    den = met.denom(params)
    B = sb["B"]
    draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = scene_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    po = pool_side(model, pb, w, cfg.mu) if pb is not None else None
    groups = {}
    for i, n in enumerate(sb["ns"].tolist()):
        groups.setdefault(n, []).append(out["sc"]["rows"][i:i + 1])
    if po is not None:
        for i, n in enumerate(pb["ns"].tolist()):
            groups.setdefault(n, []).append(po["rows"][i:i + 1])
    ns_sorted = sorted(groups)
    if len(ns_sorted) > max_groups:
        sel = torch.randperm(len(ns_sorted), generator=rng)[:max_groups].tolist()
        ns_sorted = sorted(ns_sorted[i] for i in sel)
    vecs = []
    for n in ns_sorted:
        loss = torch.cat(groups[n]).mean()
        g = AL.grads_of(loss, params, retain=True)
        vecs.append(AL.Metric.whiten_flat(g, den))
    V = torch.stack(vecs)                                     # (nV, D)
    g_S = AL.grads_of(out["S"]["rows"].mean(), params, retain=True)
    v_S = AL.Metric.whiten_flat(g_S, den)
    tg_n, t5_n, ns_n = AL.null_targets(sb["ns"], sb["th"], sb["truth5"], k, rng, dev)
    Sn = lds_rows(model, out["h2"], out["hc"], sb["th"], tg_n, t5_n, ns_n, w, cfg.mu)
    g_Sn = AL.grads_of(Sn["rows"].mean(), params, retain=False)
    v_Sn = AL.Metric.whiten_flat(g_Sn, den)
    gram = (V.double() @ V.double().T)
    gram = 0.5 * (gram + gram.T)
    ev, U = torch.linalg.eigh(gram.cpu())                     # ≤64×64 双精度: CPU 特征分解 (GPU cusolver 建句柄需显存外的 cudaMalloc,
    ev, U = ev.to(V.device), U.to(V.device)                    # 训练峰值 ~18 GiB 时曾 CUSOLVER_STATUS_INTERNAL_ERROR 崩掉 f=1.0 臂首评)
    keep = ev > 1e-6 * float(ev.max().clamp(min=1e-30))
    inv = torch.where(keep, 1.0 / ev.clamp(min=1e-30), torch.zeros_like(ev))
    ginv = (U * inv) @ U.T

    def c_of(v):
        q = (V.double() @ v.double())
        proj2 = float(q @ ginv @ q)
        nrm2 = float(v.double() @ v.double())
        return math.sqrt(max(min(proj2 / nrm2, 1.0), 0.0)) if nrm2 > 0 else 0.0

    return dict(c=c_of(v_S), c_null=c_of(v_Sn), nV=len(ns_sorted), rank=int(keep.sum()),
                L_S=float(out["S"]["rows"].mean()), L_S_null=float(Sn["rows"].mean()))


# ================================================================================================
# v3/v3.1 链前向 (spec-v3 §4-§6 + spec-v3.1 C1-C7). v2 函数 (scene_side/train_step/proj_rate/
# floor_measure) 原样保留: 诊断脚本与 A9 的参照物. β/u_S/水位闸整体挂起 (v3 §8), 链路径单次反传.
# v3.1: origin 拼接撤销 (C1, 序列恒 257) · 场景链 k=0 恒写/闸挪 k≥1 (C2/C3, 本阶段闸关) ·
# 池起链 + generation (C4) · V̂ 软标签 (C5) · Ŵ = V̂(ĥ) 经前向模型 g + L_fwd (C6) ·
# 链内共同随机数腐蚀 (C7, 单份 ChannelDraw 整链重放).
# ================================================================================================
def enc_v3(model, x, cfg, flagged=False, mask_cells=None):
    """v3.1 单图编码 (C1): [CLS] ⊕ tok(x), 长 257, 段嵌入不加载 (seg=False; seg 代码路留在 model.enc_seq).
    use_flag=1 (A18 兼容路) ⇒ v2 装配: flagged 者追加 flag. 候选/池抽笔记/随机画布皆此式."""
    tok = model.E.tokenize(x, mask_cells)
    uf = bool(getattr(cfg, "use_flag", 0))
    return model.E.enc_seq(tok, None, seg=False, use_flag=flagged and uf)


def encode_cands_v3(model, cands, cfg):
    """候选集 (B,C,side,side) 各自过主干 -> (B,C,d). v3.1: 单图裸式; 兼容路: 带 flag (v2 同款)."""
    B, C = cands.shape[:2]
    out = enc_v3(model, cands.reshape(B * C, *cands.shape[2:]), cfg, flagged=True)
    return out["cls"].view(B, C, -1)


def write_tokens_org(model, tok_cur, tok0):
    """[U] 2026-08-23「写路径加 origin，读路径不动」: 写路径两槽装配 [CLS] ⊕ (tok_cur + seg_cur) ⊕ (tok0 + seg_org)
    (长 513; v3 §4.1 origin 槽的代码路 model.enc_seq seg=True), 返回当前槽 256 个 token = 写头 D1 的输入 z.
    每个写步都走此处 (k=0 当前槽 = 空白纸, k≥1 当前槽 = 上一张纸); 读路径 (h_k → 六任务/直读) 不经此处, 仍单槽 257."""
    return model.E.enc_seq(tok_cur, tok0, seg=True)["tokens"]


def blank_note(B, draw, tpl):
    """[U] 2026-08-23「写第一张和第二张笔记都能看原题」+ 第一张的当前槽「放空白纸」: 空白纸 = 全空类 (EMPTY) 硬渲染
    (全零画布) 过同链信道 (C7 同一份 draw ⇒ 与该链各张纸同一腐蚀实现: 遮挡/模糊对全零恒等, 只剩像素噪声). 无梯度常量."""
    k0 = torch.zeros(B, GM.T, dtype=torch.long, device=draw.u.device)
    with torch.no_grad():
        return R.channel(R.render_hard(k0, tpl), draw)


def chain_side(model, sb, w, mu, cfg, draw, tpl):
    """v3.1 链前向 (C1/C2/C7): x0 → (写→渲→蚀→读)×K, 全部状态读出, BPTT 贯通 (§5.1); 读路径序列恒单槽 257
    (origin 拼接撤销, tok(x0) 缓存复用逻辑已删, A15 N/A). 训练期恒跑满 K (A14), k=0 恒写 (A22 前提).
    [U] 2026-08-23 origin_write=1「写第一张和第二张笔记都能看原题 origin … 读笔记 … 只能读笔记自身」: 每个写步 k 的写路径 =
    两槽编码 write_tokens_org(当前槽, origin 槽 x0), 写头吃其当前槽 token; 当前槽: k=0 = 空白纸 (blank_note, [U] 裁定), k≥1 = 纸 x_k;
    读路径 (场景 k=0 / 各纸) 不动 = 单槽 (origin_write=0 ⇒ 本函数逐位同旧).
    跨深度共同随机数 (C7/A24): 单份 draw/tpl 整链重放 (逐深度同一腐蚀实现); 链间 = 批行间独立.
    每步内的运算次序与 v2 scene_side 逐一对位 (编码→候选→读→写→渲→蚀→编码→读→墨费), A18 回归依赖它.
    返回 dict(states=[{rows,accs,read_pred,marg,h,...}×(K+1)], writes=[{p,k,hard,logits,x}×K], hc, l_lam)."""
    K = int(cfg.chain_k)
    assert K >= 1
    assert not int(getattr(cfg, "origin_cat", 0)), "v3.1 C1: origin 拼接已撤销 (seg 代码路只留 model.enc_seq)"
    uf = bool(getattr(cfg, "use_flag", 0))
    org_w = bool(getattr(cfg, "origin_write", 0))
    assert not (org_w and uf), "A18 兼容路 (use_flag=1) 要求 origin_write=0"
    tok0 = model.E.tokenize(sb["x1"])
    x_aux = sb.get("x_aux")
    if x_aux is not None:                                  # 加法流 ([U] 2026-08-27 双场景 N+m): 深度 0 = 两槽装配 (现成段嵌入路, 零新参数)
        assert not uf and not org_w, "加法流两槽装配与 use_flag 兼容路 / origin_write 互斥"
        e = model.E.enc_seq(tok0, model.E.tokenize(x_aux), seg=True)   # [CLS]⊕(tok(x_N)+seg_cur)⊕(tok(x_m)+seg_org): cls → 六任务/直读 (标签 s=N+m);
    else:                                                               # tokens = 当前槽 (场景 N 的格) → 写头; 纸的读路径仍单槽 (下同)
        e = model.E.enc_seq(tok0, None, seg=False, use_flag=uf)
    hc = encode_cands_v3(model, sb["cands"], cfg)
    st0 = lds_rows(model, e["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, mu)
    st0["h"] = e["cls"]
    states, writes = [st0], []
    l_lam = torch.zeros((), device=sb["x1"].device)
    if org_w:                                              # k=0 写路径: [空白纸 | origin 槽 x0] 两槽
        z = write_tokens_org(model, model.E.tokenize(blank_note(sb["B"], draw, tpl)), tok0)
    else:
        z = e["tokens"]
    for k in range(K):
        logits = write_logits(model, z, cfg)               # (B,T,4), 梯度穿过冻结的 W_D1
        r = R.render_ste(logits, tpl)
        assert torch.equal(r["x2"].detach(), r["hard"]), f"A12: 深度 {k + 1} 直通前向 ≠ 硬渲染"
        x_next = R.channel(r["x2"], draw)
        tok_next = model.E.tokenize(x_next)
        e = model.E.enc_seq(tok_next, None, seg=False)     # 读路径: 单槽 257 (不动)
        stt = lds_rows(model, e["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, mu)
        stt["h"] = e["cls"]
        states.append(stt)
        l_lam = l_lam + lam_cost(cfg.lam, r["p"], r["k"])  # §5.2' 推广到每个写步 (v3 §5.2)
        writes.append(dict(p=r["p"], k=r["k"], hard=r["hard"], logits=logits, x=x_next))
        if org_w and k + 1 < K:                            # 下一写步 (k+1 ≥ 1) 的写路径: 两槽 (当前纸 + origin 槽 x0)
            z = write_tokens_org(model, tok_next, tok0)
        else:
            z = e["tokens"]
    return dict(states=states, writes=writes, hc=hc, l_lam=l_lam)


def pool_side_v3(model, pb, w, mu, cfg):
    """池抽笔记单态读出 (v3 §6.1 第二流; v3.1 C4.1 训练默认改走池起链 chain_side, 本函数保留给
    chain_from_pool=0 兼容路与评测): 单图裸式编码. 兼容路 = v2 pool_side 逐位同."""
    hc = encode_cands_v3(model, pb["cands"], cfg)
    e = enc_v3(model, pb["x"], cfg, flagged=False)
    out = lds_rows(model, e["cls"], hc, pb["th"], pb["tg"], pb["truth5"], pb["ns"], w, mu)
    out["h"] = e["cls"]
    return out


def mask_side_v3(model, mb, cfg):
    """L_mask (v3: 随机画布单图式编码; 兼容路 = v2 裸装配). A13: 不进链, 不产生 V̂/g/闸/D1 张量."""
    e = enc_v3(model, mb["x"], cfg, flagged=False, mask_cells=mb["rep"])
    logits = model.heads.cell(e["tokens"])                 # (B,T,4)
    sel = mb["sel"]
    loss = F.cross_entropy(logits[sel], mb["k"][sel])
    with torch.no_grad():
        pred = logits.argmax(-1)
        rep, keep = mb["rep"], mb["keep"]
        m_acc = float((pred[rep] == mb["k"][rep]).float().mean()) if bool(rep.any()) else float("nan")
        k_acc = float((pred[keep] == mb["k"][keep]).float().mean()) if bool(keep.any()) else float("nan")
    return loss, dict(mask_acc=m_acc, kept_acc=k_acc)


# ---------------------------------------------------------------- V̂ 软标签 + 前向模型 + 闸 (v3.1 C3/C5/C6)
def margins(tl, t5, tg, truth5):
    """正确类边距 (C5.1, 六任务头 logit 上): m_t = logit[真值] − max_{c≠真值} logit[c]
    = log p_真值 − log p_次强 (nat); sign(m) = verifier 判决. 输出 detach — y_soft 的 sg (C5.3/A19)
    结构性成立. 列序 = TASKS (t1..t6, 与 task_accs 同)."""
    cols = []
    for t in TASKS:
        lg = (t5 if t == "t5" else tl[t]).detach()
        y = (truth5 if t == "t5" else tg[t]).unsqueeze(1)
        true_v = lg.gather(1, y).squeeze(1)
        rival = lg.scatter(1, y, float("-inf")).max(1).values
        cols.append(true_v - rival)
    return torch.stack(cols, dim=1)


class SoftScale:
    """逐任务边距尺度 s[t] (C5.2): 批均 |m| 的运行值 — 首 init_steps 步累计均值自举 ([C] 实例化:
    「初值取首 100 步均值」期间用累计到当步的均值, 当步先 update 后取用 ⇒ 首步即有定义, 无零除),
    其后 EMA (动量 m); 取用时 clamp 下界. 无梯度 (输入已 detach). 状态入检查点."""

    def __init__(self, m=0.99, clamp=1e-3, init_steps=100):
        self.m, self.clamp, self.init_steps = float(m), float(clamp), int(init_steps)
        self.n = 0
        self.s = torch.ones(6)

    def update(self, marg):
        """marg (…,6) 已 detach; 一训练步一更新 (场景链 + 池链全状态合并)."""
        a = marg.detach().abs().float().reshape(-1, 6).mean(0).cpu()
        if self.n < self.init_steps:
            self.s = a if self.n == 0 else (self.s * self.n + a) / (self.n + 1)
        else:
            self.s = self.m * self.s + (1.0 - self.m) * a
        self.n += 1

    def scale(self, dev=None):
        s = self.s.clamp(min=self.clamp)
        return s.to(dev) if dev is not None else s

    def state(self):
        return dict(n=self.n, s=self.s.clone())

    def load_state(self, d):
        self.n, self.s = int(d["n"]), d["s"].clone()


def y_soft_of(marg, sscale):
    """y_soft = σ(sg[m]/s[t]) (C5.3). marg (…,6) 已 detach; s clamp 后按末维广播."""
    return torch.sigmoid(marg / sscale.scale(marg.device))


def pred_gate(model, states, sb, cfg, k0_gated=False):
    """V̂ 逐状态 + 前向模型 ĥ_{k+1} = h_k + g(h_k) + Ŵ = V̂(ĥ) (C6) + Δ̂ + 闸 λ/p_halt (C3:
    场景链 k0_gated=False ⇒ k=0 强制写无闸, 受闸位 k=1..K−1, 停位 {1..K}; 池链 k0_gated=True ⇒
    k=0 起即有闸, 停位 {0..K}). A11: 输入只有 h_k (ptb=0 时 sg) 与 φ(θ) (同 detach), 真实成绩不进
    任何前向张量. A25: Ŵ 无独立参数, 计算图必经 V̂ 头与 g (末层零初始化 ⇒ 初始 Ŵ ≡ V̂, Δ̂ ≡ 0).
    本阶段闸关 (C3): λ/p_halt 只作只读读数, (a,b) 无损失不更新.
    返回 dict(V (B,K+1,6), W (B,K,6), Vlog, Wlog, hhat 列表, dhat (B,K), lam (B,K), p_halt (B,K+1−g0), g0)."""
    K = len(states) - 1
    det = not int(getattr(cfg, "pred_to_backbone", 0))
    hs = [stt["h"].detach() if det else stt["h"] for stt in states]
    Vlog = torch.stack([model.pred.v_logits(model.heads, h, sb["th"], detach_emb=det) for h in hs], dim=1)
    hhat = [model.fwd(hs[k]) for k in range(K)]            # C6.2 (ptb=0 时 h_k 已 sg ⇒ L_fwd 不进 Θ_E)
    Wlog = torch.stack([model.pred.v_logits(model.heads, hh, sb["th"], detach_emb=det) for hh in hhat], dim=1)
    V, W = torch.sigmoid(Vlog), torch.sigmoid(Wlog)
    dhat = W.mean(-1) - V[:, :K].mean(-1)                  # (B,K) 准确率单位 (C6.3)
    g0 = 0 if k0_gated else 1                              # 首个受闸步 (C3)
    lam_all = model.gate.lam(dhat.detach())                # (B,K); 场景链 λ_0 不入闸 (只读)
    lam = lam_all[:, g0:]
    if lam.shape[1] > 0:
        one_m = torch.cumprod(1.0 - lam, dim=1)
        prev = torch.cat([torch.ones_like(lam[:, :1]), one_m[:, :-1]], dim=1)
        p_halt = torch.cat([lam * prev, one_m[:, -1:]], dim=1)   # (B,K−g0+1), 停位 {g0..K}, 和恒 1
    else:                                                  # K=1 场景链: 无受闸位, 恒停在 K
        p_halt = torch.ones(dhat.shape[0], 1, device=dhat.device, dtype=dhat.dtype)
    return dict(V=V, W=W, Vlog=Vlog, Wlog=Wlog, hhat=hhat, dhat=dhat, lam=lam_all, p_halt=p_halt, g0=g0)


def pred_loss(pg, states, sscale, eta_fwd):
    """L_pred (C6.5) = Σ_{k=0..K} Σ_t BCE(V̂_k, y_soft_k) + η_fwd·L_fwd. y_soft = σ(sg[m]/s) (C5.3/C5.4,
    标签零梯度 A19); L_fwd = Σ_{k<K} (1 − cos(ĥ_{k+1}, sg[h_{k+1}])) (C6.4, 目标端 sg, A20).
    批均值, k/t 求和. Ŵ 无 BCE (C6: g 只由 L_fwd 训, V̂ 只在真实态上训).
    返回 (L_pred, dict(bce, fwd))."""
    marg = torch.stack([stt["marg"] for stt in states], dim=1)     # (B,K+1,6) 已 detach
    ys = y_soft_of(marg, sscale)
    lb = F.binary_cross_entropy_with_logits(pg["Vlog"], ys, reduction="none").sum(dim=(1, 2)).mean()
    K = len(states) - 1
    lf = torch.zeros((), device=marg.device)
    for k in range(K):
        lf = lf + (1.0 - F.cosine_similarity(pg["hhat"][k], states[k + 1]["h"].detach(), dim=-1)).mean()
    # lf_t = 训练真实用的 L_fwd 张量本体 (A20 对它取梯度 — 钉在真实计算图上, 不许断言侧自建公式);
    # 调用方只存 float, 图随本 dict 出栈释放.
    return lb + float(eta_fwd) * lf, dict(bce=float(lb), fwd=float(lf), lf_t=lf)


def gate_loss(pg, states, cfg):
    """L_gate (v3 §5.4 + C3 闸位): Σ_{k=g0..K} p_halt(k)(−sg[acc̄(x_k)] + c_step·k) + κ_p·KL(p_halt‖Geom(p_g));
    支撑 = 停位 {g0..K} (场景链 g0=1: k=0 恒写不可停). 成绩与 Δ̂ 皆 detach ⇒ 梯度只落 (a,b) (A17).
    本阶段闸关不启用 (C3), 器官与断言保留."""
    accbar = torch.stack([stt["accs"].mean(1) for stt in states], dim=1)   # (B,K+1), 无梯度
    g0 = pg["g0"]
    acc = accbar[:, g0:]
    L = acc.shape[1]
    ks = torch.arange(g0, g0 + L, device=acc.device, dtype=acc.dtype)
    ponder = (pg["p_halt"] * (-acc + float(cfg.c_step) * ks)).sum(1).mean()
    pgm = float(cfg.p_g)
    q = torch.tensor([pgm * (1 - pgm) ** j for j in range(L - 1)] + [(1 - pgm) ** (L - 1)],
                     device=acc.device, dtype=acc.dtype).clamp(min=1e-8)
    ph = pg["p_halt"].clamp(min=1e-8)
    kl = (ph * (ph.log() - q.log().unsqueeze(0))).sum(1).mean()
    return ponder + float(cfg.kappa_p) * kl


def first_true(mask):
    """(B,K) bool → 每行第一个 True 的下标; 无 True → K."""
    return (mask.long().cumsum(1) == 0).long().sum(1)


def kstar_det(dhat, c_step, k0_gated=False):
    """推理/评测判据 (v3 §4.5 + C3): 继续 ⇔ Δ̂_k > c_step, 从首个受闸步起判. 场景链 (k0_gated=False)
    k* ∈ {1..K} (A22: 不存在 k*=0); 池链 (k0_gated=True) k* ∈ {0..K}. (B,K) → (B,) long."""
    g0 = 0 if k0_gated else 1
    return g0 + first_true(~(dhat[:, g0:] > float(c_step)))


def kstar_sample(lam, rng, k0_gated=False):
    """训练期入池 k* ([U]「选A」抽样制 + C3 闸位): 从首个受闸步起逐步 Bernoulli(λ_k) 停.
    lam (B,K) CPU; rng = CPU generator (CRN). 场景链最小 1 (A22)."""
    g0 = 0 if k0_gated else 1
    u = torch.rand(lam[:, g0:].shape, generator=rng)
    return g0 + first_true(u < lam[:, g0:].detach().cpu())


def policy_lam(lam_cpu, cfg, rng):
    """对照臂闸策略 (v3 §11.2 / params §13 步 4; v3.1 C11: a=0 与随机闸两臂本阶段挂起, 代码保留):
    learned = 原 λ 透传 (不耗 rng); const = 恒 gate_const_lam; rand = U(gate_rand_lo, gate_rand_hi)
    逐位独立 (CPU generator, CRN). ε_write 已退役 (C2.5): eps_write=0 时夹取为无操作 (代码留给休眠臂).
    lam_cpu: (B,K) CPU."""
    mode = str(getattr(cfg, "gate_mode", "learned"))
    if mode == "learned":
        lam = lam_cpu
    elif mode == "const":
        lam = torch.full_like(lam_cpu, float(cfg.gate_const_lam))
    elif mode == "rand":
        lo, hi = float(getattr(cfg, "gate_rand_lo", 0.0)), float(cfg.gate_rand_hi)
        lam = lo + torch.rand(lam_cpu.shape, generator=rng) * (hi - lo)
    else:
        raise AssertionError(f"unknown gate_mode: {mode}")
    ew = float(getattr(cfg, "eps_write", 0.0))
    if ew > 0:
        lam = torch.cat([lam[:, :1].clamp(max=1.0 - ew), lam[:, 1:]], dim=1)
    return lam


def const_rate_match(ek, K):
    """恒定停止率 λ̄ 使截断策略平均深度 E(λ) = Σ_{k=1..K} (1−λ)^k = ek (E 单调减, 二分 50 轮).
    H-D 同均深混合零假设 (params §13 步 4 ①). 返回 (λ̄, P), P = [λ̄(1−λ̄)^k]_{k<K} ⊕ [(1−λ̄)^K], ΣP=1."""
    K = int(K)
    ek = max(0.0, min(float(ek), float(K)))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        e = sum((1 - mid) ** k for k in range(1, K + 1))
        if e < ek:
            hi = mid
        else:
            lo = mid
    lb = (lo + hi) / 2
    P = [lb * (1 - lb) ** k for k in range(K)] + [(1 - lb) ** K]
    return lb, P


def pb_top1(q):
    """Poisson–binomial top-1 (v3 §4.4 两行 DP): q (B,6) 六路独立命中概率 → (B,) 总对数众数.
    只入报表, 不进闸 (闸用期望); v3.1 C5.6: 输入 = V̂ 的软预测."""
    B = q.shape[0]
    z = torch.zeros(B, 1, device=q.device, dtype=q.dtype)
    f = torch.ones(B, 1, device=q.device, dtype=q.dtype)
    for t in range(q.shape[1]):
        qt = q[:, t:t + 1]
        f = torch.cat([f * (1 - qt), z], dim=1) + torch.cat([z, f * qt], dim=1)
    return f.argmax(1)


# ---------------------------------------------------------------- v3.1 一步 (§5.5 合成; β 挂起 ⇒ 单次反传)
def chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, const=None, sscale=None, w_pool=None,
                     relearn=None, genrl=None, pool=None, ab=None, rng_add=None):
    """v3.1 训练一步 (§5.5 + C2-C7): L = [scene_ds_weight·L̄_ds(x0) + Σ_{k≥1} L̄_ds(x_k)]|场景链
    + Σ_k L̄_ds(x_k)|池链 (C4.1) + L_mask + L_λ(两链全部写步) + η_pred·(L_pred|场景 + L_pred|池)
    [+ η_gate·L_gate]. 各路各自批均值再相加. 腐蚀 CRN (C7): 场景链单 draw 整链重放; 池链复用
    pb["draw"]/pb["tpl"] (其 x0 重渲亦同一实现, 由调用方在 render_pool_batch 传入).
    兼容路 (A18: chain_k=1 + origin_cat=0 + use_flag=1 + pred/gate 关 + chain_from_pool=0 +
    scene_ds_weight=1) 求和树按 v2 分组 — 梯度累加序敏感, 逐位回归需要.
    k* 抽样与入池由调用方做; 常数预测器 const 与边距尺度 sscale (C5.2) 同批在线更新 (无梯度).
    w = 损失权重 (含 [U] 2026-08-23 逐任务倍率); w_pool = 池成绩 σ 的加权均值用的归一 w (C4.5, 和 1), None ⇒ 同 w.
    relearn = 迭代重学甲案对象 (trainer 只在 exposure=1 且阶段 2 传入); None ⇒ 逐位同旧 (A30).
    genrl = 乙案对象 (trainer 只在 gen_relearn=1 且阶段 2 传入; None ⇒ 逐位同旧 A35); pool = 数据池 (乙案两流用).
    ab = 加法流批 ([U] 2026-08-27; build_batch 带 x_aux/n_a/m_add, 标签 s=N+m): 另跑一条 chain_side (两槽深度 0), 损失
    L_add = scene_ds_weight·L̄_ds(两槽 CLS) + Σ_{k≥1} L̄_ds(纸_k) 与其 L_λ 并入总和, 纸命中并入 w_t 人口; 信道随机流 rng_add
    (None ⇒ 用 rng); None ⇒ 逐位同旧 (无加法项)."""
    B = sb["B"]
    K = int(cfg.chain_k)
    wp = w if w_pool is None else w_pool
    pred_on = bool(getattr(cfg, "pred_on", 0))
    gate_on = bool(getattr(cfg, "gate_on", 0))
    cfp = bool(getattr(cfg, "chain_from_pool", 0))
    sdw = float(getattr(cfg, "scene_ds_weight", 1.0))
    assert not (gate_on and not pred_on), "gate_on 要求 pred_on=1 (闸吃 Δ̂)"
    assert not pred_on or sscale is not None, "pred_on 要求 sscale (C5.2 软标签尺度)"
    compat = (K == 1 and not cfg.origin_cat and getattr(cfg, "use_flag", 0)
              and not pred_on and not gate_on and not cfp)
    if compat:
        assert sdw == 1.0, "A18 兼容路要求 scene_ds_weight=1"
    sup = None
    if genrl is not None:
        assert not compat, "gen_relearn 与 A18 兼容路互斥 (A35)"
        sup = genrl.exp_mask(sb["ns"]).float()
    draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)   # C7: 场景链整链一份
    tpl = R.templates(draw.u, draw.s)
    out = chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    if sup is None:
        Ls = [stt["rows"].mean() for stt in out["states"]]
    else:                                                  # 乙案标签限域 (A36): 六任务+直读只算 N∈𝒩_exp 行
        den_s = sup.sum().clamp_min(1.0)
        Ls = [(stt["rows"] * sup).sum() / den_s for stt in out["states"]]
    out_a = L_add = La = None
    if ab is not None:                                     # 加法流 ([U] 2026-08-27): 同一模型、同一写读流程, 只换输入 (两张场景)
        assert not compat and relearn is None and genrl is None, "加法流与 A18 兼容路 / 迭代重学互斥"
        ra = rng if rng_add is None else rng_add
        draw_a = R.draw_channel(ab["B"], cfg.s, ra, dev, cfg.occ_k)     # 加法链自己的信道实现 (主流程随机流不动)
        tpl_a = R.templates(draw_a.u, draw_a.s)
        out_a = chain_side(model, ab, w, cfg.mu, cfg, draw_a, tpl_a)
        La = [stt["rows"].mean() for stt in out_a["states"]]
        asw = float(getattr(cfg, "add_sc_weight", 1.0)) * sdw       # 两槽 CLS 项权重 (诊断旋钮 × C2.3 场景权重)
        anw = float(getattr(cfg, "add_note_weight", 1.0))           # 加法纸项权重 (诊断旋钮)
        L_add = (La[0] if asw == 1.0 else asw * La[0]) + (sum(La[1:]) if anw == 1.0 else anw * sum(La[1:]))
    out_p = po = None
    sbp = None
    if pb is not None:
        if cfp:
            sbp = dict(pb)
            sbp["x1"] = pb["x"]
            out_p = chain_side(model, sbp, w, cfg.mu, cfg, pb["draw"], pb["tpl"])   # C4.1 池起链
            L_pool = sum(stt["rows"].mean() for stt in out_p["states"])
        else:
            po = pool_side_v3(model, pb, w, cfg.mu, cfg)
            if sup is None:
                L_pool = po["rows"].mean()
            else:
                sup_p = genrl.exp_mask(pb["ns"]).float()
                L_pool = (po["rows"] * sup_p).sum() / sup_p.sum().clamp_min(1.0)
    else:
        L_pool = torch.zeros((), device=dev)
    if compat:
        L_V = Ls[0] + L_pool          # A18: 求和节点的创建位置复刻 v2 (掩码图之前)
    l_mask, mstat = mask_side_v3(model, mb, cfg)
    l_lam = out["l_lam"] if out_p is None else out["l_lam"] + out_p["l_lam"]
    if out_a is not None:
        l_lam = l_lam + out_a["l_lam"]                     # 加法链写步同样计墨费
    l_tr = None
    tr_m = {}
    if relearn is not None:
        assert not compat, "exposure 与 A18 兼容路互斥 (A30)"
        l_tr, tr_m = relearn.transfer(out["writes"][-1]["x"], sb)
    l_im = l_pr = None
    gm = {}
    if genrl is not None:
        l_im, l_pr, gm = genrl.streams(model, pool)
    pg = pg_p = None
    L_pred = L_gate = None
    comp = dict(bce=0.0, fwd=0.0)
    if pred_on:
        margs = [torch.stack([stt["marg"] for stt in out["states"]], dim=1).reshape(-1, 6)]
        if out_p is not None:
            margs.append(torch.stack([stt["marg"] for stt in out_p["states"]], dim=1).reshape(-1, 6))
        sscale.update(torch.cat(margs))                    # C5.2: 先更新后取用 ([C]), 两链合并一更新
        pg = pred_gate(model, out["states"], sb, cfg, k0_gated=False)
        L_pred, c1 = pred_loss(pg, out["states"], sscale, cfg.eta_fwd)
        comp = dict(bce=c1["bce"], fwd=c1["fwd"])
        if out_p is not None:
            pg_p = pred_gate(model, out_p["states"], sbp, cfg, k0_gated=True)
            lp2, c2 = pred_loss(pg_p, out_p["states"], sscale, cfg.eta_fwd)
            L_pred = L_pred + lp2
            comp = dict(bce=comp["bce"] + c2["bce"], fwd=comp["fwd"] + c2["fwd"])
        if gate_on and str(getattr(cfg, "gate_mode", "learned")) == "learned":
            L_gate = gate_loss(pg, out["states"], cfg)
            if pg_p is not None:
                L_gate = L_gate + gate_loss(pg_p, out_p["states"], cfg)
    if compat:                        # v2 逐字次序: 弃用求和树取数值 → zero_grad → 新建同形树反传
        L_aux = l_mask + out["l_lam"]
        total = float(Ls[1] + L_V + L_aux)
        opt.zero_grad(set_to_none=True)
        (Ls[1] + L_V + L_aux).backward()
    else:
        L0 = Ls[0] if sdw == 1.0 else sdw * Ls[0]          # C2.3 场景 k=0 项 × scene_ds_weight
        tot = L0 + sum(Ls[1:]) + L_pool + l_mask + l_lam
        if L_add is not None:
            tot = tot + L_add
        if l_tr is not None:
            tot = tot + l_tr
        if l_im is not None:
            tot = tot + float(cfg.eta_im) * l_im
        if l_pr is not None:
            tot = tot + l_pr
        if L_pred is not None:
            tot = tot + float(cfg.eta_pred) * L_pred
        if L_gate is not None:
            tot = tot + float(cfg.eta_gate) * L_gate
        total = float(tot)
        opt.zero_grad(set_to_none=True)
        tot.backward()
    po_rows = (out_p["states"][0]["rows"] if out_p is not None else (po["rows"] if po is not None else None))
    po_ns = pb["ns"] if pb is not None else None
    m = dict(L=total, L_sc=float(Ls[0]), L_S=float(Ls[-1]),
             L_states=[float(x) for x in Ls],
             L_pool=(float(L_pool) if pb is not None else None), L_mask=float(l_mask),
             L_lam=float(l_lam),
             L_pred=(float(L_pred) if L_pred is not None else None),
             L_pred_bce=(comp["bce"] if pred_on else None),
             L_pred_fwd=(comp["fwd"] if pred_on else None),
             L_gate=(float(L_gate) if L_gate is not None else None),
             acc_sc=out["states"][0]["accs"].mean(0).tolist(),
             acc_S=out["states"][-1]["accs"].mean(0).tolist(),
             read_sc=float((out["states"][0]["read_pred"] == sb["ns"]).float().mean()),
             read_S=float((out["states"][-1]["read_pred"] == sb["ns"]).float().mean()),
             stamps=float(R.stamp_count(out["writes"][-1]["k"]).float().mean()),
             stamps_d=[float(R.stamp_count(wr["k"]).float().mean()) for wr in out["writes"]],
             p_empty=float(out["writes"][-1]["p"][..., R.EMPTY].mean()),
             p_empty_d=[float(wr["p"][..., R.EMPTY].mean()) for wr in out["writes"]],
             ell_ex=excess_surprise(float(Ls[-1]), po_rows, po_ns), **mstat)
    if out_a is not None:                                  # 加法流仪表 (标签 s=N+m; read_add_S_vsN = 纸直读命中 N 的占比, 「纸只写了 N」对照)
        m.update(L_add=float(L_add), L_add_states=[float(x) for x in La],
                 acc_add_sc=out_a["states"][0]["accs"].mean(0).tolist(),
                 acc_add_S=out_a["states"][-1]["accs"].mean(0).tolist(),
                 read_add_sc=float((out_a["states"][0]["read_pred"] == ab["ns"]).float().mean()),
                 read_add_S=float((out_a["states"][-1]["read_pred"] == ab["ns"]).float().mean()),
                 read_add_S_vsN=float((out_a["states"][-1]["read_pred"] == ab["n_a"]).float().mean()),
                 stamps_add=float(R.stamp_count(out_a["writes"][-1]["k"]).float().mean()),
                 p_empty_add=float(out_a["writes"][-1]["p"][..., R.EMPTY].mean()),
                 ce_add_S=out_a["states"][-1]["ce"].mean(0).tolist())
    else:
        m["L_add"] = None
    if relearn is not None:
        m.update(tr_m)
        m["eta_tr_now"] = relearn.eta_now
        m["notes_x"] = out["writes"][-1]["x"].detach()
    if genrl is not None:
        m.update(gm)
        m["L_im"] = float(l_im) if l_im is not None else None
        m["L_poolread"] = float(l_pr) if l_pr is not None else None
        m["n_sup_rows"] = int(sup.sum())
    if pg is not None:
        m["dhat_mean"] = float(pg["dhat"].mean())
        m["lam_gate_mean"] = float(pg["lam"].mean())
    if cfg.clip > 0:
        m["gnorm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
    opt.step()
    if const is not None and pred_on:
        const.update(sb["th"], torch.stack([stt["accs"] for stt in out["states"]], dim=1))
        if out_p is not None:
            const.update(pb["th"], torch.stack([stt["accs"] for stt in out_p["states"]], dim=1))
        elif po is not None:
            const.update(pb["th"], po["accs"])
    notes_accs = [out["states"][-1]["accs"]]               # w_t 命中人口: 场景末纸 + 池笔记本体 ([C] v2 同义)
    if out_p is not None:
        notes_accs.append(out_p["states"][0]["accs"])
    elif po is not None:
        notes_accs.append(po["accs"])
    if out_a is not None and int(getattr(cfg, "add_wpop", 1)):
        notes_accs.append(out_a["states"][-1]["accs"])     # 加法纸并入 w_t 命中人口 (流程同场景末纸; add_wpop=0 诊断臂不并入)
    m["accs_notes"] = torch.cat(notes_accs)
    m["ks"] = [wr["k"].detach().cpu() for wr in out["writes"]]
    m["ks_pool"] = [wr["k"].detach().cpu() for wr in out_p["writes"]] if out_p is not None else None
    m["ks_add"] = [wr["k"].detach().cpu() for wr in out_a["writes"]] if out_a is not None else None
    m["lam_rows"] = pg["lam"].detach().cpu() if pg is not None else None
    m["lam_rows_pool"] = pg_p["lam"].detach().cpu() if pg_p is not None else None
    if out_p is not None:                                  # C4.5: σ 更新量 = w_t 加权六任务均值 (w 归一和 1 ⇒ 用 wp)
        m["pool_accs"] = (out_p["states"][0]["accs"] * wp.unsqueeze(0)).sum(1)
    elif po is not None:
        m["pool_accs"] = (po["accs"] * wp.unsqueeze(0)).sum(1)
    else:
        m["pool_accs"] = None
    # 训练侧 ham12 ([U] 2026-08-23 K=2 主判据): 场景链第一写与第二写硬类别图的逐格不同占比, 批均; K<2 ⇒ None
    m["ham12"] = (float((out["writes"][0]["k"] != out["writes"][1]["k"]).float().mean()) if len(out["writes"]) >= 2 else None)
    # 训练侧 Δ_true ([U] 2026-08-23 主判据 Δ_true[1] = 纸 2 六任务 − 纸 1 六任务): 场景链相邻深度六任务命中均值之差, 批均, 长 K
    # (下标 0 = 纸 1 − 场景, 下标 1 = 纸 2 − 纸 1, …); 评测侧同口径逐链配对统计见 trainer.dtrue_stats
    accm = [stt["accs"].mean(1) for stt in out["states"]]
    m["dtrue"] = [float((accm[k + 1] - accm[k]).mean()) for k in range(len(accm) - 1)]
    # 逐任务未加权 CE (TASKS 序 t1..t6): 场景 k=0 / 链末纸 / 池笔记 ([U] 2026-08-23 t4 权重实验观测量)
    m["ce_sc"] = out["states"][0]["ce"].mean(0).tolist()
    m["ce_S"] = out["states"][-1]["ce"].mean(0).tolist()
    if out_p is not None:
        m["ce_pool"] = out_p["states"][0]["ce"].mean(0).tolist()
    elif po is not None:
        m["ce_pool"] = po["ce"].mean(0).tolist()
    else:
        m["ce_pool"] = None
    return m


# ---------------------------------------------------------------- v3 逐评读数 (§8 只读入册, 逐深度分列)
def eval_readings_chain(model, opt, params, sb, pb, cfg, w, dev, rng, k, max_groups=64):
    """β 挂起期的只读读数 (v3 §8/§11.1, 重启前置条件 1「S 逐深度分列」): 每深度 d ∈ 1..K 一份
    {𝒮̂ (双半批), 𝒮_full, τ, cos, ℓ_ex, c}; g_V = 场景深度0 行 + 池行; c 的 V-跨度 = 逐 N 组均值梯度
    (v2 proj_rate 同构). 链腐蚀 = 单 draw (C7)."""
    met = AL.Metric(opt, "adam")
    den = met.denom(params)
    B = sb["B"]
    K = int(cfg.chain_k)
    draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    po = pool_side_v3(model, pb, w, cfg.mu, cfg) if pb is not None else None
    L_V = out["states"][0]["rows"].mean() + (po["rows"].mean() if po is not None else 0.0)
    g_V = AL.grads_of(L_V, params, retain=True)
    # ---- V-跨度 (逐 N 组均值梯度, 场景深度0 ∪ 池)
    groups = {}
    for i, n in enumerate(sb["ns"].tolist()):
        groups.setdefault(n, []).append(out["states"][0]["rows"][i:i + 1])
    if po is not None:
        for i, n in enumerate(pb["ns"].tolist()):
            groups.setdefault(n, []).append(po["rows"][i:i + 1])
    ns_sorted = sorted(groups)
    if len(ns_sorted) > max_groups:
        sel = torch.randperm(len(ns_sorted), generator=rng)[:max_groups].tolist()
        ns_sorted = sorted(ns_sorted[i] for i in sel)
    vecs = []
    for n in ns_sorted:
        g = AL.grads_of(torch.cat(groups[n]).mean(), params, retain=True)
        vecs.append(AL.Metric.whiten_flat(g, den))
    V = torch.stack(vecs)
    gram = (V.double() @ V.double().T)
    gram = 0.5 * (gram + gram.T)
    ev_, U = torch.linalg.eigh(gram.cpu())                 # CPU eigh (GPU cusolver 建句柄 OOM 教训)
    ev_, U = ev_.to(V.device), U.to(V.device)
    keep = ev_ > 1e-6 * float(ev_.max().clamp(min=1e-30))
    inv = torch.where(keep, 1.0 / ev_.clamp(min=1e-30), torch.zeros_like(ev_))
    ginv = (U * inv) @ U.T

    def c_of(v):
        q = (V.double() @ v.double())
        proj2 = float(q @ ginv @ q)
        nrm2 = float(v.double() @ v.double())
        return math.sqrt(max(min(proj2 / nrm2, 1.0), 0.0)) if nrm2 > 0 else 0.0

    per_depth = []
    halves = B >= 2 and B % 2 == 0
    h = B // 2
    for d in range(1, K + 1):
        rows = out["states"][d]["rows"]
        g_S = AL.grads_of(rows.mean(), params, retain=True)
        rd = readings(g_S, g_V, den)
        if halves:
            g1 = AL.grads_of(rows[:h].mean(), params, retain=True)
            g2 = AL.grads_of(rows[h:].mean(), params, retain=True)
            rd["S_hat"] = float(AL.Metric.inner(g1, g2, den))
        rd["c"] = c_of(AL.Metric.whiten_flat(g_S, den))
        rd["ell_ex"] = excess_surprise(float(rows.mean()), po["rows"] if po is not None else None,
                                       pb["ns"] if pb is not None else None)
        rd["L_S"] = float(rows.mean())
        per_depth.append(rd)
    # ---- pooled null: 全链纸行合并 + 打乱标签, 当评重算 (硬约束 3)
    tg_n, t5_n, ns_n = AL.null_targets(sb["ns"], sb["th"], sb["truth5"], k, rng, dev)
    null_rows = []
    for d in range(1, K + 1):
        Sn = lds_rows(model, out["states"][d]["h"], out["hc"], sb["th"], tg_n, t5_n, ns_n, w, cfg.mu)
        null_rows.append(Sn["rows"])
    g_Sn = AL.grads_of(torch.cat(null_rows).mean(), params, retain=False)
    c_null = c_of(AL.Metric.whiten_flat(g_Sn, den))
    return dict(depths=per_depth, c_null=c_null, gV=float(AL.Metric.norm(g_V, den)),
                nV=len(ns_sorted), rank=int(keep.sum()))


# ================================================================ 地板与水位闸量 r (§7.5; [U] 2026-08-18 G1 归一化口径)
FLOOR_DRAWS = 3          # 每评 ≥3 次独立抽取取中位 (G1 硬要求)


@torch.no_grad()
def floor_measure(model, sb, cfg, w, dev, rng, n_draws=FLOOR_DRAWS):
    """地板 = 与 L_S **同一批 S、同一 w_t 快照、同一 θ_t** 下, 只把当步笔记换成**逐样本等占空**随机画布 (章数同、格位章型重抽)
    的 L̄_ds = 图像完全不携带 N 信息时的水平. n_draws 次独立抽取 (每次换信道实现 + 换随机画布; 笔记类别图 k 只依赖场景, 与信道
    无关, 各抽取重渲同一 k) 取中位, 散布一并入册. 闸量 r = median(L_S)/median(地板) (G1: r > LSTAR_FRAC ⇒ β 当评置零, 每评重算,
    对 w_t 迁移免疫); l_star = 0.5×地板 只作历史对照."""
    B = sb["B"]
    L_S, floors, accs = [], [], []
    hc = k = None
    for _ in range(n_draws):
        draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
        tpl = R.templates(draw.u, draw.s)
        if hc is None:
            out = scene_side(model, sb, w, cfg.mu, cfg, draw, tpl)
            hc, k = out["hc"], out["k"]
            S = out["S"]
        else:
            x2 = R.channel(R.render_hard(k, tpl), draw)                # 同一 k, 新信道实现 (直通前向 = 硬渲染, A2)
            S = lds_rows(model, model.encode(x2, is_scene=False)["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, cfg.mu)
        L_S.append(float(S["rows"].mean()))
        k_rand = R.random_canvas_like(k, rng)
        x = R.channel(R.render_hard(k_rand, tpl), draw)
        fl = lds_rows(model, model.encode(x, is_scene=False)["cls"], hc, sb["th"], sb["tg"], sb["truth5"], sb["ns"], w, cfg.mu)
        floors.append(float(fl["rows"].mean()))
        accs.append(fl["accs"].mean(0))
    fm = float(torch.tensor(floors, dtype=torch.float64).median())
    lm = float(torch.tensor(L_S, dtype=torch.float64).median())
    return dict(floor=fm, L_S=lm, r=lm / fm, floors=floors, L_S_draws=L_S, n_draws=n_draws, B=B,
                floor_sd=(float(torch.tensor(floors, dtype=torch.float64).std()) if n_draws > 1 else 0.0),
                floor_acc=torch.stack(accs).mean(0).tolist(), l_star=AL.LSTAR_FRAC * fm)


def gate_open(fl):
    """水位闸 (§7.5 G1): r = L_S/地板_当评 ≤ LSTAR_FRAC(0.5) ⇒ 开 (β 生效); r > 0.5 ⇒ 关 (β 当评置零, 下评重测).
    语义与原 L* = 0.5×地板 逐字相同, 只是地板每评当场重算. fl = floor_measure 返回值."""
    return bool(fl["r"] <= AL.LSTAR_FRAC)
