# tests/test_nc_align.py
"""对齐最小化 v4.1 判别测试 (align.py + trainer 接线):
度规约定 / R 谱合成 / 对齐量 A / 留出轮换 / 闭式 c 对 autograd 对表 (§6.2 承重点) /
c 通道性质 + X4 重建 + 状态往返 / 打乱标签零假设构造 / 有限差分对真 HVP (GA2 接线) /
Hutchinson 与 HVP 二次型精确 / X1 留出纯洁性逐位 / X3 κ 项双向钉子 / 定标码 /
盲 CE 地板 (GA5) / 扩张步长 1 / 装配级冒烟 (含非零梯度断言) / 检查点往返."""
import math

import pytest
import torch
import torch.nn.functional as F

from symemerge.numcode import align as AL
from symemerge.numcode import data as D
from symemerge.numcode import geometry as GM
from symemerge.numcode import grpo as GR
from symemerge.numcode import trainer as TR
from symemerge.numcode.model import NumCodeModel

DEV = torch.device("cpu")


def _fake_adam(params, steps=100, seed=0):
    """造一个走过 steps 步的 AdamW 状态 (随机正的二阶矩), 供度规测试."""
    opt = torch.optim.AdamW(params, lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for p in params:
        opt.state[p] = dict(step=torch.tensor(float(steps)),
                            exp_avg=torch.randn(p.shape, generator=g) * 1e-3,
                            exp_avg_sq=torch.rand(p.shape, generator=g) * 1e-4 + 1e-8)
    return opt


# ---------------------------------------------------------------- 度规
def test_metric_convention_A():
    """F=diag(√v̂+ε): inner = Σab/(√v̂+ε); 白化欧氏内积 = inner; precond = a/(√v̂+ε)."""
    ps = [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(7))]
    opt = _fake_adam(ps)
    met = AL.Metric(opt, "adam")
    den = met.denom(ps)
    b2 = opt.param_groups[0]["betas"][1]
    for p, d in zip(ps, den):
        vh = opt.state[p]["exp_avg_sq"] / (1 - b2 ** 100)
        assert torch.allclose(d, vh.sqrt() + opt.param_groups[0]["eps"])
    a = [torch.randn_like(p) for p in ps]
    b = [torch.randn_like(p) for p in ps]
    ref = sum(((x * y) / d).sum() for x, y, d in zip(a, b, den))
    assert torch.allclose(AL.Metric.inner(a, b, den), ref)
    wa, wb = AL.Metric.whiten_flat(a, den), AL.Metric.whiten_flat(b, den)
    assert torch.allclose(wa @ wb, ref, atol=1e-5)
    pc = AL.Metric.precond(a, den)
    assert torch.allclose(pc[0], a[0] / den[0])
    ident = AL.Metric(None, "identity").denom(ps)
    assert all(bool((d == 1).all()) for d in ident)


def test_metric_no_state_falls_back_to_ones():
    ps = [torch.nn.Parameter(torch.randn(4))]
    opt = torch.optim.AdamW(ps, lr=1e-3)          # 无状态
    den = AL.Metric(opt, "adam").denom(ps)
    assert bool((den[0] == 1).all())


# ---------------------------------------------------------------- R 谱 / A
def test_gram_R_identity_and_rank1():
    Fn = 6
    X = torch.eye(Fn) * 2.0
    assert abs(AL.gram_R(X, demean=False)["R"] - Fn) < 1e-6      # 正交等模 → R=F
    assert abs(AL.gram_R(X)["R"] - (Fn - 1)) < 1e-5              # 去均值后 F−1
    v = torch.randn(10)
    Y = torch.stack([(i + 1) * v for i in range(Fn)])           # 秩 1
    assert abs(AL.gram_R(Y)["R"] - 1.0) < 1e-5
    eig = AL.gram_R(Y)["eig"]
    assert eig[0] > 0 and all(e < 1e-4 * eig[0] for e in eig[1:])


def test_alignment_A_top_mode():
    """A = F·Σ c_k² λ̂_k: 等谱 ⇒ A=1 (任意 ℓ); 单模谱 λ̂=(1,0,0,0), U=I,
    ℓ=(3,1,1,1) ⇒ ℓ̃=(1.5,−.5,−.5,−.5), c_0²=2.25/3 ⇒ A = 4×0.75 = 3."""
    Fn = 4
    U = torch.eye(Fn)
    assert abs(AL.alignment_A([1.0] * Fn, U, torch.randn(Fn)) - 1.0) < 1e-5
    assert abs(AL.alignment_A([4.0, 0.0, 0.0, 0.0], U,
                              torch.tensor([3.0, 1.0, 1.0, 1.0])) - 3.0) < 1e-5


def test_R_truncation_curve():
    X = torch.eye(5)
    curve = AL.R_truncation(X, [5, 3, 1, 4, 2])
    assert [f for f, _ in curve] == [2, 3, 4, 5]
    assert all(abs(r - (f - 1)) < 1e-5 for f, r in curve)


# ---------------------------------------------------------------- 留出轮换
def test_holdout_sched_rotation_and_coverage():
    ns = list(range(1, 13))
    hs = AL.HoldoutSched(ns, 0.15, 10, seed=3)
    assert hs.size() == 2
    seen, Ss = set(), []
    for step in range(0, 60):
        S = hs.current(step)
        assert len(S) == 2 and S <= set(ns)
        if step % 10 == 0:
            Ss.append(tuple(sorted(S)))
        seen |= S
    assert seen == set(ns)                       # 6 轮 × 2 = 全覆盖
    assert len(set(Ss)) == 6                     # 每轮不同
    assert hs.current(0 + 5) == hs.current(0 + 9)   # 周期内不变
    hs.on_expand([13])
    assert 13 in hs.ns and 13 in hs.order
    hs.frac = 0.0
    hs.enabled = False
    assert hs.current(1000) == set()
    hs2 = AL.HoldoutSched(ns, 0.15, 10, seed=3)
    hs2.load_state(hs.state())
    assert hs2.S == hs.S and hs2.ptr == hs.ptr and hs2.ns == hs.ns


def test_holdout_sched_min_size_one():
    hs = AL.HoldoutSched([1, 2, 3, 4], 0.15, 5, seed=0)
    assert hs.size() == 1
    assert len(hs.current(0)) == 1


# ---------------------------------------------------------------- 闭式 c 对表 (§6.2)
@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return NumCodeModel()


@pytest.fixture(scope="module")
def groups4():
    """四组, 数量恰 1,2,3,4 (S/V 划分可控)."""
    rng = torch.Generator().manual_seed(11)
    return [D.sample_group(rng, 4, n=n) for n in (1, 2, 3, 4)]


def _route(net, groups, g, rng, cfg_s=0.15, occ=4):
    b = TR.build_group_batch(groups, g, DEV)
    with torch.no_grad():
        st = net.encode_scene(b["x1"])["tokens"].repeat_interleave(g, 0)
        progs, _, _ = net.writer.rollout(st, 12, torch.ones(len(groups) * g),
                                         rng=torch.Generator().manual_seed(5))
    x2 = GM.render_channel(progs, cfg_s, rng, occ)
    o = AL.canvas_route(net, x2, b["th"], b["tg"], b["truth5"], b["cands"],
                        b["ns_rep"], 2.0, 0.05)
    return b, x2, o


def test_c_closed_form_matches_autograd(net, groups4):
    """§6.2 承重点: 逐 rollout Σ_t w_t CE_t 对 t1..t6 权重/偏置的 autograd 梯度 ==
    闭式 a⊗x / Δ⊗h (含 φ(θ) 拼接列与 t5 双线性)."""
    rng = torch.Generator().manual_seed(7)
    b, x2, o = _route(net, groups4, 2, rng)
    w = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.2, 0.1])
    ch = AL.CChannel(net, 0.99, DEV)
    parts = ch.parts(net.heads, o["h"].detach(), {t: v.detach() for t, v in o["tl"].items()},
                     o["t5"].detach(), o["hc"].detach(), b["tg"], b["truth5"], b["th"], w)
    direct = AL.head_grad_direct(net, o, w, b["tg"], b["truth5"])
    B = o["h"].shape[0]
    for i in range(B):
        for t, wi, bi, shp in ch.blocks:
            a, x = parts[t]
            Gw = torch.outer(a[i], x[i])
            assert torch.allclose(Gw, direct[i][wi], atol=1e-5, rtol=1e-4), t
            if bi is not None:
                assert torch.allclose(a[i], direct[i][bi], atol=1e-5, rtol=1e-4), t


def test_c_channel_properties_and_roundtrip(net, groups4):
    """([U] 2026-08-17 改动 4 窗均值口径) 全同部件 ⇒ c=1; 入册只进本窗累加, roll() 前不就绪
    (首评前 c 恒 0); roll() 后 bar = 窗均值 (无 EMA: 两窗各自独立, 第二窗整体换新, 旧窗不残留);
    本窗缺席的数量保留上一份 bar; 状态往返."""
    rng = torch.Generator().manual_seed(8)
    b, x2, o = _route(net, groups4, 2, rng)
    w = torch.full((6,), 1 / 6)
    ch = AL.CChannel(net, 0.9, DEV)
    parts = ch.parts(net.heads, o["h"].detach(), {t: v.detach() for t, v in o["tl"].items()},
                     o["t5"].detach(), o["hc"].detach(), b["tg"], b["truth5"], b["th"], w)
    ns = b["ns_rep"]
    dom = list(D.train_ns(4))
    den = AL.Metric(None, "identity").denom(ch.hp)
    assert not ch.ready(dom)
    assert bool((ch.c_of(parts, ns, den, dom) == 0).all())    # 未就绪 → 0
    # 全同部件: 每 rollout 的 (a, x) 都取第 0 行 → G^(i) 全同 → c = 1
    same = {t: (a[:1].expand_as(a).clone(), x[:1].expand_as(x).clone())
            for t, (a, x) in parts.items()}
    ns_all = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    ch.update(same, ns_all)
    assert not ch.ready(dom)                                  # 入册 ≠ 就绪: 要等 roll()
    assert bool((ch.c_of(same, ns_all, den, dom) == 0).all())
    assert ch.roll() == 4 and ch.ready(dom) and ch.rolls == 1
    c = ch.c_of(same, ns_all, den, dom)
    assert torch.allclose(c, torch.ones_like(c), atol=1e-4)
    # 真部件: c ∈ [-1, 1]
    c2 = ch.c_of(parts, ns, den, dom)
    assert bool((c2.abs() <= 1.0 + 1e-5).all())
    sd = ch.state()
    ch2 = AL.CChannel(net, 0.9, DEV)
    ch2.load_state(sd)
    assert torch.allclose(ch2.c_of(same, ns_all, den, dom), c)
    # 无 EMA: 第二窗只入册真部件 (数量 1..4 各两行) 两次 → roll 后 bar == 真部件的逐数量均值,
    # 与旧窗 (全同部件) 无关; 与手算逐位同 (窗内两次相同入册的均值 = 一次的值)
    ch.update(parts, ns)
    ch.update(parts, ns)
    ch.roll()
    for t, wi, bi, shp in ch.blocks:
        a, x = parts[t]
        for n in (1, 2, 3, 4):
            rows = (ns == n).nonzero().squeeze(1)
            Gw = torch.einsum("bc,bj->cj", a[rows], x[rows]) / rows.shape[0]
            assert torch.allclose(ch.bar[t + ".w"][n], Gw, atol=1e-6), (t, n)
    # 缺席数量保留上一份 bar: 第三窗只入册数量 1 → roll 后 2..4 的 bar 不变, 1 换新
    before = {k: v.clone() for k, v in ch.bar.items()}
    ch.update(same, torch.tensor([1, 1]))
    assert ch.roll() == 1
    for k in ch.bar:
        assert torch.equal(ch.bar[k][2:5], before[k][2:5])
        assert not torch.equal(ch.bar[k][1], before[k][1])
    assert int(ch.n_bar[1]) == 1 and int(ch.n_bar[2]) == 2 and ch.ready(dom)


def test_null_targets_relabels(groups4):
    b = TR.build_group_batch(groups4, 2, DEV)
    rng = torch.Generator().manual_seed(1)
    tgn, t5n, nsn = AL.null_targets(b["ns_rep"], b["th"], b["truth5"], 4, rng, DEV)
    assert bool((nsn != b["ns_rep"]).all())
    assert bool((t5n != b["truth5"]).all())
    for i in range(nsn.shape[0]):
        theta = dict(tau=int(b["th"]["tau"][i]) + 1, p=D.P_CHOICES[int(b["th"]["p"][i])],
                     m=D.M_CHOICES[int(b["th"]["m"][i])])
        tt = D.targets(int(nsn[i]), theta)
        for t in AL.TASKS_LIN:
            assert int(tgn[t][i]) == tt[t]


# ---------------------------------------------------------------- 曲率
def test_hvp_and_hutchinson_quadratic():
    """L = ½ xᵀAx: H·v = Av; 对角 A 的单次 Rademacher 估计 = tr(A) 精确."""
    A = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.nn.Parameter(torch.randn(4))
    loss_fn = lambda: 0.5 * x @ A @ x
    v = [torch.randn(4)]
    hv = AL.hvp(loss_fn, [x], v)
    assert torch.allclose(hv[0], A @ v[0], atol=1e-6)
    tr = AL.hutchinson_trace(loss_fn, [x], torch.Generator().manual_seed(0))
    assert abs(tr - 10.0) < 1e-5


def test_cross_term_fd_matches_hvp(net, groups4):
    """GA2 接线: u_S 有限差分 ≈ H_S F⁻¹ĝ_V / ‖g_S‖ (真 HVP double backward), 余弦 ≥ .95,
    相对误差 ≤ .1 (小 δ, 单位度规)."""
    rng = torch.Generator().manual_seed(9)
    cvp = AL.cv_params(net)
    den = AL.Metric(None, "identity").denom(cvp)
    bV, x2V, oV = _route(net, groups4[:2], 1, rng)
    bS, x2S, oS = _route(net, groups4[2:], 1, rng)
    g_V = AL.grads_of(oV["lrow"].mean(), cvp)
    g_S = AL.grads_of(oS["lrow"].mean(), cvp)

    def recompute():
        o = AL.canvas_route(net, x2S, bS["th"], bS["tg"], bS["truth5"], bS["cands"],
                            bS["ns_rep"], 2.0, 0.05)
        return AL.grads_of(o["lrow"].mean(), cvp)

    u, diag = AL.cross_term(cvp, den, g_V, g_S, recompute, delta_abs=1e-3)
    assert not diag["degenerate"]
    for p, o in zip(cvp, [p.detach().clone() for p in cvp]):
        assert torch.equal(p.detach(), o)                       # 逐位复原
    nV = AL.Metric.norm(g_V, den)
    dirn = AL.Metric.precond([g / nV for g in g_V], den)

    def loss_S():
        o = AL.canvas_route(net, x2S, bS["th"], bS["tg"], bS["truth5"], bS["cands"],
                            bS["ns_rep"], 2.0, 0.05)
        return o["lrow"].mean()

    hv = AL.hvp(loss_S, cvp, dirn)
    nS = AL.Metric.norm(g_S, den)
    ref = [h / nS for h in hv]
    cos = float(AL.Metric.cos(u, ref, den))
    rel = float(AL.Metric.norm([a - b for a, b in zip(u, ref)], den)
                / AL.Metric.norm(ref, den))
    assert cos >= AL.GA2_COS and rel <= AL.GA2_REL, (cos, rel)


# ---------------------------------------------------------------- 定标码 / 地板
def test_cal_programs():
    ns = [1, 2, 5, 17, 20]
    un = AL.cal_table("unary", ns, 0)
    assert all(len(un[n]) == n for n in ns)
    pl = AL.cal_table("place", ns, 0)
    for n in ns:                                    # 基数-4 位值解码回 n
        v = 0
        for aid in pl[n]:
            cell, st = GM.aid_unpack(aid)
            v += (st + 1) * (GM.S + 1) ** cell
        assert v == n
    ct = AL.cal_table("const", ns, 0)
    assert all(ct[n] == ct[ns[0]] for n in ns) and len(ct[1]) == 12
    rd = AL.cal_table("random", ns, 0)
    assert len(set(tuple(sorted(rd[n])) for n in ns)) == len(ns)
    for tab in (un, pl, ct, rd):
        GM.raster([tab[n] for n in ns])              # 皆可光栅化 (无重复格)


def test_blind_task_ce_k4_and_floor():
    ce = AL.blind_task_ce(4)
    assert abs(ce["t2"] - math.log(2)) < 1e-9
    assert abs(ce["t6"] - math.log(4)) < 1e-9
    assert abs(ce["t5"] - math.log(4)) < 1e-9
    h2 = lambda q: -(q * math.log(q) + (1 - q) * math.log(1 - q))
    assert abs(ce["t1"] - (h2(0.75) + h2(0.5) + h2(0.25)) / 3) < 1e-9
    fl = AL.abstain_floor(4, 0.05, 1.0)
    assert abs(fl["floor"] - (ce["l_task_blind"] + 0.05)) < 1e-9
    assert abs(fl["l_star"] - AL.LSTAR_FRAC * fl["floor"]) < 1e-9


def test_snr_from_cos():
    assert AL.snr_from_cos(0.5) == pytest.approx(math.sqrt(2.0))
    assert AL.snr_from_cos(-0.2) == 0.0


# ---------------------------------------------------------------- 装配级
def _cfg(**kw):
    base = dict(phase=2, k=4, n_groups=4, g=2, n_greedy=1, batch_mask=2, batch_scenes=4,
                conf_gate=0.0, s=0.15, occ_k=4, holdout_frac=0.3, t_rotate=10,
                reader_lag=0.999, reward_m=1, align_every=1)
    base.update(kw)
    return TR.TrainCfg(**base)


def _lag(model):
    lag = NumCodeModel()
    lag.load_state_dict(model.state_dict())
    for p in lag.parameters():
        p.requires_grad_(False)
    return lag.eval()


def _step(model, cfg, groups, sc, ns, align, hold, s_scale=None, seed=3):
    model.zero_grad(set_to_none=True)
    out = TR.phase2_losses(model, cfg, groups, sc, ns, DEV,
                           torch.Generator().manual_seed(seed),
                           torch.Generator().manual_seed(seed + 1), 0,
                           GR.TaskWeights(), lag=_lag(model), align=align, hold=hold,
                           s_scale=s_scale)
    (out["l_grpo"] + out["l_count"]).backward()
    main = {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}
    ad = TR.apply_align_grads(model, align, out, cfg, align.opt, DEV)
    full = {n: (p.grad.detach().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}
    return out, main, full, ad


def test_x1_holdout_purity_bitwise(groups4):
    """X1: S 的精确梯度不进 optimizer -- 把 S 逐行损失乘 3.7 (g_S 随机化), β=0 下
    全部 .grad 逐位不变; β>0 下主梯度逐位不变、修正只经 u_S 通道 (g_S 置零 ⇒ 退化
    ⇒ 修正为零). 全程 S 组 GRPO 照常发钱 (l_grpo 有图)."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    hold = {1, 2}                                                # S={1,2}, V={3,4}
    rng = torch.Generator().manual_seed(2)
    sc, ns = D.sample_scene_batch(rng, 4, 4)
    out1, main1, full1, _ = _step(model, cfg, groups4, sc, ns, align, hold)
    out2, main2, full2, _ = _step(model, cfg, groups4, sc, ns, align, hold, s_scale=3.7)
    assert out1["align"]["g_S"] is not None
    for n in full1:
        if full1[n] is None:
            assert full2[n] is None
        else:
            assert torch.equal(full1[n], full2[n]), n            # β=0: 逐位不变
    assert not torch.equal(out1["align"]["g_S"][0], out2["align"]["g_S"][0])
    cfg_b = _cfg(beta_align=0.05)
    align_b = TR.AlignCtx(model, opt, cfg_b, DEV, list(D.train_ns(4)))
    o3, main3, full3, ad3 = _step(model, cfg_b, groups4, sc, ns, align_b, hold)
    o4, main4, full4, ad4 = _step(model, cfg_b, groups4, sc, ns, align_b, hold, s_scale=0.0)
    assert ad3["beta_eff"] > 0 and ad4["beta_eff"] == 0.0        # g_S=0 ⇒ 退化 ⇒ 无修正
    cv_names = {n for n, p in model.named_parameters()
                if any(p is q for q in align_b.cvp)}
    for n in full3:
        if full3[n] is None:
            continue
        if main3[n] is None:                                    # 头参数在主反传前无 grad
            assert main4[n] is None, n
        else:
            assert torch.equal(main3[n], main4[n]), n           # 主梯度不看 g_S
        if n not in cv_names:
            assert torch.equal(full3[n], full4[n]), n           # 非 Θ_cv 无修正
    assert any(not torch.equal(full3[n], full4[n]) for n in cv_names)   # 修正只经 u_S


def test_x3_kappa_two_way(groups4):
    """X3: (a) 毁掉影子读者 ⇒ c 变 ⇒ r 变 (κ>0, 缓冲就绪); (b) 毁掉 c 通道 (κ=0 或
    c=None) ⇒ acc/λ 项分毫不差 (奖励逐位等于旧式)."""
    accs = torch.rand(8, 6).round()
    w = torch.full((6,), 1 / 6)
    n_st = torch.randint(0, 10, (8,))
    c = torch.rand(8) * 2 - 1
    r0 = GR.reward(accs, w, n_st, 0.005, 0.02)
    assert torch.equal(GR.reward(accs, w, n_st, 0.005, 0.02, c_align=c, kappa=0.0), r0)
    assert torch.equal(GR.reward(accs, w, n_st, 0.005, 0.02, c_align=None, kappa=0.3), r0)
    r1 = GR.reward(accs, w, n_st, 0.005, 0.02, c_align=c, kappa=0.3)
    assert torch.allclose(r1 - r0, 0.3 * (accs @ w) * c)          # 乘性调制
    c2 = c.clone()
    c2[0] += 0.5
    assert not torch.equal(GR.reward(accs, w, n_st, 0.005, 0.02, c_align=c2, kappa=0.3), r1)
    # 装配级 (旧余弦口径 c_mode=cos): κ>0 但缓冲未就绪 ⇒ kappa_on=0 且 r 与 κ=0 逐位同
    torch.manual_seed(0)
    model = NumCodeModel()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(2)
    sc, ns = D.sample_scene_batch(rng, 4, 4)
    res = {}
    for kap in (0.0, 0.3):
        cfg = _cfg(kappa=kap, c_mode="cos")
        align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
        out, *_ = _step(model, cfg, groups4, sc, ns, align, {groups4[0]["n"]})
        res[kap] = out["metrics"]["r_mean"]
        assert out["metrics"].get("kappa_on", 0.0) == 0.0
    assert res[0.0] == res[0.3]
    # 投影率口径 (c_mode=proj, 默认): S 行存在且 V 子空间存在 ⇒ kappa_on=1, κ 项只落 S 行 (V 行 c=0);
    # S 为空 ⇒ kappa_on=0
    captured = {}
    orig_reward = GR.reward

    def rec_reward(accs, weights, n_stamps, lam, lam0=0.0, probe_hits=None, alpha=0.0,
                   c_align=None, kappa=0.0):
        captured["c"] = None if c_align is None else c_align.detach().clone()
        return orig_reward(accs, weights, n_stamps, lam, lam0, probe_hits, alpha, c_align, kappa)
    TR.GR.reward = rec_reward
    try:
        cfg = _cfg(kappa=0.3)
        align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
        hold = {groups4[0]["n"], groups4[1]["n"]}                     # S = {1,2}, V = {3,4}
        out, *_ = _step(model, cfg, groups4, sc, ns, align, hold)
        assert out["metrics"]["kappa_on"] == 1.0
        c = captured["c"]
        isS = torch.tensor([gr["n"] in hold for gr in groups4]).repeat_interleave(cfg.g)
        assert c is not None and bool((c[~isS] == 0).all())           # V 行 κ 项为 0
        assert bool((c[isS] >= 0).all()) and bool((c[isS] <= 1).all()) and float(c[isS].max()) > 0
        assert out["metrics"]["c_V"] is None
        assert out["metrics"]["c_mean"] == pytest.approx(float(c[isS].mean()), abs=1e-5)
        assert out["metrics"]["c_rank"] >= 1 and out["metrics"]["c_kV"] == 2
        assert out["metrics"]["c_null"] is not None                    # align_every=1 ⇒ 每步有零假设
        out0, *_ = _step(model, cfg, groups4, sc, ns, align, set())    # S 空 ⇒ 关
        assert out0["metrics"]["kappa_on"] == 0.0 and out0["metrics"]["c_mean"] is None
    finally:
        TR.GR.reward = orig_reward


def test_phase2_align_smoke_and_nonzero_grads(groups4):
    """装配级: 分路前向 + 手工装配 -- Θ_cv 全体收到非零梯度 (nonzero-gradient 断言),
    S 计数流不 apply (l_count_S 记账), 逐步仪表齐全."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg(beta_align=0.02)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    hold = {groups4[0]["n"]}
    rng = torch.Generator().manual_seed(2)
    sc, ns = D.sample_scene_batch(rng, 4, 8)
    ns[0] = groups4[0]["n"]                                      # 计数流里放一个 S
    out, main, full, ad = _step(model, cfg, groups4, sc, ns, align, hold)
    A = out["align"]
    for k in ("cos_cv", "tau", "S_S", "S_V", "tau_null", "cos_sc", "cos_joint",
              "cos_null"):
        assert k in A["diag"], k
    assert out["metrics"]["l_count_S"] is not None
    assert "abst_S" in out["metrics"] and "c_mean" in out["metrics"]
    names = {id(p): n for n, p in model.named_parameters()}
    for p in align.cvp:
        if names[id(p)] == "E.mask_tok":                         # 只经素养流 (本测试不含)
            continue
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, names[id(p)]
    assert ad["u"] > 0 and ad["beta_eff"] > 0
    assert ad["corr_frac"] <= AL.BETA_CAP + 1e-6                 # §10 β 夹取
    # 计数头 (非 Θ_cv) 也有梯度 (V 计数流照常)
    assert float(model.heads.count[0].weight.grad.abs().sum()) > 0
    fl = align.flush()                                            # 窗由 run() 喂, 此处空
    assert "c_hist" in fl and "c_by_n" in fl and sum(fl["c_hist"]) > 0


def test_expand_step_one(tmp_path, monkeypatch):
    cfg = TR.TrainCfg(phase=2, k=8, out=str(tmp_path), expand_step=1)
    monkeypatch.setattr(TR, "measure_extrap0",
                        lambda *a, **k: dict(per_n={}, mean=0.0))
    latch = TR.BestLatch()
    ex = TR.do_expand(NumCodeModel(), cfg, None, latch, 100, DEV)
    assert ex["k"] == [8, 9] and cfg.k == 9
    cfg2 = TR.TrainCfg(phase=2, k=8, out=str(tmp_path))            # 0 ⇒ 常数 +2
    ex2 = TR.do_expand(NumCodeModel(), cfg2, None, latch, 100, DEV)
    assert ex2["k"] == [8, 10]


def test_align_final_and_first_seen_accumulate():
    class _A:
        pass
    a = _A()
    a.first_seen = [dict(n=13, jump=1, l_ex=0.5, l_ex_count=0.1),
                    dict(n=14, jump=1, l_ex=0.25, l_ex_count=0.1),
                    dict(n=57, jump=13, l_ex=0.9, l_ex_count=0.2)]
    a.R_hist = [(500, 12, 6.0), (1000, 12, 6.5), (1500, 13, 7.0), (2500, 14, 7.2)]
    a.cchan = _A()
    a.cchan.rolls = 0
    fin = TR.align_final(a)
    assert [r["L_preq"] for r in fin["L_preq"]] == [0.5, 0.75]
    assert len(fin["first_seen_jump_gt1"]) == 1
    assert fin["R_F"] == [(12, 6.5), (13, 7.0), (14, 7.2)]
    assert fin["dR_dF"] == [(13, 0.5), (14, 0.2)]
    assert fin["d2R"] == [(14, -0.3)]


def _tiny_es(rng, k=4, per=2):
    es = dict(m1_pool={n: [D.sample_group(rng, k, n=n) for _ in range(per)]
                       for n in D.train_ns(k)})
    mp = [GM.sample_uniform_prog(rng, 20) for _ in range(8)]
    from symemerge.numcode import losses as L
    es["mask_x"] = GM.channel(GM.raster(mp, 0.1, rng), 0.1, rng)
    es["mask_truth"] = GM.cell_truth(mp).view(8, GM.T)
    sel = L.sample_mask_cells(8, rng)
    es["mask_rep"], es["mask_keep"] = TR.split_mask_keep(sel, rng, 0.15)
    es["mask_sel"] = sel
    es["groups"] = [D.sample_group(rng, k) for _ in range(6)]
    return es


def test_align_eval_and_first_seen_smoke():
    """每评仪表 M1-M8 出数 (R/谱/A/分块/截断/对照列/曲率/窗汇总) + M6 首见行结构."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg(k=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    rng = torch.Generator().manual_seed(4)
    es = _tiny_es(rng)
    align.win.append(dict(cos_cv=0.1, tau=0.2, L_S=1.0))
    ev = TR.align_eval(model, cfg, align, es, DEV, 500, {2}, 0.0)
    m1 = ev["M1"]
    assert len(m1["eig"]) == 4 and m1["R"] >= 1.0 - 1e-6
    assert len(m1["R_f"]) == 3 and "A" in m1 and "R_bb" in m1 and "R_head" in m1
    assert "R_sc" in m1 and "R_rep_cls" in m1 and "R_rep_tok" in m1
    assert m1["R_ctrl_mask"] is not None
    assert ev["M7"]["S"] is not None or ev["M7"]["V"] is not None
    assert ev["cos_cv"] == pytest.approx(0.1) and ev["n_steps"] == 1
    assert len(align.R_hist) == 1 and align.last_ell
    assert align.last_acc and set(align.last_acc) == set(align.last_ell)
    # M6: 假装 4 是新引入 (地板 = 其余三数中位); 新生模型无一达标 ⇒ floor_kind="all"
    old = {n: l for n, l in align.last_ell.items() if n != 4}
    align.last_ell = old
    align.last_ell_count = {n: l for n, l in align.last_ell_count.items() if n != 4}
    align.last_acc = {n: 0.0 for n in old}
    rows = TR.first_seen_rows(model, cfg, align, es, DEV, [4], 3)
    assert rows[0]["n"] == 4 and rows[0]["jump"] == 1
    assert rows[0]["l_ex"] == pytest.approx(rows[0]["l_note"] - rows[0]["floor"], abs=1e-3)
    assert rows[0]["floor_kind"] == "all" and rows[0]["n_qualified"] == 0
    assert rows[0]["floor"] == rows[0]["floor_all"]
    # 改动 4: 每评 roll -- 窗内入册的数量换成窗均值 (无 X4 重建; v_med 不再触发任何动作)
    align.cchan.n_acc[1] = 3
    ev2 = TR.align_eval(model, cfg, align, es, DEV, 1000, {2}, 5.0)
    assert ev2.get("cchan_rolled") == 1 and int(align.cchan.n_bar[1]) == 3
    assert int(align.cchan.n_acc[1]) == 0 and "cchan_rebuild" not in ev2
    assert not hasattr(AL, "REBUILD_V") and not hasattr(align.cchan, "rebuild")


def test_first_seen_floor_qualified_numbers():
    """改动 3 配套 ([U]): ℓ_ex 地板 = 已达标数量 (上评 M1 六任务均值 ≥ FLOOR_ACC=.95) 的 ℓ_n
    中位, 不是全部已引入数量; 无达标数量时退化为全部并标注. 合成 last_ell/last_acc 手算."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg(k=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    rng = torch.Generator().manual_seed(4)
    es = _tiny_es(rng)
    align.last_ell = {1: 0.10, 2: 0.50, 3: 2.00}       # 全部中位 = .50
    align.last_ell_count = {1: 0.01, 2: 0.02, 3: 0.03}
    align.last_acc = {1: 1.0, 2: 0.80, 3: 0.97}         # 达标 {1,3} ⇒ 中位 (排序 [.1, 2.0]) 取上位 2.0
    rows = TR.first_seen_rows(model, cfg, align, es, DEV, [4], 3)
    r = rows[0]
    assert r["floor_kind"] == "qualified" and r["n_qualified"] == 2
    assert r["floor"] == pytest.approx(2.0) and r["floor_all"] == pytest.approx(0.5)
    assert r["l_ex"] == pytest.approx(r["l_note"] - 2.0, abs=1e-3)
    assert r["floor_count"] == pytest.approx(0.03)      # 计数版地板同口径 (达标 {1,3} 中位)
    assert TR.FLOOR_ACC == TR.EXPAND_SCORE == 0.95


def test_m1_eval_only_ctx_empty_S():
    """B 臂 (α=0 复跑, [C] m1_eval=1): holdout_frac=0 下只评用 AlignCtx -- S 恒空,
    align_eval 仍出 M1 (R/谱/对照列) 与 M7-V, M7-S 为 None, 无逐步窗 ⇒ c 直方/逐数量 c
    不入册 (N/A 留空, 不填 0); 训练路径由 run() 传 align=None (逐位旧制, B0-a 断言)."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg(k=4, holdout_frac=0.0, m1_eval=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    assert align.hold.size() == 0 and align.hold.current(0) == set()
    assert align.hold.current(10 ** 6) == set()
    rng = torch.Generator().manual_seed(4)
    es = _tiny_es(rng)
    ev = TR.align_eval(model, cfg, align, es, DEV, 500, align.hold.current(0), 0.0)
    assert ev["S"] == []
    m1 = ev["M1"]
    assert len(m1["eig"]) == 4 and m1["R"] >= 1.0 - 1e-6 and m1["R_ctrl_mask"] is not None
    assert ev["M7"]["S"] is None and ev["M7"]["V"] is not None
    for k in ("c_hist", "c_by_n", "capped", "n_steps", "L_S", "tau", "cos_cv"):
        assert k not in ev, k
    assert len(align.R_hist) == 1
    assert "m1_eval" in TR.TrainCfg.__dataclass_fields__ and TR.TrainCfg().m1_eval == 0


def test_ckpt_align_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = NumCodeModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = _cfg()
    al = TR.AlignCtx(model, opt, cfg, DEV, [1, 2, 3, 4])
    al.hold.current(0)
    al.first_seen.append(dict(n=5, jump=1, l_ex=0.3))
    al.R_hist.append((500, 4, 3.2))
    al.cchan.cnt[2] = 4
    al.last_ell, al.last_acc = {1: 0.2, 2: 0.3}, {1: 1.0, 2: 0.5}
    al.last_expand, al.expands = 1500, [dict(step=1500, k=[4, 5], why="timer")]
    p = str(tmp_path / "ck.pt")
    lag = _lag(model)
    with torch.no_grad():
        next(lag.parameters()).add_(1.0)                          # 影子 ≠ 在线
    misc = dict(tpl_hist=[{"1": [0] * 4}], expands=al.expands, cctl=1, sctl=2, xctl=1,
                stop=dict(s_score=2, s_count=1))
    TR.save_ckpt(p, model, opt, GR.TaskWeights(), 10, cfg, dict(score=0.5, step=10),
                 align=al, lag=lag, misc=misc)
    meta = TR.load_ckpt(p, model)
    al2 = TR.AlignCtx(model, opt, cfg, DEV, [1, 2, 3, 4])
    al2.load_state(meta["align_sd"])
    assert al2.hold.S == al.hold.S and al2.first_seen == al.first_seen
    assert al2.R_hist == al.R_hist and int(al2.cchan.cnt[2]) == 4
    assert al2.last_ell == al.last_ell and al2.last_acc == al.last_acc
    assert al2.last_expand == 1500 and al2.expands == al.expands
    # L* 门态: l_star>0 继承检查点; l_star=0 ⇒ 恒开 (第四阶段步骤 2 教训)
    al.beta_gate = False
    sd = al.state()
    cfg_g = _cfg(l_star=1.7)
    al4 = TR.AlignCtx(model, opt, cfg_g, DEV, [1, 2, 3, 4]); al4.load_state(sd)
    assert al4.beta_gate is False
    cfg_0 = _cfg(l_star=0.0)
    al5 = TR.AlignCtx(model, opt, cfg_0, DEV, [1, 2, 3, 4]); al5.load_state(sd)
    assert al5.beta_gate is True
    # 恢复保真 (分阶段 --resume): 影子权重与 misc 原样回来
    lag2 = NumCodeModel()
    lag2.load_state_dict(meta["lag_sd"])
    assert torch.equal(next(lag2.parameters()), next(lag.parameters()))
    assert not torch.equal(next(lag2.parameters()), next(model.parameters()))
    assert meta["misc"]["stop"] == dict(s_score=2, s_count=1) and meta["misc"]["cctl"] == 1
    assert meta["misc"]["tpl_hist"] == [{"1": [0] * 4}]


# ---------------------------------------------------------------- [U] 2026-08-17 正式跑五改动
def test_c2_count_full_band_and_x1_sc_route_purity(groups4):
    """改动 2: 计数流 L_count 全带 apply (S 样本不再过滤: l_count == 全行 CE 均值, S 计数样本
    梯度进 .grad); 改动 1 X1-sc: 场景路由 S 损失只经 u_S^sc 进 .grad -- β_sc=0 下把 S 侧
    (画布路由 S 行 + 场景路由 S 损失) 乘 3.7, 全部 .grad 逐位不变; β_sc>0 (β_cv=0) 下主梯度
    逐位不变, 修正只落 Θ_sc (E ∪ count_head), 非 Θ_sc 参数逐位同; g_S^sc 置零 ⇒ 退化 ⇒ 无修正."""
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = _cfg()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    align = TR.AlignCtx(model, opt, cfg, DEV, list(D.train_ns(4)))
    hold = {1, 2}                                                # S={1,2}, V={3,4}
    rng = torch.Generator().manual_seed(2)
    sc, ns = D.sample_scene_batch(rng, 4, 8)
    ns[0], ns[1] = 1, 2                                          # 计数流里放两个 S 样本
    out1, main1, full1, ad1 = _step(model, cfg, groups4, sc, ns, align, hold)
    # 改动 2: l_count = 全行均值 (与 S 无关; 旧制 = 过滤掉 S 行的均值, 现值 ≠ 旧值), l_count_S
    # 记账非空; 场景路由 S 样本 = 计数流 S 样本 + 2 个 S 组场景
    with torch.no_grad():
        lc_row = F.cross_entropy(model.heads.count(model.encode_scene(sc)["cls"]), ns - 1,
                                 reduction="none")
    mS = (ns == 1) | (ns == 2)
    assert torch.allclose(out1["l_count"], lc_row.mean(), atol=1e-6)
    assert not torch.allclose(out1["l_count"], lc_row[~mS].mean(), atol=1e-6)
    assert out1["metrics"]["l_count_S"] == pytest.approx(float(lc_row[mS].mean()), abs=1e-5)
    assert out1["metrics"]["n_S_count"] == int(mS.sum())
    assert out1["metrics"]["n_S_sc"] == int(mS.sum()) + 2
    A = out1["align"]
    assert A["g_S_sc"] is not None and A["g_V_sc"] is not None and A["L_S_sc"] is not None
    for k in ("cos_sc", "tau_sc"):
        assert k in A["diag"] and A["diag"][k] is not None
    assert ad1["u_sc"] > 0 and ad1["beta_eff_sc"] == 0.0            # 算而不 apply
    # X1-sc (β_sc=0): S 侧缩放 3.7 ⇒ .grad 逐位不变
    out2, main2, full2, ad2 = _step(model, cfg, groups4, sc, ns, align, hold, s_scale=3.7)
    for n in full1:
        if full1[n] is None:
            assert full2[n] is None, n
        else:
            assert torch.equal(full1[n], full2[n]), n
    assert not torch.equal(out1["align"]["g_S_sc"][0], out2["align"]["g_S_sc"][0])
    # β_sc>0: 修正只落 Θ_sc; g_S^sc=0 (s_scale=0) ⇒ 退化 ⇒ 无修正
    cfg_b = _cfg(beta_sc=0.05)
    align_b = TR.AlignCtx(model, opt, cfg_b, DEV, list(D.train_ns(4)))
    o3, main3, full3, ad3 = _step(model, cfg_b, groups4, sc, ns, align_b, hold)
    o4, main4, full4, ad4 = _step(model, cfg_b, groups4, sc, ns, align_b, hold, s_scale=0.0)
    assert ad3["beta_eff_sc"] > 0 and ad4["beta_eff_sc"] == 0.0 and ad4["degenerate_sc"] == 1.0
    assert ad3.get("beta_eff", 0.0) == 0.0                           # β_cv=0: 画布路由不施压
    assert ad3["corr_frac_sc"] <= AL.BETA_CAP + 1e-6
    sc_names = {n for n, p in model.named_parameters() if any(p is q for q in align_b.scp)}
    for n in full3:
        if full3[n] is None:
            continue
        if main3[n] is not None:
            assert torch.equal(main3[n], main4[n]), n               # 主梯度不看 S 侧
        if n not in sc_names:
            assert torch.equal(full3[n], full4[n]), n               # 非 Θ_sc 无修正
    assert any(not torch.equal(full3[n], full4[n]) for n in sc_names)
    assert any(not torch.equal(full3[n], full4[n]) for n in sc_names if n.startswith("heads.count"))
    # [C] L* 水平门只管画布路由: 门关 (beta_gate=False) 时 β_sc 仍施加, β_cv 置零
    cfg_g = _cfg(beta_sc=0.05, beta_align=0.05)
    align_g = TR.AlignCtx(model, opt, cfg_g, DEV, list(D.train_ns(4)))
    align_g.beta_gate = False
    _, _, _, adg = _step(model, cfg_g, groups4, sc, ns, align_g, hold)
    assert adg["beta_eff_sc"] > 0 and adg["beta_eff"] == 0.0


def test_c3_expand_timer_and_hole_skip(tmp_path, monkeypatch):
    """改动 3: 推进条件 = 稳定触发 OR 距上次推进满 every 步; step=1 时前沿按训练带数量 +1,
    跨空洞段直接跳到下一个训练带 N (首见跳距 >1); 定时推进不重启锚宽限 ([C]), 稳定触发照旧."""
    x = TR.ExpandCtl(every=1500)
    assert x.offer(0.5, 9.0, False, step=1000, last_expand=0) is None
    assert x.offer(0.5, 9.0, False, step=1500, last_expand=0) == "timer"
    assert x.offer(0.5, 9.0, False, step=2000, last_expand=1500) is None
    x.offer(0.96, 1.0, False, step=2500, last_expand=1500)
    assert x.offer(0.96, 1.0, False, step=3000, last_expand=1500) == "stable"   # 稳定优先记名
    assert TR.next_k(12, 1) == 13 and TR.next_k(44, 1) == 57 and TR.next_k(82, 1) == 95
    assert TR.next_k(112, 1) == GM.K_MAX and TR.next_k(8, 2) == 10 and TR.next_k(127, 2) == 128
    monkeypatch.setattr(TR, "measure_extrap0", lambda *a, **k: dict(per_n={}, mean=0.0))
    cfg = TR.TrainCfg(phase=2, k=44, out=str(tmp_path), expand_step=1)
    latch = TR.BestLatch()
    ex = TR.do_expand(NumCodeModel(), cfg, None, latch, 100, DEV, why="timer")
    assert ex["k"] == [44, 57] and cfg.k == 57 and ex["why"] == "timer"
    assert [n for n in D.train_ns(57) if n not in D.train_ns(44)] == [57]
    # 锚: timer 不进宽限, stable 进宽限
    torch.manual_seed(0)
    model = NumCodeModel()
    ctl = TR.AnchorCtl(model, 0.005)
    ctl.tpl, ctl.score, ctl.grace_done, ctl.grace_until = {}, 0.9, True, 2000
    cfg2 = TR.TrainCfg(phase=2, k=12, out=str(tmp_path), expand_step=1)
    TR.do_expand(model, cfg2, ctl, latch, 3000, DEV, why="timer")
    assert ctl.grace_done is True and ctl.grace_until == 2000
    TR.do_expand(model, cfg2, ctl, latch, 4500, DEV, why="stable")
    assert ctl.grace_done is False and ctl.grace_until == 4500 + TR.ANCHOR_GRACE


def test_stop_ctl_rules():
    """[U] 2026-08-17 停跑: 六任务 <.85 连续三评 / count 训练带 exact <.80 连续两评 / 并轨置位;
    单评下探不停; 计数器状态往返."""
    s = TR.StopCtl()
    assert s.offer(0.80, 0.95, False) == []
    assert s.offer(0.80, 0.95, False) == []
    assert s.offer(0.90, 0.95, False) == []                    # 断开清零
    assert s.offer(0.80, 0.95, False) == []
    assert s.offer(0.80, 0.95, False) == []
    assert s.offer(0.80, 0.95, False) == ["score<0.85x3"]
    s2 = TR.StopCtl()
    assert s2.offer(0.95, 0.70, False) == []
    assert s2.offer(0.95, 0.70, False) == ["count_train<0.8x2"]
    assert TR.StopCtl().offer(0.95, 0.95, True) == ["collapse_flag"]
    s3 = TR.StopCtl()
    s3.offer(0.80, 0.70, False)
    s4 = TR.StopCtl()
    s4.load_state(s3.state())
    assert s4.offer(0.80, 0.70, False) == ["count_train<0.8x2"] and s4.s_score == 2


def test_ga1_report_shape():
    torch.manual_seed(0)
    model = NumCodeModel()
    ps = AL.cv_params(model)
    opt = _fake_adam(ps, steps=50)
    ef = [torch.rand_like(p) for p in ps]
    rep = AL.ga1_report(model, opt, ps, ef)
    assert "global" in rep and "task_heads" in rep and "read_head" in rep
    assert isinstance(rep["pass"], bool)
    # 完全同序 ⇒ ρ=1 且过门
    vh = [AL.Metric(opt, "adam").adam_vhat(p) for p in ps]
    rep2 = AL.ga1_report(model, opt, ps, vh)
    assert rep2["global"] > 0.999 and rep2["pass"]


# ---------------------------------------------------------------- [U] 2026-08-17 第四阶段
def _flat_row(ch, parts, i):
    vs = []
    for t, wi, bi, shp in ch.blocks:
        a, x = parts[t]
        vs.append(torch.outer(a[i], x[i]).reshape(-1))
        if bi is not None:
            vs.append(a[i])
    return torch.cat(vs)


def test_c_proj_geometry(net):
    """改动 2 投影率 c = ‖P_V ĝ‖/‖ĝ‖ 的几何钉子 (手造闭式部件, 单位度规): S 行 = 某 V 数量唯一
    rollout 的复制 ⇒ c=1; S 行与全部 V 行在 x 上支撑不交 (a 相同) ⇒ c=0; 一般随机行 ∈ [0,1] 且与
    显式 QR 正交基 ‖QQᵀĝ‖/‖ĝ‖ 一致; V 行恒 0; 秩 = 独立 V 数量数."""
    torch.manual_seed(1)
    ch = AL.CChannel(net, 0.9, DEV)
    den = AL.Metric(None, "identity").denom(ch.hp)
    B = 5
    parts = {}
    for t, wi, bi, shp in ch.blocks:
        C, d = shp
        a = torch.zeros(B, C)
        x = torch.zeros(B, d)
        a[0, 1 % C] = 1.0
        x[0, d // 2:] = torch.randn(d - d // 2)
        a[1] = a[0]
        x[1] = x[0]
        a[2, 0] = 0.7
        x[2, d // 2:] = torch.randn(d - d // 2)
        a[3] = a[0]
        x[3] = x[0]                                    # S 行 3 = V 数量 1 的复制 ⇒ c=1
        a[4] = a[0]
        x[4, :d // 2] = torch.randn(d // 2)            # S 行 4: x 支撑与 V 不交 ⇒ 权重块内积 0
        parts[t] = (a, x)
    ns = torch.tensor([1, 1, 2, 3, 4])
    isS = torch.tensor([False, False, False, True, True])
    c, basis, diag = ch.c_proj(parts, ns, den, isS)
    assert bool((c[:3] == 0).all())
    assert abs(float(c[3]) - 1.0) < 1e-4, float(c[3])
    assert diag["kV"] == 2 and diag["rank"] == 2 and diag["nS"] == 2
    # 显式正交基对表 (单位度规 ⇒ 欧氏): V 数量均值向量拉平 → QR → ‖QQᵀĝ‖/‖ĝ‖ (行 4 含偏置块, 非零但同法)
    Vm = torch.stack([0.5 * (_flat_row(ch, parts, 0) + _flat_row(ch, parts, 1)), _flat_row(ch, parts, 2)], 1)
    Q, _ = torch.linalg.qr(Vm)
    for i in (3, 4):
        g_ = _flat_row(ch, parts, i)
        ref = float((Q @ (Q.T @ g_)).norm() / g_.norm())
        assert abs(float(c[i]) - ref) < 1e-4, (i, float(c[i]), ref)
    # 一般随机部件: ∈ [0,1], V 行 0, 与显式 QR 一致
    torch.manual_seed(2)
    parts2 = {t: (torch.randn(B, shp[0]), torch.randn(B, shp[1])) for t, wi, bi, shp in ch.blocks}
    c2, basis2, d2 = ch.c_proj(parts2, ns, den, isS)
    assert bool((c2[isS] >= 0).all()) and bool((c2[isS] <= 1).all()) and bool((c2[~isS] == 0).all())
    Vm2 = torch.stack([0.5 * (_flat_row(ch, parts2, 0) + _flat_row(ch, parts2, 1)), _flat_row(ch, parts2, 2)], 1)
    Q2, _ = torch.linalg.qr(Vm2)
    for i in (3, 4):
        g_ = _flat_row(ch, parts2, i)
        ref = float((Q2 @ (Q2.T @ g_)).norm() / g_.norm())
        assert abs(float(c2[i]) - ref) < 1e-3, (i, float(c2[i]), ref)
    # 秩不足: V 数量 2 = 数量 1 的倍数 ⇒ rank 1, 投影不变
    parts3 = {t: (parts[t][0].clone(), parts[t][1].clone()) for t in parts}
    for t in parts3:
        parts3[t][0][2] = 2.0 * parts3[t][0][0]
        parts3[t][1][2] = parts3[t][1][0]
    c3, _, d3 = ch.c_proj(parts3, ns, den, isS)
    assert d3["rank"] == 1 and abs(float(c3[3]) - 1.0) < 1e-4


def test_holdout_retreat_and_stop_rules_stage4():
    """改动 1 前沿退回: 留出调度只留 1..12, S 含撤下数量时立即重轮换, 全覆盖仍成立; 停跑新规则:
    count 线可调 (.90) / dmin≤1 即停 / c 窗均 ≤ 零假设即停 (None 不判)."""
    hs = AL.HoldoutSched(list(range(1, 15)), 0.15, 10, seed=3)
    hs.current(0)
    hs.retreat(list(range(1, 13)))
    assert hs.ns == list(range(1, 13)) and all(n <= 12 for n in hs.order) and hs.ptr <= len(hs.order)
    S1 = hs.current(1)
    assert S1 <= set(range(1, 13)) and len(S1) == 2
    seen = set()
    for st in range(0, 200):
        seen |= hs.current(st)
    assert seen == set(range(1, 13))
    s = TR.StopCtl(stop_count=0.90, stop_dmin=1, stop_cnull=1)
    assert s.offer(0.95, 0.89, False, dmin=5, c_mean=0.3, c_null=0.1) == []
    assert s.offer(0.95, 0.89, False, dmin=5, c_mean=0.3, c_null=0.1) == ["count_train<0.9x2"]
    assert TR.StopCtl(0.9, 1, 1).offer(0.95, 0.99, False, dmin=1, c_mean=0.3, c_null=0.1) == ["dmin<=1"]
    assert TR.StopCtl(0.9, 1, 1).offer(0.95, 0.99, False, dmin=5, c_mean=0.1, c_null=0.1) == ["c<=c_null"]
    assert TR.StopCtl(0.9, 0, 0).offer(0.95, 0.99, False, dmin=1, c_mean=0.0, c_null=0.5) == []
    assert TR.StopCtl(0.9, 1, 1).offer(0.95, 0.99, False, dmin=5, c_mean=None, c_null=0.5) == []
    for f in ("c_mode", "k_force", "stop_count", "stop_dmin", "stop_cnull"):
        assert f in TR.TrainCfg.__dataclass_fields__
    assert TR.TrainCfg().c_mode == "proj" and TR.TrainCfg().stop_count == 0.80
