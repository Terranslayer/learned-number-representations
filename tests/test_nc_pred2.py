# tests/test_nc_pred2.py -- 预测子阶段的判别性测试. v2 部分 (渲染/信道/器官/池/候选包/一步装配/读数)
# 原样保留 (诊断路回归); v3 部分 (spec-v3): 序列装配兼容逐位 / K 跳链 BPTT 与段嵌入非零梯度 /
# V̂-Ŵ-闸 (形状, p_halt, ε_write 夹取, k* 规则, Poisson-binomial) / 常数预测器 / 池 depth 字段 /
# 平台期+零入流停跑 / 接线断言 A1–A18 (含 A18 K=1 逐位平价) / 链训练器 3 步 + 恢复.
import math
import os

import pytest
import torch

import symemerge.numcode.data as D
import symemerge.numcode.geometry as GM
import symemerge.numcode.pred.render as R
import symemerge.numcode.pred.step as ST
import symemerge.numcode.pred.trainer as TR
from symemerge.numcode.grpo import TaskWeights
from symemerge.numcode.pred.data import KitMaker, build_batch, make_kits
from symemerge.numcode.pred.model import PredModel
from symemerge.numcode.pred.pool import DataPool

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rng(s=0):
    return torch.Generator().manual_seed(s)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return PredModel().to(DEV)


@pytest.fixture(scope="module")
def cfg():
    return TR.TrainCfg(b_scene=4, b_pool=4, b_mask=4, workers=0, kit_workers=0)


@pytest.fixture(scope="module")
def batches(net, cfg):
    rng = _rng(3)
    items = [D.sample_group(rng, 9) for _ in range(cfg.b_scene)]
    sb = build_batch(items, DEV)
    pool = DataPool(64)
    with torch.no_grad():
        enc = net.encode(sb["x1"], is_scene=True)
        kk = net.D1(enc["tokens"]).argmax(-1).cpu()
    pool.admit(kk, sb["ns"].cpu(), sb["zseed"], 0)
    idx = pool.sample(cfg.b_pool, cfg.t_pool, cfg.eps_pool, rng)
    kits = make_kits(pool.labels(idx).tolist(), 9, 11)
    draw_p = R.draw_channel(idx.shape[0], cfg.s, rng, DEV, cfg.occ_k)   # v3.1 C7: 池链共享 draw
    pb = ST.render_pool_batch(pool, idx, kits, cfg, rng, DEV, draw=draw_p)
    pb["draw"], pb["tpl"] = draw_p, R.templates(draw_p.u, draw_p.s)
    pb["pgen"] = pool.gens(idx)
    pb["pzseed"] = pool.zseed[idx.cpu()].clone()
    mb = ST.make_mask_batch(cfg.b_mask, cfg, rng, DEV)
    return sb, pb, mb, pool


# ================================================================ 渲染 / 信道
def test_templates_match_geometry_alpha():
    """逐格模板 = geometry.stamp_alpha (同抖动/尺寸/角度), 三章型."""
    rng = _rng(1)
    u = torch.rand(2, GM.T, 4, generator=rng) * 2 - 1
    s = 0.7
    tpl = R.templates(u, s)                                     # (2,T,4,P,P) CPU
    n = 2 * GM.T
    types = torch.arange(n) % 3
    jy = (u[..., 0] * GM.JITTER_PX * s).reshape(-1)
    jx = (u[..., 1] * GM.JITTER_PX * s).reshape(-1)
    sc = (1.0 + u[..., 2] * GM.SIZE_JIT * s).reshape(-1)
    an = (u[..., 3] * math.radians(GM.ROT_DEG) * s).reshape(-1)
    ref = GM.stamp_alpha(types, jy, jx, sc, an)                 # (n,P,P)
    got = tpl.reshape(n, 4, GM.P, GM.P)[torch.arange(n), types + 1]
    assert torch.allclose(got, ref, atol=1e-6)
    assert bool((tpl[:, :, 0] == 0).all())                       # 空类全零


def test_render_hard_matches_raster_and_classes_roundtrip():
    rng = _rng(2)
    progs = [GM.sample_uniform_prog(rng, 40) for _ in range(3)]
    k = R.classes_from_progs(progs)
    tpl0 = R.templates(torch.zeros(3, GM.T, 4), 0.0)
    x = R.render_hard(k, tpl0)
    assert torch.equal(x, GM.raster(progs))
    assert torch.equal(R.stamp_count(k), torch.tensor([len(p) for p in progs]))
    assert torch.equal(GM.cell_truth(progs).view(3, GM.T), k)


def test_ste_forward_identity_and_soft_gradient():
    """§4.4: 前向逐位 = 硬渲染; 反向 = 软渲染的梯度. 信道后仍逐位同 (同输入)."""
    rng = _rng(4)
    B = 3
    draw = R.draw_channel(B, 0.25, rng, DEV, occ_k=8)
    tpl = R.templates(draw.u, draw.s)
    logits = torch.randn(B, GM.T, 4, device=DEV, requires_grad=True)
    r = R.render_ste(logits, tpl)
    hard = R.render_hard(logits.argmax(-1), tpl)
    assert torch.equal(r["x2"], hard)
    y = R.channel(r["x2"], draw)
    assert torch.equal(y, R.channel(hard, draw))
    wgt = torch.arange(GM.SIDE, device=DEV).float()
    g_ch = torch.autograd.grad((y * wgt).sum(), logits, retain_graph=True)[0]
    assert float(g_ch.abs().sum()) > 0                            # 信道后仍有梯度回到 p
    # 信道前: 直通的反向 = 软渲染的反向 (∂x2/∂p = ∂x_soft/∂p, §4.4)
    g_ste = torch.autograd.grad((r["x2"] * wgt).sum(), logits)[0]
    l2 = torch.randn(B, GM.T, 4, device=DEV, requires_grad=True)
    l2.data.copy_(logits.data)
    soft = R.render_soft(torch.softmax(l2, -1), tpl)
    g_soft = torch.autograd.grad((soft * wgt).sum(), l2)[0]
    assert float(g_ste.abs().sum()) > 0
    assert torch.allclose(g_ste, g_soft, atol=1e-5, rtol=1e-4)
    # 硬类别冻结: 换 logits 但传 k_hard ⇒ hard 不变, p 变
    r2 = R.render_ste(logits + torch.randn_like(logits), tpl, k_hard=r["k"])
    assert torch.equal(r2["hard"], r["hard"])
    assert not torch.equal(r2["p"], r["p"])


def test_channel_pieces():
    rng = _rng(5)
    B = 4
    d0 = R.draw_channel(B, 0.0, rng, DEV)
    x = torch.rand(B, GM.SIDE, GM.SIDE, device=DEV)
    assert torch.equal(R.channel(x, d0), x)                       # s=0 恒等
    d = R.draw_channel(B, 0.25, rng, DEV, occ_k=48)
    assert int((d.keep == 0).sum(1)[0]) == 48                    # 恒开遮 48 格
    y = R.channel(torch.ones(B, GM.SIDE, GM.SIDE, device=DEV), d)
    kp = R.keep_pixels(d.keep)
    assert float(y[kp == 0].max()) <= 0.02 * 0.25 * 5 + 1e-6      # 被遮块只剩噪声
    assert d.noise.shape == (B, GM.SIDE, GM.SIDE) and 0 <= d.sigma <= GM.BLUR_SIG * 0.25
    d2 = R.draw_channel(B, 0.25, rng, DEV, occ_k=0)
    assert bool(((d2.keep == 0).sum(1) <= GM.OCCLUDE_MAX).all())  # 概率遮挡 1..2 格


def test_random_canvas_stratified_and_like():
    rng = _rng(6)
    k = R.random_canvas_classes(2000, rng)
    occ = (k != 0).float().mean(1)
    assert 0.45 < float(occ.mean()) < 0.55                       # ρ~U[0,1]
    assert float(occ.min()) < 0.02 and float(occ.max()) > 0.98    # 覆盖空到满
    assert set(k.unique().tolist()) == {0, 1, 2, 3}
    k2 = R.random_canvas_like(k[:50], rng)
    assert torch.equal(R.stamp_count(k2), R.stamp_count(k[:50]))
    assert not torch.equal(k2, k[:50])


# ================================================================ 器官
def test_model_flag_and_frozen_d1(net):
    x = torch.rand(2, GM.SIDE, GM.SIDE, device=DEV)
    a = net.encode(x, is_scene=True)
    b = net.encode(x, is_scene=False)
    assert a["tokens"].shape == (2, GM.T, 256) and b["tokens"].shape == (2, GM.T, 256)
    assert not torch.allclose(a["cls"], b["cls"])                # flag 有影响
    assert all(not p.requires_grad for p in net.D1.parameters())
    ids = {id(p) for p in net.trainable_params()}
    assert not any(id(p) in ids for p in net.D1.parameters())
    assert net.heads.read[-1].out_features == GM.K_MAX            # 无 ⊥
    assert net.heads.cell.out_features == 4
    lg = net.D1(a["tokens"])
    assert lg.shape == (2, GM.T, 4)
    assert not hasattr(net.E, "modal")


# ================================================================ 数据池
def test_pool_fifo_median_sampling_ema_state():
    p = DataPool(5)
    ks = torch.randint(0, 4, (3, GM.T))
    p.admit(ks, torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30]), step=0)
    assert p.size == 3 and float(p.sigma[:3].unique()) == 0.5    # 空池初值
    p.update_scores(torch.tensor([0, 1]), torch.tensor([1.0, 0.0]), eta=0.5)
    assert torch.allclose(p.sigma[:3], torch.tensor([0.75, 0.25, 0.5]))
    p.admit(ks[:1], torch.tensor([4]), torch.tensor([40]), step=1)
    assert float(p.sigma[3]) == 0.5                              # 中位数 = median(.75,.25,.5)
    p.admit(ks[:2], torch.tensor([5, 6]), torch.tensor([50, 60]), step=2)   # 满 5, 逐出最老 (n=1)
    assert p.size == 5 and p.n_evict == 1
    assert 1 not in p.n[:5].tolist() and 6 in p.n[:5].tolist()
    pr = p.probs(0.3, 0.1)
    s = p.sigma[:5]
    ref = 0.9 * torch.softmax(s / 0.3, 0) + 0.1 / 5
    assert torch.allclose(pr, ref)
    idx = p.sample(3, 0.3, 0.1, _rng(0))
    assert idx.shape == (3,) and len(set(idx.tolist())) == 3
    q = DataPool(5)
    q.load_state(p.state())
    assert torch.equal(q.k[:5], p.k[:5]) and torch.equal(q.sigma[:5], p.sigma[:5]) and q.head == p.head
    assert 1.0 < p.prob_ratio(0.3, 0.1) < 10
    st = p.stats(0.3, 0.1)
    assert st["size"] == 5 and "sig_med" in st


# ================================================================ 数据
def test_kits_and_batch():
    rng = _rng(7)
    g = D.sample_group(rng, 9, n=4, with_scene=False)
    assert g["scene"] is None and g["n"] == 4 and "zseed" in g and g["cands"].shape[0] == D.N_CAND
    kits = make_kits([3, 5, 5], 9, 123)
    assert [k["n"] for k in kits] == [3, 5, 5]
    pb = build_batch(kits, DEV, with_scene=False)
    assert "x1" not in pb and pb["cands"].shape == (3, D.N_CAND, GM.SIDE, GM.SIDE)
    km = KitMaker(0)
    km.submit([2, 2], 9, 5)
    assert km.has_pending()
    out = km.collect()
    assert [o["n"] for o in out] == [2, 2] and not km.has_pending()
    a = make_kits([2], 9, 5)[0]
    assert torch.equal(a["cands"], out[0]["cands"])               # 同种子同包
    g2 = D.sample_group(_rng(8), 9)
    assert g2["scene"] is not None and g2["zseed"] > 0


# ================================================================ 一步装配
def test_lds_rows_and_scene_side_shapes(net, cfg, batches):
    sb, pb, mb, pool = batches
    w = TaskWeights().weights(DEV)
    rng = _rng(9)
    draw = R.draw_channel(sb["B"], cfg.s, rng, DEV, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    cfg_l = TR.TrainCfg(**{**cfg.__dict__, "lam": 1e-3})       # λ>0 才有 L_λ 梯度可查
    out = ST.scene_side(net, sb, w, cfg.mu, cfg_l, draw, tpl)
    assert out["S"]["rows"].shape == (sb["B"],) and out["S"]["accs"].shape == (sb["B"], 6)
    assert out["p"].shape == (sb["B"], GM.T, 4) and out["x2"].shape == (sb["B"], GM.SIDE, GM.SIDE)
    # 写路径梯度非零: L_λ 只经 p → E; L_S 对 D1 输出 logits 的梯度非零 (直通)
    g_e = torch.autograd.grad(out["l_lam"], list(net.E.cnn.parameters()), retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in g_e if g is not None) > 0
    g_l = torch.autograd.grad(out["S"]["rows"].mean(), out["logits"], retain_graph=True)[0]
    assert float(g_l.abs().sum()) > 0
    po = ST.pool_side(net, pb, w, cfg.mu)
    assert po["rows"].shape == (pb["B"],)
    loss, ms = ST.mask_side(net, mb)
    assert loss.dim() == 0 and 0 <= ms["mask_acc"] <= 1
    ell = ST.excess_surprise(1.0, po["rows"], pb["ns"])
    assert isinstance(ell, float)


def test_train_step_split_equals_joint():
    """β=0: 分路取梯度 (g_S+g_V+g_aux) 与合并反传给出同一更新 (CPU, 容差内)."""
    torch.manual_seed(1)
    cfg = TR.TrainCfg(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, s=0.25, occ_k=48)
    dev = torch.device("cpu")
    outs = []
    for split in (False, True):
        torch.manual_seed(1)
        m = PredModel().to(dev)
        params = m.trainable_params()
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
        rng = _rng(3)
        items = [D.sample_group(rng, 9) for _ in range(2)]
        sb = build_batch(items, dev)
        pool = DataPool(8)
        with torch.no_grad():
            kk = m.D1(m.encode(sb["x1"], True)["tokens"]).argmax(-1)
        pool.admit(kk, sb["ns"], sb["zseed"], 0)
        idx = pool.sample(2, cfg.t_pool, cfg.eps_pool, rng)
        kits = make_kits(pool.labels(idx).tolist(), 9, 1)
        pb = ST.render_pool_batch(pool, idx, kits, cfg, rng, dev)
        mb = ST.make_mask_batch(2, cfg, rng, dev)
        w = TaskWeights().weights(dev)
        mm = ST.train_step(m, opt, params, sb, pb, mb, cfg, w, dev, rng, beta=0.0, split=split, want_S=split)
        # v3 新参数 (seg/pred/gate) 不在 v2 路径图内: 合并反传留 None, 分路补零 — 同义, 统一按零比较
        outs.append((torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                for p in params]).clone(), mm))
    # 比梯度而非比 Adam 首步后的参数 (首步 ≈ lr·sign(g), 近零梯度的舍入差会翻号)
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6, rtol=1e-4), \
        float((outs[0][0] - outs[1][0]).abs().max())
    al = outs[1][1]["al"]
    assert "S_hat" in al and "tau" in al and "cos" in al and al["gS"] > 0 and al["gV"] > 0


def test_train_step_beta_and_hard_freeze(net, cfg, batches):
    sb, pb, mb, pool = batches
    params = net.trainable_params()
    opt = torch.optim.AdamW(params, lr=1e-4)
    w = TaskWeights().weights(DEV)
    rng = _rng(10)
    ST.train_step(net, opt, params, sb, pb, mb, cfg, w, DEV, rng, beta=0.0, split=False)   # 造优化器状态
    m = ST.train_step(net, opt, params, sb, pb, mb, cfg, w, DEV, rng, beta=1e-3, split=True, want_S=True)
    al = m["al"]
    assert "u" in al and al["u"] >= 0 and "beta_eff" in al and al["u_mode"] == "exact" and "S_hat" in al
    if not al.get("degenerate"):
        assert al["beta_eff"] * al["u"] <= 0.1 + 1e-6            # 夹取
        assert al["JTw"] > 0 and abs(al["beta_ceil"] - 0.1 / al["u"]) < 1e-9 * max(1.0, al["beta_ceil"])
    assert m["stamps"] >= 0 and m["pool_accs"].shape == (pb["B"],)
    # 固定剂量 ([C] 2026-08-18): beta_frac=f ⇒ β = f·β_ceil 每步重算, 修正项 F⁻¹ 范数 = f·0.1·‖g_V‖ (corr_frac = f·0.1)
    m2 = ST.train_step(net, opt, params, sb, pb, mb, cfg, w, DEV, rng, beta=0.0, beta_frac=0.5, split=True)
    al2 = m2["al"]
    assert al2["u_mode"] == "exact" and abs(al2["f"] - 0.5) < 1e-12 and abs(al2["corr_frac"] - 0.05) < 1e-5
    assert abs(al2["beta_eff"] * al2["u"] - 0.05) < 1e-5 and al2["capped"] == 0
    # 旧 FD 路径只为诊断脚本保留 (u_mode=fd), 仍可跑
    cfg_fd = TR.TrainCfg(**{**cfg.__dict__, "u_mode": "fd"})
    m3 = ST.train_step(net, opt, params, sb, pb, mb, cfg_fd, w, DEV, rng, beta=1e-3, split=True)
    assert m3["al"]["u_mode"] == "fd" and "delta" in m3["al"]


def test_exact_cross_term_is_true_transpose_f64():
    """精确 u_S 是显式场 g(Θ) 的真转置 (D-CHK-2 C11 的实现内自检, float64 CPU): ⟨Jᵀw, a⟩ 对 [⟨g(Θ+εa),w⟩ − ⟨g(Θ−εa),w⟩]/2ε,
    g 在扰动参数处按同一硬画布 (k_hard 冻结) 求值. 相对差 ≤ 1e-4 (float64 中心差分)."""
    import symemerge.numcode.align as AL
    torch.manual_seed(2)
    dev = torch.device("cpu")
    m = PredModel().double().to(dev)
    params = m.trainable_params()
    cfg = TR.TrainCfg(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0)
    rng = _rng(4)
    items = [D.sample_group(rng, 9) for _ in range(2)]
    sb = build_batch(items, dev)
    sb["x1"], sb["cands"] = sb["x1"].double(), sb["cands"].double()
    d0 = R.draw_channel(2, cfg.s, rng, dev, cfg.occ_k)
    draw = R.ChannelDraw(u=d0.u.double(), keep=d0.keep.double(), sigma=0.0, noise=d0.noise.double(), s=d0.s)   # 模糊核 float32, 双精度下关模糊
    tpl = R.templates(draw.u, draw.s)
    w = TaskWeights().weights(dev).double()
    den = [torch.ones_like(p) for p in params]
    with AL._math_sdpa():
        out = ST.scene_side(m, sb, w, cfg.mu, cfg, draw, tpl, field=True)
        k0 = out["k"]
        fg = ST.field_grads(out, params)
        g_V = AL.grads_of(out["sc"]["rows"].mean(), params, retain=True)
        nS = float(AL.Metric.norm(fg["g_S"], den))
        u, dg = ST.exact_cross_term(params, den, g_V, fg["g_S"], fg, out["y"])
        JTw = [x * dg["gS"] for x in u]
        nV = dg["gV"]
        wdir = [g / nV for g in g_V]                              # F⁻¹ĝ_V (den = 1)
        a = [g / nS for g in fg["g_S"]]                            # 第二方向 (与 dchk 同构), L2 长 1
        exact = float(sum((x * y).sum() for x, y in zip(JTw, a)))

        def field_dot(sign, eps=1e-4):
            saved = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, ai in zip(params, a):
                    p.add_(ai, alpha=sign * eps)
            try:
                o2 = ST.scene_side(m, sb, w, cfg.mu, cfg, draw, tpl, k_hard=k0, field=True)
                assert torch.equal(o2["hard"], out["hard"])
                g2 = ST.field_grads(o2, params)["g_S"]
                return float(sum((x * y).sum() for x, y in zip(g2, wdir)))
            finally:
                with torch.no_grad():
                    for p, s0 in zip(params, saved):
                        p.copy_(s0)
        fd = (field_dot(+1) - field_dot(-1)) / (2 * 1e-4)
    assert exact != 0.0 and dg["u"] > 0
    assert abs(exact - fd) <= 1e-4 * abs(fd), (exact, fd)


def test_proj_rate_and_floor(net, cfg, batches):
    sb, pb, mb, pool = batches
    params = net.trainable_params()
    opt = torch.optim.AdamW(params, lr=1e-4)
    w = TaskWeights().weights(DEV)
    c = ST.proj_rate(net, opt, params, sb, pb, cfg, w, DEV, _rng(11), 9)
    assert 0.0 <= c["c"] <= 1.0 and 0.0 <= c["c_null"] <= 1.0 and c["rank"] >= 1 and c["nV"] >= 1
    fl = ST.floor_measure(net, sb, cfg, w, DEV, _rng(12))
    assert fl["floor"] > 0 and abs(fl["l_star"] - 0.5 * fl["floor"]) < 1e-9
    # G1 归一化口径 ([U] 2026-08-18): ≥3 抽中位, r = median(L_S)/median(地板), 闸线 r ≤ .5 开
    assert fl["n_draws"] == 3 and len(fl["floors"]) == 3 and len(fl["L_S_draws"]) == 3
    assert fl["floor"] == sorted(fl["floors"])[1] and fl["L_S"] == sorted(fl["L_S_draws"])[1]
    assert abs(fl["r"] - fl["L_S"] / fl["floor"]) < 1e-12 and ST.gate_open(fl) == (fl["r"] <= 0.5)
    assert ST.gate_open(dict(r=0.5)) and not ST.gate_open(dict(r=0.500001))
    fl5 = ST.floor_measure(net, sb, cfg, w, DEV, _rng(12), n_draws=5)
    assert fl5["n_draws"] == 5 and len(fl5["floors"]) == 5
    # 同一 rng 起点: 5 抽的前 3 抽 = 3 抽的全部 (同批同 k, 逐次换信道 + 换画布; GPU 非严格模式只到容差)
    assert torch.allclose(torch.tensor(fl5["floors"][:3]), torch.tensor(fl["floors"]), rtol=1e-4, atol=1e-5)
    assert torch.allclose(torch.tensor(fl5["L_S_draws"][:3]), torch.tensor(fl["L_S_draws"]), rtol=1e-4, atol=1e-5)


# ================================================================ 接线断言 A1–A28 (A15 = N/A)
def test_wiring_checks_all_pass():
    from symemerge.numcode.pred.checks import run_all
    res = run_all(DEV, seed=0, verbose=True)
    bad = {k: v for k, v in res.items() if v[0] not in ("PASS", "N/A")}
    assert not bad, bad
    assert res["A15"][0] == "N/A"                       # v3.1 C1: origin 拼接撤销


# ================================================================ 训练器
def test_winagg_and_latch():
    wa = TR.WinAgg()
    wa.add(dict(a=1.0, b=[1.0, 3.0], c=None, d=dict(x=2.0)))
    wa.add(dict(a=3.0, b=[3.0, 5.0], d=dict(x=4.0)))
    m = wa.mean()
    assert m["a"] == 2.0 and m["b"] == [2.0, 4.0] and m["d"]["x"] == 3.0
    l = TR.BestLatch()
    assert not l.offer(0.9, 1) and l.offer(0.8, 2) and l.best == 0.8 and l.step == 2
    assert not l.offer(0.99, 3)                                   # 单点尖峰不锁
    assert l.offer(0.95, 4) and l.best == 0.95


def test_trainer_smoke_and_resume(tmp_path):
    """v3.1 链训练器冒烟: K=2 + V̂/g + 池起链 3 步 (闸关 C3) → 评测行含 逐深度/三分解/前向模型/池代际
    块; 恢复续 2 步 (含 s[t] 尺度状态)."""
    cfg = TR.TrainCfg(steps=3, b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                      eval_every=3, log_every=1, viz_every=3, eval_per_n=1, table_per_n=1,
                      out=str(tmp_path / "run"), chain_k=2, pred_on=1, gate_on=0,
                      chain_from_pool=1, rho_admit=1.0, rho_admit_gen=1.0)
    fin = TR.run(cfg, device=str(DEV))
    assert fin["step"] == 3 and os.path.exists(tmp_path / "run" / "ckpt_last.pt")
    import json as _json
    rows = [l for l in open(tmp_path / "run" / "log.jsonl")]
    evs = [_json.loads(r) for r in rows if '"eval": true' in r]
    ev = evs[-1]
    assert len(ev["depths"]) == 2 and ev["depths"][0]["depth"] == 1 and "stamps_mean" in ev["depths"][1]
    assert "gate" in ev and ev["gate"]["E_kstar"] == 2.0 and ev["gate"]["inflow"] == 1.0   # 闸关恒跑满 (C3)
    pr = ev["pred"]
    assert "brier_V" in pr and "brier_V_hard" in pr and "d_const" in pr and "d_const_hard" in pr  # C10.3 软硬双口径
    assert "corr3" in pr and set(pr["corr3"]) == {"dt", "ot", "do", "per_k"}               # C10.1 三分解
    assert "r" in pr["corr3"]["dt"] and "rho" in pr["corr3"]["dt"] and "r_thresh" in pr["corr3"]["dt"]
    assert "cos_fwd" in pr and "cos_base" in pr and len(pr["var_h"]) == 3                   # C10.2/C10.4
    assert "ham12" in pr and pr["ham12"] is not None and "med" in pr["ham12"]               # C10.6
    assert "s_scale" in pr and len(pr["s_scale"]) == 6                                      # C5.2
    assert "readings" in ev and len(ev["readings"]["depths"]) == 2 and "c" in ev["readings"]["depths"][0]
    assert "plateau" in ev and "zero_inflow" in ev["plateau"]
    assert "span_meas" in ev and "adm_window" in ev                                         # C10.8 池跨度实测
    ps = ev["pool_stats"]
    assert "gen_hist" in ps and "gen_max" in ps and "n_hist" in ps and "pair_ham" in ps     # C10.7/C10.8
    assert set(ps.get("depth_hist", {})) <= {"1", "2"} and ps["n_admit"] >= 4
    assert ev["pool"] is not None and "gen_mean" in ev["pool"]
    ck = torch.load(tmp_path / "run" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert ck.get("const") is not None and "t1" in ck["const"]
    assert ck.get("sscale") is not None and int(ck["sscale"]["n"]) == 3                     # s[t] 状态入检查点
    cfg2 = TR.TrainCfg(**{**cfg.__dict__, "steps": 5})
    fin2 = TR.run(cfg2, device=str(DEV), resume=str(tmp_path / "run" / "ckpt_last.pt"))
    assert fin2["step"] == 5
    # β 挂起 (spec-v3 §8): beta_frac>0 被强制 0, 照常跑
    cfg3 = TR.TrainCfg(**{**cfg.__dict__, "steps": 6, "beta_frac": 0.5})
    fin3 = TR.run(cfg3, device=str(DEV), resume=str(tmp_path / "run" / "ckpt_last.pt"))
    assert fin3["step"] == 6


def test_stop_rules_grace_and_slope():
    """[U] 2026-08-18 切换后豁免 ≥ 池跨度 + 斜率条件; v3.1 C2.2/A21: 跌落判据重基到 note_1 侧
    (场景侧退出判定), 状态键 s_d1/hist_d1."""
    cfg = TR.TrainCfg(n_warm=100, switch_grace=0, pool_cap=200, rho_admit=0.5, b_scene=4)
    span = cfg.pool_cap / (cfg.rho_admit * cfg.b_scene)                # 老口径跨度 = 100 (本测只用此值)
    st = TR.new_state(cfg)
    st.update(stage=2, k=128, switch_step=1000, stage1_level=0.96)
    assert TR.is_exempt(st, cfg, 1000, span) and TR.is_exempt(st, cfg, 1099, span)
    assert not TR.is_exempt(st, cfg, 1100, span)                       # 切换 + 一个跨度后不再豁免
    cfg2 = TR.TrainCfg(**{**cfg.__dict__, "switch_grace": 250})
    assert TR.is_exempt(st, cfg2, 1249, span) and not TR.is_exempt(st, cfg2, 1250, span)   # 取 max(值, 跨度)
    st1 = dict(st, stage=1)
    assert TR.is_exempt(st1, cfg, 5000, span)                          # 阶段 1 恒豁免
    # 豁免期内: 计数清零、不触发
    assert TR.update_stops(st, 0.40, 0.40, exempt=True) == [] and st["s_d1"] == 0 and st["s_notes"] == 0
    # 非豁免: note_1 侧连续两评低于 收敛水平−.1 且非上升 -> 触发
    st = TR.new_state(cfg); st.update(stage=2, stage1_level=0.96)
    assert TR.update_stops(st, 0.40, 0.50, False) == []
    assert TR.update_stops(st, 0.40, 0.45, False) == ["d1_score<level-0.1x2"]
    # 斜率条件: 连续两评上升 (h[-3] < h[-2] < h[-1]) 时不计数也不触发
    st = TR.new_state(cfg); st.update(stage=2, stage1_level=0.96)
    TR.update_stops(st, 0.40, 0.45, False)
    st["s_d1"], st["s_notes"] = 0, 0             # 造「上升中」的历史: .45 -> .50 -> .55
    TR.update_stops(st, 0.42, 0.50, False)
    hits = TR.update_stops(st, 0.44, 0.55, False)
    assert hits == [] and st["s_d1"] == 0 and st["s_notes"] == 0      # 两侧都在连续两评上升 -> 清零
    hits = TR.update_stops(st, 0.44, 0.55, False)                       # 平台一评: 非上升 -> 计 1
    assert hits == [] and st["s_d1"] == 1 and st["s_notes"] == 1
    hits = TR.update_stops(st, 0.44, 0.54, False)
    assert hits == ["d1_score<level-0.1x2"] and st["s_notes"] == 2
    assert TR.rising2([0.1, 0.2, 0.3]) and not TR.rising2([0.1, 0.3, 0.3]) and not TR.rising2([0.2, 0.3])


def test_eval_decisions_a21_and_curriculum():
    """v3.1 C2.4/A21: 判定函数只吃 k*/note_1 两侧; 课程切换 = 双侧连续两评 ≥ θ_boot; 切换改写状态
    (stage1_level 记 note_1 侧) 并重置闩."""
    import inspect
    sig = list(inspect.signature(TR.eval_decisions).parameters)
    assert not any("scene" in s or s in ("sc", "score_sc") for s in sig)       # A21: 无场景侧参数
    cfg = TR.TrainCfg(theta_boot=0.9, t_stage1=99999, n_warm=0)
    st = TR.new_state(cfg)
    latch = TR.BestLatch()
    d = TR.eval_decisions(st, cfg, latch, 0.95, 0.85, 500, 10.0, 1.0)          # note_1 未到线
    assert not d["want_switch"] and st["boot_streak"] == 0
    d = TR.eval_decisions(st, cfg, latch, 0.95, 0.92, 1000, 10.0, 1.0)
    assert not d["want_switch"] and st["boot_streak"] == 1
    d = TR.eval_decisions(st, cfg, latch, 0.96, 0.93, 1500, 10.0, 1.0)         # 双侧连续两评 -> 切换
    assert d["want_switch"] and d["why"] == "boot" and st["stage"] == 2
    assert abs(st["stage1_level"] - (0.92 + 0.93) / 2) < 1e-9                  # 收敛水平 = note_1 侧
    assert latch.best == -1.0 and st["hist_kside"] == [0.96]                   # 闩重置; 平台窗史重记后含当评


def test_lam_cost_masked_by_hard_ink():
    # [U] 2026-08-20 §5.2': 只对实际落章的格计费; 全空零成本零梯度, 章格梯度非零, 空格梯度恒零
    torch.manual_seed(0)
    B = 3
    tpl = R.templates(torch.zeros(B, GM.T, 4), 0.0)
    lg = torch.randn(B, GM.T, R.NCLS) * 0.1
    lg[..., R.EMPTY] += 5.0
    lg.requires_grad_(True)
    r = R.render_ste(lg, tpl)
    l0 = ST.lam_cost(0.02, r["p"], r["k"])
    g0 = torch.autograd.grad(l0, lg)[0]
    assert float(l0) == 0.0 and float(g0.abs().sum()) == 0.0
    lg2 = torch.randn(B, GM.T, R.NCLS) * 0.1
    lg2[:, : GM.T // 2, R.EMPTY] += 5.0
    lg2.requires_grad_(True)
    r2 = R.render_ste(lg2, tpl)
    l2 = ST.lam_cost(0.02, r2["p"], r2["k"])
    ink = r2["k"] != R.EMPTY
    assert int(ink.sum()) > 0
    g2 = torch.autograd.grad(l2, lg2)[0]
    assert float(l2) > 0.0
    assert float(g2[ink].abs().sum()) > 0.0
    assert float(g2[~ink].abs().sum()) == 0.0
    ref = 0.02 * ((1.0 - r2["p"][..., R.EMPTY].detach()) * ink.float()).sum(-1).mean()
    assert torch.allclose(l2.detach(), ref)


def test_lam_schedule_onset_and_ladder():
    # [U] 2026-08-20 步3: 无时程 = 恒 lam; 有时程 = 起征前 0, 起征后按级距升, 末级保持; 触发要连续两评
    cfg = TR.TrainCfg(lam=0.5, lam_ladder="", lam_level_steps=100)
    st = TR.new_state(cfg)
    assert TR.lam_at(cfg, st, 1) == 0.5
    cfg = TR.TrainCfg(lam=0.0, lam_ladder="5e-4,2e-3,8e-3", lam_level_steps=100, lam_onset_buffer=50,
                      lam_onset_stamps=5.0, lam_onset_evals=2)
    st = TR.new_state(cfg)
    assert TR.lam_at(cfg, st, 999) == 0.0
    assert not TR.lam_trigger(cfg, st, 500, 4.0)
    assert not TR.lam_trigger(cfg, st, 1000, 6.0)
    assert st["lam_streak"] == 1
    assert TR.lam_trigger(cfg, st, 1500, 7.0)
    assert st["lam_onset"] == 1550
    assert not TR.lam_trigger(cfg, st, 2000, 9.0)
    assert TR.lam_at(cfg, st, 1549) == 0.0
    assert TR.lam_at(cfg, st, 1550) == 5e-4
    assert TR.lam_at(cfg, st, 1649) == 5e-4
    assert TR.lam_at(cfg, st, 1650) == 2e-3
    assert TR.lam_at(cfg, st, 1750) == 8e-3
    assert TR.lam_at(cfg, st, 99999) == 8e-3
    st2 = TR.new_state(cfg)
    TR.lam_trigger(cfg, st2, 500, 6.0)
    assert not TR.lam_trigger(cfg, st2, 1000, 3.0)
    assert st2["lam_streak"] == 0


def test_iconic_rate():
    # [C] 2026-08-20 对位率: 落章格中对应场景格有墨的比例; 全空样本 NaN
    B = 3
    x1 = torch.zeros(B, GM.SIDE, GM.SIDE)
    x1[:, 0:8, 0:8] = 1.0
    k = torch.zeros(B, GM.T, dtype=torch.long)
    k[0, 0] = 1
    k[1, 5] = 1
    ic = TR.iconic_rate(k, x1)
    assert float(ic[0]) == 1.0
    assert float(ic[1]) == 0.0
    assert math.isnan(float(ic[2]))
    k[0, 5] = 2
    ic2 = TR.iconic_rate(k, x1)
    assert abs(float(ic2[0]) - 0.5) < 1e-9


# ================================================================ v3/v3.1 (spec-v3 + spec-v3.1): 装配 / 链 / 预测
@pytest.fixture(scope="module")
def cnet():
    torch.manual_seed(0)
    return PredModel().to("cpu")


def test_enc_seq_compat_bitwise_matches_forward(cnet):
    """A18 逐位基础 (CPU): tokenize+enc_seq 兼容装配与 v2 forward 同输入逐位同输出 (场景/笔记/掩码三式)."""
    x = torch.rand(3, GM.SIDE, GM.SIDE)
    a = cnet.E.enc_seq(cnet.E.tokenize(x), None, seg=False, use_flag=True)
    b = cnet.E(x, is_scene=True)
    assert torch.equal(a["cls"], b["cls"]) and torch.equal(a["tokens"], b["tokens"])
    a2 = cnet.E.enc_seq(cnet.E.tokenize(x), None, seg=False, use_flag=False)
    b2 = cnet.E(x, is_scene=False)
    assert torch.equal(a2["cls"], b2["cls"])
    rep = torch.zeros(3, GM.T, dtype=torch.bool)
    rep[:, :5] = True
    a3 = cnet.E.enc_seq(cnet.E.tokenize(x, rep), None, seg=False)
    b3 = cnet.E(x, is_scene=False, mask_cells=rep)
    assert torch.equal(a3["cls"], b3["cls"]) and torch.equal(a3["tokens"], b3["tokens"])


def test_v31_organ_registry():
    """v3.1 器官注册 (C6.1/C6.2): 无独立 Ŵ 头 (A25); 前向模型 g = MLP(d→64→d) 末层零初始化;
    共享参数相对序 = v2 (新参数只插入不重排 — A18 的 clip 求和序前提); 段嵌入保留且零初始化 (C1 不加载)."""
    m2 = PredModel()
    names = [n for n, p in m2.named_parameters() if p.requires_grad]
    assert not any(n.startswith("pred.W") for n in names)                    # C6.1 删 Ŵ 头
    shared = [n for n in names if not n.startswith(("pred.", "fwd.", "gate."))
              and n not in ("E.seg_cur", "E.seg_org")]
    v2_order = (["E.pos", "E.cls", "E.flag", "E.mask_tok"]
                + [n for n in names if n.startswith("E.cnn.")] + [n for n in names if n.startswith("E.enc.")]
                + [n for n in names if n.startswith("heads.")])
    assert shared == v2_order
    i_heads = max(i for i, n in enumerate(names) if n.startswith("heads."))
    assert all(names.index(n) > i_heads for n in names if n.startswith(("pred.", "fwd.", "gate.")))
    assert float(m2.E.seg_cur.abs().sum()) == 0.0 and float(m2.E.seg_org.abs().sum()) == 0.0
    assert abs(m2.gate.slope() - 1.0) < 1e-6 and float(m2.gate.b) == 0.0
    assert set(m2.pred.V.keys()) == {"t1", "t2", "t3", "t4", "t5", "t6"}
    assert m2.pred.V["t1"].in_features == 256 + 64 and m2.pred.V["t2"].in_features == 256
    assert m2.fwd.l1.in_features == 256 and m2.fwd.l1.out_features == 64      # C6.2 瓶颈
    assert m2.fwd.l2.out_features == 256
    assert float(m2.fwd.l2.weight.abs().sum()) == 0.0 and float(m2.fwd.l2.bias.abs().sum()) == 0.0
    m3 = TR.build_model(TR.TrainCfg(d_bottleneck=32))
    assert m3.fwd.l1.out_features == 32                                       # 旋钮穿线


def test_chain_side_single_slot_bptt_crn(net, cfg, batches):
    """v3.1 C1/C7: 单槽序列 (origin_cat=1 即断言报错); 单份 draw 整链重放; BPTT 贯通 (末状态损失对
    第 1 次写 logits 非零梯度); 状态含边距 marg; L_λ 逐写步累计非负."""
    sb, pb, mb, pool = batches
    w = TaskWeights().weights(DEV)
    rng = _rng(21)
    cfg3 = TR.TrainCfg(**{**cfg.__dict__, "chain_k": 2, "lam": 1e-3})
    draw = R.draw_channel(sb["B"], cfg3.s, rng, DEV, cfg3.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = ST.chain_side(net, sb, w, cfg3.mu, cfg3, draw, tpl)
    assert len(out["states"]) == 3 and len(out["writes"]) == 2
    assert all("marg" in s and s["marg"].shape == (sb["B"], 6) for s in out["states"])
    g = torch.autograd.grad(out["states"][2]["rows"].mean(), out["writes"][0]["logits"], retain_graph=True)[0]
    assert float(g.abs().sum()) > 0
    assert float(out["l_lam"]) >= 0.0
    # C7/A24: 两写步的画布均可由同一 draw 重放复原 (逐位)
    for k12 in range(2):
        assert torch.equal(out["writes"][k12]["x"],
                           R.channel(R.render_hard(out["writes"][k12]["k"], tpl), draw))
    cfg_bad = TR.TrainCfg(**{**cfg.__dict__, "origin_cat": 1})
    with pytest.raises(AssertionError):
        ST.chain_side(net, sb, w, cfg3.mu, cfg_bad, draw, tpl)


def test_margins_softscale_ysoft():
    """C5.1 边距 = logit[真值] − 次强; C5.2 s[t] 自举均值→EMA→clamp; C5.3 y_soft = σ(m/s)."""
    tl = dict(t1=torch.tensor([[2.0, 0.5]]), t2=torch.tensor([[0.0, 1.0]]),
              t3=torch.tensor([[0.1, 0.9, 3.0]]), t4=torch.tensor([[1.0, -1.0, 0.0]]),
              t6=torch.tensor([[0.0, 0.0, 5.0]]))
    t5 = torch.tensor([[0.2, 0.7, 0.1]])
    tg = dict(t1=torch.tensor([0]), t2=torch.tensor([0]), t3=torch.tensor([2]),
              t4=torch.tensor([1]), t6=torch.tensor([2]))
    truth5 = torch.tensor([1])
    m = ST.margins(tl, t5, tg, truth5)
    exp = torch.tensor([[1.5, -1.0, 2.1, -2.0, 0.5, 5.0]])
    assert torch.allclose(m, exp, atol=1e-6) and not m.requires_grad
    ss = ST.SoftScale(m=0.5, clamp=0.2, init_steps=2)
    ss.update(torch.tensor([[1.0] * 6]))
    assert torch.allclose(ss.s, torch.ones(6))                     # 自举第 1 步 = 本批均值
    ss.update(torch.tensor([[3.0] * 6]))
    assert torch.allclose(ss.s, torch.full((6,), 2.0))             # 自举第 2 步 = 累计均值 (1+3)/2
    ss.update(torch.tensor([[4.0] * 6]))
    assert torch.allclose(ss.s, torch.full((6,), 3.0))             # EMA: .5*2 + .5*4
    ss2 = ST.SoftScale(clamp=0.5)
    ss2.update(torch.tensor([[0.01] * 6]))
    assert torch.allclose(ss2.scale(), torch.full((6,), 0.5))      # clamp 下界
    ys = ST.y_soft_of(torch.tensor([[0.0] * 6]), ss2)
    assert torch.allclose(ys, torch.full((1, 6), 0.5))             # m=0 ⇒ y_soft=.5
    st = ss.state()
    ss3 = ST.SoftScale()
    ss3.load_state(st)
    assert ss3.n == ss.n and torch.equal(ss3.s, ss.s)


def test_fwd_model_zero_init_and_pred_gate_shapes(batches, cfg):
    """C6.2 末层零初始化 ⇒ ĥ ≡ h 逐位、Ŵ ≡ V̂[:, :K]、Δ̂ ≡ 0 (初始闸无信号的结构事实);
    C3 闸位: 场景链 p_halt 停位 {1..K} (B,K), 池链 {0..K} (B,K+1); 行和恒 1."""
    sb, pb, mb, pool = batches
    torch.manual_seed(1)
    m = PredModel().to(DEV)
    w = TaskWeights().weights(DEV)
    cfg3 = TR.TrainCfg(**{**cfg.__dict__, "chain_k": 2, "pred_on": 1})
    rng = _rng(22)
    draw = R.draw_channel(sb["B"], cfg3.s, rng, DEV, cfg3.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = ST.chain_side(m, sb, w, cfg3.mu, cfg3, draw, tpl)
    B = sb["B"]
    pg = ST.pred_gate(m, out["states"], sb, cfg3, k0_gated=False)
    assert pg["V"].shape == (B, 3, 6) and pg["W"].shape == (B, 2, 6) and pg["dhat"].shape == (B, 2)
    for k in range(2):
        assert torch.equal(pg["hhat"][k], out["states"][k]["h"].detach())      # ĥ = h (零初始化)
    assert torch.equal(pg["W"], pg["V"][:, :2]) and float(pg["dhat"].abs().max()) == 0.0
    assert pg["g0"] == 1 and pg["p_halt"].shape == (B, 2)                       # 场景链停位 {1,2}
    s1 = pg["p_halt"].sum(1)
    assert torch.allclose(s1, torch.ones_like(s1), atol=1e-6)
    pgp = ST.pred_gate(m, out["states"], sb, cfg3, k0_gated=True)
    assert pgp["g0"] == 0 and pgp["p_halt"].shape == (B, 3)                     # 池链停位 {0,1,2}
    s2 = pgp["p_halt"].sum(1)
    assert torch.allclose(s2, torch.ones_like(s2), atol=1e-6)
    # K=1 场景链边界 (审查 Minor 后补钉): 受闸位为空 ⇒ p_halt 单列全 1, k* 恒 1; 池链照常从 0 判
    pg1 = ST.pred_gate(m, out["states"][:2], sb, cfg3, k0_gated=False)
    assert pg1["dhat"].shape == (B, 1) and pg1["p_halt"].shape == (B, 1)
    assert torch.allclose(pg1["p_halt"], torch.ones_like(pg1["p_halt"]))
    assert ST.kstar_det(pg1["dhat"], 0.0, k0_gated=False).tolist() == [1] * B
    assert ST.kstar_sample(torch.rand(B, 1, generator=_rng(9)), _rng(9), k0_gated=False).tolist() == [1] * B
    assert set(ST.kstar_det(torch.full((B, 1), 9.0), 0.0, k0_gated=True).tolist()) == {1}   # 池链 K=1 可继续到 1
    with torch.no_grad():
        for p in m.fwd.l2.parameters():
            p.normal_(0.0, 0.05)
    pg2 = ST.pred_gate(m, out["states"], sb, cfg3, k0_gated=False)
    assert float(pg2["dhat"].abs().max()) > 0.0                                 # 扰动后 Δ̂ 可达非零


def test_pred_loss_routing_and_a20(batches, cfg):
    """C6.5 路由: ptb=0 ⇒ L_pred (BCE+L_fwd) 只到 V̂ 头与 g (Θ_E/任务头零) — 新可学部件非零梯度断言;
    C6.4/A20: L_fwd 目标端 sg (末态 h 零梯度), ptb=1 下经 ĥ 到 h_0 非零."""
    sb, pb, mb, pool = batches
    torch.manual_seed(2)
    m = PredModel().to(DEV)
    with torch.no_grad():
        for p in m.fwd.l2.parameters():
            p.normal_(0.0, 0.05)                                # 扰动末层: l1 的梯度可达
    w = TaskWeights().weights(DEV)
    rng = _rng(23)
    for ptb in (0, 1):
        cfg3 = TR.TrainCfg(**{**cfg.__dict__, "chain_k": 2, "pred_on": 1, "pred_to_backbone": ptb})
        draw = R.draw_channel(sb["B"], cfg3.s, rng, DEV, cfg3.occ_k)
        tpl = R.templates(draw.u, draw.s)
        out = ST.chain_side(m, sb, w, cfg3.mu, cfg3, draw, tpl)
        ss = ST.SoftScale()
        ss.update(torch.stack([s_["marg"] for s_ in out["states"]], dim=1))
        pg = ST.pred_gate(m, out["states"], sb, cfg3, k0_gated=False)
        lp, comp = ST.pred_loss(pg, out["states"], ss, cfg3.eta_fwd)
        assert torch.isfinite(lp) and comp["bce"] > 0 and comp["fwd"] >= 0
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        g = torch.autograd.grad(lp, [p for _, p in named], allow_unused=True, retain_graph=True)
        if ptb == 0:
            leaks = [n for (n, _), gg in zip(named, g)
                     if not n.startswith(("pred.", "fwd.")) and gg is not None and float(gg.abs().sum()) > 0]
            assert not leaks, leaks
            for pfx in ("fwd.l1", "fwd.l2", "pred.V"):
                got = sum(float(gg.abs().sum()) for (n, _), gg in zip(named, g)
                          if n.startswith(pfx) and gg is not None)
                assert got > 0, pfx                              # 非零梯度断言 (新可学部件)
        else:
            e_nz = sum(float(gg.abs().sum()) for (n, _), gg in zip(named, g)
                       if n.startswith("E.") and gg is not None)
            assert e_nz > 0                                      # ptb=1: L_pred 进 Θ_E (A26 方向)
            # A20: 对 pred_loss 真实返回的 L_fwd 张量取梯度 (审查 Important 后改法, 不自建公式) —
            # 末态 h 纯目标端 (sg) ⇒ 零; h_0 经 ĥ ⇒ 非零. pred_loss 内 detach 被删则此处即红.
            lf = comp["lf_t"]
            hs = [s_["h"] for s_ in out["states"]]
            gh = torch.autograd.grad(lf, hs, allow_unused=True)
            assert gh[-1] is None or float(gh[-1].abs().sum()) == 0.0
            assert gh[0] is not None and float(gh[0].abs().sum()) > 0


def test_first_true_kstar_and_pbtop1():
    m = torch.tensor([[False, True], [True, False], [False, False]])
    assert ST.first_true(m).tolist() == [1, 0, 2]
    dh = torch.tensor([[0.5, -0.1], [-0.2, 0.3], [0.1, 0.2]])
    # 池链 (k0_gated=True): 从 k=0 判 (v3 原语义)
    assert ST.kstar_det(dh, 0.0, k0_gated=True).tolist() == [1, 0, 2]
    assert ST.kstar_det(dh, 0.4, k0_gated=True).tolist() == [1, 0, 0]
    # 场景链 (默认): 从 k=1 判, k* ≥ 1 (A22)
    assert ST.kstar_det(dh, 0.0).tolist() == [1, 2, 2]
    assert ST.kstar_det(dh, 0.4).tolist() == [1, 1, 1]
    assert int(ST.kstar_det(torch.full((5, 2), -9.0), 0.0).min()) == 1
    rng = _rng(0)
    assert ST.kstar_sample(torch.ones(5, 2), rng).tolist() == [1] * 5           # 场景链逼停 → 1 不是 0
    assert ST.kstar_sample(torch.zeros(5, 2), rng).tolist() == [2] * 5
    assert ST.kstar_sample(torch.ones(5, 2), rng, k0_gated=True).tolist() == [0] * 5
    import itertools
    torch.manual_seed(3)
    q = torch.rand(4, 6)
    top = ST.pb_top1(q)
    for b in range(4):
        dist = torch.zeros(7)
        for bits in itertools.product([0, 1], repeat=6):
            p = 1.0
            for j, y in enumerate(bits):
                p *= float(q[b, j]) if y else 1.0 - float(q[b, j])
            dist[sum(bits)] += p
        assert int(top[b]) == int(dist.argmax())


def test_const_pred_ema_and_predict():
    cp = TR.ConstPred(eta=0.5)
    th = dict(tau=torch.tensor([2, 2, 7]), p=torch.tensor([0, 1, 0]), m=torch.tensor([1, 1, 2]))
    accs = torch.tensor([[1., 1, 1, 1, 1, 1], [0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]])
    cp.update(th, accs)
    assert abs(float(cp.val["t1"][2]) - 0.5) < 1e-6
    assert abs(float(cp.val["t1"][7]) - 0.75) < 1e-6
    assert abs(float(cp.val["t2"][0]) - (0.5 * 0.5 + 0.5 * (2 / 3))) < 1e-6
    p = cp.predict(th, 3)
    assert p.shape == (3, 6) and abs(float(p[0, 0]) - 0.5) < 1e-6
    cp2 = TR.ConstPred(eta=1.0)
    a3 = torch.zeros(1, 3, 6)
    a3[0, 0] = 1.0
    cp2.update(dict(tau=torch.tensor([5]), p=torch.tensor([0]), m=torch.tensor([0])), a3)
    assert abs(float(cp2.val["t2"][0]) - 1 / 3) < 1e-6              # (B,K+1,6) 状态维边缘化
    cp3 = TR.ConstPred()
    cp3.load_state(cp.state())
    assert torch.equal(cp3.val["t1"], cp.val["t1"])


def test_pool_depth_field_and_state():
    p = DataPool(4)
    ks = torch.randint(0, 4, (3, GM.T))
    p.admit(ks, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 0, depths=torch.tensor([1, 2, 1]))
    assert p.depth[:3].tolist() == [1, 2, 1]
    st = p.stats(0.5, 0.1)
    assert st["depth_hist"] == {"1": 2, "2": 1}
    q = DataPool(4)
    q.load_state(p.state())
    assert q.depth[:3].tolist() == [1, 2, 1]
    old = {k: v for k, v in p.state().items() if k not in ("depth", "gen")}   # 旧检查点无 depth/gen 字段
    r = DataPool(4)
    r.load_state(old)
    assert r.depth[:3].tolist() == [0, 0, 0] and r.gen[:3].tolist() == [0, 0, 0]
    p.admit(ks[:1], torch.tensor([4]), torch.tensor([4]), 1)        # 省略 depths/gens → 0
    assert int(p.depth[3]) == 0 and int(p.gen[3]) == 0


def test_pool_gen_stratified_and_hamming():
    """C4.2 gen 字段 / C4.4 分层抽取 (跨层均匀配额, 层内成绩加权) / C10.8 成对汉明距."""
    p = DataPool(300)
    rngk = _rng(7)
    for n in range(1, 10):                                          # 9 个 N 层, 各 30 条, σ 层内错开
        ks = torch.randint(0, 4, (30, GM.T), generator=rngk)
        p.admit(ks, torch.full((30,), n), torch.arange(30), 0, gens=torch.full((30,), n % 3))
    p.sigma[:p.size] = torch.rand(p.size, generator=rngk)
    idx = p.sample_stratified(27, 0.5, 0.1, _rng(1), band_w=1)
    ns = p.labels(idx).tolist()
    from collections import Counter
    c = Counter(ns)
    assert set(c) == set(range(1, 10)) and all(v == 3 for v in c.values())    # 27/9 = 每层恰 3
    idx2 = p.sample_stratified(8, 0.5, 0.1, _rng(2), band_w=1)                # 层数 > B: 8 层各 1
    assert len(set(p.labels(idx2).tolist())) == 8
    idx3 = p.sample_stratified(20, 0.5, 0.1, _rng(3), band_w=16)              # 带宽 16: N 1..9 同带
    assert idx3.shape == (20,)
    st = p.stats(0.5, 0.1, step=100, rng=_rng(4))
    assert st["gen_hist"] and st["gen_max"] == 2 and len(st["n_hist"]) == 9
    assert "age_q" in st and "pair_ham" in st and 0.0 <= st["pair_ham"]["overall"] <= 1.0
    q = DataPool(300)
    q.load_state(p.state())
    assert torch.equal(q.gen[:q.size], p.gen[:p.size])
    p2 = DataPool(4)
    p2.admit(torch.zeros(2, GM.T, dtype=torch.long), torch.tensor([1, 1]), torch.tensor([0, 0]), 0)
    assert p2.pair_hamming(64, _rng(5))["overall"] == 0.0                     # 同码全对 → 汉明 0


def test_plateau_zero_inflow_rule():
    """[U] 2026-08-20: 平台期 ∧ 零入流才停; 任一单独不停; 豁免与连续两评上升不触发."""
    cfg = TR.TrainCfg(w_plat=3, eps_plat=0.01)
    st = {}
    r = None
    for s, i in [(0.5, 1.0), (0.6, 1.0), (0.59, 0.0), (0.60, 0.0)]:
        r = TR.plateau_zero_inflow(st, cfg, s, i, exempt=False)
    assert not r["hit"]                                             # 窗前最优 .5, 窗内 .6 > .51 ⇒ 非平台
    r = TR.plateau_zero_inflow(st, cfg, 0.58, 0.0, exempt=False)
    assert r["plateau"] and r["zero_inflow"] and r["hit"] and abs(r["prev_best"] - 0.6) < 1e-9
    st2 = dict(hist_kside=[0.5, 0.6, 0.59, 0.60], inflow_hist=[1.0, 1.0, 0.0, 0.0])
    r2 = TR.plateau_zero_inflow(st2, cfg, 0.58, 0.5, exempt=False)
    assert r2["plateau"] and not r2["zero_inflow"] and not r2["hit"]   # 平台但有入流 ⇒ 不停
    st3 = dict(hist_kside=[0.5, 0.6, 0.59, 0.60], inflow_hist=[0.0] * 4)
    r3 = TR.plateau_zero_inflow(st3, cfg, 0.58, 0.0, exempt=True)
    assert not r3["hit"] and not r3["plateau"] and r3["zero_inflow"]   # 豁免 ⇒ 只记账
    st4 = dict(hist_kside=[0.5, 0.6, 0.55, 0.57], inflow_hist=[0.0] * 4)
    r4 = TR.plateau_zero_inflow(st4, cfg, 0.59, 0.0, exempt=False)
    assert not r4["hit"] and r4["zero_inflow"]                         # 连续两评上升 ⇒ 不触发


def test_three_decomp_metrics():
    """C10.1 度量件: 秩 (并列平均) / Pearson / pair_metrics / corr3_block."""
    r = TR._rankdata(torch.tensor([3.0, 1.0, 1.0, 2.0]))
    assert torch.allclose(r, torch.tensor([3.0, 0.5, 0.5, 2.0], dtype=torch.float64))
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(TR._pearson(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(TR._pearson(x, -x) + 1.0) < 1e-9
    assert math.isnan(TR._pearson(x, torch.ones(4)))                 # 常量列 → nan (Δ̂≡0 初始的正确行为)
    y = torch.tensor([1.0, 2.0, 3.0, 3.5])
    pm = TR.pair_metrics(x, y)
    assert pm["n"] == 4 and abs(pm["rho"] - 1.0) < 1e-6 and pm["sign"] == 1.0
    assert abs(pm["mae"] - float((x - y).abs().mean())) < 1e-6
    assert abs(pm["r_thresh"] - 1.96) < 1e-9                          # n=4 ⇒ 1.96/√1
    dh = torch.tensor([[0.1, -0.2], [0.3, 0.1], [-0.1, 0.2], [0.2, -0.3]])
    do = dh * 0.9
    rt = torch.tensor([[1 / 6, -1 / 6], [1 / 6, 0.0], [-1 / 6, 1 / 6], [1 / 6, -1 / 6]])
    c3 = TR.corr3_block(dh, do, rt)
    assert set(c3) == {"dt", "ot", "do", "per_k"} and len(c3["per_k"]) == 2
    assert c3["do"]["r"] > 0.99                                       # Δ̂ 与 Δ_oracle 构造同向
    assert c3["dt"]["n"] == 8 and c3["dt"]["r_thresh"] == round(1.96 / math.sqrt(5), 4)


def test_chain_train_step_cpu_smoke():
    """chain_train_step (v3.1: pred 开 + 池起链, 闸关, CPU): 有限损失; L_pred 两组件; 常数器与 s[t]
    同批更新; 场景/池两链的 k 图与 σ 加权更新量接线."""
    torch.manual_seed(4)
    dev = torch.device("cpu")
    m = PredModel().to(dev)
    params = m.trainable_params()
    opt = torch.optim.AdamW(params, lr=1e-4)
    cfg = TR.TrainCfg(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                      chain_k=2, pred_on=1, gate_on=0, chain_from_pool=1, lam=1e-3)
    rng = _rng(5)
    items = [D.sample_group(rng, 9) for _ in range(2)]
    sb = build_batch(items, dev)
    pool = DataPool(8)
    with torch.no_grad():
        kk = m.D1(m.encode(sb["x1"], True)["tokens"]).argmax(-1)
    pool.admit(kk, sb["ns"], sb["zseed"], 0)
    idx = pool.sample(2, cfg.t_pool, cfg.eps_pool, rng)
    kits = make_kits(pool.labels(idx).tolist(), 9, 1)
    draw_p = R.draw_channel(2, cfg.s, rng, dev, cfg.occ_k)
    pb = ST.render_pool_batch(pool, idx, kits, cfg, rng, dev, draw=draw_p)
    pb["draw"], pb["tpl"] = draw_p, R.templates(draw_p.u, draw_p.s)
    pb["pgen"] = pool.gens(idx)
    pb["pzseed"] = pool.zseed[idx].clone()
    mb = ST.make_mask_batch(2, cfg, rng, dev)
    w = TaskWeights().weights(dev)
    const = TR.ConstPred(eta=0.5)
    ss = ST.SoftScale()
    mm = ST.chain_train_step(m, opt, params, sb, pb, mb, cfg, w, dev, rng, const=const, sscale=ss)
    assert math.isfinite(mm["L"]) and mm["L_pred"] is not None and mm["L_gate"] is None
    assert mm["L_pred_bce"] > 0 and mm["L_pred_fwd"] >= 0
    assert len(mm["L_states"]) == 3 and len(mm["ks"]) == 2 and mm["ks"][0].shape == (2, GM.T)
    assert mm["ks_pool"] is not None and mm["ks_pool"][0].shape == (2, GM.T)
    assert ss.n == 1                                                  # 一步一更新 (两链合并)
    assert any(float((v - 0.5).abs().sum()) > 0 for v in const.val.values())
    assert mm["pool_accs"].shape == (2,) and bool((mm["pool_accs"] <= 1.0).all())
    assert abs(float(w.sum()) - 1.0) < 1e-6                           # w 归一 ⇒ σ 加权均值 ∈ [0,1]
    # 无池批: 照常跑
    mm2 = ST.chain_train_step(m, opt, params, sb, None, mb, cfg, w, dev, rng, const=const, sscale=ss)
    assert mm2["L_pool"] is None and mm2["ks_pool"] is None


def test_flow_constraint_defaults():
    """C4.3 定值判据在登记默认下成立: ρ_admit_gen·B_pool ≤ ρ_admit·B_scene (闸关 Pr[k*≥1]=1)."""
    cfg = TR.TrainCfg()
    assert cfg.rho_admit_gen * cfg.b_pool <= cfg.rho_admit * cfg.b_scene + 1e-9


# ================================================================ 闸期小件 (步 4 遗产, 器官保留): 初值旋钮 / c_step 时程 / 对照臂
def test_gate_init_knobs():
    """闸初值旋钮穿线: 必须经 build_model, 不许直接 PredModel(...) —— pin 打在可达输入上."""
    m = TR.build_model(TR.TrainCfg(gate_a0=92.0, gate_b0=0.25))
    assert abs(m.gate.slope() - 92.0) / 92.0 < 1e-6
    assert float(m.gate.b) == 0.25
    m0 = TR.build_model(TR.TrainCfg())
    assert abs(m0.gate.slope() - 1.0) < 1e-6


def test_c_ladder_schedule():
    # c_ladder 空 ⇒ c_at 恒等于 cfg.c_step (透传, 与 lam_at 同构)
    cfg = TR.TrainCfg(c_step=0.5, c_ladder="", c_level_steps=100)
    st = TR.new_state(cfg)
    assert TR.c_at(cfg, st, 1) == 0.5
    cfg = TR.TrainCfg(c_ladder="4e-3,8e-3,1.6e-2", c_level_steps=2000, c_onset_buffer=500,
                      c_onset_stamps=5, c_onset_evals=2)
    st = TR.new_state(cfg)
    assert TR.c_at(cfg, st, 999) == 0.0                       # 起征前恒 0
    assert not TR.c_trigger(cfg, st, 500, 4.0)                 # 未过阈: 不计入 streak
    assert not TR.c_trigger(cfg, st, 1000, 6.0)                # 首评过阈: streak=1, 未连续两评
    assert st["c_streak"] == 1
    assert TR.c_trigger(cfg, st, 1500, 6.0)                    # 次评过阈: streak=2 -> 触发
    assert st["c_onset"] == 2000
    assert not TR.c_trigger(cfg, st, 2000, 9.0)                # 已起征后恒 False
    assert TR.c_at(cfg, st, 1999) == 0.0
    assert TR.c_at(cfg, st, 2000) == 4e-3
    assert TR.c_at(cfg, st, 3999) == 4e-3
    assert TR.c_at(cfg, st, 4000) == 8e-3
    assert TR.c_at(cfg, st, 6000) == 1.6e-2
    assert TR.c_at(cfg, st, 99999) == 1.6e-2
    st2 = TR.new_state(cfg)
    TR.c_trigger(cfg, st2, 500, 6.0)
    assert not TR.c_trigger(cfg, st2, 1000, 4.0)                # stamps 不过阈 -> streak 清零
    assert st2["c_streak"] == 0


def test_policy_lam_modes():
    lam = torch.full((4, 2), 0.7)
    cfg_l = TR.TrainCfg(gate_mode="learned", eps_write=0.1)
    out_l = ST.policy_lam(lam, cfg_l, _rng(1))
    assert torch.equal(out_l, lam)                              # .7 未超 1-eps=.9, 透传不变
    cfg_c = TR.TrainCfg(gate_mode="const", gate_const_lam=0.95, eps_write=0.1)
    out_c = ST.policy_lam(lam, cfg_c, _rng(2))
    assert torch.allclose(out_c[:, 0], torch.full((4,), 0.9))    # 列0 夹到 1-eps=.9 (休眠臂路)
    assert torch.allclose(out_c[:, 1], torch.full((4,), 0.95))   # 列1 不夹
    cfg_r = TR.TrainCfg(gate_mode="rand", gate_rand_hi=1.0, eps_write=0.1)
    out_r1 = ST.policy_lam(lam, cfg_r, _rng(3))
    out_r2 = ST.policy_lam(lam, cfg_r, _rng(3))
    assert torch.equal(out_r1, out_r2)                          # 同种子两次逐位同 (CRN)
    assert float(out_r1.min()) >= 0.0 and float(out_r1.max()) < 1.0
    assert float(out_r1[:, 0].max()) <= 0.9 + 1e-6
    cfg_r0 = TR.TrainCfg(gate_mode="rand", gate_rand_hi=1.0, gate_rand_lo=0.0, eps_write=0.1)
    assert torch.equal(ST.policy_lam(lam, cfg_r0, _rng(3)), out_r1)   # lo=0 与旧路逐位同 (同 rng 消耗)
    cfg_rl = TR.TrainCfg(gate_mode="rand", gate_rand_lo=0.6, gate_rand_hi=1.0, eps_write=0.1)
    out_rl = ST.policy_lam(lam, cfg_rl, _rng(5))
    assert float(out_rl[:, 1].min()) >= 0.6 and float(out_rl.max()) < 1.0   # 列1 落 [lo, hi)
    assert float(out_rl[:, 0].max()) <= 0.9 + 1e-6
    cfg_e0 = TR.TrainCfg(gate_mode="const", gate_const_lam=0.95, eps_write=0.0)
    out_e0 = ST.policy_lam(lam, cfg_e0, _rng(6))
    assert torch.allclose(out_e0, torch.full((4, 2), 0.95))      # C2.5 退役: eps=0 ⇒ 不夹
    cfg_bad = TR.TrainCfg(gate_mode="bogus")
    with pytest.raises(AssertionError):
        ST.policy_lam(lam, cfg_bad, _rng(4))


def test_const_rate_match():
    K = 2
    for lbar in (0.2, 0.5, 0.8):
        ek = sum((1 - lbar) ** k for k in range(1, K + 1))
        lb, P = ST.const_rate_match(ek, K)
        assert abs(lb - lbar) < 1e-6
        assert abs(sum(P) - 1.0) < 1e-6
    lb0, _ = ST.const_rate_match(0.0, K)
    assert abs(lb0 - 1.0) < 1e-6
    lb2, _ = ST.const_rate_match(2.0, K)
    assert abs(lb2 - 0.0) < 1e-6


def test_gate_mode_no_gate_loss():
    """const 臂不产生 L_gate, (a,b) 冻结 (无梯度); learned 臂产生 L_gate 且 (a,b) 非 None 梯度."""
    def _one(gate_mode):
        torch.manual_seed(4)
        dev = torch.device("cpu")
        m = PredModel().to(dev)
        with torch.no_grad():
            for p in m.fwd.l2.parameters():
                p.normal_(0.0, 0.02)                            # Δ̂ ≠ 0 (learned 臂 a 的可达梯度)
        params = m.trainable_params()
        opt = torch.optim.AdamW(params, lr=1e-4)
        cfg = TR.TrainCfg(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                          chain_k=2, pred_on=1, gate_on=1, lam=1e-3, chain_from_pool=1,
                          gate_mode=gate_mode, gate_const_lam=0.5)
        rng = _rng(5)
        items = [D.sample_group(rng, 9) for _ in range(2)]
        sb = build_batch(items, dev)
        pool = DataPool(8)
        with torch.no_grad():
            kk = m.D1(m.encode(sb["x1"], True)["tokens"]).argmax(-1)
        pool.admit(kk, sb["ns"], sb["zseed"], 0)
        idx = pool.sample(2, cfg.t_pool, cfg.eps_pool, rng)
        kits = make_kits(pool.labels(idx).tolist(), 9, 1)
        draw_p = R.draw_channel(2, cfg.s, rng, dev, cfg.occ_k)
        pb = ST.render_pool_batch(pool, idx, kits, cfg, rng, dev, draw=draw_p)
        pb["draw"], pb["tpl"] = draw_p, R.templates(draw_p.u, draw_p.s)
        pb["pgen"] = pool.gens(idx)
        pb["pzseed"] = pool.zseed[idx].clone()
        mb = ST.make_mask_batch(2, cfg, rng, dev)
        w = TaskWeights().weights(dev)
        mm = ST.chain_train_step(m, opt, params, sb, pb, mb, cfg, w, dev, rng, const=TR.ConstPred(),
                                 sscale=ST.SoftScale())
        return mm, m

    mm_c, m_c = _one("const")
    assert mm_c["L_gate"] is None and m_c.gate.a.grad is None
    mm_l, m_l = _one("learned")
    assert mm_l["L_gate"] is not None and m_l.gate.a.grad is not None


def test_a26():
    from symemerge.numcode.pred.checks import a26
    status, evidence = a26(torch.device("cpu"))
    assert status == "PASS", evidence


# ================================================================ [U] 2026-08-21 写头共模扣除 + λ PI 伺服 + 只跑阶段 1
def test_write_head_common_mode():
    """[U] 2026-08-21 写头共模扣除: d1_cm=1 ⇒ ℓ_{j,c} = w_c·(z_j − α z̄) + b_c (z̄ = 沿格均值), α 初 1 可学, b_c 初 0 可学,
    W_D1 仍冻结 (A1 不动), 旧冻结偏置不参与; d1_cm=0 ⇒ 旧式 ℓ = W_D1 z + b 逐位不变且**不注册** α/b_c (旧检查点键兼容);
    同种子两式 W_D1 逐位同 (新参数不耗 RNG); 新参数梯度非零 (非零梯度断言); 旋钮经 build_model 穿线."""
    torch.manual_seed(0)
    m0 = PredModel()
    torch.manual_seed(0)
    m1 = PredModel(d1_cm=True)
    assert not hasattr(m0.D1, "alpha") and not hasattr(m0.D1, "bias_c")
    assert "D1.alpha" not in m0.state_dict() and "D1.bias_c" not in m0.state_dict()
    assert "D1.alpha" in m1.state_dict() and "D1.bias_c" in m1.state_dict()
    assert torch.equal(m0.D1.lin.weight, m1.D1.lin.weight) and torch.equal(m0.D1.lin.bias, m1.D1.lin.bias)
    assert float(m1.D1.alpha) == 1.0 and tuple(m1.D1.alpha.shape) == () and float(m1.D1.bias_c.abs().sum()) == 0.0
    assert tuple(m1.D1.bias_c.shape) == (4,)
    assert m1.D1.alpha.requires_grad and m1.D1.bias_c.requires_grad
    assert not m1.D1.lin.weight.requires_grad and not m1.D1.lin.bias.requires_grad
    ids = {id(p) for p in m1.trainable_params()}
    assert id(m1.D1.alpha) in ids and id(m1.D1.bias_c) in ids
    assert id(m1.D1.lin.weight) not in ids and id(m1.D1.lin.bias) not in ids
    assert len(m1.d1_weights()) == 2 and len(m0.d1_weights()) == 2            # W_D1 + 冻结偏置 (A1 口径), 与 cm 无关
    z = torch.randn(3, GM.T, 256)
    lg0 = m0.D1(z)
    assert torch.allclose(lg0, z @ m0.D1.lin.weight.T + m0.D1.lin.bias, atol=1e-6)   # 旧式
    lg1 = m1.D1(z)
    ref = (z - z.mean(1, keepdim=True)) @ m1.D1.lin.weight.T                 # α=1, b_c=0: 旧偏置不参与
    assert lg1.shape == (3, GM.T, 4) and torch.allclose(lg1, ref, atol=1e-5)
    assert torch.allclose(lg1.sum(1), torch.zeros(3, 4), atol=1e-3)         # 逐类沿格求和为零 (共模已扣)
    with torch.no_grad():
        m1.D1.alpha.fill_(0.0)
    assert torch.allclose(m1.D1(z), z @ m1.D1.lin.weight.T, atol=1e-5)       # α=0 取回旧式 (无偏置项)
    with torch.no_grad():
        m1.D1.alpha.fill_(1.0)
        m1.D1.bias_c.copy_(torch.tensor([1.0, -1.0, 0.5, 0.0]))
    lg2 = m1.D1(z)
    assert torch.allclose(lg2 - ref, torch.tensor([1.0, -1.0, 0.5, 0.0]).expand(3, GM.T, 4), atol=1e-5)
    loss = torch.nn.functional.softmax(lg2, -1)[..., R.EMPTY].mean()
    ga, gb = torch.autograd.grad(loss, [m1.D1.alpha, m1.D1.bias_c])
    assert float(ga.abs().sum()) > 0.0 and float(gb.abs().sum()) > 0.0       # 非零梯度断言
    m2 = TR.build_model(TR.TrainCfg(d1_cm=1))
    assert hasattr(m2.D1, "alpha") and not hasattr(TR.build_model(TR.TrainCfg()).D1, "alpha")


def test_lam_servo_pi_tracks_target():
    """[U] 2026-08-21 λ 用 PI 伺服在目标章数上 (不设定值): 章数高于目标 ⇒ log λ 单调升 (稳态每步 k_i·e); 低于 ⇒ 降;
    误差 e = clip(log((ŝ+1)/(s*+1)), ±eclip); EMA 首步自举; 夹取 [lo, hi]; 状态可存取 (续跑)."""
    cfg = TR.TrainCfg(lam_servo=1, lam_target=5.0, lam_kp=0.05, lam_ki=1e-3, lam_ema=0.9,
                      lam_init=1e-4, lam_lo=1e-6, lam_hi=0.1, lam_eclip=2.0)
    sv = TR.LamServo(cfg)
    assert abs(sv.lam - 1e-4) < 1e-12 and sv.s_ema is None and sv.n == 0
    assert sv.error() == 0.0                                   # 无观测 ⇒ 误差 0
    l0 = sv.lam
    sv.update(200.0)
    assert sv.s_ema == 200.0 and sv.n == 1                     # 首步自举
    assert sv.error() == 2.0                                   # log(201/6) = 3.51 → 夹 2
    assert sv.lam > l0
    prev = sv.lam
    for _ in range(50):
        sv.update(200.0)
        assert sv.lam > prev
        prev = sv.lam
    before = math.log(sv.lam)
    sv.update(200.0)                                           # 稳态: P 项 0, Δlogλ = k_i·e
    assert abs(math.log(sv.lam) - before - 1e-3 * 2.0) < 1e-9
    sv_eq = TR.LamServo(cfg)
    sv_eq.update(5.0)
    assert abs(sv_eq.lam - 1e-4) < 1e-15                       # 章数 = 目标 ⇒ 不动
    sv2 = TR.LamServo(cfg)
    for _ in range(20):
        sv2.update(0.0)
    assert sv2.lam < 1e-4 and abs(sv2.error() - max(-2.0, math.log(1.0 / 6.0))) < 1e-12
    sv3 = TR.LamServo(cfg)
    for _ in range(5000):
        sv3.update(256.0)
    assert abs(sv3.lam - 0.1) < 1e-12                          # 上夹
    sv4 = TR.LamServo(cfg)
    for _ in range(5000):
        sv4.update(0.0)
    assert abs(sv4.lam - 1e-6) < 1e-15                         # 下夹
    d = sv.state()
    sv5 = TR.LamServo(cfg)
    sv5.load_state(d)
    assert sv5.lam == sv.lam and sv5.s_ema == sv.s_ema and sv5.n == sv.n and sv5.e_prev == sv.e_prev
    rd = sv.readings()
    assert set(rd) >= {"lam", "s_ema", "e", "loglam", "n"} and rd["lam"] == sv.lam


def test_stage1_only_never_switches():
    """[U] 2026-08-21 阶段 1 只跑 N∈{1..9}: stage1_only=1 ⇒ boot 连续两评达标与 T_stage1 到期都不切换 (阶段 1 全程豁免);
    =0 照旧切."""
    for so, expect in ((1, False), (0, True)):
        cfg = TR.TrainCfg(theta_boot=0.5, t_stage1=10, stage1_only=so)
        st = TR.new_state(cfg)
        latch = TR.BestLatch()
        TR.eval_decisions(st, cfg, latch, 0.9, 0.9, 5, 100.0, 1.0)
        d2 = TR.eval_decisions(st, cfg, latch, 0.9, 0.9, 6, 100.0, 1.0)
        assert d2["want_switch"] is expect and st["stage"] == (2 if expect else 1)
        if so:
            d3 = TR.eval_decisions(st, cfg, latch, 0.1, 0.1, 10, 100.0, 1.0)   # T_stage1 到期亦不切
            assert not d3["want_switch"] and st["stage"] == 1 and d3["exempt"] and not d3["hits"]


def test_trainer_smoke_cm_servo(tmp_path):
    """共模写头 + λ 伺服 + 只跑阶段 1 的链训练器冒烟 (K=1, pred/闸/池起链关): 3 步 → 日志行带 lam/log10_lam/stamps_ema,
    评测行带 servo 块, 检查点含伺服状态 + D1.alpha/bias_c 键; 恢复续 2 步伺服状态连续 (首步 λ = 存档 log λ)."""
    import json as _json
    cfg = TR.TrainCfg(steps=3, b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                      eval_every=3, log_every=1, viz_every=3, eval_per_n=1, table_per_n=1,
                      out=str(tmp_path / "run"), chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0,
                      d1_cm=1, lam_servo=1, lam_init=1e-3, lam_target=5.0, stage1_only=1, rho_admit=1.0)
    fin = TR.run(cfg, device=str(DEV))
    assert fin["step"] == 3 and fin["stage"] == 1
    rows = [_json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert len(steps) == 3 and all("lam" in r and "log10_lam" in r and "stamps_ema" in r for r in steps)
    assert abs(steps[0]["lam"] - 1e-3) < 1e-9                  # 首步用 λ₀
    evs = [r for r in rows if r.get("eval")]
    ev = evs[-1]
    assert "servo" in ev and set(ev["servo"]) >= {"lam", "s_ema", "e", "loglam", "n"} and ev["servo"]["n"] == 3
    assert abs(ev["lam_now"] - steps[-1]["lam"]) < 5e-6          # 评测行 λ (全精度) = 末步所用 λ (窗均 5 位舍入)
    ck = torch.load(tmp_path / "run" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert "D1.alpha" in ck["model"] and "D1.bias_c" in ck["model"]
    assert ck["st"]["lam_servo"]["n"] == 3
    assert ck["cfg"]["d1_cm"] == 1 and ck["cfg"]["lam_servo"] == 1
    cfg2 = TR.TrainCfg(**{**cfg.__dict__, "steps": 5})
    fin2 = TR.run(cfg2, device=str(DEV), resume=str(tmp_path / "run" / "ckpt_last.pt"))
    assert fin2["step"] == 5
    rows2 = [_json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    s4 = [r for r in rows2 if r.get("step") == 4 and "L" in r and not r.get("eval")][0]
    assert abs(s4["log10_lam"] - ck["st"]["lam_servo"]["loglam"] / math.log(10.0)) < 1e-4   # 伺服状态续上
    # 兼容: 旧式 (d1_cm=0) 模型无新键, 去掉新键后严格装载照旧
    m_old = TR.build_model(TR.TrainCfg())
    sd = {k: v for k, v in ck["model"].items() if k not in ("D1.alpha", "D1.bias_c")}
    m_old.load_state_dict(sd)


def test_a27():
    from symemerge.numcode.pred.checks import a27
    status, evidence = a27(torch.device("cpu"))
    assert status == "PASS", evidence


def test_switch_at_resume(tmp_path):
    """[U] 2026-08-22「延用 B 臂设置，切阶段 2」: --resume 时 switch_at_resume=1 且检查点仍在阶段 1 ⇒ 续跑第一步前当场切大数量阶段
    (why='resume', k=K_MAX, stage1_level = 检查点末两评 note_1 均, 闩重置, switch_step = 续跑起点), 存 ckpt_stage1_end.pt, 日志记 switch 行,
    阶段 2 每评照记 α (cm 块); 已在阶段 2 的检查点再续 = 无动作 (switch_step 不变); 与 stage1_only 互斥."""
    import json as _json
    base = dict(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1, viz_every=10 ** 9,
                eval_per_n=1, table_per_n=1, chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0034,
                rho_admit=1.0, out=str(tmp_path / "run"))
    fin = TR.run(TR.TrainCfg(steps=3, stage1_only=1, **base), device=str(DEV))
    assert fin["stage"] == 1 and fin["step"] == 3
    ck = torch.load(tmp_path / "run" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    lds = ck["st"]["last_d1_scores"]
    assert len(lds) >= 1
    cfg2 = TR.TrainCfg(steps=5, stage1_only=0, switch_at_resume=1, **base)
    fin2 = TR.run(cfg2, device=str(DEV), resume=str(tmp_path / "run" / "ckpt_last.pt"))
    assert fin2["step"] == 5 and fin2["stage"] == 2 and fin2["k"] == GM.K_MAX
    assert fin2["switch"]["why"] == "resume" and fin2["switch"]["step"] == 3
    assert abs(fin2["switch"]["stage1_level"] - round(sum(lds) / len(lds), 4)) < 1e-9
    assert os.path.exists(tmp_path / "run" / "ckpt_stage1_end.pt")
    rows = [_json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    sw = [r for r in rows if r.get("resume_switch")]
    assert len(sw) == 1 and sw[0]["step"] == 3 and sw[0]["switch"]["why"] == "resume" and sw[0]["switch"]["k"] == GM.K_MAX
    ev = [r for r in rows if r.get("eval") and r["step"] == 5][0]
    assert ev["stage"] == 2 and ev["k"] == GM.K_MAX and len(ev["per_n"]) == len(D.train_ns(GM.K_MAX))
    assert "cm" in ev and "alpha" in ev["cm"] and ev["cm"]["frozen"] is False            # 阶段 2 每评照记 α
    ck2 = torch.load(tmp_path / "run" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert ck2["st"]["stage"] == 2 and ck2["st"]["switch_step"] == 3 and ck2["st"]["latch"]["best"] == -1.0   # 闩重置后单评未锁
    # 已在阶段 2 的检查点再续: 无动作, switch_step 仍 3
    fin3 = TR.run(TR.TrainCfg(steps=6, stage1_only=0, switch_at_resume=1, **base), device=str(DEV),
                  resume=str(tmp_path / "run" / "ckpt_last.pt"))
    assert fin3["stage"] == 2 and fin3["switch"]["step"] == 3 and fin3["switch"]["why"] == "resume"
    rows3 = [_json.loads(l) for l in open(tmp_path / "run" / "log.jsonl")]
    assert len([r for r in rows3 if r.get("resume_switch")]) == 1
    with pytest.raises(AssertionError):
        TR.run(TR.TrainCfg(steps=7, stage1_only=1, switch_at_resume=1, **base), device=str(DEV),
               resume=str(tmp_path / "run" / "ckpt_last.pt"))


def test_alpha_freeze(tmp_path):
    """[U] 2026-08-22「…立刻冻结 α 重启」预备: alpha_freeze=1 ⇒ 共模写头 α 整跑逐位不变 (反传后 α.grad 置 None ⇒ AdamW 跳过: 无动量无权重衰减),
    b_c 照常学; =0 对照 α 一步即动 (A27 同证); 续跑冻结 = 保持检查点值逐位; 评测行 cm.frozen=True; 要求 d1_cm=1."""
    import json as _json
    base = dict(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1, viz_every=10 ** 9,
                eval_per_n=1, table_per_n=1, chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0034,
                stage1_only=1, rho_admit=1.0)
    TR.run(TR.TrainCfg(steps=3, alpha_freeze=0, out=str(tmp_path / "free"), **base), device=str(DEV))
    ck0 = torch.load(tmp_path / "free" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    a0 = ck0["model"]["D1.alpha"].clone()
    assert float(a0) != 1.0                                                            # 对照: 不冻结一步即动
    TR.run(TR.TrainCfg(steps=3, alpha_freeze=1, out=str(tmp_path / "frz"), **base), device=str(DEV))
    ck1 = torch.load(tmp_path / "frz" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert float(ck1["model"]["D1.alpha"]) == 1.0                                       # 冻结: 出厂值逐位不变
    assert float(ck1["model"]["D1.bias_c"].abs().sum()) > 0                             # b_c 照常学
    rows = [_json.loads(l) for l in open(tmp_path / "frz" / "log.jsonl")]
    ev = [r for r in rows if r.get("eval")][-1]
    assert ev["cm"]["frozen"] is True and ev["cm"]["alpha"] == 1.0
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert all(math.isfinite(r["gnorm"]) for r in steps)                                # 梯度裁剪对 grad=None 参数无碍
    # 续跑冻结: 从不冻结检查点 (α≠1) 续 2 步, α 保持检查点值逐位
    fin2 = TR.run(TR.TrainCfg(steps=5, alpha_freeze=1, out=str(tmp_path / "free"), **base), device=str(DEV),
                  resume=str(tmp_path / "free" / "ckpt_last.pt"))
    ck2 = torch.load(tmp_path / "free" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert fin2["step"] == 5 and torch.equal(ck2["model"]["D1.alpha"], a0)
    with pytest.raises(AssertionError):
        TR.run(TR.TrainCfg(steps=1, alpha_freeze=1, out=str(tmp_path / "bad"),
                           **{k: v for k, v in base.items() if k != "d1_cm"}, d1_cm=0), device=str(DEV))


def test_task_wmul(tmp_path):
    """[U] 2026-08-23「将 t4 权重设置为 3 倍续跑 3000 步，重点观察 t4 的损失变化」: task_wmul="1,1,1,3,1,1" ⇒
    损失权重 w = 归一 w × 倍率 (t4 位恰 3×, 其余位逐位同归一 w); 池成绩 σ 用的归一 w 不动 (C4.5); 缺省 (空) ⇒ weights() 逐位同旧;
    日志步行/评测行带 w_norm 与逐任务未加权 CE (ce_sc/ce_S/ce_pool, scene.ce/notes.ce/pool.ce); lds_rows 的 rows == Σ w_t ce_t + μ·read_row;
    倍率个数 ≠ 6 或 ≤0 ⇒ 断言."""
    import json as _json
    # ---- 权重代数
    tw0 = TaskWeights()
    tw3 = TaskWeights(mul=[1, 1, 1, 3, 1, 1])
    acc = torch.tensor([[1.0, 0.5, 0.8, 0.2, 0.9, 0.3]])
    for _ in range(20):
        tw0.update(acc)
        tw3.update(acc)
    assert torch.equal(tw0.weights(), tw0.weights_norm())                               # 缺省: 逐位同旧
    assert tw0.mul is None
    assert torch.equal(tw3.weights_norm(), tw0.weights_norm())                          # 归一 w 不受倍率影响
    assert abs(float(tw3.weights_norm().sum()) - 1.0) < 1e-6
    w3, wn = tw3.weights(), tw3.weights_norm()
    assert torch.equal(w3[[0, 1, 2, 4, 5]], wn[[0, 1, 2, 4, 5]])                         # 其余位逐位同
    assert abs(float(w3[3]) - 3.0 * float(wn[3])) < 1e-6                                 # t4 位恰 3×
    with pytest.raises(AssertionError):
        TaskWeights(mul=[1, 1, 1, 3, 1])
    with pytest.raises(AssertionError):
        TaskWeights(mul=[1, 1, 1, 0, 1, 1])
    # ---- 训练 3 步: 日志行带 ce_*/w/w_norm, 评测行带 scene.ce/notes.ce/pool.ce + task_wmul; 倍率从命令行来 (续跑亦然)
    base = dict(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1, viz_every=10 ** 9,
                eval_per_n=1, table_per_n=1, chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0034,
                stage1_only=1, rho_admit=1.0)
    TR.run(TR.TrainCfg(steps=3, task_wmul="1,1,1,3,1,1", out=str(tmp_path / "m3"), **base), device=str(DEV))
    rows = [_json.loads(l) for l in open(tmp_path / "m3" / "log.jsonl")]
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert len(steps) == 3
    for r in steps:
        assert len(r["ce_sc"]) == 6 and len(r["ce_S"]) == 6 and all(math.isfinite(x) and x >= 0 for x in r["ce_sc"] + r["ce_S"])
        assert r.get("ce_pool") is None or (len(r["ce_pool"]) == 6 and all(math.isfinite(x) for x in r["ce_pool"]))   # 无池批次 ⇒ None ⇒ 窗口均值丢键
        assert len(r["w"]) == 6 and len(r["w_norm"]) == 6
        assert abs(r["w"][3] - 3.0 * r["w_norm"][3]) < 2e-3 and all(abs(r["w"][i] - r["w_norm"][i]) < 1e-9 for i in (0, 1, 2, 4, 5))
    ev = [r for r in rows if r.get("eval")][-1]
    assert ev["task_wmul"] == [1.0, 1.0, 1.0, 3.0, 1.0, 1.0] and len(ev["w_norm"]) == 6
    for side in (ev["scene"], ev["notes"], ev["depths"][0]):
        assert set(side["ce"]) == {"t1", "t2", "t3", "t4", "t5", "t6"} and all(math.isfinite(v) for v in side["ce"].values())
    if ev.get("pool"):
        assert set(ev["pool"]["ce"]) == {"t1", "t2", "t3", "t4", "t5", "t6"}
    # ---- 缺省跑: 日志行无 w_norm (旧 schema), 有 ce_*; 评测行无 task_wmul
    TR.run(TR.TrainCfg(steps=3, out=str(tmp_path / "m0"), **base), device=str(DEV))
    rows0 = [_json.loads(l) for l in open(tmp_path / "m0" / "log.jsonl")]
    s0 = [r for r in rows0 if "L" in r and not r.get("eval") and not r.get("final")]
    assert all("w_norm" not in r and len(r["ce_S"]) == 6 for r in s0)
    assert "task_wmul" not in [r for r in rows0 if r.get("eval")][-1]
    # ---- lds_rows 分解: rows == Σ_t w_t ce_t + μ read_row
    ck = torch.load(tmp_path / "m0" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    cfg = TR.TrainCfg(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = TR.build_model(cfg).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    es = TR.build_eval_sets(cfg, cfg.k1)
    sb = build_batch(es["groups"][:4], DEV)
    w = torch.tensor([0.1, 0.2, 0.05, 0.4, 0.15, 0.1], device=DEV)
    with torch.no_grad():
        draw = R.draw_channel(sb["B"], cfg.s, _rng(1), DEV, cfg.occ_k)
        tpl = R.templates(draw.u, draw.s)
        out = ST.scene_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    sc = out["sc"]
    recon = (sc["ce"] * w.unsqueeze(0)).sum(1) + cfg.mu * sc["read_row"]
    assert torch.allclose(recon, sc["rows"], atol=1e-5, rtol=1e-5)
    assert sc["ce"].shape == (sb["B"], 6)


def test_ham12_top_level_without_pred(tmp_path):
    """[U] 2026-08-23 K=2 主判据 ham12 (第一写 vs 第二写硬类别图逐格不同占比) 不依赖 pred_on: chain_k=2 + pred/gate 关 ⇒
    评测行顶层 ham12 {mean, med, q[3], nz_frac, n} (n = 评测条数), 步行 ham12 ∈ [0,1]; chain_k=1 ⇒ 评测行无 ham12、步行无 ham12 键."""
    import json as _json
    base = dict(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1, viz_every=10 ** 9,
                eval_per_n=1, table_per_n=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0034,
                stage1_only=1, rho_admit=1.0)
    TR.run(TR.TrainCfg(steps=3, chain_k=2, out=str(tmp_path / "k2"), **base), device=str(DEV))
    rows = [_json.loads(l) for l in open(tmp_path / "k2" / "log.jsonl")]
    ev = [r for r in rows if r.get("eval")][-1]
    h = ev["ham12"]
    assert set(h) == {"mean", "med", "q", "nz_frac", "n"} and len(h["q"]) == 3 and h["n"] == ev["n_items"]
    assert 0.0 <= h["med"] <= 1.0 and 0.0 <= h["mean"] <= 1.0 and 0.0 <= h["nz_frac"] <= 1.0
    assert "pred" not in ev                                                              # pred 关: 旧位置不存在, 顶层仍有
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert len(steps) == 3 and all(0.0 <= r["ham12"] <= 1.0 for r in steps)
    TR.run(TR.TrainCfg(steps=3, chain_k=1, out=str(tmp_path / "k1"), **base), device=str(DEV))
    rows1 = [_json.loads(l) for l in open(tmp_path / "k1" / "log.jsonl")]
    assert "ham12" not in [r for r in rows1 if r.get("eval")][-1]
    assert all("ham12" not in r for r in rows1 if "L" in r and not r.get("eval") and not r.get("final"))


def test_origin_write_path_reads_single_slot():
    """[U] 2026-08-23「写路径加 origin，读路径不动」+「写第一张和第二张笔记都能看原题 origin … 读笔记 … 只能读笔记自身」(第一张当前槽 = 空白纸):
    origin_write=1 ⇒ (i) 场景读出 h_0 与 origin_write=0 逐位同 (CPU); 第一写 logits = write_tokens_org(tok(空白纸), tok(x0)) 两槽装配逐位同,
    空白纸 = 全空类硬渲染过同链信道 (全零画布 + 同 draw 噪声), 第一写对 origin_write=0 不同, 换 origin 槽 (x0 按批滚动) ⇒ 第一写 logits 改变;
    (ii) 各深度读出 h_k = 单槽 enc_v3(x_k) 逐位同 (读路径不动); (iii) 第二写 logits = write_tokens_org(tok(x1), tok(x0)) 逐位同, 换 origin 槽 ⇒ 改变;
    (iv) 深度 2 损失对 seg_cur/seg_org 梯度 |g| 和 > 0 (非零梯度断言; origin_write=0 下二者 grad 为 None); (v) use_flag=1 兼容路与 origin_write 互斥."""
    import dataclasses
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = PredModel(d1_cm=True).to(dev)
    cfg0 = TR.TrainCfg(b_scene=4, b_pool=4, b_mask=4, workers=0, kit_workers=0, chain_k=2, pred_on=0, gate_on=0,
                       chain_from_pool=0, d1_cm=1, lam=0.0034, origin_write=0)
    cfg1 = dataclasses.replace(cfg0, origin_write=1)
    rng = _rng(5)
    items = [D.sample_group(rng, 9) for _ in range(4)]
    sb = build_batch(items, dev)
    w = TaskWeights().weights(dev)
    draw = R.draw_channel(sb["B"], cfg0.s, rng, dev, cfg0.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out0 = ST.chain_side(model, sb, w, cfg0.mu, cfg0, draw, tpl)
    out1 = ST.chain_side(model, sb, w, cfg1.mu, cfg1, draw, tpl)
    # (i) 场景读出不动; 第一写 = [空白纸 | x0] 两槽装配
    assert torch.equal(out0["states"][0]["h"], out1["states"][0]["h"])
    xb = ST.blank_note(sb["B"], draw, tpl)
    assert xb.shape == sb["x1"].shape and torch.equal(xb, R.channel(torch.zeros_like(sb["x1"]), draw))   # 全空类 = 全零画布 + 同 draw 信道
    with torch.no_grad():
        z0a = ST.write_tokens_org(model, model.E.tokenize(xb), model.E.tokenize(sb["x1"]))
        z0b = ST.write_tokens_org(model, model.E.tokenize(xb), model.E.tokenize(sb["x1"].roll(1, 0)))
        assert torch.equal(ST.write_logits(model, z0a, cfg1), out1["writes"][0]["logits"].detach())
        assert not torch.equal(ST.write_logits(model, z0a, cfg1), ST.write_logits(model, z0b, cfg1))   # 第一写吃 origin 槽
    assert not torch.equal(out0["writes"][0]["logits"], out1["writes"][0]["logits"])
    # (ii) 读路径 = 单槽 257
    with torch.no_grad():
        for d in (1, 2):
            h_single = ST.enc_v3(model, out1["writes"][d - 1]["x"].detach(), cfg1, flagged=False)["cls"]
            assert torch.equal(out1["states"][d]["h"].detach(), h_single)
    # (iii) 第二写 = [x1 | x0] 两槽装配, 吃 origin 槽
    x1 = out1["writes"][0]["x"].detach()
    with torch.no_grad():
        za = ST.write_tokens_org(model, model.E.tokenize(x1), model.E.tokenize(sb["x1"]))
        zb = ST.write_tokens_org(model, model.E.tokenize(x1), model.E.tokenize(sb["x1"].roll(1, 0)))
        assert za.shape == (sb["B"], GM.T, za.shape[-1])
        assert torch.equal(ST.write_logits(model, za, cfg1), out1["writes"][1]["logits"].detach())
        assert not torch.equal(ST.write_logits(model, za, cfg1), ST.write_logits(model, zb, cfg1))
    # (iv) 非零梯度到段嵌入 (只经写路径)
    model.zero_grad(set_to_none=True)
    out1["states"][2]["rows"].mean().backward()
    assert model.E.seg_cur.grad is not None and float(model.E.seg_cur.grad.abs().sum()) > 0
    assert model.E.seg_org.grad is not None and float(model.E.seg_org.grad.abs().sum()) > 0
    model.zero_grad(set_to_none=True)
    out0["states"][2]["rows"].mean().backward()
    assert model.E.seg_cur.grad is None and model.E.seg_org.grad is None
    # (v) 互斥
    with pytest.raises(AssertionError):
        ST.chain_side(model, sb, w, cfg1.mu, dataclasses.replace(cfg1, chain_k=1, use_flag=1), draw, tpl)


def test_dtrue_top_level_and_origin_write_run(tmp_path):
    """[U] 2026-08-23 主判据 Δ_true[1] = 纸 2 六任务 − 纸 1 六任务: 评测行顶层 dtrue = K 项 {mean, se, lo95, t, pos_frac, neg_frac, n}
    (不依赖 pred_on; n = 评测条数; dtrue[d].mean = depths[d].score − 上一深度 score; lo95 = mean − 1.96·se), 步行 dtrue = K 项批均 ∈ [−1,1];
    origin_write=1 走通训练/评测/检查点/续跑 (run_start cfg 记 origin_write=1)."""
    import json as _json
    base = dict(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1, viz_every=10 ** 9,
                eval_per_n=1, table_per_n=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0034,
                stage1_only=1, rho_admit=1.0)
    TR.run(TR.TrainCfg(steps=3, chain_k=2, origin_write=1, out=str(tmp_path / "ow"), **base), device=str(DEV))
    rows = [_json.loads(l) for l in open(tmp_path / "ow" / "log.jsonl")]
    assert rows[0]["run_start"]["cfg"]["origin_write"] == 1
    ev = [r for r in rows if r.get("eval")][-1]
    dt = ev["dtrue"]
    assert len(dt) == 2 and all(set(d) == {"mean", "se", "lo95", "t", "pos_frac", "neg_frac", "n"} for d in dt)
    assert all(d["n"] == ev["n_items"] for d in dt)
    assert abs(dt[0]["mean"] - (ev["depths"][0]["score"] - ev["scene"]["score"])) < 3e-4
    assert abs(dt[1]["mean"] - (ev["depths"][1]["score"] - ev["depths"][0]["score"])) < 3e-4
    assert all(abs(d["lo95"] - (d["mean"] - 1.96 * d["se"])) < 1e-4 for d in dt)
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert len(steps) == 3 and all(len(r["dtrue"]) == 2 and all(-1.0 <= x <= 1.0 for x in r["dtrue"]) for r in steps)
    TR.run(TR.TrainCfg(steps=4, chain_k=2, origin_write=1, out=str(tmp_path / "ow"), **base), device=str(DEV),
           resume=str(tmp_path / "ow" / "ckpt_last.pt"))
    rows2 = [_json.loads(l) for l in open(tmp_path / "ow" / "log.jsonl")]
    assert [r for r in rows2 if "L" in r and not r.get("eval") and not r.get("final")][-1]["step"] == 4


def test_d1_learn_unfreeze_rownorm(tmp_path):
    """[U] 2026-08-23「W_D1 解冻，行归一化，lr = 0.1 × 主 lr」: d1_learn=1 ⇒ W_D1 在且只在第 2 参数组
    (lr = d1_lr_mult × 主 lr, wd=0), 行 L2 单位范数 (载入时 + 每步后), 一步后值改变 (动量非零 = 非零梯度经历),
    评测行带 d1 漂移块; d1_learn=0 同种子 W_D1 逐位 = 初值 (A1 口径), 参数组仍 1 组."""
    cfg = TR.TrainCfg(steps=2, b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                      eval_every=2, log_every=1, viz_every=10 ** 9, eval_per_n=1, table_per_n=1,
                      out=str(tmp_path / "run"), chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0,
                      d1_cm=1, d1_learn=1, d1_lr_mult=0.1)
    torch.manual_seed(cfg.seed)
    w_init = TR.build_model(cfg).D1.lin.weight.detach().clone()      # 同种子冻结初值 (解冻不耗 RNG)
    fin = TR.run(cfg, device=str(DEV))
    assert fin["step"] == 2
    ck = torch.load(tmp_path / "run" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    W = ck["model"]["D1.lin.weight"]
    assert len(ck["opt"]["param_groups"]) == 2
    g1 = ck["opt"]["param_groups"][1]
    assert g1["lr"] == pytest.approx(cfg.lr * 0.1) and g1["weight_decay"] == 0.0 and len(g1["params"]) == 1
    assert torch.allclose(W.norm(dim=1), torch.ones(W.shape[0]), atol=1e-5)      # 行单位范数 (post-hook)
    assert not torch.equal(W, w_init)                                            # 一步后值改变
    s1 = ck["opt"]["state"][g1["params"][0]]
    assert float(s1["exp_avg"].abs().sum()) > 0                                  # 非零梯度经历 (动量非零)
    import json as _json
    evs = [_json.loads(r) for r in open(tmp_path / "run" / "log.jsonl") if '"eval": true' in r]
    ev = evs[-1]
    assert "d1" in ev and len(ev["d1"]["drift"]) == W.shape[0]
    assert max(ev["d1"]["wnorm"]) == pytest.approx(1.0, abs=1e-5)
    cfg0 = TR.TrainCfg(**{**cfg.__dict__, "d1_learn": 0, "out": str(tmp_path / "run0")})
    TR.run(cfg0, device=str(DEV))
    ck0 = torch.load(tmp_path / "run0" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert torch.equal(ck0["model"]["D1.lin.weight"], w_init)                    # 关 = 逐位同旧 (A1)
    assert len(ck0["opt"]["param_groups"]) == 1


def test_d1_learn_resume_group_orders(tmp_path):
    """W_D1 解冻的恢复兼容两路: (a) 旧检查点 (opt 1 组) 带 d1_learn=1 恢复 ⇒ 载后建组续跑; (b) 解冻检查点
    (opt 2 组) 再恢复 ⇒ 先建组再载 (W_D1 动量随载), 续跑; 两路收官均 2 组且行单位范数."""
    base = TR.TrainCfg(steps=2, b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0,
                       eval_every=2, log_every=1, viz_every=10 ** 9, eval_per_n=1, table_per_n=1,
                       out=str(tmp_path / "a"), chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, d1_cm=1)
    TR.run(base, device=str(DEV))
    cka = torch.load(tmp_path / "a" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert len(cka["opt"]["param_groups"]) == 1
    cfg1 = TR.TrainCfg(**{**base.__dict__, "steps": 4, "d1_learn": 1, "d1_lr_mult": 0.1,
                          "out": str(tmp_path / "b")})
    fin1 = TR.run(cfg1, device=str(DEV), resume=str(tmp_path / "a" / "ckpt_last.pt"))
    assert fin1["step"] == 4
    ckb = torch.load(tmp_path / "b" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert len(ckb["opt"]["param_groups"]) == 2
    cfg2 = TR.TrainCfg(**{**cfg1.__dict__, "steps": 6, "out": str(tmp_path / "c")})
    fin2 = TR.run(cfg2, device=str(DEV), resume=str(tmp_path / "b" / "ckpt_last.pt"))
    assert fin2["step"] == 6
    ckc = torch.load(tmp_path / "c" / "ckpt_last.pt", map_location="cpu", weights_only=True)
    assert len(ckc["opt"]["param_groups"]) == 2
    g1 = ckc["opt"]["param_groups"][1]
    s1 = ckc["opt"]["state"][g1["params"][0]]
    assert float(s1["exp_avg"].abs().sum()) > 0                                  # 动量随载 + 续积
    W = ckc["model"]["D1.lin.weight"]
    assert torch.allclose(W.norm(dim=1), torch.ones(W.shape[0]), atol=1e-5)


# ================================================================ 加法流 ([U] 2026-08-27 双场景 N+m → 一张纸; 流程同主流程, 不新造器官)
def test_add_group_labels_and_firewall():
    """加法流数据项 (sample_add_group): n == n_a + m, m ∈ {1,2,3}; n_a / m / n 与候选 N 皆在训练域 (留出防火墙);
    标签 = targets(n_a + m, θ) (内容对表, 不靠索引); 两张场景各 (SIDE,SIDE); build_batch 带 x_aux/n_a/m_add;
    add_pairs(k) 恰为全部合法 (N, m) 对, add_sums(k) = 其和的集合; 指定 (n_a, m) 时逐字采用."""
    for k in (9, GM.K_MAX):
        tns = set(D.train_ns(k))
        rng = _rng(100 + k)
        pairs = set(D.add_pairs(k))
        assert pairs == {(n, m) for m in D.M_CHOICES for n in D.train_ns(k) if (n + m) in tns and m in tns}
        assert set(D.add_sums(k)) == {n + m for n, m in pairs} and list(D.add_sums(k)) == sorted(set(D.add_sums(k)))
        seen = set()
        for _ in range(60):
            it = D.sample_add_group(rng, k)
            assert it["m"] in D.M_CHOICES and it["n"] == it["n_a"] + it["m"]
            assert it["n_a"] in tns and it["m"] in tns and it["n"] in tns
            assert (it["n_a"], it["m"]) in pairs
            assert all(n in tns for n in it["cand_ns"])
            assert it["targets"] == D.targets(it["n"], it["theta"])
            assert it["scene"].shape == (GM.SIDE, GM.SIDE) and it["scene_m"].shape == (GM.SIDE, GM.SIDE)
            assert it["cands"].shape == (D.N_CAND, GM.SIDE, GM.SIDE) and it["cand_kinds"][it["truth5"]] == "pos"
            seen.add(it["m"])
        assert seen == set(D.M_CHOICES)
        it2 = D.sample_add_group(rng, k, n_a=5, m=3)
        assert (it2["n_a"], it2["m"], it2["n"]) == (5, 3, 8)
    items = [D.sample_add_group(_rng(7), 9) for _ in range(3)]
    b = build_batch(items, torch.device("cpu"))
    assert b["x1"].shape == (3, GM.SIDE, GM.SIDE) and b["x_aux"].shape == (3, GM.SIDE, GM.SIDE)
    assert torch.equal(b["ns"], b["n_a"] + b["m_add"]) and b["B"] == 3
    b0 = build_batch([D.sample_group(_rng(7), 9)], torch.device("cpu"))
    assert "x_aux" not in b0


def test_add_chain_two_slot_write_and_grads():
    """加法流前向 (chain_side 带 x_aux): (i) 深度 0 读出 h = 两槽装配 [CLS]⊕(tok(x_N)+seg_cur)⊕(tok(x_m)+seg_org) 的 CLS 逐位同;
    (ii) 写 logits = 该两槽装配当前槽 token 过写头, 换 x_m (批内滚动) ⇒ 写 logits 改变 (写者确实看到第二张场景);
    (iii) 纸读出 = 单槽 enc_v3(纸) 逐位同 (读路径不动); (iv) 纸损失对 seg_cur / seg_org / 眼部首层 |g| 和 > 0;
    (v) 去掉 x_aux 的同批 = 单槽旧路逐位同; (vi) chain_train_step 带 ab: 有限损失 + 加法项仪表; ab=None ⇒ L_add None."""
    import dataclasses
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = PredModel(d1_cm=True).to(dev)
    cfg = TR.TrainCfg(b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, chain_k=1, pred_on=0, gate_on=0,
                      chain_from_pool=0, d1_cm=1, lam=0.0034, add_on=1, b_add=4, add_workers=0)
    rng = _rng(5)
    ab = build_batch([D.sample_add_group(rng, 9) for _ in range(4)], dev)
    w = TaskWeights().weights(dev)
    draw = R.draw_channel(ab["B"], cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = ST.chain_side(model, ab, w, cfg.mu, cfg, draw, tpl)
    with torch.no_grad():
        e2 = model.E.enc_seq(model.E.tokenize(ab["x1"]), model.E.tokenize(ab["x_aux"]), seg=True)
        e2r = model.E.enc_seq(model.E.tokenize(ab["x1"]), model.E.tokenize(ab["x_aux"].roll(1, 0)), seg=True)
    assert torch.equal(out["states"][0]["h"].detach(), e2["cls"])                                   # (i)
    assert torch.equal(out["writes"][0]["logits"].detach(), ST.write_logits(model, e2["tokens"], cfg).detach())   # (ii)
    assert not torch.equal(ST.write_logits(model, e2["tokens"], cfg), ST.write_logits(model, e2r["tokens"], cfg))
    with torch.no_grad():                                                                            # (iii)
        h1 = ST.enc_v3(model, out["writes"][0]["x"].detach(), cfg, flagged=False)["cls"]
    assert torch.equal(out["states"][1]["h"].detach(), h1)
    model.zero_grad(set_to_none=True)                                                                # (iv)
    out["states"][1]["rows"].mean().backward()
    for p in (model.E.seg_cur, model.E.seg_org, model.E.cnn[0].weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    ab1 = {k: v for k, v in ab.items() if k not in ("x_aux", "n_a", "m_add")}                        # (v)
    out1 = ST.chain_side(model, ab1, w, cfg.mu, cfg, draw, tpl)
    with torch.no_grad():
        e1 = model.E.enc_seq(model.E.tokenize(ab["x1"]), None, seg=False)
    assert torch.equal(out1["states"][0]["h"].detach(), e1["cls"])
    assert not torch.equal(out1["writes"][0]["logits"], out["writes"][0]["logits"])
    # (vi) 一步训练
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=1e-4)
    sb = build_batch([D.sample_group(rng, 9) for _ in range(2)], dev)
    mb = ST.make_mask_batch(2, cfg, rng, dev)
    mm = ST.chain_train_step(model, opt, params, sb, None, mb, cfg, w, dev, rng, ab=ab, rng_add=_rng(9))
    assert math.isfinite(mm["L"]) and mm["L_add"] is not None and math.isfinite(mm["L_add"])
    assert len(mm["L_add_states"]) == 2 and len(mm["acc_add_S"]) == 6 and len(mm["acc_add_sc"]) == 6
    assert mm["ks_add"] is not None and mm["ks_add"][0].shape == (4, GM.T)
    assert mm["accs_notes"].shape[0] == 2 + 4                                                       # 加法纸进 w_t 命中人口
    mm0 = ST.chain_train_step(model, opt, params, sb, None, mb, dataclasses.replace(cfg, add_on=0), w, dev, rng)
    assert mm0["L_add"] is None and mm0["ks_add"] is None and mm0["accs_notes"].shape[0] == 2


def test_trainer_smoke_add_on(tmp_path):
    """加法流训练器冒烟 (K=1, 共模写头, 只跑阶段 1, 从零 3 步 + 续跑 2 步): 步行带 L_add/acc_add_S/stamps_add/inflow_add;
    评测行带 add 块 (scene / notes / null_single / per_m / per_n, notes 含 read_vs_N); 加法纸入池 (ρ=1 ⇒ 3×(B_scene+B_add) 条);
    R6 配对图 notes_add_*.png 落盘; run_start cfg 记 add_on=1; 续跑走通."""
    import glob as _glob
    import json as _json
    cfg = TR.TrainCfg(steps=3, b_scene=2, b_pool=2, b_mask=2, workers=0, kit_workers=0, eval_every=3, log_every=1,
                      viz_every=3, eval_per_n=1, table_per_n=1, out=str(tmp_path / "add"), chain_k=1, pred_on=0,
                      gate_on=0, chain_from_pool=0, d1_cm=1, lam=0.0, stage1_only=1, rho_admit=1.0,
                      add_on=1, b_add=2, add_workers=0)
    fin = TR.run(cfg, device=str(DEV))
    assert fin["step"] == 3 and fin["pool"]["size"] == 3 * (2 + 2)
    rows = [_json.loads(l) for l in open(tmp_path / "add" / "log.jsonl")]
    assert rows[0]["run_start"]["cfg"]["add_on"] == 1 and rows[0]["run_start"]["cfg"]["b_add"] == 2
    steps = [r for r in rows if "L" in r and not r.get("eval") and not r.get("final")]
    assert len(steps) == 3 and all(("L_add" in r) and ("acc_add_S" in r) and ("stamps_add" in r) and ("inflow_add" in r) for r in steps)
    assert all(r["inflow_add"] == 1.0 for r in steps)
    ev = [r for r in rows if r.get("eval")][-1]
    ad = ev["add"]
    assert set(ad) >= {"scene", "notes", "null_single", "per_m", "per_n", "n_items"}
    assert set(ad["notes"]) >= {"acc", "score", "read", "read_vs_N", "acc5_vs_N", "stamps_mean", "iconic", "ham_vs_single"}
    assert set(ad["notes"]["ham_vs_single"]) == {"mean", "med", "nz_frac"} and 0.0 <= ad["notes"]["ham_vs_single"]["mean"] <= 1.0
    assert set(ad["null_single"]) >= {"acc", "score", "read"}
    assert ad["n_items"] == len(D.add_sums(9)) * 1 and all(str(m) in ad["per_m"] for m in D.M_CHOICES)   # 评测集 = 每个和 s 各 eval_per_n 项
    assert 0.0 <= ad["notes"]["score"] <= 1.0 and 0.0 <= ad["null_single"]["score"] <= 1.0
    assert _glob.glob(str(tmp_path / "add" / "notes_add_3*.png"))
    fin2 = TR.run(TR.TrainCfg(**{**cfg.__dict__, "steps": 5}), device=str(DEV), resume=str(tmp_path / "add" / "ckpt_last.pt"))
    assert fin2["step"] == 5 and fin2["pool"]["size"] == 5 * (2 + 2)
