# tests/test_nc_trainer.py -- 训练器装配层 (spec §7 Phase 1/2) 的判别性测试.
# 模块级部件 (合法屏蔽/GRPO 助手/损失) 已在 test_nc_model.py 钉住; 此处只测装配:
# 梯度路由的组合正确性, 组包共享, 过滤, 素养仪器的零假设, 评测不动参数, 统计与恢复.
import math

import pytest
import torch

import symemerge.numcode.data as D
import symemerge.numcode.geometry as GM
import symemerge.numcode.trainer as TR
from symemerge.numcode.model import NumCodeModel


def _rng(s=0):
    return torch.Generator().manual_seed(s)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return NumCodeModel()


@pytest.fixture(scope="module")
def groups2():
    """2 个 K=4 组 (共享场景/θ/候选集), 静态部分."""
    r = _rng(7)
    return [D.sample_group(r, 4) for _ in range(2)]


# ---------------------------------------------------------------- 遮蔽/保留桶
def test_split_mask_keep_partition():
    r = _rng(1)
    from symemerge.numcode.losses import sample_mask_cells
    sel = sample_mask_cells(64, r)
    rep, keep = TR.split_mask_keep(sel, r, keep_frac=0.15)
    assert bool((rep & keep).sum() == 0)
    assert bool(((rep | keep) == sel).all())
    frac = float(keep.sum()) / float(sel.sum())
    assert 0.10 < frac < 0.20          # ~15%, 大样本
    # 同种子确定性
    r1, r2 = _rng(9), _rng(9)
    a = TR.split_mask_keep(sel, r1, 0.15)
    b = TR.split_mask_keep(sel, r2, 0.15)
    assert bool((a[0] == b[0]).all()) and bool((a[1] == b[1]).all())
    # keep_frac=0 退化: replace == selected
    rep0, keep0 = TR.split_mask_keep(sel, _rng(3), 0.0)
    assert bool((rep0 == sel).all()) and bool(keep0.sum() == 0)


def test_mask_posterior_matches_bruteforce():
    """小 T 上与 math.comb 直接求和逐位核对 (独立式实现)."""
    T, lmax = 16, 12
    truth = torch.zeros(1, T, dtype=torch.long)
    truth[0, :5] = 1                       # 5 枚章
    rep = torch.zeros(1, T, dtype=torch.bool)
    rep[0, 3:9] = True                     # 遮 6 格, 可见 10 格, 其中 3 章可见
    q = float(TR.mask_posterior_q(truth, rep, lmax)[0])
    V = 10
    mv = 3                                 # truth[0, :3] 可见有章; 3..4 被遮
    M = T - V
    num, den = 0.0, 0.0
    for L in range(0, lmax + 1):
        if L < mv or T - L < V - mv:
            continue
        w = math.comb(L, mv) * math.comb(T - L, V - mv)
        num += w * (L - mv) / M
        den += w
    assert abs(q - num / den) < 1e-9


def test_mask_oracle_beats_blind_empty():
    """真实几何: oracle 期望准确率 >= 盲猜全空 (同一批评测画布上测)."""
    r = _rng(2)
    progs = [GM.sample_uniform_prog(r) for _ in range(32)]
    truth = GM.cell_truth(progs).view(32, GM.T)
    from symemerge.numcode.losses import sample_mask_cells
    sel = sample_mask_cells(32, r)
    rep, _ = TR.split_mask_keep(sel, r, 0.15)
    res = TR.mask_oracle(truth, rep, GM.l_max(GM.K_MAX))
    blind = float((truth[rep] == 0).float().mean())
    # oracle 只在期望意义下必胜盲猜 (预测"有章"的画布按期望 1/3 记分,
    # 单批实现值可小幅落后), 判据带小容差
    assert res["acc"] >= blind - 0.02
    assert 0.0 < res["acc"] < 1.0 and res["ce"] > 0.0


# ---------------------------------------------------------------- 空白零假设
def test_blank_nulls_exact_k4():
    """k=4 手推精确值: 训练域 {1,2,3,4}, τ~U{1..3}, p∈{3,5,7}, m∈{1,2,3}."""
    nu = TR.blank_nulls(4)
    assert abs(nu["t1"] - 2.0 / 3.0) < 1e-12
    assert abs(nu["t2"] - 0.5) < 1e-12
    assert abs(nu["t3"] - 0.5) < 1e-12
    assert abs(nu["t4"] - 1.0 / 3.0) < 1e-12
    assert abs(nu["t5"] - 0.25) < 1e-12       # 画布盲上界: 剔除 2 条同 Z 硬负
    assert abs(nu["t6"] - 0.25) < 1e-12
    nu128 = TR.blank_nulls(128)
    W = len(D.train_ns(128))
    assert abs(nu128["t6"] - 1.0 / W) < 1e-12
    assert all(0.0 < v < 1.0 for v in nu128.values())


def test_anneal_c():
    cfg = TR.TrainCfg(phase=2, k=4)
    assert abs(TR.anneal_c(cfg, 0) - cfg.c_hi) < 1e-9
    assert abs(TR.anneal_c(cfg, cfg.c_anneal_steps) - cfg.c_lo) < 1e-9
    assert abs(TR.anneal_c(cfg, 10 ** 9) - cfg.c_lo) < 1e-9
    mid = TR.anneal_c(cfg, cfg.c_anneal_steps // 2)
    assert cfg.c_lo < mid < cfg.c_hi


# ---------------------------------------------------------------- 组包与路由
def test_build_group_batch_shared_within_group(groups2):
    b = TR.build_group_batch(groups2, g=3, dev=torch.device("cpu"))
    assert b["x1"].shape[0] == 2 and b["cands"].shape[:2] == (2, D.N_CAND)
    for t in ("t1", "t2", "t3", "t4", "t6"):
        v = b["tg"][t].view(2, 3)
        assert bool((v == v[:, :1]).all())          # 组内冻结
    for kk in ("tau", "p", "m"):
        v = b["th"][kk].view(2, 3)
        assert bool((v == v[:, :1]).all())
    v = b["truth5"].view(2, 3)
    assert bool((v == v[:, :1]).all())
    assert b["ns_rep"].shape == (6,)


def _vary_accs(monkeypatch):
    """新生态下组内优势恒 0 (全 rollout 被 lmax 截断 -> 章数相同; 新生读者命中
    向量相同), 策略梯度自然为 0 -- 这是真实初态而非接线错. 测接线时注入确定性
    变化的命中向量 (0/1 交替), 保证组内优势非零."""
    real = TR.GR.task_accs

    def fake(tl, t5, tg, truth5):
        a = real(tl, t5, tg, truth5)
        alt = (torch.arange(a.shape[0]) % 2).float().unsqueeze(1)
        return alt.expand_as(a).contiguous()

    monkeypatch.setattr(TR.GR, "task_accs", fake)


def test_phase2_gradient_routing(net, groups2, monkeypatch):
    """装配级: GRPO 损失不进主干; 监督损失不进写者."""
    _vary_accs(monkeypatch)
    # conf_gate=0: 随机计数头置信度 ~1/128, 默认门 0.9 会把全部组过滤掉
    cfg = TR.TrainCfg(phase=2, k=4, n_groups=2, g=3, n_greedy=1, batch_mask=2,
                      conf_gate=0.0)
    sc, ns = D.sample_scene_batch(_rng(11), 4, 2)
    tw = TR.GR.TaskWeights()
    out = TR.phase2_losses(net, cfg, groups2, sc, ns, torch.device("cpu"),
                           _rng(12), _rng(13), step=0, tw=tw)
    l_sup = out["l_count"] + out["l_task"] + cfg.mu * out["l_read"]
    bb = [p for p in net.E.parameters() if p.requires_grad]
    wr = [p for p in net.writer.parameters() if p.requires_grad]
    gs = torch.autograd.grad(l_sup, wr, retain_graph=True, allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in gs), \
        "监督梯度泄进写者"
    gg = torch.autograd.grad(out["l_grpo"], bb, retain_graph=True,
                             allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in gg), \
        "策略梯度泄进主干"
    gw = torch.autograd.grad(out["l_grpo"], wr, retain_graph=True,
                             allow_unused=True)
    tot = sum(float(g.abs().sum()) for g in gw if g is not None)
    assert tot > 0.0, "写者没有拿到策略梯度"
    gb = torch.autograd.grad(l_sup, bb, allow_unused=True)
    tot = sum(float(g.abs().sum()) for g in gb if g is not None)
    assert tot > 0.0, "主干没有拿到监督梯度"


def test_conf_gate_zeroes_grpo(net, groups2, monkeypatch):
    """置信门=1.1 -> 全组被过滤, GRPO 对写者零梯度; 门=0 -> 有梯度."""
    _vary_accs(monkeypatch)
    sc, ns = D.sample_scene_batch(_rng(21), 4, 2)
    wr = [p for p in net.writer.parameters() if p.requires_grad]
    outs = {}
    for gate in (1.1, 0.0):
        cfg = TR.TrainCfg(phase=2, k=4, n_groups=2, g=3, n_greedy=1,
                          conf_gate=gate)
        out = TR.phase2_losses(net, cfg, groups2, sc, ns, torch.device("cpu"),
                               _rng(22), _rng(23), step=0,
                               tw=TR.GR.TaskWeights())
        g = torch.autograd.grad(out["l_grpo"], wr, allow_unused=True)
        outs[gate] = sum(float(x.abs().sum()) for x in g if x is not None)
    assert outs[1.1] == 0.0
    assert outs[0.0] > 0.0


# ---------------------------------------------------------------- 评测仪器
def test_eval_does_not_mutate_params(net):
    sc, ns = D.sample_scene_batch(_rng(31), 4, 8)
    before = [p.detach().clone() for p in net.parameters()]
    TR.eval_count(net, sc, ns, torch.device("cpu"))
    tn = [(D.sample_n(_rng(41 + i), 4), D.sample_theta(_rng(51 + i), 4))
          for i in range(16)]
    TR.blank_control(net, 4, tn, torch.device("cpu"))
    after = list(net.parameters())
    assert all(bool(torch.equal(a, b)) for a, b in zip(before, after))


def test_blank_control_on_random_net_within_null(net):
    """随机网络 + 空白画布: 各任务准确率不超过盲最优 + 4σ (泄漏警报不响)."""
    r = _rng(61)
    tn = [(D.sample_n(r, 4), D.sample_theta(r, 4)) for _ in range(256)]
    res = TR.blank_control(net, 4, tn, torch.device("cpu"))
    for t, acc in res["acc"].items():
        nu = res["null"][t]
        sig = math.sqrt(nu * (1 - nu) / len(tn))
        assert acc <= nu + 4 * sig + 1e-9, (t, acc, nu)
    assert res["alarm"] == []


def test_stream_firewall():
    st = TR.SceneStream(128, seed=5)
    it = iter(st)
    for _ in range(64):
        _, n = next(it)
        assert D.band_of(int(n)) == "train"
    gs = TR.GroupStream(4, seed=6)
    git = iter(gs)
    for _ in range(3):
        gr = next(git)
        assert gr["n"] in (1, 2, 3, 4)
        assert all(cn in (1, 2, 3, 4) for cn in gr["cand_ns"])


# ---------------------------------------------------------------- 码表统计
def test_table_stats_monotone_and_null():
    counts = {1: [1] * 8, 2: [2] * 8, 3: [3] * 8, 4: [4] * 8}
    st = TR.table_stats(counts, n_perm=500, seed=0)
    assert st["mono_frac"] == 1.0
    assert abs(st["slope"] - 1.0) < 1e-9 and st["r2"] > 0.999
    assert st["mono_p"] < 0.1                  # 整行置换零假设: 4 行全序仅识别
    #                                            排列达 1.0, 真 p = 1/24 ≈ 0.042
    assert st["median"] == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    assert st["rho"] == 1.0                    # F3 报数项: 全序表 Spearman = 1
    flat = {n: [3] * 8 for n in (1, 2, 3, 4)}
    stf = TR.table_stats(flat, n_perm=200, seed=0)
    assert stf["mono_frac"] == 0.0 and abs(stf["slope"]) < 1e-9
    assert stf["rho"] is None                  # 常数表 ρ 无定义, 不许报 0 冒充
    # 行内常数的阶跃表 = p3d 终局形态. 修正前 (打散全表单元格) 在这种表上把
    # mono_p 压到 ~0 (p3d 实测 slope .138/r2 .0023 却报 .000); 整行置换的正解
    # p = P(高行不在首位) = 3/4 -- 本 case 逮的就是这个伪显著
    step = {1: [0] * 8, 2: [0] * 8, 3: [0] * 8, 4: [13] * 8}
    sts = TR.table_stats(step, n_perm=400, seed=0)
    assert sts["mono_p"] > 0.5
    assert sts["rho"] is not None and sts["rho"] > 0


# ---------------------------------------------------------------- 检查点
def test_ckpt_roundtrip(tmp_path, net):
    cfg = TR.TrainCfg(phase=1, k=128)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    tw = TR.GR.TaskWeights()
    tw.ema += 0.25
    p = str(tmp_path / "ck.pt")
    TR.save_ckpt(p, net, opt, tw, step=123, cfg=cfg, best=0.5)
    m2 = NumCodeModel()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    tw2 = TR.GR.TaskWeights()
    meta = TR.load_ckpt(p, m2, opt2, tw2)
    assert meta["step"] == 123 and meta["best"] == 0.5
    assert bool(torch.equal(tw2.ema, tw.ema))
    for a, b in zip(net.parameters(), m2.parameters()):
        assert bool(torch.equal(a.detach(), b.detach()))
