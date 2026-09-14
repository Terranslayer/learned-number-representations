# tests/test_nc_s114.py
"""s114 改动判别测试 (params.md §10e, [U] s114 计划 C1-C7): 奖励探针项 /
双探针 (probe_in 岭闭式入环 + probe_out 单隐层体外) / 扩张步长 +2 /
并轨记录器连续两评限定 / 探针检查点往返."""
import torch

from symemerge.numcode import data as D
from symemerge.numcode import geometry as GM
from symemerge.numcode import grpo as GR
from symemerge.numcode import probe as PB
from symemerge.numcode import trainer as TR
from symemerge.numcode.model import NumCodeModel


def _pp(seed=0):
    return PB.ProbePair(TR.TrainCfg(seed=seed), torch.device("cpu"))


_P1 = [GM.aid_pack(17, 2)]                       # 两个可分"码字"
_P2 = [GM.aid_pack(120, 1), GM.aid_pack(200, 1)]


# ---------------------------------------------------------------- C1 奖励项
def test_reward_probe_term_exact_and_backcompat():
    torch.manual_seed(0)
    accs = torch.rand(6, 6)
    w = torch.full((6,), 1.0 / 6)
    n = torch.tensor([0, 1, 2, 3, 4, 5])
    base = GR.reward(accs, w, n, 0.005, 0.01)
    h = torch.rand(6)
    r = GR.reward(accs, w, n, 0.005, 0.01, probe_hits=h, alpha=0.1)
    assert torch.allclose(r, base + 0.1 * h)     # r_probe 只加 α 倍, 无其它变形
    r0 = GR.reward(accs, w, n, 0.005, 0.01, probe_hits=None, alpha=0.1)
    assert torch.allclose(r0, base)              # 无探针 = s113 旧制逐位一致


# ---------------------------------------------------------------- C2 probe_in
def test_probe_cold_start_returns_zeros():
    pp = _pp()
    x = torch.zeros(8, GM.SIDE, GM.SIDE)
    ns = torch.ones(8, dtype=torch.long)
    h = pp.reward_margin([x], ns, 4)
    assert torch.all(h == 0.0)                   # 缓冲不足: 组内常数, 优势零效应
    assert pp.sigma is None                      # 冷启动步不更新 σ


def test_probe_in_margin_on_separable_codes():
    """v3 边际制 ([U] p3c-alpha2): 可分码真类归一边际强正, 错标签强负
    (margin 可负 = 读成别人负酬); m 抽均值路径同验; σ EMA 有值."""
    pp = _pp()
    rng = torch.Generator().manual_seed(0)
    for _ in range(6):                           # 6×128 = 768 >= MIN_FIT
        x = GM.render_channel([_P1] * 64 + [_P2] * 64, 0.15, rng, 4)
        pp.push(x, torch.tensor([1] * 64 + [2] * 64))
    xa = GM.render_channel([_P1] * 64 + [_P2] * 64, 0.15, rng, 4)
    xb = GM.render_channel([_P1] * 64 + [_P2] * 64, 0.15, rng, 4)
    ns = torch.tensor([1] * 64 + [2] * 64)
    m = pp.reward_margin([xa], ns, 4)
    assert float(m.mean()) > 1.0                 # 真类 margin/σ 强正
    assert pp.sigma is not None and pp.sigma > 0.0
    wrong = torch.tensor([2] * 64 + [1] * 64)
    mw = pp.reward_margin([xa], wrong, 4)
    assert float(mw.mean()) < -1.0               # 读成别人 = 负酬
    mm = pp.reward_margin([xa, xb], ns, 4)       # m=2 均值路径
    assert float(mm.mean()) > 1.0


# ---------------------------------------------------------------- C3 probe_out
def test_probe_out_trains_and_is_outside_reward_path():
    pp = _pp()
    rng = torch.Generator().manual_seed(1)
    x = GM.render_channel([_P1] * 32 + [_P2] * 32, 0.15, rng, 4)
    ns = torch.tensor([1] * 32 + [2] * 32)
    for _ in range(10):                          # 640 样本, 越过拟合下限
        pp.step_update(x, ns)
    before = [p.detach().clone() for p in pp.out.parameters()]
    pp.step_update(x, ns)
    delta = sum(float((a - b).abs().sum())
                for a, b in zip(pp.out.parameters(), before))
    assert delta > 0.0                           # 非零更新断言 (新学习部件法则)
    sig0 = pp.sigma
    h1 = pp.reward_margin([x], ns, 4)
    with torch.no_grad():
        for p in pp.out.parameters():
            p.fill_(123.0)                       # 毁掉 probe_out
    pp.sigma = sig0                              # σ 对齐 (它随每次调用演化)
    h2 = pp.reward_margin([x], ns, 4)
    assert torch.equal(h1, h2)                   # 永不进奖励: 同缓冲同分毫不差


def test_probe_eval_acc_smoke():
    pp = _pp()
    rng = torch.Generator().manual_seed(5)
    tab = {1: [_P1] * 4, 2: [_P2] * 4}
    cfg = TR.TrainCfg(phase=2, k=4, s=0.15, occ_k=4)
    out = pp.eval_acc(tab, cfg, rng, torch.device("cpu"))
    assert out["pin"] is None and out["buf_n"] == 0   # 冷启动停表不崩
    assert 0.0 <= out["pout"] <= 1.0


# ---------------------------------------------------------------- 检查点
def test_probe_state_roundtrip():
    pp = _pp()
    rng = torch.Generator().manual_seed(2)
    x = GM.render_channel([_P1] * 64, 0.15, rng, 4)
    ns = torch.ones(64, dtype=torch.long)
    for _ in range(9):
        pp.step_update(x, ns)
    st = pp.state()
    pp2 = _pp(seed=99)                           # 不同种子起点, 全由 state 覆盖
    pp2.load_state(st)
    assert pp2.n == pp.n and pp2.ptr == pp.ptr and pp2.sigma == pp.sigma
    assert torch.equal(pp.reward_margin([x], ns, 4),
                       pp2.reward_margin([x], ns, 4))
    assert all(torch.equal(a.detach(), b.detach()) for a, b in
               zip(pp.out.parameters(), pp2.out.parameters()))


def test_ckpt_probe_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = NumCodeModel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    tw = GR.TaskWeights()
    cfg = TR.TrainCfg(phase=2, k=4, alpha=0.1)
    pp = PB.ProbePair(cfg, torch.device("cpu"))
    rng = torch.Generator().manual_seed(3)
    pp.push(GM.render_channel([_P1] * 64, 0.15, rng, 4),
            torch.ones(64, dtype=torch.long))
    p = str(tmp_path / "ck.pt")
    TR.save_ckpt(p, m, opt, tw, 100, cfg, dict(score=0.5, step=100), probe=pp)
    meta = TR.load_ckpt(p, m)                    # weights_only 路径可读
    pp2 = PB.ProbePair(cfg, torch.device("cpu"))
    pp2.load_state(meta["probe_sd"])
    assert pp2.n == 64 and pp2.ptr == 64


# ---------------------------------------------------------------- C5 扩张 +2
def test_expand_step_plus2_sequence(tmp_path, monkeypatch):
    calls = []

    def fake_extrap(model, cfg, new_ns, dev, seed, m=32):
        calls.append((list(new_ns), cfg.k))
        return dict(per_n={}, mean=0.0)

    monkeypatch.setattr(TR, "measure_extrap0", fake_extrap)
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, l_max_override=40, occ_k=48,
                      out=str(tmp_path))
    latch = TR.BestLatch()
    assert TR.EXPAND_STEP == 2
    ex1 = TR.do_expand(model, cfg, None, latch, 1500, torch.device("cpu"))
    assert ex1["k"] == [4, 6] and cfg.k == 6 and calls[0] == ([5, 6], 6)
    ex2 = TR.do_expand(model, cfg, None, latch, 3000, torch.device("cpu"))
    assert ex2["k"] == [6, 8] and cfg.k == 8 and calls[1] == ([7, 8], 8)


# ---------------------------------------------------------------- v2 仪表
def test_read_per_n_and_reset_head_smoke():
    """逐 N 直读 (P1-P4 判读) 与 C3b 重置读头 (记录项) 冒烟: 值域合法,
    模型参数不被仪表改动 (探针局部)."""
    torch.manual_seed(0)
    m = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, s=0.15, occ_k=4)
    tab = {1: [_P1] * 4, 2: [_P2] * 4}
    before = [p.detach().clone() for p in m.parameters()]
    rn = TR.read_per_n(m, tab, cfg, torch.Generator().manual_seed(6),
                       torch.device("cpu"))
    assert set(rn) == {1, 2} and all(0.0 <= v <= 1.0 for v in rn.values())
    rh = TR.reset_head_probe(m, tab, cfg, torch.device("cpu"), seed=7,
                             steps=50, r_train=2, r_test=1, bs=16)
    assert 0.0 <= rh["plateau"] <= 1.0 and len(rh["curve"]) == 2
    after = list(m.parameters())
    assert all(bool(torch.equal(a, b)) for a, b in zip(before, after))


# ---------------------------------------------------------------- 定居仪表
def test_settle_ctl_streak_and_reset():
    s = TR.SettleCtl()
    assert s.offer(-1) is False                  # 首评无 shift (哨兵 -1)
    assert s.offer(0) is False
    assert s.offer(0) is False
    assert s.offer(0) is True                    # 连续三评零变动 = settled
    assert s.offer(0) is True                    # 持续保持
    assert s.offer(2) is False                   # 码动了: 清零
    s2 = TR.SettleCtl()
    s2.offer(0)
    s2.offer(0)
    s2.reset()                                   # 扩张清零
    assert s2.offer(0) is False


# ---------------------------------------------------------------- C7 并轨限定
def test_collapse_ctl_streak_and_reset():
    c = TR.CollapseCtl()
    assert c.offer(1, 0.5) is False              # 首次成立: 只累计
    assert c.offer(1, 0.5) is True               # 连续两评成立: 置位
    assert c.offer(1, 0.5) is True               # 持续成立: 保持
    assert c.offer(2, 0.5) is False              # 条件破: 清零
    assert c.offer(1, 0.5) is False
    c2 = TR.CollapseCtl()
    c2.offer(1, 0.5)
    c2.reset()                                   # 扩张时清零 (受扰评不作证据)
    assert c2.offer(1, 0.5) is False


# ---------------------------------------------------------------- 装配级
def test_phase2_losses_probe_wiring_smoke():
    """alpha>0 + probe: met 带 r_probe (冷启动 0.0), draw0 入册 B*g 条,
    全损失仍可反传 (探针不进模型梯度图)."""
    dev = torch.device("cpu")
    torch.manual_seed(0)
    m = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, n_groups=2, g=3, n_greedy=1, batch_mask=2,
                      conf_gate=0.0, alpha=0.1, s=0.15, occ_k=4)
    pp = PB.ProbePair(cfg, dev)
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    out = TR.phase2_losses(m, cfg, groups, sc, ns, dev,
                           torch.Generator().manual_seed(3),
                           torch.Generator().manual_seed(4), 0,
                           GR.TaskWeights(), probe=pp)
    met = out["metrics"]
    assert met["r_probe"] == 0.0                 # 冷启动: 缓冲不足全零
    assert pp.n == 6                             # draw0 入册 = n_groups*g
    tot = (out["l_grpo"] + out["l_task"] + out["l_count"]
           + 0.05 * out["l_read"])
    tot.backward()
