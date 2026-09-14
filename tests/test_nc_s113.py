# tests/test_nc_s113.py
"""s113 四改动判别测试 (params.md §10d, [U] 2026-08-12): 稳定窗最优闩 / 并轨
记录器 / 结构性非法画布 ⊥ 监督 (P4 重定义) / K 稳定触发扩张 + extrap@0."""
import pytest
import torch

from symemerge.numcode import data as D
from symemerge.numcode import geometry as GM
from symemerge.numcode import grpo as GR
from symemerge.numcode import trainer as TR
from symemerge.numcode.model import NumCodeModel


# ---------------------------------------------------------------- l_max 接线
def test_l_max_of_takes_max_of_override_and_formula():
    # K<=16: 覆写 40 是地板 (与 p2e 现制逐字同); K>=32: 公式接管 (spec §3.2 l_max>=K)
    assert TR.l_max_of(TR.TrainCfg(k=4, l_max_override=40)) == 40
    assert TR.l_max_of(TR.TrainCfg(k=16, l_max_override=40)) == 40
    assert TR.l_max_of(TR.TrainCfg(k=32, l_max_override=40)) == GM.l_max(32)
    assert TR.l_max_of(TR.TrainCfg(k=4, l_max_override=0)) == GM.l_max(4)


# ---------------------------------------------------------------- 非法画布
def test_illegal_prog_structurally_unreachable():
    """§10d 改3 构造论证: 墨章数下界 = l_max+occ_k+8, 经 occ_k 遮挡后可见章数仍
    严格 > l_max -- 任何合法码字的腐蚀轨道到不了."""
    rng = torch.Generator().manual_seed(0)
    lmax, occ = 40, 48
    for _ in range(64):
        prog = GM.sample_illegal_prog(rng, lmax, occ)
        cells = {a // GM.S for a in prog}
        assert len(cells) == len(prog)            # 格不放回
        assert 96 <= len(cells) <= 192            # lo=40+48+8, hi=min(248, lo+96)
    prog = GM.sample_illegal_prog(rng, lmax, occ)
    x = GM.raster([prog])
    GM.occlude_k(x, occ, rng)
    v = x.view(1, GM.G, GM.P, GM.G, GM.P)
    inked = int((v.amax(dim=(2, 4)) > 0.5).sum())
    assert inked > lmax


def test_illegal_prog_asserts_when_capacity_exhausted():
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(AssertionError):
        GM.sample_illegal_prog(rng, 200, 48)      # lo=256 顶满画布, 构造失效


def test_ood_bot_loss_routes_to_read_head_only():
    """结构断言 (与被删的空白版同构): ⊥ 监督只达直读头与画布主干; 任务头/计数头/
    写者不可达."""
    torch.manual_seed(0)
    m = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, s=0.15, occ_k=12, ood_bot=4,
                      l_max_override=40)
    loss = TR.ood_bot_loss(m, cfg, torch.Generator().manual_seed(1),
                           torch.device("cpu"))
    loss.backward()
    g_read = sum(float(p.grad.abs().sum()) for p in m.heads.read.parameters()
                 if p.grad is not None)
    assert g_read > 0.0                        # 新学习路径非零梯度断言
    g_bb = sum(float(p.grad.abs().sum()) for p in m.E.parameters()
               if p.grad is not None)
    assert g_bb == 0.0                         # 主干 detach (p3a 实测: 稠密画布经
    #   共享主干的梯度冲击 500 步内搅坏合法读音流形, 读者先坏写者随崩; §10d 修正案)
    assert m.heads.t1.weight.grad is None      # 任务头结构上不可达
    assert m.heads.count[0].weight.grad is None
    assert m.writer.head.weight.grad is None


def test_ensure_ood_progs_migration_and_idempotent():
    cfg = TR.TrainCfg(phase=2, k=4, occ_k=48, l_max_override=40, seed=3)
    es = {"groups": []}
    TR.ensure_ood_progs(es, cfg, cache=None)
    assert len(es["ood_progs"]) == 128
    for pr in es["ood_progs"][:8]:
        assert len({a // GM.S for a in pr}) >= 96
    es2 = {"groups": [], "ood_progs": ["sentinel"]}
    TR.ensure_ood_progs(es2, cfg)
    assert es2["ood_progs"] == ["sentinel"]      # 已有则不动 (CRN 幂等)


def test_ood_abstain_range():
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, s=0.15, occ_k=12, seed=0,
                      l_max_override=40)
    rng = torch.Generator().manual_seed(777)
    es = {"ood_progs": [GM.sample_illegal_prog(rng, 40, 12) for _ in range(4)]}
    r = TR.ood_abstain(model, es, cfg, torch.device("cpu"))
    assert 0.0 <= r <= 1.0


def test_ood_capacity_guard_at_top_k():
    # 评审 Critical#1: K=128 档 lo=256>hi=248, 必须停表而非断言崩溃
    assert GM.illegal_capacity_ok(40, 48) is True
    assert GM.illegal_capacity_ok(200, 48) is False
    cfg = TR.TrainCfg(phase=2, k=128, occ_k=48, l_max_override=40, seed=3)
    es = {"groups": []}
    TR.ensure_ood_progs(es, cfg)               # 不崩
    assert es["ood_progs"] == []
    assert TR.ood_abstain(None, es, cfg, torch.device("cpu")) is None


# ---------------------------------------------------------------- 改1 并轨记录器
def test_collapse_flag_truth_table():
    assert TR.collapse_flag(1, 0.80) is True
    assert TR.collapse_flag(0, 0.84) is True
    assert TR.collapse_flag(1, 0.85) is False    # 成绩线严格小于
    assert TR.collapse_flag(2, 0.50) is False    # 间距未到并轨临界


# ---------------------------------------------------------------- 改2 稳定窗闩
def test_best_latch_stability_window():
    lt = TR.BestLatch()
    assert lt.offer(1.0, 500) is False           # 首评无前评, 占不了闩位
    assert lt.offer(0.5, 1000) is True           # 首个合格对子即锁存, 闩值取较小者
    assert lt.best == 0.5                        #   -- 尖峰 1.0 没占到闩位
    assert lt.offer(0.99, 1500) is False         # pair = min(0.5, 0.99) 不超现闩
    assert lt.offer(0.98, 2000) is True          # 连续两评 >= 0.98
    assert lt.best == 0.98 and lt.step == 2000
    assert lt.offer(1.0, 2500) is False          # pair=0.98 不超现闩
    assert lt.offer(1.0, 3000) is True           # 连续两评 1.0
    assert lt.best == 1.0
    lt.reset()
    assert lt.best == -1.0 and lt.prev is None
    assert lt.offer(1.0, 3500) is False          # 重置后重新要两评
    assert lt.offer(1.0, 4000) is True           # 新档首个稳定对子重新锁存
    assert lt.best == 1.0


# ---------------------------------------------------------------- 改4 扩张
def test_expand_ctl_streak_semantics():
    """(2026-08-17 改动 3 后 offer 返回 None / "stable" / "timer"; every=0 = 旧制仅稳定触发)"""
    x = TR.ExpandCtl()
    assert x.offer(0.96, 1.0, False) is None
    assert x.offer(0.96, 1.0, False) == "stable"   # 连续两评触发
    assert x.offer(0.96, 1.0, False) is None       # 触发后清零重计
    x2 = TR.ExpandCtl()
    assert x2.offer(0.96, 3.0, False) is None      # v 超线
    assert x2.offer(0.96, None, False) is None     # 首评无 v
    assert x2.offer(0.96, 1.0, True) is None       # 回滚/过渡评不作证据
    assert x2.offer(0.96, 1.0, False) is None      # 从零重计
    assert x2.offer(0.94, 1.0, False) is None      # 成绩不到线清零
    # every=0 时给了 step/last_expand 也不定时推进 (旧制不变)
    assert x2.offer(0.5, 9.0, False, step=99999, last_expand=0) is None


def test_eval_disturbed_truth_table():
    assert TR.eval_disturbed(None) is False
    assert TR.eval_disturbed({}) is False
    assert TR.eval_disturbed({"transition": True}) is True
    assert TR.eval_disturbed({"event": "reset_fast"}) is True
    assert TR.eval_disturbed({"event": "accept_current"}) is True
    assert TR.eval_disturbed({"event": "refresh"}) is False


def test_do_expand_order_and_bookkeeping(tmp_path, monkeypatch):
    """§10d 改4 防火墙顺序: extrap@0 在 cfg.k 更新后测 (l_max 按新档), 新 N 集合
    正确; 闩重置且旧档 best 另存; 锚进宽限期且历史最优快照作废."""
    calls = {}

    def fake_extrap(model, cfg, new_ns, dev, seed, m=32):
        calls["new_ns"] = list(new_ns)
        calls["k_at_measure"] = cfg.k
        return dict(per_n={}, mean=0.0)

    monkeypatch.setattr(TR, "measure_extrap0", fake_extrap)
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, l_max_override=40, occ_k=48,
                      out=str(tmp_path), anchor=1, beta=0.005, k_expand=1)
    ctl = TR.AnchorCtl(model, 0.005)
    ctl.tpl, ctl.score, ctl.grace_done = {}, 0.9, True
    ctl.best = ({}, {}, 0.9)
    latch = TR.BestLatch()
    latch.best, latch.step, latch.prev = 0.99, 1500, 0.99
    (tmp_path / "ckpt_best.pt").write_bytes(b"x")
    ex = TR.do_expand(model, cfg, ctl, latch, 2000, torch.device("cpu"))
    assert cfg.k == 6 and ex["k"] == [4, 6]      # s114 C5: 步长 +4 -> +2
    assert calls["new_ns"] == [5, 6]
    assert calls["k_at_measure"] == 6
    assert ex["prev_best"]["score"] == 0.99
    assert latch.best == -1.0 and latch.prev is None
    assert (tmp_path / "ckpt_best_k4.pt").exists()
    assert ctl.best is None and ctl.grace_done is False
    assert ctl.grace_until == 2000 + TR.ANCHOR_GRACE
    # 宽限期语义: 糟糕评测也不触发任何闸事件
    out = ctl.on_eval(model, 2500, 0.1, 0, 99.0,
                      {n: [0] * GM.T for n in range(1, 9)}, None)
    assert out.get("grace") is True and "event" not in out
    # 宽限期末: 当前策略立新锚
    out = ctl.on_eval(model, 4000, 0.5, 0, 99.0,
                      {n: [0] * GM.T for n in range(1, 9)}, None)
    assert out.get("event") == "grace_anchor"


def test_anchor_state_roundtrip_with_grace_until():
    torch.manual_seed(0)
    model = NumCodeModel()
    ctl = TR.AnchorCtl(model, 0.005)
    ctl.kl_tgt = 1.0
    ctl.calib = [(1.0, 1.0)]
    ctl.on_expand(3000)
    assert ctl.kl_tgt is None and ctl.calib == []
    st = ctl.state()
    ctl2 = TR.AnchorCtl(model, 0.005)
    ctl2.load_state(st)
    assert ctl2.grace_until == 3000 + TR.ANCHOR_GRACE
    st.pop("grace_until")                        # 旧检查点缺新键 -> 保默认
    ctl3 = TR.AnchorCtl(model, 0.005)
    ctl3.load_state(st)
    assert ctl3.grace_until == TR.ANCHOR_GRACE


def test_measure_extrap0_smoke():
    torch.manual_seed(0)
    model = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=8, l_max_override=40, s=0.15, occ_k=12)
    out = TR.measure_extrap0(model, cfg, [5, 6], torch.device("cpu"),
                             seed=7, m=2)
    assert set(out["per_n"]) == {5, 6}
    for v in out["per_n"].values():
        assert 0.0 <= v["read"] <= 1.0 and v["stamps_med"] >= 0.0
    assert 0.0 <= out["mean"] <= 1.0


# ---------------------------------------------------------------- 改6 检查点
def test_ckpt_best_dict_roundtrip(tmp_path):
    # 评审 Important#6: best 以 {score, step} 入检查点, resume 双点口径不丢 step
    torch.manual_seed(0)
    m = NumCodeModel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    tw = GR.TaskWeights()
    cfg = TR.TrainCfg(phase=2, k=4)
    p = str(tmp_path / "ck.pt")
    TR.save_ckpt(p, m, opt, tw, 1500, cfg, dict(score=0.9, step=1500))
    meta = TR.load_ckpt(p, m)
    assert meta["best"] == {"score": 0.9, "step": 1500}
