# tests/test_nc_s111.py
"""s111 第二修复轮四改动 + s111b 五条闸逻辑修正的判别测试 (params.md §10):
固定锚状态机 (宽限期建锚/合取快速闸/同锚回滚封顶改收编/强制刷新移出回滚路径/
β 校准始终运行+加压乘性衰减) / KL token 平均 / 奖励 m 次独立腐蚀均值 (draw0 与
监督共源) / 空白画布 -> ⊥ 梯度路由 / 位移仪表 / 检查点含锚状态往返."""
import math

import pytest
import torch

import symemerge.numcode.data as D
import symemerge.numcode.geometry as GM
import symemerge.numcode.grpo as GR
import symemerge.numcode.trainer as TR
from symemerge.numcode.model import NumCodeModel


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_cfg(**kw):
    base = dict(phase=2, k=4, n_groups=2, g=2, n_greedy=1, s=0.15,
                occ_k=12, soft_gate=1, reader_lag=0.999, beta=0.02,
                lam=0.02, l_max_override=10)
    base.update(kw)
    return TR.TrainCfg(**base)


def _base_tpl():
    return {n: [0] * GM.T for n in (1, 2, 3, 4)}


def _far_tpl(cells=25):
    return {n: [1] * cells + [0] * (GM.T - cells) for n in (1, 2, 3, 4)}


def _armed_ctl(m, score=0.95):
    """宽限期已收口、锚已建档的状态机 (单测直接从受控运行域开始)."""
    ctl = TR.AnchorCtl(m, 0.02)
    ctl.tpl = _base_tpl()
    ctl.score = score
    ctl.grace_done = True
    return ctl


# ================================================================ KL token 平均
def test_k3_kl_token_mean_removes_length_bias():
    # 两条 rollout 每 token KL 相同, 长度 1 vs 5: token 平均下贡献相等
    # (序列求和的旧制会给长程序 5 倍惩罚 = 未登记的长度压力, §10 [U])
    lp = torch.zeros(2, 5)
    lp_ref = torch.full((2, 5), -0.5)
    keep = torch.tensor([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    c = math.exp(-0.5) + 0.5 - 1.0
    assert abs(float(GR.k3_kl(lp, lp_ref, keep)) - c) < 1e-6


# ================================================================ 宽限期 (s111b#2)
def test_anchor_ctl_grace_holds_fire_then_files_new_anchor():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = TR.AnchorCtl(m, 0.02)
    ctl.tpl = _base_tpl()                      # 发射建档 (旧锚)
    ctl.score = 0.99
    w0 = m.writer.head.weight.clone()
    far = _far_tpl(200)                        # D 中位 200, 成绩 0.3: 平时必回滚
    o = ctl.on_eval(m, 500, 0.3, 1, 50.0, far, 0.5)
    assert o.get("grace") and "event" not in o
    assert torch.equal(m.writer.head.weight, w0)       # 宽限期内绝不回滚
    assert ctl.calib == []                             # 冲击数据不进校准散点
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    new = _far_tpl(8)
    o2 = ctl.on_eval(m, TR.ANCHOR_GRACE, 0.97, 9, 1.0, new, 0.1)
    assert o2["event"] == "grace_anchor"               # 期末稳定点设为新锚
    assert torch.equal(ctl.ref.head.weight, m.writer.head.weight)
    assert ctl.tpl == new and ctl.score == 0.97
    assert ctl.grace_done and ctl.last_refresh == TR.ANCHOR_GRACE


# ================================================================ 合取快速闸 (s111b#1)
def test_anchor_ctl_conjunction_spares_good_restructure():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    w0 = m.writer.head.weight.clone()
    # p2c 评499 的情形: D=25 > 20 但成绩 0.97 ≥ 0.85 且 dmin 10 ≥ 6 -> 不回滚
    o = ctl.on_eval(m, 3000, 0.97, 10, 1.0, _far_tpl(25), None)
    assert "event" not in o and o["D"]["med"] == 25.0
    assert torch.equal(m.writer.head.weight, w0)
    assert ctl.streak_ok == 0                  # 大位移照旧挡慢刷新 (不追认)


@pytest.mark.parametrize("score,dmin", [
    (0.7, 10),        # 成绩 < 0.85
    (0.97, 2),        # dmin < 6
])
def test_anchor_ctl_fast_reset_needs_bad_quality(score, dmin):
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    w_anchor = ctl.ref.head.weight.clone()
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    o = ctl.on_eval(m, 3000, score, dmin, 1.0, _far_tpl(25), None)
    assert o["event"] == "reset_fast"
    assert torch.equal(m.writer.head.weight, w_anchor)
    assert ctl.resets_since_refresh == 1


def test_anchor_ctl_reset_clears_writer_opt_state_and_boost_decays():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for p in m.writer.parameters():            # 制造非空优化器动量
        p.grad = torch.zeros_like(p)
    opt.step()
    assert any(p in opt.state for p in m.writer.parameters())
    o = ctl.on_eval(m, 3000, 0.7, 10, 1.0, _far_tpl(25), None, opt=opt)
    assert o["event"] == "reset_fast"
    assert not any(p in opt.state for p in m.writer.parameters())  # 动量已清
    # s111b#3: 加压乘性衰减 3 -> 1, 尺度 ANCHOR_BOOST_STEPS, 无悬崖
    assert ctl.beta_eff(3000) == pytest.approx(0.02 * TR.ANCHOR_BOOST)
    mid = 0.02 * TR.ANCHOR_BOOST ** 0.5        # 半程 = 3^0.5 倍
    assert ctl.beta_eff(3000 + TR.ANCHOR_BOOST_STEPS // 2) == pytest.approx(
        mid, rel=1e-3)
    assert ctl.beta_eff(3000 + TR.ANCHOR_BOOST_STEPS) == pytest.approx(
        0.02, rel=1e-4)
    assert ctl.beta_eff(3000 + 5 * TR.ANCHOR_BOOST_STEPS) == pytest.approx(
        0.02)                                  # 衰减夹在 1, 不掉穿基准
    o2 = ctl.on_eval(m, 3100, 0.1, 0, 99.0, _far_tpl(25), None)
    assert o2.get("transition") and "event" not in o2  # 过渡期不参与任何判据


def test_anchor_ctl_fallback_reset_three_below():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m, score=0.95)
    base = _base_tpl()
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    w_anchor = ctl.ref.head.weight.clone()
    assert "event" not in ctl.on_eval(m, 500, 0.75, 10, 1.0, base, None)
    assert "event" not in ctl.on_eval(m, 1000, 0.75, 10, 1.0, base, None)
    o = ctl.on_eval(m, 1500, 0.75, 10, 1.0, base, None)   # 连续三评 < 0.95-0.15
    assert o["event"] == "reset_fall"
    assert torch.equal(m.writer.head.weight, w_anchor)
    ctl2 = _armed_ctl(m, score=0.95)
    ctl2.on_eval(m, 500, 0.75, 10, 1.0, base, None)
    ctl2.on_eval(m, 1000, 0.92, 10, 1.0, base, None)      # 单次回升即断连
    assert ctl2.streak_below == 0


# ================================================================ 同锚封顶 (s111b#5)
def test_anchor_ctl_third_gate_fire_accepts_current():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    ctl.resets_since_refresh = TR.ANCHOR_RESET_CAP     # 同锚已连滚两次
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    w_cur = m.writer.head.weight.clone()
    cur = _far_tpl(25)
    o = ctl.on_eval(m, 5000, 0.7, 10, 1.0, cur, None)  # 第三次触发
    assert o["event"] == "accept_current"              # 不回滚, 收编为新锚
    assert torch.equal(m.writer.head.weight, w_cur)    # 写者原地不动
    assert torch.equal(ctl.ref.head.weight, w_cur)     # 锚 = 当前策略
    assert ctl.tpl == cur and ctl.score == 0.7
    assert ctl.resets_since_refresh == 0


# ================================================================ 慢刷新 (语义不变)
def test_anchor_ctl_slow_refresh_needs_streak_and_tmin():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    base = _base_tpl()
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)          # 当前写者 != 锚
    o1 = ctl.on_eval(m, 500, 0.95, 10, 1.0, base, None)
    assert "event" not in o1 and ctl.streak_ok == 1
    o2 = ctl.on_eval(m, 1000, 0.95, 10, 1.0, base, None)
    assert "event" not in o2 and ctl.streak_ok == 2   # 连续两评但 T_min 未到: 持有
    o3 = ctl.on_eval(m, 2000, 0.96, 10, 1.0, base, None)
    assert o3["event"] == "refresh"
    assert torch.equal(ctl.ref.head.weight, m.writer.head.weight)
    assert ctl.score == 0.96 and ctl.streak_ok == 0


@pytest.mark.parametrize("score,dmin,v", [
    (0.85, 10, 1.0),      # 条件1 破: 成绩 < 0.90 绝对下限
    (0.95, 7, 1.0),       # 条件2 破: dmin < 8
    (0.95, 10, 3.5),      # 条件3a 破: v > 3
])
def test_anchor_ctl_any_condition_breaks_streak(score, dmin, v):
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    base = _base_tpl()
    ctl.on_eval(m, 500, 0.95, 10, 1.0, base, None)
    assert ctl.streak_ok == 1
    o = ctl.on_eval(m, 1000, score, dmin, v, base, None)
    assert ctl.streak_ok == 0 and "event" not in o


def test_anchor_ctl_mid_D_blocks_refresh_without_reset():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    # D 中位 13: 破刷新条件3b (>12) 但不过快速闸位移项 (<=20)
    o = ctl.on_eval(m, 500, 0.95, 10, 1.0, _far_tpl(13), None)
    assert o["D"]["med"] == 13.0
    assert "event" not in o and ctl.streak_ok == 0


# ================================================================ 强制刷新 (s111b#4)
def test_anchor_ctl_forced_refresh_goes_to_best_not_current():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    torch.manual_seed(7)
    best_m = NumCodeModel()                    # 权重与 m 不同
    ctl.note_best(best_m, {n: [2] * GM.T for n in (1, 2, 3, 4)}, 0.99)
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    o = ctl.on_eval(m, 5500, 0.95, 10, 9.0, _base_tpl(), None)  # 超 T_max
    assert o["event"] == "refresh_forced" and o["forced_to"] == "best"
    assert torch.equal(ctl.ref.head.weight, best_m.writer.head.weight)
    assert not torch.equal(ctl.ref.head.weight, m.writer.head.weight)
    assert ctl.score == 0.99 and ctl.tpl[1][0] == 2


def test_anchor_ctl_forced_refresh_not_blocked_by_reset():
    """s111b#4 判别: p2c 死锁 = 回滚早返回挡死强制刷新. 现在同评先强制刷新
    (锚 -> 历史最优), 再按新锚判闸 -> 回滚落到新锚上."""
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    torch.manual_seed(7)
    best_m = NumCodeModel()
    ctl.note_best(best_m, {n: [2] * GM.T for n in (1, 2, 3, 4)}, 0.99)
    with torch.no_grad():
        m.writer.head.weight.add_(1.0)
    # 超 T_max 且当前质量差 (成绩 0.5, 对新锚 D 也必大): 两个动作同评发生
    o = ctl.on_eval(m, 5500, 0.5, 10, 9.0, _far_tpl(25), None)
    assert o["event"] == "refresh_forced+reset_fast"
    assert torch.equal(ctl.ref.head.weight, best_m.writer.head.weight)
    assert torch.equal(m.writer.head.weight, best_m.writer.head.weight)
    assert ctl.resets_since_refresh == 1       # 对新锚的第一次回滚


# ================================================================ β 校准/自适应 (s111b#3)
def test_anchor_ctl_beta_calibration_window_after_grace():
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    base = _base_tpl()
    g = TR.ANCHOR_GRACE
    ctl.on_eval(m, g + 500, 0.95, 10, None, base, 0.008)   # 首评无 v: 不进散点
    assert ctl.kl_tgt is None and ctl.calib == []
    ctl.on_eval(m, g + 1000, 0.95, 10, 1.5, base, 0.010)   # v<=2 入选
    ctl.on_eval(m, g + 1500, 0.95, 10, 4.0, base, 0.030)   # v>2 落选
    o = ctl.on_eval(m, g + 2000, 0.95, 10, 2.0, base, 0.012)  # 到窗: tgt=max(入选)
    assert o["kl_tgt"] == pytest.approx(0.012)
    ctl.on_eval(m, g + 2500, 0.95, 10, 1.0, base, 0.030)   # > 1.5*tgt -> ×2
    assert ctl.beta == pytest.approx(0.04)
    ctl.on_eval(m, g + 3000, 0.95, 10, 1.0, base, 0.004)   # < tgt/1.5 -> ÷2
    assert ctl.beta == pytest.approx(0.02)
    for i in range(20):                                     # 夹取下限
        ctl.on_eval(m, g + 3500 + i * 500, 0.95, 10, 9.9, base, 1e-9)
    assert ctl.beta == pytest.approx(TR.BETA_MIN)


def test_anchor_ctl_beta_adapts_even_during_boost():
    """s111b#3 判别: p2c 死锁 = 每次回滚重挂加压窗把校准/自适应无限顺延.
    现在自适应始终运行, 加压只是乘性衰减的倍率."""
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    ctl.kl_tgt = 0.01
    ctl.on_eval(m, 3000, 0.7, 10, 1.0, _far_tpl(25), None)  # 触发回滚, 加压启动
    b0 = ctl.beta
    o = ctl.on_eval(m, 3500, 0.95, 10, 1.0, _base_tpl(), 10.0)  # 加压未衰完
    assert ctl.beta == pytest.approx(min(b0 * 2, TR.BETA_MAX))  # 照样自适应


# ================================================================ s112: 空白惩罚 / β 固定
def test_reward_lam0_blank_penalty():
    accs = torch.zeros(3, 6)
    w = torch.full((6,), 1.0 / 6.0)
    n = torch.tensor([0, 1, 5])
    r = GR.reward(accs, w, n, 0.005, lam0=0.01)
    assert float(r[0]) == pytest.approx(-0.01)   # 空白付 λ0 (零投影点定价)
    assert float(r[1]) == pytest.approx(-0.005)  # 一章只付 λ -- 空白被严格劣汰
    assert float(r[2]) == pytest.approx(-0.025)
    r0 = GR.reward(accs, w, n, 0.005)            # 缺省 λ0=0 = 旧制不变
    assert float(r0[0]) == 0.0


def test_anchor_ctl_kl_win_none_freezes_beta():
    """beta_adapt=0 的接线语义: 训练器喂 kl_win=None -> 校准与自适应整块跳过."""
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = _armed_ctl(m)
    ctl.kl_tgt = 0.01                            # 即使目标已在, 也不动
    for i in range(6):
        ctl.on_eval(m, TR.ANCHOR_GRACE + 500 + i * 500, 0.95, 10, 1.0,
                    _base_tpl(), None)
    assert ctl.beta == pytest.approx(0.02) and ctl.calib == []


# ================================================================ KL 参考接线
def test_phase2_losses_kl_reads_anchor_not_shadow():
    dev = _dev()
    torch.manual_seed(0)
    m = NumCodeModel().to(dev)
    lag = NumCodeModel().to(dev)
    lag.load_state_dict(m.state_dict())        # 影子 == 在线 => 对影子 KL 恒 0
    for p in lag.parameters():
        p.requires_grad_(False)
    lag.eval()
    ctl = TR.AnchorCtl(m, 0.02)
    with torch.no_grad():
        for p in ctl.ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # 锚被扰动 => 对锚 KL > 0
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    rng_dev = (torch.Generator(device=dev).manual_seed(2)
               if dev.type == "cuda" else torch.Generator().manual_seed(2))
    out_l = TR.phase2_losses(m, _tiny_cfg(), groups, sc, ns, dev,
                             torch.Generator().manual_seed(3), rng_dev,
                             100, GR.TaskWeights(), lag=lag)
    assert abs(out_l["metrics"]["kl"]) < 1e-6            # 默认参考 = 影子写者
    out_a = TR.phase2_losses(m, _tiny_cfg(), groups, sc, ns, dev,
                             torch.Generator().manual_seed(3), rng_dev,
                             100, GR.TaskWeights(), ref_writer=ctl.ref, lag=lag)
    assert out_a["metrics"]["kl"] > 1e-4                 # ref_writer 给定时用锚
    out_0 = TR.phase2_losses(m, _tiny_cfg(), groups, sc, ns, dev,
                             torch.Generator().manual_seed(3), rng_dev,
                             100, GR.TaskWeights(), ref_writer=ctl.ref, lag=lag,
                             beta=0.0)
    assert out_0["metrics"]["kl"] == 0.0                 # β 覆写 0: KL 不算


# ================================================================ 奖励 m 次腐蚀
def test_reward_m_draws_shared_draw0_and_mean(monkeypatch):
    dev = _dev()
    torch.manual_seed(0)
    m = NumCodeModel().to(dev)
    lag = NumCodeModel().to(dev)
    lag.load_state_dict(m.state_dict())
    for p in lag.parameters():
        p.requires_grad_(False)
    lag.eval()
    chan_calls = []
    orig_chan = GM.channel

    def spy_chan(canvas, strength, rng, occ_k=0):
        chan_calls.append(occ_k)
        return orig_chan(canvas, strength, rng, occ_k)

    monkeypatch.setattr(GM, "channel", spy_chan)
    seen = dict(online=[], lag=[])
    orig_on, orig_lag = m.encode_canvas, lag.encode_canvas
    m.encode_canvas = lambda x, **kw: (seen["online"].append(x),
                                       orig_on(x, **kw))[1]
    lag.encode_canvas = lambda x, **kw: (seen["lag"].append(x),
                                         orig_lag(x, **kw))[1]
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    rng_dev = (torch.Generator(device=dev).manual_seed(2)
               if dev.type == "cuda" else torch.Generator().manual_seed(2))
    out = TR.phase2_losses(m, _tiny_cfg(reward_m=4), groups, sc, ns, dev,
                           torch.Generator().manual_seed(3), rng_dev,
                           100, GR.TaskWeights(), lag=lag)
    assert chan_calls == [12, 12, 12, 12]      # m 次独立腐蚀, 全带恒开遮挡
    assert len(seen["lag"]) == 4               # 影子读全部 m 次
    assert len(seen["online"]) == 1            # 在线监督只读 draw0
    assert seen["lag"][0].data_ptr() == seen["online"][0].data_ptr()  # 共源
    a = torch.tensor(out["metrics"]["acc"])
    q = a * 16                                 # 4 rollout × 4 抽 0/1 -> 1/16 格点
    assert float((q - q.round()).abs().max()) < 1e-2


# ================================================================ 位移仪表
def test_modal_templates_and_shift():
    tab = {1: [[GM.aid_pack(0, 0)]] * 3,
           2: [[GM.aid_pack(0, 1)]] * 3,
           3: [[GM.aid_pack(5, 0), GM.aid_pack(9, 2)]] * 3}
    tpl = TR.modal_templates(tab)
    assert tpl[1][0] == 1 and tpl[2][0] == 2 and tpl[3][5] == 1
    assert tpl[3][9] == 3
    prev = {1: [0] * GM.T, 2: [0] * GM.T, 3: [0] * GM.T, 4: [0] * GM.T}
    cur = {n: list(v) for n, v in prev.items()}
    cur[2][5] = 1
    cur[3][0], cur[3][1], cur[3][2] = 1, 2, 3
    out = TR.template_shift(prev, cur)
    assert out["per_n"] == {1: 0, 2: 1, 3: 3, 4: 0}
    assert out["med"] == 0.5                   # [0,0,1,3] 偶数取中两值均值
    assert out["max"] == 3
    assert TR.template_shift({}, cur) is None  # 无交集: 不产出


# ================================================================ 检查点往返
def test_ckpt_roundtrip_with_anchor_state(tmp_path):
    torch.manual_seed(0)
    m = NumCodeModel()
    ctl = TR.AnchorCtl(m, 0.02)
    ctl.tpl = {1: [0] * GM.T}
    ctl.score = 0.9
    ctl.kl_tgt = 0.01
    ctl.last_refresh = 3000
    ctl.beta = 0.08
    ctl.grace_done = True
    ctl.resets_since_refresh = 1
    ctl.boost_from = 2500
    ctl.calib = [(1.0, 0.01)]
    ctl.note_best(m, {1: [1] * GM.T}, 0.97)
    with torch.no_grad():
        ctl.ref.head.weight.add_(0.5)          # 锚 != 当前写者
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = str(tmp_path / "ck.pt")
    TR.save_ckpt(p, m, opt, GR.TaskWeights(), 7, TR.TrainCfg(), 0.5,
                 anchor=ctl.state())
    torch.manual_seed(1)
    m2 = NumCodeModel()
    ctl2 = TR.AnchorCtl(m2, 0.02)
    meta = TR.load_ckpt(p, m2)
    assert meta["step"] == 7 and meta["anchor_sd"] is not None
    ctl2.load_state(meta["anchor_sd"])
    assert torch.equal(ctl2.ref.head.weight, ctl.ref.head.weight)
    assert ctl2.beta == 0.08 and ctl2.kl_tgt == 0.01
    assert ctl2.last_refresh == 3000
    assert ctl2.grace_done is True and ctl2.resets_since_refresh == 1
    assert ctl2.boost_from == 2500 and list(ctl2.calib) == [(1.0, 0.01)]
    assert ctl2.best is not None and ctl2.best[2] == 0.97
    p0 = str(tmp_path / "ck0.pt")
    TR.save_ckpt(p0, m, opt, GR.TaskWeights(), 8, TR.TrainCfg(), 0.5)
    assert TR.load_ckpt(p0, m2)["anchor_sd"] is None      # 旧制无锚键
