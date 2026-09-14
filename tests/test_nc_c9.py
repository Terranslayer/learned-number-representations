# tests/test_nc_c9.py
"""C9 退化组守卫判别测试 ([U] p3d, params.md §10e-补2; p3e 两处一行修正):
forced 臂随机位置禁众数动作 (含 STOP 拦截提前消费) / div_nat 状态机
(触发-退出-撤除-再触发) / 采纳记录 / 装配级接线 / 检查点往返."""
import torch

from symemerge.numcode import data as D
from symemerge.numcode import geometry as GM
from symemerge.numcode import grpo as GR
from symemerge.numcode import trainer as TR
from symemerge.numcode.model import NumCodeModel


def _bias_writer(m, aid, val=20.0):
    """把写者输出头偏成众数 = aid (softmax 后概率 ~1), 造可控退化策略."""
    with torch.no_grad():
        m.writer.head.bias.zero_()
        m.writer.head.bias[aid] = val


# ---------------------------------------------------------------- forced 臂
def test_rollout_ban_mode_at_position():
    """众数=STOP 的退化策略 (p3d N5 形态): 禁 0 位众数 ⇒ 受迫行必落墨;
    未禁行立即 STOP (空程序). 消费一次后自由 ⇒ 正常质量短程序 (恰 1 章)."""
    torch.manual_seed(0)
    m = NumCodeModel()
    _bias_writer(m, GM.A_STOP)
    rng = torch.Generator().manual_seed(1)
    sc, _ = D.sample_scene_batch(rng, 4, 4)
    with torch.no_grad():
        st = m.encode_scene(sc)["tokens"].repeat_interleave(8, 0)
    temps = torch.ones(32)
    ban = torch.zeros(32, dtype=torch.bool)
    ban[::2] = True
    rd = torch.Generator().manual_seed(2)
    progs, acts, lens = m.writer.rollout(
        st, 12, temps, rng=rd, ban_mode=ban,
        ban_pos=torch.zeros(32, dtype=torch.long))
    for i in range(32):
        if ban[i]:
            assert len(progs[i]) == 1        # 0 位被禁写一章, 消费后自由 STOP
        else:
            assert len(progs[i]) == 0        # 未禁行照常空程序
    _, lp, keep = m.writer.logp_of(acts, lens, st)   # 真策略 log π 可计
    assert torch.isfinite((lp * keep).sum())


def test_rollout_ban_mode_nonstop_mode():
    """众数=某图章 (p3d N4 已有墨形态, 旧「首动作禁 STOP」的空操作案例):
    禁 0 位众数 ⇒ 受迫行首动作 != 众数; 未禁行首动作 == 众数."""
    torch.manual_seed(0)
    m = NumCodeModel()
    aid = GM.aid_pack(17, 1)
    _bias_writer(m, aid)
    rng = torch.Generator().manual_seed(1)
    sc, _ = D.sample_scene_batch(rng, 4, 2)
    with torch.no_grad():
        st = m.encode_scene(sc)["tokens"].repeat_interleave(8, 0)
    temps = torch.ones(16)
    ban = torch.zeros(16, dtype=torch.bool)
    ban[::2] = True
    rd = torch.Generator().manual_seed(2)
    progs, _, _ = m.writer.rollout(
        st, 12, temps, rng=rd, ban_mode=ban,
        ban_pos=torch.zeros(16, dtype=torch.long))
    for i in range(16):
        if ban[i]:
            assert progs[i][:1] != [aid]     # 空程序 (采到 STOP) 也是合法偏离
        else:
            assert progs[i][0] == aid


def test_rollout_ban_mode_stop_interception():
    """u 未到就采样出 STOP ⇒ 该步改禁 STOP 提前消费 (恰偏离一次):
    立即-STOP 策略 + u=5 ⇒ 受迫行在 0 位被拦截落墨, 之后自由 STOP,
    长度恰 1 (若拦截不消费而是硬撑到 u, 长度会是 6 -- 此测试逮它)."""
    torch.manual_seed(0)
    m = NumCodeModel()
    _bias_writer(m, GM.A_STOP)
    rng = torch.Generator().manual_seed(1)
    sc, _ = D.sample_scene_batch(rng, 4, 2)
    with torch.no_grad():
        st = m.encode_scene(sc)["tokens"].repeat_interleave(8, 0)
    temps = torch.ones(16)
    ban = torch.ones(16, dtype=torch.bool)
    rd = torch.Generator().manual_seed(2)
    progs, _, _ = m.writer.rollout(
        st, 12, temps, rng=rd, ban_mode=ban,
        ban_pos=torch.full((16,), 5, dtype=torch.long))
    for i in range(16):
        assert len(progs[i]) == 1


# ---------------------------------------------------------------- 状态机
def test_c9_guard_state_machine():
    g = TR.C9Guard([5, 6])
    step = 0
    for _ in range(TR.C9_TRIG - 1):                  # 49 步低 div: 未触发
        step += 1
        assert g.observe({5: 0.0, 6: 0.5}, step) == []
    assert g.arms(5) == 0
    step += 1
    evs = g.observe({5: 0.0, 6: 0.5}, step)          # 第 50 步触发
    assert evs == [dict(n=5, event="trigger", step=step)]
    assert g.state[5] == "active" and g.arms(5) == TR.C9_ARMS
    assert g.state[6] == "idle" and g.arms(6) == 0   # 邻类不受扰
    g.observe({6: 0.5}, step + 1)                    # N5 缺席: streak 冻结
    assert g.state[5] == "active"
    for _ in range(TR.C9_EXIT - 1):                  # 199 步高 div: 仍 active
        step += 1
        g.observe({5: 0.3}, step)
    assert g.state[5] == "active"
    step += 1
    evs = g.observe({5: 0.3}, step)                  # 第 200 步 -> 撤除
    assert evs[0]["event"] == "ramp" and evs[0]["dur"] > 0
    assert g.arms(5) == 2                            # 撤除前半 2 臂
    for _ in range(TR.C9_RAMP // 2):
        step += 1
        g.observe({5: 0.3}, step)
    assert g.arms(5) == 1                            # 撤除后半 1 臂
    for _ in range(TR.C9_RAMP // 2):
        step += 1
        evs = g.observe({5: 0.3}, step)
    assert g.state[5] == "idle" and g.arms(5) == 0   # 满窗归零
    assert evs[0]["event"] == "off" and "adopt" in evs[0]


def test_c9_guard_retrigger_during_ramp():
    g = TR.C9Guard([5])
    step = 0
    for _ in range(TR.C9_TRIG):
        step += 1
        g.observe({5: 0.0}, step)
    for _ in range(TR.C9_EXIT):
        step += 1
        g.observe({5: 0.3}, step)
    assert g.state[5] == "ramp"
    for _ in range(TR.C9_TRIG):                      # 撤除期 div 又塌 -> 回 active
        step += 1
        evs = g.observe({5: 0.0}, step)
    assert g.state[5] == "active"
    assert evs[0]["event"] == "retrigger"


def test_c9_adoption_counter_and_state_roundtrip():
    g = TR.C9Guard([5, 6])
    g.note_adoption(5, [0.2, -0.1])
    g.note_adoption(5, [0.3, 0.0])
    a = g._adopt(5)
    assert a == dict(pos=2, tot=4, frac=0.5)         # >0 计正采纳, 0 不计
    sd = g.state_dict()
    g2 = TR.C9Guard([5, 6])
    g2.load_state(sd)
    assert g2._adopt(5) == a and g2.state == g.state


# ---------------------------------------------------------------- 装配级
def test_phase2_losses_c9_wiring_smoke():
    """守卫置 active: out.c9 带 div 与事件流, forced 臂采纳有记录, 全损失可反传."""
    dev = torch.device("cpu")
    torch.manual_seed(0)
    m = NumCodeModel()
    cfg = TR.TrainCfg(phase=2, k=4, n_groups=2, g=4, n_greedy=1, batch_mask=2,
                      conf_gate=0.0, c9=1, s=0.15, occ_k=4)
    guard = TR.C9Guard(list(D.train_ns(4)))
    for n in D.train_ns(4):
        guard.state[n] = "active"                    # 全类强制 active
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    out = TR.phase2_losses(m, cfg, groups, sc, ns, dev,
                           torch.Generator().manual_seed(3),
                           torch.Generator().manual_seed(4), 0,
                           GR.TaskWeights(), guard=guard)
    assert out["c9"] is not None and "div" in out["c9"]
    assert "div_mix" in out["c9"]                # p3e: 自然臂/全臂两口径并记
    assert sum(guard.adopt_tot.values()) == 2 * TR.C9_ARMS   # 每组末 2 臂入册
    tot = (out["l_grpo"] + out["l_task"] + out["l_count"]
           + 0.05 * out["l_read"])
    tot.backward()


def test_ckpt_c9_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = NumCodeModel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    cfg = TR.TrainCfg(phase=2, k=4, c9=1)
    g = TR.C9Guard([1, 2, 3, 4])
    g.state[3] = "active"
    g.note_adoption(3, [0.5])
    p = str(tmp_path / "ck.pt")
    TR.save_ckpt(p, m, opt, GR.TaskWeights(), 10, cfg,
                 dict(score=0.5, step=10), c9=g.state_dict())
    meta = TR.load_ckpt(p, m)
    g2 = TR.C9Guard([1, 2, 3, 4])
    g2.load_state(meta["c9_sd"])
    assert g2.state[3] == "active" and g2._adopt(3)["pos"] == 1
