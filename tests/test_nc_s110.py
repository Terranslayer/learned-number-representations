# tests/test_nc_s110.py
"""s110 修复轮六改的判别测试 (params.md §9): 恒开遮挡 / c 随课程 / 软门 /
影子 EMA / 奖励读影子 (接线断言) / 评测缓存键换制度换键 / 空白弃权仪表."""
import math

import torch

import symemerge.numcode.data as D
import symemerge.numcode.geometry as GM
import symemerge.numcode.grpo as GR
import symemerge.numcode.trainer as TR
from symemerge.numcode.model import NumCodeModel


def test_occlude_k_exact_count_and_untouched_rest():
    rng = torch.Generator().manual_seed(0)
    x = torch.ones(3, GM.SIDE, GM.SIDE)
    GM.occlude_k(x, 5, rng)
    v = x.view(3, GM.G, GM.P, GM.G, GM.P)
    for b in range(3):
        zero_blocks = int((v[b].sum(dim=(1, 3)) == 0).sum())
        assert zero_blocks == 5
    assert float(x.sum()) == 3 * (GM.T - 5) * GM.P * GM.P


def test_channel_occ_k_dispatch_and_crn():
    prog = [GM.aid_pack(c, c % GM.S) for c in range(0, 120, 3)]
    base = GM.raster([prog])
    a = GM.channel(base, 0.15, torch.Generator().manual_seed(7), 12)
    b = GM.channel(base, 0.15, torch.Generator().manual_seed(7), 12)
    assert torch.equal(a, b)                       # CRN 确定性
    c = GM.channel(base, 0.15, torch.Generator().manual_seed(7))
    assert not torch.equal(a, c)                   # 新旧遮挡路径不同
    va = a.view(1, GM.G, GM.P, GM.G, GM.P)[0].amax(dim=(1, 3))
    stamped = {p // GM.S for p in prog}
    dark = [cell for cell in stamped
            if float(va[cell // GM.G, cell % GM.G]) < 0.05]
    assert len(dark) >= 1                          # 恒开遮挡确实打中过章格


def test_anneal_c_delta_follows_curriculum():
    cfg = TR.TrainCfg(phase=2, k=4, c_delta=0.3)
    assert abs(TR.anneal_c(cfg, 10 ** 6) - (math.log(4.0) - 0.3)) < 1e-9
    assert abs(TR.anneal_c(cfg, 0) - cfg.c_hi) < 1e-9
    legacy = TR.TrainCfg(phase=2, k=4)
    assert abs(TR.anneal_c(legacy, 10 ** 6) - math.log(8.0)) < 1e-9


def test_gate_weights_soft_vs_hard():
    conf = torch.tensor([0.95, 0.5])
    hard = TR.gate_weights(conf, TR.TrainCfg(soft_gate=0), 2)
    soft = TR.gate_weights(conf, TR.TrainCfg(soft_gate=1), 2)
    assert hard.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.allclose(soft, torch.tensor([0.95, 0.95, 0.5, 0.5]))


def test_ema_update_moves_and_converges():
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 3)
    b = torch.nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.weight.add_(1.0)
    TR.ema_update(b, a, 0.9)
    gap = float((b.weight - a.weight).abs().max())
    assert 0.85 < gap < 0.95                       # 一步移动 (1-decay)=0.1
    for _ in range(300):
        TR.ema_update(b, a, 0.9)
    # fp32 逐步乘加的舍入噪声地板 ~1e-6, 容差取其百倍仍远小于任何行为意义
    assert float((b.weight - a.weight).abs().max()) < 1e-4


def test_eval_cache_path_new_regime_new_key():
    c1 = TR.TrainCfg(phase=2, k=4, out="outputs/nc_p2b", s=0.15, occ_k=12)
    assert "_sw0.15_ok12" in TR._eval_cache_path(c1)
    c0 = TR.TrainCfg(phase=2, k=4, out="outputs/nc_p2", s=0.05)
    assert TR._eval_cache_path(c0).endswith("nc_evalsets_s0_p2_k4.pt")


def _tiny_cfg(**kw):
    base = dict(phase=2, k=4, n_groups=2, g=2, n_greedy=1, s=0.15,
                occ_k=12, soft_gate=1, reader_lag=0.999, beta=0.02, lam=0.005)
    base.update(kw)
    return TR.TrainCfg(**base)


def test_phase2_losses_reward_reads_lagged_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = NumCodeModel().to(dev)
    lag = NumCodeModel().to(dev)
    lag.load_state_dict(m.state_dict())
    for p in lag.parameters():
        p.requires_grad_(False)
    lag.eval()
    calls = {"canvas": 0}
    orig = lag.encode_canvas

    def spy(x, **kw):
        calls["canvas"] += 1
        return orig(x, **kw)

    lag.encode_canvas = spy
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    rng_dev = (torch.Generator(device=dev).manual_seed(2)
               if dev.type == "cuda" else torch.Generator().manual_seed(2))
    tw = GR.TaskWeights()
    out = TR.phase2_losses(m, _tiny_cfg(), groups, sc, ns, dev,
                           torch.Generator().manual_seed(3), rng_dev,
                           100, tw, lag=lag)
    assert calls["canvas"] == 1                    # 奖励确实过影子前向
    met = out["metrics"]
    assert "acc" in met and "acc_on" in met and len(met["acc_on"]) == 6
    kl = met["kl"]
    assert kl == kl and kl >= 0.0                  # β>0 路径产出有限 KL
    tot = out["l_grpo"] + out["l_task"] + out["l_count"] + 0.05 * out["l_read"]
    tot.backward()                                 # 装配级可反传


def test_phase2_losses_legacy_path_unchanged_keys():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = NumCodeModel().to(dev)
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    rng_dev = (torch.Generator(device=dev).manual_seed(2)
               if dev.type == "cuda" else torch.Generator().manual_seed(2))
    cfg = _tiny_cfg(occ_k=0, soft_gate=0, reader_lag=0.0, beta=0.0, lam=0.0,
                    s=0.05)
    out = TR.phase2_losses(m, cfg, groups, sc, ns, dev,
                           torch.Generator().manual_seed(3), rng_dev,
                           100, GR.TaskWeights())
    assert "acc_on" not in out["metrics"]          # 旧制无影子字段
    assert out["metrics"]["kl"] == 0.0


def test_l_max_override():
    assert TR.l_max_of(TR.TrainCfg(phase=2, k=4)) == GM.l_max(4) == 14
    assert TR.l_max_of(TR.TrainCfg(phase=2, k=4, l_max_override=28)) == 28


def test_min_pair_hamming_counts_modal_diffs():
    tab = {1: [[0 * GM.S + 0]] * 3,                  # N=1: 格0 点
           2: [[0 * GM.S + 1]] * 3,                  # N=2: 格0 横杠 (同格换型 = 1 格差)
           3: [[5 * GM.S + 0, 9 * GM.S + 2]] * 3}    # N=3: 格5 点 + 格9 竖杠
    out = TR.min_pair_hamming(tab)
    assert out["pairs"]["1-2"] == 1
    assert out["pairs"]["1-3"] == 3                  # 格0 消失 + 格5/9 新增
    assert out["pairs"]["2-3"] == 3
    assert out["min"] == 1


def test_wiring_reader_on_corrupted_and_mask_legacy(monkeypatch):
    """s110 两条接线裁定的钉子: phase2 损失与奖励共用同一次带 occ_k 的腐蚀前向
    (读者在腐蚀画布上训练); 素养流的信道不带 occ_k (旧配方)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = NumCodeModel().to(dev)
    calls = []
    orig = GM.channel

    def spy(canvas, strength, rng, occ_k=0):
        calls.append(occ_k)
        return orig(canvas, strength, rng, occ_k)

    monkeypatch.setattr(GM, "channel", spy)
    rng = torch.Generator().manual_seed(1)
    groups = [D.sample_group(rng, 4) for _ in range(2)]
    sc, ns = D.sample_scene_batch(rng, 4, 2)
    rng_dev = (torch.Generator(device=dev).manual_seed(2)
               if dev.type == "cuda" else torch.Generator().manual_seed(2))
    cfg = _tiny_cfg(occ_k=48, l_max_override=28)
    out = TR.phase2_losses(m, cfg, groups, sc, ns, dev,
                           torch.Generator().manual_seed(3), rng_dev,
                           100, GR.TaskWeights())
    assert calls == [48]                     # 恰一次画布腐蚀, 损失与奖励同源
    calls.clear()
    TR.mask_forward(m, cfg, torch.Generator().manual_seed(4), dev)
    assert calls == [0]                      # 素养通道不跟 occ_k


def test_blank_control_reports_read_abstain():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = NumCodeModel().to(dev).eval()
    rng = torch.Generator().manual_seed(0)
    tn = [(D.sample_n(rng, 4), D.sample_theta(rng, 4)) for _ in range(8)]
    out = TR.blank_control(m, 4, tn, dev)
    assert "read_abstain_blank" in out
    assert 0.0 <= out["read_abstain_blank"] <= 1.0
