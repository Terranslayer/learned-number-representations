# tests/test_nc_relearn.py -- 迭代重学甲案 (spec 2026-08-24-iterated-relearning-phase-spec.md;
# [U] 2026-08-24「先跑甲，写头d1保持冻结」): 旋钮存在性 / A30 钉值平价 (CPU) / 曝光子集分层配额 /
# 梯度路由与曝光防火墙 (A31/A32) / 教学隔离与期界重置与状态往返 (A33) / 考试冒烟.
import torch

import symemerge.numcode.data as D
import symemerge.numcode.geometry as GM
import symemerge.numcode.pred.checks as C
import symemerge.numcode.pred.relearn as RL
import symemerge.numcode.pred.step as ST
import symemerge.numcode.pred.trainer as TR
from symemerge.numcode.grpo import TaskWeights
from symemerge.numcode.pred.data import build_batch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HEAD (f711e63) 上按 §6 采值脚本在 pod CPU 采得的一步损失, 逐位 (A30 钉值; 改码前后必须同值)
A30_PIN = 9.853376388549805


def test_cfg_fields_and_import():
    cfg = TR.TrainCfg(exposure=1, e_exp=20, t_period=2000, eta_tr=0.3, tr_warm=0.25,
                      sr_lr=3e-4, n_s=1, b_teach=16, b_teach_sc=8, sr_mu=0.05)
    assert cfg.exposure == 1 and cfg.e_exp == 20
    r = RL.Relearner()
    from symemerge.numcode.pred.model import Backbone, Heads
    assert isinstance(r.E, Backbone) and isinstance(r.heads, Heads)


def test_a30_pin_parity_cpu():
    dev = torch.device("cpu")
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(1)
    cfg = C._small_cfg(d1_cm=1, chain_k=1, chain_from_pool=0)
    model = TR.build_model(cfg).to(dev)
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    w = TaskWeights().weights(dev)
    sb, pb, mb, pool = C._mini_batches(model, cfg, dev, rng)
    m = ST.chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng)
    assert A30_PIN is not None, "先按 brief §6 在 HEAD 采钉值回填"
    assert m["L"] == A30_PIN
    assert "notes_x" not in m and "n_tr" not in m and "sr_loss" not in m


def test_draw_exposure_quotas_and_firewall():
    g = torch.Generator().manual_seed(7)
    exp = RL.draw_exposure(g, GM.K_MAX, 20)
    assert len(exp) == 20 and len(set(exp)) == 20
    quotas = [sum(1 for n in exp if lo <= n <= hi) for lo, hi in RL.BANDS]
    assert quotas == [2, 4, 4, 6, 4]
    train = set(D.train_ns(GM.K_MAX))
    assert set(exp) <= train
    for n in exp:                       # 留出段 (空洞 45-56/83-94, 外推 113-128) 零出现
        assert not (45 <= n <= 56 or 83 <= n <= 94 or 113 <= n <= 128)
    g2 = torch.Generator().manual_seed(7)
    assert RL.draw_exposure(g2, GM.K_MAX, 20) == exp


def test_transfer_routing_and_fences():
    verdict, info = C.a31(DEV, seed=0)
    assert verdict == "PASS", info
    verdict, info = C.a32(DEV, seed=0)
    assert verdict == "PASS", info


def test_rollover_and_state_roundtrip():
    verdict, info = C.a33(DEV, seed=0)
    assert verdict == "PASS", info
    cfg = C._small_cfg(exposure=1, e_exp=3, t_period=50)
    rl = RL.Relearn(cfg, DEV)
    rl.rollover(9, 0)
    st = rl.state()
    rl2 = RL.Relearn(cfg, DEV)
    rl2.load_state(st)
    assert rl2.period == rl.period and rl2.exp == rl.exp
    same = all(torch.equal(a.cpu(), b.cpu()) for a, b in
               zip(rl.reader.state_dict().values(), rl2.reader.state_dict().values()))
    assert same


def test_exam_smoke_and_isolation():
    verdict, info = C.a34(DEV, seed=0)
    assert verdict == "PASS", info
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(11)
    cfg = C._small_cfg(d1_cm=1, exposure=1, e_exp=3, chain_k=1, chain_from_pool=0)
    model = TR.build_model(cfg).to(DEV)
    w = TaskWeights().weights(DEV)
    rl = RL.Relearn(cfg, DEV)
    rl.rollover(9, 0)
    rl.exp = (1, 2, 3)
    items = [D.sample_group(rng, 9, n=n) for n in (1, 2, 3, 5, 6, 7, 8, 9)]
    ev = rl.exam(model, dict(groups=items), cfg, w, DEV, torch.Generator().manual_seed(4242))
    assert ev["unexp"]["n"] == 5 and ev["exp_set"]["n"] == 3
    assert set(ev["unexp"]["acc"]) == {"t1", "t2", "t3", "t4", "t6"}
    assert len(ev["unexp"]["tol"]) == 4 and ev["scene_unexp"]["n"] == 5


# ---------------- 乙案 (gen_relearn; spec §2.2 乙执行细则; [U] 2026-08-24「既然已经澄清，那就根据讨论试一试乙吧」)


def test_gen_cfg_fields_and_alloff_pin():
    cfg = TR.TrainCfg(gen_relearn=1, t_gen=3000, eta_im=1.0, b_im=16, b_poolread=16, gen_keep_cnn=1)
    assert cfg.gen_relearn == 1 and cfg.t_gen == 3000
    v, info = C.a35(torch.device("cpu"), seed=0)
    assert v == "PASS", info
    assert info["L"] == A30_PIN


def test_gen_label_firewall():
    v, info = C.a36(DEV, seed=0)
    assert v == "PASS", info


def test_gen_rollover_reset_and_keep():
    v, info = C.a37(DEV, seed=0)
    assert v == "PASS", info


def test_gen_imitation_and_poolread_routing():
    v, info = C.a38(DEV, seed=0)
    assert v == "PASS", info


def test_gen_state_roundtrip_and_bank_determinism():
    cfg = C._small_cfg(gen_relearn=1, e_exp=3, t_gen=50, d1_cm=1)
    from symemerge.numcode.pred.model import PredModel
    model = PredModel(d1_cm=True).to(DEV)
    g1 = RL.GenRelearn(cfg, DEV)
    g1.rollover(model, 9, 1)
    st = g1.state()
    g2 = RL.GenRelearn(cfg, DEV)
    g2.load_state(st)
    assert g2.period == g1.period and g2.exp == g1.exp and g2.k == g1.k
    assert set(g2.bank) == set(g1.bank)
    assert all(torch.equal(g2.bank[n], g1.bank[n]) for n in g1.bank)


def test_gen_exam_smoke_via_main_model():
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(11)
    cfg = C._small_cfg(d1_cm=1, gen_relearn=1, e_exp=3, chain_k=1, chain_from_pool=0)
    from symemerge.numcode.pred.model import PredModel
    model = PredModel(d1_cm=True).to(DEV)
    w = TaskWeights().weights(DEV)
    grl = RL.GenRelearn(cfg, DEV)
    grl.period = 0
    grl.k = 9
    grl.exp = (1, 2, 3)
    grl.bank = {}
    items = [D.sample_group(rng, 9, n=n) for n in (1, 2, 3, 5, 6, 7, 8, 9)]
    ev = grl.exam(model, dict(groups=items), cfg, w, DEV, torch.Generator().manual_seed(4242))
    assert ev["unexp"]["n"] == 5 and ev["exp_set"]["n"] == 3
    assert ev["scene_unexp"]["n"] == 5 and ev["period"] == 0


def test_gen_warm_and_ink_weight():
    cfg = TR.TrainCfg(gen_relearn=1, im_wink=8.0, gen_warm=0.25)
    assert cfg.im_wink == 8.0 and cfg.gen_warm == 0.25
    v, info = C.a39(DEV, seed=0)
    assert v == "PASS", info
