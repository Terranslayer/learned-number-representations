# tests/test_nc_model.py -- 多任务协同分支: 主干/头/写者/损失/GRPO
import math

import pytest
import torch
import torch.nn as nn

from symemerge.numcode import geometry as GM
from symemerge.numcode import grpo as R
from symemerge.numcode import losses as L
from symemerge.numcode import model as M


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return M.NumCodeModel()


def test_backbone_shapes_and_modality(net):
    x = torch.rand(2, GM.SIDE, GM.SIDE)
    s = net.encode_scene(x)
    c = net.encode_canvas(x)
    assert s["cls"].shape == (2, M.D_MODEL)
    assert s["tokens"].shape == (2, GM.T, M.D_MODEL)
    # 同一输入, 两种模态标记必须给出不同表征 (spec §2.1)
    assert not torch.allclose(s["cls"], c["cls"])


def test_cnn_token_cell_alignment(net):
    # spec §2.1 关键几何约束: 一 token 一格, token 中心 = 格心 (相位对齐).
    # 走纯卷积级联测 (GN/SiLU 的随机不对称会扰动 argmax, 但相位是卷积的性质):
    # k4/s2/p1 下单章响应 argmax 正中承载格且左右/上下邻差 ~1x;
    # k3 的半格相位错会把邻差推到 1e8 量级 (pod 实测), 双判据都逮得住.
    gi, gj = 7, 9
    convs = nn.Sequential(net.E.cnn[0], net.E.cnn[3], net.E.cnn[6], net.E.cnn[9])
    canv = GM.raster([[GM.aid_pack(gi * GM.G + gj, GM.STAMP_VBAR)]])
    with torch.no_grad():
        f0 = convs(torch.zeros(1, 1, GM.SIDE, GM.SIDE))
        f1 = convs(canv.unsqueeze(1))
    delta = (f1 - f0)[0].norm(dim=0)               # (G, G)
    idx = int(delta.flatten().argmax())
    assert (idx // GM.G, idx % GM.G) == (gi, gj), (idx // GM.G, idx % GM.G)
    for a, b in (((gi, gj - 1), (gi, gj + 1)), ((gi - 1, gj), (gi + 1, gj))):
        lo = min(float(delta[a]), float(delta[b]))
        hi = max(float(delta[a]), float(delta[b]))
        assert hi / max(lo, 1e-9) < 2.0, (a, b, lo, hi)


def test_heads_structural(net):
    h = net.heads
    # 计数头 2 层上限 (spec §2.3); 直读头 2 层 (spec §2.5)
    assert [type(x) for x in h.count] == [nn.Linear, nn.GELU, nn.Linear]
    assert [type(x) for x in h.read] == [nn.Linear, nn.GELU, nn.Linear]
    assert h.read[-1].out_features == M.K + 1
    # 任务头严格线性 (spec §2.4)
    for t in (h.t1, h.t2, h.t3, h.t4, h.t6):
        assert isinstance(t, nn.Linear)
    assert isinstance(h.t5, nn.Linear) and h.t5.bias is None
    assert h.cell.out_features == GM.S + 1


def test_task_logits_shapes(net):
    h = torch.randn(3, M.D_MODEL)
    idx = dict(tau=torch.tensor([0, 5, 100]), p=torch.tensor([0, 1, 2]),
               m=torch.tensor([0, 1, 2]))
    lg = net.heads.task_logits(h, idx)
    for t, n in (("t1", 2), ("t2", 2), ("t3", M.K), ("t4", 7), ("t6", M.K + 3)):
        assert lg[t].shape == (3, n)
    sc = net.heads.t5_scores(h, torch.randn(3, 6, M.D_MODEL))
    assert sc.shape == (3, 6)


def test_mask_replacement(net):
    rng = torch.Generator().manual_seed(1)
    canv = GM.raster([GM.sample_uniform_prog(rng, lmax=60)])
    mask = torch.zeros(1, GM.T, dtype=torch.bool)
    mask[0, torch.randperm(GM.T, generator=rng)[:70]] = True
    with torch.no_grad():
        a = net.encode_canvas(canv)
        b = net.encode_canvas(canv, mask_cells=mask)
    assert not torch.allclose(a["cls"], b["cls"])
    assert net.heads.cell(b["tokens"]).shape == (1, GM.T, GM.S + 1)


def test_legal_bias():
    seq = torch.tensor([[M.BOS, GM.aid_pack(5, 1), GM.aid_pack(9, 0)]])
    bias = M.legal_bias(seq)
    assert bias.shape == (1, 3, GM.VOCAB)
    assert float(bias[0, 0].abs().sum()) == 0.0           # BOS 位: 尚无已盖格
    for st in range(GM.S):        # 位置 1 (输入=格5动作) 预测 a_1: 格 5 已盖
        assert float(bias[0, 1, GM.aid_pack(5, st)]) < -1e8
    assert float(bias[0, 1, GM.aid_pack(9, 0)]) == 0.0
    for st in range(GM.S):        # 位置 2 预测 a_2: 格 5 与格 9 都已盖
        assert float(bias[0, 2, GM.aid_pack(5, st)]) < -1e8
        assert float(bias[0, 2, GM.aid_pack(9, st)]) < -1e8
    assert float(bias[0, 2, GM.A_STOP]) == 0.0            # STOP 恒合法


def test_writer_rollout_legal(net):
    torch.manual_seed(2)
    st = torch.randn(16, GM.T, M.D_MODEL)
    temps = torch.tensor([0.0, 0.0] + [1.5] * 14)
    progs, acts, lens = net.writer.rollout(
        st, 20, temps, rng=torch.Generator().manual_seed(3))
    assert len(progs) == 16
    for b, p in enumerate(progs):
        assert len(p) <= 20
        assert all(0 <= a < GM.A_STOP for a in p)
        assert int(lens[b]) == len(p) + 1                 # 全部序列以 STOP 收尾
        cells = [a // GM.S for a in p]
        assert len(cells) == len(set(cells)), f"重复落格 (含连续重复): {p}"
    GM.raster(progs)                                      # 光栅化自带无重复格 assert
    # 贪心行不依赖采样流: 换 rng 种子, 前两行程序不变
    progs2, _, _ = net.writer.rollout(
        st, 20, temps, rng=torch.Generator().manual_seed(99))
    assert progs2[0] == progs[0] and progs2[1] == progs[1]


def test_writer_logp(net):
    torch.manual_seed(4)
    st = torch.randn(2, GM.T, M.D_MODEL)
    progs, acts, lens = net.writer.rollout(
        st, 6, torch.tensor([1.0, 0.0]), rng=torch.Generator().manual_seed(5))
    lp_sum, lp, keep = net.writer.logp_of(acts, lens, st)
    assert lp_sum.shape == (2,)
    assert bool(torch.isfinite(lp_sum).all())
    assert torch.equal(keep.sum(-1), lens)
    assert torch.allclose((lp * keep).sum(-1), lp_sum)


def test_gradient_routing(net):
    # spec §2.7: 策略梯度只进写者; 精确梯度进主干
    net.zero_grad(set_to_none=True)
    x = torch.rand(2, GM.SIDE, GM.SIDE)
    with torch.no_grad():
        st = net.encode_scene(x)["tokens"]
    progs, acts, lens = net.writer.rollout(
        st, 6, torch.tensor([1.0, 1.0]), rng=torch.Generator().manual_seed(6))
    lp_sum, lp, keep = net.writer.logp_of(acts, lens, st)
    (-(torch.tensor([1.0, -1.0]) * lp_sum).mean()).backward()
    e_g = sum(float(p.grad.abs().sum()) for p in net.E.parameters()
              if p.grad is not None)
    w_g = sum(float(p.grad.abs().sum()) for p in net.writer.parameters()
              if p.grad is not None)
    assert e_g == 0.0, "策略梯度漏进主干"
    assert w_g > 0.0, "写者没接到策略梯度"
    net.zero_grad(set_to_none=True)
    out = net.encode_scene(x)
    L.count_loss(net.heads.count(out["cls"]), torch.tensor([3, 4])).backward()
    e_g2 = sum(float(p.grad.abs().sum()) for p in net.E.parameters()
               if p.grad is not None)
    assert e_g2 > 0.0, "计数损失没流入主干"
    net.zero_grad(set_to_none=True)
    # 带梯度的场景表征直接喂写者必须被拦下 (assert 把关)
    st_grad = net.encode_scene(x)["tokens"]
    with pytest.raises(AssertionError):
        net.writer.logits(torch.full((2, 1), M.BOS, dtype=torch.long), st_grad)


def test_read_loss_limits():
    c = 2.0
    ns = torch.tensor([7])
    all_abstain = torch.zeros(1, M.K + 1)
    all_abstain[0, M.ABSTAIN] = 50.0
    assert abs(float(L.read_loss(all_abstain, ns, c)) - c) < 0.01
    confident = torch.zeros(1, M.K + 1)
    confident[0, 6] = 50.0                                # 类 6 = N 7
    assert float(L.read_loss(confident, ns, c)) < 0.01


def test_task_and_mask_losses(net):
    torch.manual_seed(7)
    B = 3
    h = torch.randn(B, M.D_MODEL)
    idx = dict(tau=torch.tensor([4, 4, 4]), p=torch.tensor([0, 0, 0]),
               m=torch.tensor([0, 0, 0]))
    lg = net.heads.task_logits(h, idx)
    tg = dict(t1=torch.tensor([0, 1, 0]), t2=torch.tensor([1, 0, 1]),
              t3=torch.tensor([4, 5, 6]), t4=torch.tensor([2, 0, 1]),
              t6=torch.tensor([6, 7, 8]))
    sc = net.heads.t5_scores(h, torch.randn(B, 6, M.D_MODEL))
    lt = L.task_loss(lg, sc, tg, torch.tensor([0, 3, 5]))
    assert bool(torch.isfinite(lt)) and float(lt) > 0.0
    truth = torch.zeros(B, GM.G, GM.G, dtype=torch.long)
    truth[:, 0, 0] = 2
    mask = torch.zeros(B, GM.T, dtype=torch.bool)
    mask[:, :5] = True
    easy = torch.zeros(B, GM.T, GM.S + 1)
    easy[:, 0, 2] = 50.0
    easy[:, 1:, 0] = 50.0
    assert float(L.mask_loss(easy, truth, mask)) < 0.01
    rand_lg = torch.zeros(B, GM.T, GM.S + 1)
    assert abs(float(L.mask_loss(rand_lg, truth, mask)) - math.log(GM.S + 1)) < 0.01


def test_mask_cells_sampler():
    rng = torch.Generator().manual_seed(8)
    mc = L.sample_mask_cells(16, rng)
    frac = mc.float().mean(-1)
    assert bool((frac >= 0.14).all()) and bool((frac <= 0.41).all())


def test_grpo_helpers():
    r = torch.arange(8.0)
    adv = R.advantages(r, 2, 4)
    assert torch.allclose(adv.view(2, 4).sum(1), torch.zeros(2), atol=1e-6)
    lp = torch.randn(4, 6)
    keep = torch.ones(4, 6, dtype=torch.bool)
    assert float(R.k3_kl(lp, lp, keep)) == 0.0
    assert float(R.k3_kl(lp, lp + torch.randn(4, 6) * 0.3, keep)) > 0.0
    accs = torch.ones(4, 6)
    w = torch.full((6,), 1 / 6)
    rw = R.reward(accs, w, torch.tensor([0, 2, 4, 6]), lam=0.1)
    assert torch.allclose(rw, torch.tensor([1.0, 0.8, 0.6, 0.4]))
    tw = R.TaskWeights(decay=0.5)
    solved = torch.zeros(8, 6)
    solved[:, 0] = 1.0
    for _ in range(20):
        tw.update(solved)
    w = tw.weights()
    assert abs(float(w.sum()) - 1.0) < 1e-5
    assert float(w[0]) < float(w[1])                      # 已解决任务权重退出


def test_grpo_loss_direction(net):
    lp_sum = torch.tensor([-5.0, -3.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    loss = R.grpo_loss(lp_sum, adv, torch.tensor(0.0), beta=0.1)
    loss.backward()
    # 正优势 -> 提升该程序 logπ (梯度为负方向)
    assert float(lp_sum.grad[0]) < 0.0 and float(lp_sum.grad[1]) > 0.0


def test_theta_to_idx():
    idx = M.theta_to_idx(dict(tau=17, p=5, m=3))
    assert int(idx["tau"]) == 16 and int(idx["p"]) == 1 and int(idx["m"]) == 2
