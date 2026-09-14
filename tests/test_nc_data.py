# tests/test_nc_data.py -- 多任务协同分支: 留出区间/抽样/任务族/候选集
import torch

from symemerge.numcode import data as D
from symemerge.numcode import geometry as GM


def test_band_partition():
    bands = {"train": 0, "hole": 0, "extrap": 0}
    for n in range(1, D.K + 1):
        bands[D.band_of(n)] += 1
    assert bands == {"train": 88, "hole": 24, "extrap": 16}
    frac_hole = bands["hole"] / D.K
    assert 0.10 <= frac_hole <= 0.20            # spec §4.5 空洞总宽 10-20%
    assert D.EXTRAP[1] - D.EXTRAP[0] + 1 >= 16  # 任何基数 <=16 的完整进位周期


def test_band_edges():
    for n, b in ((44, "train"), (45, "hole"), (56, "hole"), (57, "train"),
                 (82, "train"), (83, "hole"), (94, "hole"), (95, "train"),
                 (112, "train"), (113, "extrap"), (128, "extrap")):
        assert D.band_of(n) == b, (n, b)


def test_train_ns_stages():
    assert D.train_ns(4) == (1, 2, 3, 4)
    assert D.train_ns(16) == tuple(range(1, 17))
    t64 = D.train_ns(64)
    assert t64 == tuple(range(1, 45)) + tuple(range(57, 65))
    assert len(D.train_ns(128)) == 88
    assert set(D.hole_ns()) | set(D.extrap_ns()) | set(D.train_ns(128)) \
        == set(range(1, 129))


def test_delta_distribution():
    rng = torch.Generator().manual_seed(0)
    draws = [D.sample_delta(rng) for _ in range(500)]
    assert all(1 <= abs(d) <= 4 for d in draws)
    f1 = sum(1 for d in draws if abs(d) == 1) / 500
    assert 0.5 <= f1 <= 0.7                     # 登记值 0.6 (spec §4.2 重偏 ±1)
    assert any(d > 0 for d in draws) and any(d < 0 for d in draws)


def test_delta_in_band():
    rng = torch.Generator().manual_seed(1)
    for k, n in ((4, 4), (16, 1), (64, 44), (128, 112)):
        for _ in range(20):
            d = D.sample_delta_in(rng, k, n)
            assert (n + d) in set(D.train_ns(k)), (k, n, d)


def test_theta_and_targets():
    rng = torch.Generator().manual_seed(2)
    for _ in range(50):
        th = D.sample_theta(rng, 16)
        assert 1 <= th["tau"] <= 15
        assert th["p"] in D.P_CHOICES and th["m"] in D.M_CHOICES
    tg = D.targets(5, dict(tau=5, p=3, m=2))
    assert tg == dict(t1=0, t2=1, t3=4, t4=2, t6=6)
    assert D.targets(6, dict(tau=5, p=3, m=1))["t1"] == 1
    # 类索引全域不越头 (params.md §6: 类数基座一次定死)
    for n in D.train_ns(128):
        for tau in (1, 127):
            for p in D.P_CHOICES:
                for m in D.M_CHOICES:
                    tg = D.targets(n, dict(tau=tau, p=p, m=m))
                    for t, v in tg.items():
                        assert 0 <= v < D.TASK_NCLS[t], (n, tau, p, m, t, v)


def test_render_scene_deterministic():
    a = D.render_scene(torch.Generator().manual_seed(7), 9)
    b = D.render_scene(torch.Generator().manual_seed(7), 9)
    assert a.shape == (D.SCENE_SIDE, D.SCENE_SIDE)
    assert torch.equal(a, b)
    assert float(a.sum()) > 0.0
    assert float(a.max()) <= 1.0 and float(a.min()) >= 0.0


def test_scene_batch():
    scenes, ns = D.sample_scene_batch(torch.Generator().manual_seed(3), 16, 3)
    assert scenes.shape == (3, D.SCENE_SIDE, D.SCENE_SIDE)
    assert all(int(n) in set(D.train_ns(16)) for n in ns)


def test_group_bundle():
    g = D.sample_group(torch.Generator().manual_seed(5), 16)
    assert g["scene"].shape == (D.SCENE_SIDE, D.SCENE_SIDE)
    assert g["cands"].shape == (D.N_CAND, D.SCENE_SIDE, D.SCENE_SIDE)
    assert sorted(g["cand_kinds"]) == sorted(
        ["pos", "hard_area", "hard_perim", "rand", "rand", "rand"])
    assert g["cand_kinds"][g["truth5"]] == "pos"
    assert g["cand_ns"][g["truth5"]] == g["n"]
    tset = set(D.train_ns(16))
    assert g["n"] in tset and all(n in tset for n in g["cand_ns"])
    for kind, n in zip(g["cand_kinds"], g["cand_ns"]):
        if kind.startswith("hard"):
            assert n != g["n"]                  # 硬负必须换 N
    assert set(g["targets"]) == {"t1", "t2", "t3", "t4", "t6"}
    # 组包确定性 (CRN)
    g2 = D.sample_group(torch.Generator().manual_seed(5), 16)
    assert torch.equal(g["cands"], g2["cands"]) and g2["truth5"] == g["truth5"]


def test_group_bundle_respects_firewall_at_k128():
    g = D.sample_group(torch.Generator().manual_seed(6), 128)
    tset = set(D.train_ns(128))
    assert g["n"] in tset and all(n in tset for n in g["cand_ns"])


def test_geometry_data_side_agree():
    assert D.SCENE_SIDE == GM.SIDE              # 共享主干单一输入几何
