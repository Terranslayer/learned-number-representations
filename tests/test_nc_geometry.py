# tests/test_nc_geometry.py -- 多任务协同分支: 画布几何/光栅化/腐蚀信道
import math

import pytest
import torch

from symemerge.numcode import geometry as GM


def test_capacity_laws():
    # spec §3.2: 一元码必须装得下 -- T >= K_max, 每档 L_max >= K
    assert GM.SIDE == GM.G * GM.P
    assert GM.T == GM.G * GM.G
    assert GM.T >= GM.K_MAX
    for k in GM.K_STAGES:
        assert GM.l_max(k) >= k
        assert GM.l_max(k) <= GM.T
    assert GM.VOCAB == GM.T * GM.S + 1
    assert GM.A_STOP == GM.T * GM.S


def test_worst_reach_within_cell():
    # params.md §1 验算: 满腐蚀最坏离心距 <= P/2 (邻格像素中心零覆盖条件)
    for t in range(GM.S):
        assert GM.worst_reach(t) <= GM.P / 2 + 1e-9, (t, GM.worst_reach(t))


def test_stamp_never_crosses_cell_empirically():
    # 极端腐蚀组合逐一渲染, pad=2 外环必须一滴墨都没有
    vals = (-GM.JITTER_PX, 0.0, GM.JITTER_PX)
    scs = (1.0 - GM.SIZE_JIT, 1.0, 1.0 + GM.SIZE_JIT)
    angs = (-math.radians(GM.ROT_DEG), 0.0, math.radians(GM.ROT_DEG))
    for t in range(GM.S):
        for jy in vals:
            for jx in vals:
                for sc in scs:
                    for an in angs:
                        a = GM.stamp_alpha(
                            torch.tensor([t]), torch.tensor([jy]),
                            torch.tensor([jx]), torch.tensor([sc]),
                            torch.tensor([an]), pad=2)
                        ring = a[0].clone()
                        ring[2:-2, 2:-2] = 0.0
                        assert float(ring.sum()) == 0.0, (t, jy, jx, sc, an)


def test_stamp_ink_mass_sane():
    # 名义渲染下每种章都得有肉眼级的墨量 (可分性的必要非充分条件; 正式门在 Phase 0)
    z = torch.zeros(1)
    one = torch.ones(1)
    for t, lo in ((GM.STAMP_DOT, 9.0), (GM.STAMP_DASH, 4.0), (GM.STAMP_VBAR, 6.0)):
        a = GM.stamp_alpha(torch.tensor([t]), z, z, one, z)
        assert float(a.sum()) >= lo, (t, float(a.sum()))


def test_aid_roundtrip():
    for cell in (0, 1, GM.G, GM.T - 1):
        for st in range(GM.S):
            c2, s2 = GM.aid_unpack(GM.aid_pack(cell, st))
            assert (c2, s2) == (cell, st)


def test_raster_ink_exactly_in_stamped_cells():
    cells = {0: GM.STAMP_DOT, GM.T - 1: GM.STAMP_VBAR,
             5 * GM.G + 7: GM.STAMP_DASH}
    prog = [GM.aid_pack(c, s) for c, s in cells.items()]
    canv = GM.raster([prog, []])
    assert canv.shape == (2, GM.SIDE, GM.SIDE)
    blocks = canv[0].view(GM.G, GM.P, GM.G, GM.P).permute(0, 2, 1, 3) \
                    .reshape(GM.T, -1).sum(1)
    for c in range(GM.T):
        if c in cells:
            assert float(blocks[c]) > 3.0, c
        else:
            assert float(blocks[c]) == 0.0, c
    assert float(canv[1].abs().sum()) == 0.0


def test_raster_block_layout():
    # reshape 布局回归: 格 (gi, gj) 的墨必须落在行 [gi*P, gi*P+P) 列 [gj*P, gj*P+P)
    gi, gj = 2, 11
    canv = GM.raster([[GM.aid_pack(gi * GM.G + gj, GM.STAMP_VBAR)]])
    ys, xs = torch.nonzero(canv[0] > 0.0, as_tuple=True)
    assert int(ys.min()) >= gi * GM.P and int(ys.max()) < gi * GM.P + GM.P
    assert int(xs.min()) >= gj * GM.P and int(xs.max()) < gj * GM.P + GM.P
    # 章心在格心: 墨的质心离格心 < 1px
    m = canv[0]
    cy = float((ys.float() * m[ys, xs]).sum() / m[ys, xs].sum())
    cx = float((xs.float() * m[ys, xs]).sum() / m[ys, xs].sum())
    assert abs(cy - (gi * GM.P + (GM.P - 1) / 2)) < 1.0
    assert abs(cx - (gj * GM.P + (GM.P - 1) / 2)) < 1.0


def test_raster_rejects_duplicates_and_stop():
    with pytest.raises(AssertionError):
        GM.raster([[GM.aid_pack(3, 0), GM.aid_pack(3, 1)]])
    with pytest.raises(AssertionError):
        GM.raster([[GM.A_STOP]])


def test_raster_corruption_deterministic_and_varying():
    prog = [GM.aid_pack(17, 0), GM.aid_pack(200, 1), GM.aid_pack(100, 2)]
    a = GM.raster([prog], 1.0, torch.Generator().manual_seed(7))
    b = GM.raster([prog], 1.0, torch.Generator().manual_seed(7))
    c = GM.raster([prog], 1.0, torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    with pytest.raises(AssertionError):
        GM.raster([prog], 0.5, None)


def test_cell_truth_matches_raster():
    rng = torch.Generator().manual_seed(3)
    progs = [GM.sample_uniform_prog(rng) for _ in range(4)]
    truth = GM.cell_truth(progs)
    canv = GM.raster(progs)
    blocks = canv.view(-1, GM.G, GM.P, GM.G, GM.P).permute(0, 1, 3, 2, 4) \
                 .reshape(-1, GM.T, GM.P * GM.P).sum(-1)
    assert truth.shape == (4, GM.G, GM.G)
    flat = truth.view(4, GM.T)
    assert bool(((blocks > 0.0) == (flat > 0)).all())
    for b, prog in enumerate(progs):
        for aid in prog:
            cell, st = GM.aid_unpack(aid)
            assert int(flat[b, cell]) == st + 1


def test_uniform_prog_legal():
    rng = torch.Generator().manual_seed(11)
    lens = []
    for _ in range(50):
        prog = GM.sample_uniform_prog(rng, lmax=40)
        lens.append(len(prog))
        cells = [GM.aid_unpack(a)[0] for a in prog]
        assert len(set(cells)) == len(cells)
        assert all(0 <= a < GM.A_STOP for a in prog)
        assert len(prog) <= 40
    assert max(lens) > 20 and min(lens) < 15   # U{0..40} 不该缩在一角


def test_occlude_zeroes_blocks():
    canvas = torch.ones(3, GM.SIDE, GM.SIDE)
    out = GM.occlude(canvas.clone(), 1.0, torch.Generator().manual_seed(5))
    v = out.view(3, GM.G, GM.P, GM.G, GM.P).permute(0, 1, 3, 2, 4) \
           .reshape(3, GM.T, -1)
    zeroed = (v.sum(-1) == 0.0).sum(1)
    for b in range(3):
        assert 1 <= int(zeroed[b]) <= GM.OCCLUDE_MAX, int(zeroed[b])
    # 未遮的格保持原样
    assert float(v.max()) == 1.0


def test_blur_identity_and_mass():
    canvas = GM.raster([[GM.aid_pack(8 * GM.G + 8, GM.STAMP_DOT)]])
    assert torch.equal(GM.blur(canvas, 0.0), canvas)
    out = GM.blur(canvas, GM.BLUR_SIG)
    # 内部章的墨量在模糊下近守恒 (零填充只在画布边缘损墨)
    assert abs(float(out.sum()) - float(canvas.sum())) < 0.05 * float(canvas.sum())


def test_channel_range_and_identity():
    rng = torch.Generator().manual_seed(9)
    progs = [GM.sample_uniform_prog(rng) for _ in range(2)]
    clean = GM.raster(progs)
    assert torch.equal(GM.channel(clean, 0.0, None), clean)
    out = GM.render_channel(progs, 1.0, torch.Generator().manual_seed(13))
    assert out.shape == clean.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(out, clean)
