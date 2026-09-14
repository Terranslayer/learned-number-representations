#!/usr/bin/env python
"""迭代重学甲案 步 1 验尺 (只读; spec 2026-08-24-iterated-relearning-phase-spec.md §6.2/§8 步 1).

转移测验仪器的资格考 + 零假设表当场实测:
  码 = CAL 见证码三件 (随机码/一元码/位值码; 台架见证, 永不给智能体看) + 起点码 (检查点自己写的纸).
  每码: 教一个全新新读者 (同构主干重初始化) 于曝光子集 (draw_exposure, e_exp=20, 分带 2/4/4/6/4),
  固定预算 (steps_teach × batch 32), 考 未曝光 68 数: LIN 五任务 / t246 / 直读容差 / 五分带.
  验尺门: 一元码与位值码转移须显著高于随机码地板; 容量门: 教学集内成绩到线.

用法 (pod GPU):
  PYTHONPATH=. ~/venv/bin/python outputs/eval_relearn_gauge.py \
      --ckpt outputs/nc_cm_k1_stage2_afrz_k2_orgw2/ckpt_last.pt \
      --out outputs/nc_relearn_gauge_29000 [--budgets 300,1000] [--seed 0]

产物: <out>/gauge.json (零假设表) + 逐码逐预算行打印.
"""
import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import symemerge.numcode.data as D                     # noqa: E402
import symemerge.numcode.geometry as GM                # noqa: E402
import symemerge.numcode.pred.relearn as RL            # noqa: E402
import symemerge.numcode.pred.render as R              # noqa: E402
import symemerge.numcode.pred.step as ST               # noqa: E402
import symemerge.numcode.pred.trainer as TR            # noqa: E402
from symemerge.numcode.grpo import TaskWeights         # noqa: E402
from symemerge.numcode.pred.data import LIN, build_batch   # noqa: E402

EMPTY = 0          # 类别 0 = 空 (WriteHead 四类 {空, 点, 横, 竖})


# ---------------------------------------------------------------- CAL 见证码 (逐 N 类别图; 台架见证)
def cal_random(rng, n_cells=20):
    """随机码 (查表刻度): 每 N 一张固定随机稀疏图 (n_cells 格, 随机非空类); 同 N 各实例同图."""
    tabs = {}
    for n in D.train_ns(GM.K_MAX):
        k = torch.zeros(GM.T, dtype=torch.long)
        cells = torch.randperm(GM.T, generator=rng)[:n_cells]
        k[cells] = torch.randint(1, R.NCLS, (n_cells,), generator=rng)
        tabs[n] = k
    return tabs


def cal_unary(**_):
    """一元码: 前 N 格 (行主序) 各一枚点章 (类 1)."""
    return {n: torch.cat([torch.ones(n, dtype=torch.long),
                          torch.zeros(GM.T - n, dtype=torch.long)])
            for n in D.train_ns(GM.K_MAX)}


def cal_posval(**_):
    """位值码 (基 4, 章型当数字; §11 CAL4 同义): N = Σ d_j·4^j, 位 j 的格 = 第 0 行第 2j 列,
    格类 = 数位值 d_j (0=空/1=点/2=横/3=竖)."""
    tabs = {}
    for n in D.train_ns(GM.K_MAX):
        k = torch.zeros(GM.T, dtype=torch.long)
        v = int(n)
        for j in range(4):
            k[2 * j] = v % 4
            v //= 4
        tabs[n] = k
    return tabs


# ---------------------------------------------------------------- 教学与考试
def batch_from_tabs(tabs, ns_pick, cfg, rng_ch, rng_th, dev, k_max):
    """从类别图表抽一批: 每行一个 N 的实例 (重渲 + 信道), θ 新抽."""
    kk = torch.stack([tabs[int(n)] for n in ns_pick]).to(dev)
    draw = R.draw_channel(kk.shape[0], cfg.s, rng_ch, dev, cfg.occ_k)
    with torch.no_grad():
        x = R.channel(R.render_classes(kk, draw), draw)
    ns = torch.as_tensor([int(n) for n in ns_pick], device=dev)
    th, tg = RL._theta_targets(ns, k_max, rng_th, dev)
    return x, ns, th, tg


def teach_and_exam(tabs, exp, cfg, seed, steps_teach, dev, batch=32, exam_per_n=8):
    """教新读者于曝光子集 → 考未曝光. 返回 dict(教学末窗 acc, 未曝光块, 曝光块)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reader = RL.Relearner().to(dev)
    opt = torch.optim.AdamW(reader.parameters(), lr=float(cfg.sr_lr), weight_decay=float(cfg.wd))
    rng_pick = torch.Generator().manual_seed(seed + 1)
    rng_ch = torch.Generator().manual_seed(seed + 2)
    rng_th = torch.Generator().manual_seed(seed + 3)
    exp_t = list(exp)
    tail = []
    for t in range(int(steps_teach)):
        pick = [exp_t[int(i)] for i in torch.randint(0, len(exp_t), (batch,), generator=rng_pick)]
        x, ns, th, tg = batch_from_tabs(tabs, pick, cfg, rng_ch, rng_th, dev, GM.K_MAX)
        l, accs, _ = RL.sr_rows(reader, x, ns, th, tg, float(cfg.sr_mu))
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), float(cfg.clip))
        opt.step()
        if t >= int(steps_teach) - 20:
            tail.append(float(accs.mean()))
    # 考试: 未曝光/曝光 各 exam_per_n 实例每 N, 固定考试 CRN
    rng_ex_ch = torch.Generator().manual_seed(seed + 4242)
    rng_ex_th = torch.Generator().manual_seed(seed + 4243)
    out = {}
    for name, ns_set in (("unexp", [n for n in D.train_ns(GM.K_MAX) if n not in exp]),
                         ("exp", exp_t)):
        A, RD, NS = [], [], []
        rows = [n for n in ns_set for _ in range(exam_per_n)]
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                x, ns, th, tg = batch_from_tabs(tabs, rows[i:i + 64], cfg, rng_ex_ch, rng_ex_th, dev, GM.K_MAX)
                _, accs, rd = RL.sr_rows(reader, x, ns, th, tg, 0.0)
                A.append(accs.cpu())
                RD.append(rd.cpu())
                NS.append(ns.cpu())
        a, rd, ns = torch.cat(A), torch.cat(RD), torch.cat(NS)
        t246 = a[:, list(RL.T246)].mean(1)
        err = (rd - ns).abs()
        bands = {}
        for lo, hi in RL.BANDS:
            bm = (ns >= lo) & (ns <= hi)
            if int(bm.sum()) > 0:
                bands[f"{lo}-{hi}"] = round(float(a[bm].mean()), 4)
        out[name] = dict(
            acc={t: round(float(a[:, i].mean()), 4) for i, t in enumerate(LIN)},
            mean=round(float(a.mean()), 4), se=round(float(a.mean(1).std() / max(len(ns), 1) ** 0.5), 4),
            t246=round(float(t246.mean()), 4), t246_se=round(float(t246.std() / max(len(ns), 1) ** 0.5), 4),
            tol=[round(float((err <= e).float().mean()), 4) for e in (0, 1, 3, 5)],
            bands=bands, n=int(len(ns)))
    out["teach_acc_tail20"] = round(sum(tail) / max(len(tail), 1), 4)
    return out


def origin_tabs(model, cfg, dev, w, seed):
    """起点码: 检查点自己在评测场景上写的纸 (逐 N 8 张类别图; 评测 CRN 同口径 seed+4242)."""
    es = TR.build_eval_sets(cfg, GM.K_MAX)
    rng_eval = torch.Generator().manual_seed(int(cfg.seed) + 4242)
    tabs = {}     # n -> list[Tensor(T)]
    items = es["groups"]
    with torch.no_grad():
        for i in range(0, len(items), 64):
            sb = build_batch(items[i:i + 64], dev)
            draw = R.draw_channel(sb["B"], cfg.s, rng_eval, dev, cfg.occ_k)
            tpl = R.templates(draw.u, draw.s)
            out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
            kk = out["writes"][-1]["k"].cpu()
            for b in range(sb["B"]):
                tabs.setdefault(int(sb["ns"][b]), []).append(kk[b])
    return tabs


class MultiTabs:
    """逐 N 多实例类别图表 (起点码): [int(n)] 轮转返回实例 (教学多样性)."""

    def __init__(self, tabs):
        self.tabs = {n: list(v) for n, v in tabs.items()}
        self.i = {n: 0 for n in tabs}

    def __getitem__(self, n):
        lst = self.tabs[n]
        j = self.i[n] % len(lst)
        self.i[n] = self.i[n] + 1
        return lst[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/nc_cm_k1_stage2_afrz_k2_orgw2/ckpt_last.pt")
    ap.add_argument("--out", default="outputs/nc_relearn_gauge_29000")
    ap.add_argument("--budgets", default="300,1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e_exp", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    cfg = TR.TrainCfg(chain_k=1, d1_cm=1, origin_write=0, pred_on=0, gate_on=0, chain_from_pool=0,
                      alpha_freeze=1, lam=0.0034, lam_servo=0, seed=a.seed,
                      workers=0, kit_workers=0)
    model = TR.build_model(cfg).to(dev)
    TR.load_ckpt(a.ckpt, model)
    model.eval()
    w = TaskWeights().weights(dev)
    # 曝光子集 (验尺用固定期 0 种子, 与训练机制同式)
    exp = RL.draw_exposure(torch.Generator().manual_seed(RL._mix(a.seed + 9001, 0)), GM.K_MAX, a.e_exp)
    print(f"[gauge] dev={dev} ckpt={a.ckpt} e_exp={a.e_exp} 𝒩_exp={list(exp)}", flush=True)
    rng_cal = torch.Generator().manual_seed(a.seed + 71)
    codes = dict(random=cal_random(rng_cal), unary=cal_unary(), posval=cal_posval())
    codes["origin"] = MultiTabs(origin_tabs(model, cfg, dev, w, a.seed))
    res = dict(ckpt=a.ckpt, e_exp=a.e_exp, exp=list(exp), seed=a.seed,
               cfg=dataclasses.asdict(cfg), codes={})
    for name, tabs in codes.items():
        res["codes"][name] = {}
        for bud in [int(x) for x in a.budgets.split(",")]:
            r = teach_and_exam(tabs, exp, cfg, a.seed + 100 * bud, bud, dev)
            res["codes"][name][str(bud)] = r
            u, e = r["unexp"], r["exp"]
            print(f"[gauge] {name:7s} 预算 {bud:5d} | 教学末 {r['teach_acc_tail20']} 曝光考 {e['mean']} | "
                  f"未曝光 五任务 {u['mean']}±{u['se']} t246 {u['t246']}±{u['t246_se']} "
                  f"直读容差 {u['tol']} 分带 {u['bands']}", flush=True)
    with open(os.path.join(a.out, "gauge.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"[gauge] 写出 {a.out}/gauge.json", flush=True)


if __name__ == "__main__":
    main()
