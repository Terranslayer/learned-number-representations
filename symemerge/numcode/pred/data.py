# symemerge/numcode/pred/data.py
"""数据流 (spec-v2 §6.1, §6.7): 场景流 (程序生成, 带候选集/θ/标签) · 池抽笔记的候选包 (kit) ·
随机画布 (render.random_canvas_classes, 在 step 里就地抽) · 加法流 ([U] 2026-08-27: 场景 N + 场景 m → 一张纸, 真值 N+m).

场景项 = data.sample_group(rng, k) 一份 (查询场景 (N,Z) + 六候选 + θ + 标签 + zseed);
池抽笔记的 kit = data.sample_group(rng, k, n=N, with_scene=False) (只要候选/θ/标签; 任务 5 按 §4.2
笔记行: 正确项 场景 (N,Z'), 硬负 场景 (N',Z), 随机负 3 -- 与场景行同一构造器, 只是 query 换成笔记).

kit 的 N 由池抽取决定 (主进程的池状态), 故 kit 用进程池按「下一步的抽取」提前一步生产 (KitMaker):
第 t 步末 (入池 + 成绩更新后) 抽第 t+1 步的条目并提交渲染, 第 t+1 步开头收取. 抽取分布与 §6.5 逐字一致.

留出防火墙 (§6.6): 一切经 data.train_ns(k) 抽 N, 留出 N 的场景不进任何 batch (A3).
所有随机性来自 CPU generator (CRN).
"""
import concurrent.futures as cf
import multiprocessing as mp

import torch

from .. import data as D
from .. import geometry as GM

_PHI = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_MOD = 2 ** 62
TASKS = ("t1", "t2", "t3", "t4", "t5", "t6")
LIN = ("t1", "t2", "t3", "t4", "t6")


def th_idx(items, dev):
    """项列表 -> θ 表索引 dict (tau-1 / P_CHOICES 下标 / M_CHOICES 下标)."""
    return dict(
        tau=torch.tensor([it["theta"]["tau"] - 1 for it in items], device=dev),
        p=torch.tensor([D.P_CHOICES.index(it["theta"]["p"]) for it in items], device=dev),
        m=torch.tensor([D.M_CHOICES.index(it["theta"]["m"]) for it in items], device=dev))


def build_batch(items, dev, with_scene=True):
    """项列表 -> 训练张量 dict(x1?, ns, th, tg, truth5, cands, zseed)."""
    out = dict(
        ns=torch.tensor([it["n"] for it in items], dtype=torch.long, device=dev),
        th=th_idx(items, dev),
        tg={t: torch.tensor([it["targets"][t] for it in items], device=dev) for t in LIN},
        truth5=torch.tensor([it["truth5"] for it in items], device=dev),
        cands=torch.stack([it["cands"] for it in items]).to(dev),
        zseed=torch.tensor([int(it.get("zseed", 0)) for it in items], dtype=torch.long),
        B=len(items))
    if with_scene:
        out["x1"] = torch.stack([it["scene"] for it in items]).to(dev)
        if "scene_m" in items[0]:                          # 加法流项 ([U] 2026-08-27): 第二张场景 (m 个物体) + 元数据
            out["x_aux"] = torch.stack([it["scene_m"] for it in items]).to(dev)
            out["n_a"] = torch.tensor([it["n_a"] for it in items], dtype=torch.long, device=dev)
            out["m_add"] = torch.tensor([it["m"] for it in items], dtype=torch.long, device=dev)
    return out


class SceneStream(torch.utils.data.IterableDataset):
    """无限场景项流: N ~ U(train_ns(k)), Z 全新; 每项含候选集/θ/标签."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = torch.Generator().manual_seed((self.seed * _PHI + (wid + 7) * _MIX) % _MOD)
        while True:
            yield D.sample_group(rng, self.k)


def make_scene_loader(k, seed, B, workers):
    return torch.utils.data.DataLoader(
        SceneStream(k, seed), batch_size=B, num_workers=workers,
        collate_fn=lambda x: x, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None)


class AddStream(torch.utils.data.IterableDataset):
    """加法流无限项流 ([U] 2026-08-27 双场景 N+m): 每项 = 场景 (N, Z1) + 场景 (m, Z2) + 和 s 的候选包/θ/标签.
    独立于场景流 (自己的 seed/worker 派生) ⇒ 开加法流不改变场景流抽到的项 (主流程输入与基线逐项同)."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = torch.Generator().manual_seed((self.seed * _PHI + (wid + 7) * _MIX) % _MOD)
        while True:
            yield D.sample_add_group(rng, self.k)


def make_add_loader(k, seed, B, workers):
    return torch.utils.data.DataLoader(
        AddStream(k, seed), batch_size=B, num_workers=workers,
        collate_fn=lambda x: x, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None)


def _noop(i):
    return i


def make_kits(ns, k, seed):
    """(工作进程) 给定标记列表 -> kit 列表 (无场景). 纯 CPU."""
    rng = torch.Generator().manual_seed(int(seed) % _MOD)
    return [D.sample_group(rng, k, n=int(n), with_scene=False) for n in ns]


class KitMaker:
    """池抽笔记候选包的提前一步生产者: submit(ns, k, seed) 立即返回, collect() 收取上次提交.
    workers=0 时同步生产 (测试/评测用)."""

    def __init__(self, workers=4, chunk=8):
        self.workers = int(workers)
        self.chunk = int(chunk)
        self.ex = None
        if self.workers > 0:
            ctx = mp.get_context("fork")
            self.ex = cf.ProcessPoolExecutor(self.workers, mp_context=ctx)
            # 预热: 立刻 fork 出全部工作进程 (在调用方初始化 CUDA 之前), 之后只做纯 CPU 渲染
            futs = [self.ex.submit(_noop, i) for i in range(self.workers)]
            for f in futs:
                f.result()
        self._pending = None

    def submit(self, ns, k, seed):
        assert self._pending is None, "上一份 kit 尚未 collect"
        ns = [int(n) for n in ns]
        if self.ex is None:
            self._pending = ("sync", make_kits(ns, k, seed))
            return
        futs = []
        for i in range(0, len(ns), self.chunk):
            futs.append(self.ex.submit(make_kits, ns[i:i + self.chunk], k, seed + 7919 * i))
        self._pending = ("async", futs)

    def has_pending(self):
        return self._pending is not None

    def collect(self):
        kind, obj = self._pending
        self._pending = None
        if kind == "sync":
            return obj
        out = []
        for f in obj:
            out += f.result()
        return out

    def make_sync(self, ns, k, seed):
        return make_kits(ns, k, seed)

    def close(self):
        if self.ex is not None:
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = None
