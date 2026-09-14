# symemerge/numcode/probe.py
"""双探针 (params.md §10e, [U] s114 计划 C2/C3): 像素级 N 可分性直接入奖励.

p3b 定位的瓶颈: 信息在墨里 (像素探针 0.875 对 chance 0.125) 而在位读者只提取
0.695, 读写自举耦合下落墨对空白差价仅 +0.005 (λ0−λ), 点火 ~1 数量/4–5k 步
(常数台账). 探针奖励绕开在位读者: "画布上像素可分" 当步即付 α, 不等读者追上.

probe_in (入环): 下采样原始像素 (avg_pool2d 4x, 与 pixel_probe 仪表同构) 上的
线性读头, 每步在 FIFO 缓冲上岭回归闭式重训 -- 真"每步重训": 无优化器状态无滞后.
先拟合后入册: 当步样本不进当步拟合, 打分恒为缓冲外样本. 逐 rollout
r_probe = margin/σ (v3 [U] p3c-alpha2 边际制: margin = 真类后验减最强竞争者,
可负、饱和区仍有梯度; 滑动 σ 带下限归一防码动稀释), 后验 = 岭 one-hot 输出
(指示矩阵回归的类后验最小二乘估计), 对 m 份独立腐蚀渲染取均值 (§10 改4 同款).

probe_out (体外): 独立种子 + 独立结构 (单隐层 MLP), 同缓冲在线训练, 永不进奖励.
它与在位读者的缺口 = 读者侧欠账 (预测 Z5); 它与 probe_in 的差 = 写者是否在钻
入环探针的具体权重 (同数据不同结构/种子, 拉开 = 钻空子).

奖励路由: r_probe 是 no-grad 标量, 只经 GRPO 优势进写者; 探针参数不在模型
优化器里; 特征取自原始像素, 结构上不经主干 -- 对主干/读头零梯度.
"""
import torch
import torch.nn.functional as F

from . import data as D
from . import geometry as GM

POOL = 4                # 特征 = avg_pool2d(4) 展平 (128/4=32)^2 = 1024 维,
#                         与 pixel_probe 仪表同一下采样 (信息口径一致)
CAP = 4096              # FIFO 容量 = 16 步窗 (256 rollout/步); 码漂移尺度
#                         ~1 数量/4–5k 步 (常数台账, p3b) ⇒ 窗内分布近平稳
MIN_FIT = 512           # 缓冲低于此不拟合: r_probe 全零 = 组内常数, 优势无效应
RIDGE_LAM = 1.0         # 岭系数; 发车前在 p3b 终局码表上实测对 300 步逻辑探针
#                         的差距, 复核记录在 params.md §10e 常数复核表
OUT_HIDDEN = 256
OUT_LR = 1e-3
OUT_WD = 1e-4
OUT_INNER = 8           # probe_out 每外步内步数 (×512 批 ≈ 每 500 步评测窗 4k 次更新)
OUT_BS = 512
SEED_OFF = 31337        # probe_out 独立种子偏移
SIGMA_FLOOR = 0.05      # [U] p3c-alpha2: 边际归一的滑动 σ 下限
SIGMA_DECAY = 0.99      # [C] σ 的 EMA 衰减 (~100 步窗; p3c-alpha 稀释尺度数百步)


def pool_feats(x):
    """(B, SIDE, SIDE) 画布 -> (B, 1024) 池化像素特征."""
    return F.avg_pool2d(x.unsqueeze(1), POOL).flatten(1)


class ProbePair:
    """probe_in (岭闭式, 入环) + probe_out (单隐层, 体外) + 共享 FIFO 缓冲."""

    def __init__(self, cfg, dev):
        self.dev = dev
        fd = (GM.SIDE // POOL) ** 2
        self.fd = fd
        self.X = torch.zeros(CAP, fd, device=dev)
        self.y = torch.zeros(CAP, dtype=torch.long, device=dev)
        self.n = 0                       # 已填样本数
        self.ptr = 0                     # FIFO 写指针
        self.W = None                    # (fd+1, K_MAX) 岭解, 每步重解
        self.sigma = None                # margin 批 σ 的 EMA (v3 归一分母)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + SEED_OFF)   # 独立种子, 不扰训练流
            self.out = torch.nn.Sequential(
                torch.nn.Linear(fd, OUT_HIDDEN), torch.nn.GELU(),
                torch.nn.Linear(OUT_HIDDEN, GM.K_MAX))
        self.out = self.out.to(dev)
        self.opt = torch.optim.Adam(self.out.parameters(), lr=OUT_LR,
                                    weight_decay=OUT_WD)
        self.rng = torch.Generator().manual_seed(cfg.seed + SEED_OFF + 1)

    # ---------------------------------------------------------- probe_in
    @torch.no_grad()
    def _fit(self):
        """岭回归闭式重训: W = (X'X + λI)^{-1} X'Y, 偏置经特征增广."""
        if self.n < MIN_FIT:
            self.W = None
            return
        X = self.X[:self.n]
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=self.dev)], 1)
        Y = F.one_hot(self.y[:self.n], GM.K_MAX).float()
        A = X1.T @ X1
        A += RIDGE_LAM * torch.eye(A.shape[0], device=self.dev)
        self.W = torch.linalg.solve(A, X1.T @ Y)

    def _pred(self, feats, ns_cols):
        """受限 argmax (只在当前课程档训练域的列上比) -> 预测 N 值."""
        X1 = torch.cat([feats, torch.ones(feats.shape[0], 1,
                                          device=self.dev)], 1)
        sc = (X1 @ self.W)[:, ns_cols]
        return ns_cols[sc.argmax(-1)] + 1

    @torch.no_grad()
    def reward_margin(self, x2_list, ns, k):
        """先拟合 (缓冲 = 本步之前) 再打分 (v3, [U] p3c-alpha2):
        r_probe = margin / max(σ_slide, SIGMA_FLOOR),
        margin = p̂(N|c) − max_{M≠N, M∈训练域} p̂(M|c), 对 m 份腐蚀渲染取均值后归一.
        p̂ 仍为岭 one-hot 后验估计 (v2). margin 特性: 可负 (读成别人 = 负酬);
        饱和区仍有梯度 (真类到顶后 runner-up 质量还在动, [U] 登记理由);
        滑动 σ 归一防「码一动差价变薄」(p3c-alpha 实测重组期 r_probe 0.69→0.34
        自稀释). σ = 喂出量 (m 抽均值后 margin) 的逐步批标准差 EMA, 先更新后归一,
        冷启动步不更新."""
        self._fit()
        if self.W is None:
            return torch.zeros(ns.shape[0], device=self.dev)
        cols = torch.tensor(D.train_ns(k), device=self.dev) - 1
        pos = torch.searchsorted(cols, ns - 1).unsqueeze(1)
        margins = []
        for x2 in x2_list:
            f = pool_feats(x2)
            X1 = torch.cat([f, torch.ones(f.shape[0], 1, device=self.dev)], 1)
            sc = (X1 @ self.W)[:, cols]
            pN = sc.gather(1, pos).squeeze(1)
            rival = sc.scatter(1, pos, float("-inf")).max(-1).values
            margins.append(pN - rival)
        m = torch.stack(margins).mean(0)
        sig = float(m.std())
        self.sigma = (sig if self.sigma is None
                      else SIGMA_DECAY * self.sigma + (1 - SIGMA_DECAY) * sig)
        return m / max(self.sigma, SIGMA_FLOOR)

    # ---------------------------------------------------------- 缓冲/probe_out
    @torch.no_grad()
    def push(self, x2, ns):
        f = pool_feats(x2)
        B = f.shape[0]
        idx = (self.ptr + torch.arange(B, device=self.dev)) % CAP
        self.X[idx] = f
        self.y[idx] = ns - 1
        self.ptr = int((self.ptr + B) % CAP)
        self.n = min(self.n + B, CAP)

    def train_out(self):
        if self.n < MIN_FIT:
            return
        for _ in range(OUT_INNER):
            idx = torch.randint(0, self.n, (OUT_BS,),
                                generator=self.rng).to(self.dev)
            self.opt.zero_grad()
            F.cross_entropy(self.out(self.X[idx]), self.y[idx]).backward()
            self.opt.step()

    def step_update(self, x2, ns):
        """打分之后调用: 入册第 0 抽渲染 (与在线监督共源) + probe_out 在线更新."""
        self.push(x2, ns)
        self.train_out()

    # ---------------------------------------------------------- 评测/检查点
    @torch.no_grad()
    def eval_acc(self, tab, cfg, rng, dev):
        """贪心码表独立信道渲染一次 -> 两探针在环状态的受限 argmax 准确率
        (与 pixel_probe 同源数据, 但不重训 -- 报在环权重的当下水平)."""
        xs, ys = [], []
        for n in sorted(tab):
            x = GM.render_channel(tab[n], cfg.s, rng, cfg.occ_k)
            xs.append(x)
            ys += [n] * len(tab[n])
        f = pool_feats(torch.cat(xs).to(dev))
        y = torch.tensor(ys, device=dev)
        cols = torch.tensor(sorted(tab), device=dev) - 1
        out = dict(buf_n=self.n)
        if self.W is None:
            out["pin"] = None
        else:
            out["pin"] = round(float((self._pred(f, cols) == y)
                                     .float().mean()), 4)
        so = self.out(f)[:, cols]
        pred = cols[so.argmax(-1)] + 1
        out["pout"] = round(float((pred == y).float().mean()), 4)
        return out

    def state(self):
        return dict(X=self.X.cpu(), y=self.y.cpu(), n=self.n, ptr=self.ptr,
                    sigma=self.sigma, out=self.out.state_dict(),
                    opt=self.opt.state_dict(), rng=self.rng.get_state())

    @torch.no_grad()
    def load_state(self, d):
        self.X.copy_(d["X"].to(self.dev))
        self.y.copy_(d["y"].to(self.dev))
        self.n, self.ptr = int(d["n"]), int(d["ptr"])
        self.sigma = d.get("sigma")      # 旧检查点缺此键 → 冷启动重估
        self.out.load_state_dict(d["out"])
        self.opt.load_state_dict(d["opt"])
        self.rng.set_state(d["rng"])
