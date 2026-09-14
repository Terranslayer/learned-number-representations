# symemerge/numcode/pred/pool.py
"""数据池 (spec-v2 §6.3-6.5 + spec-v3.1 C4): 每条 = {逐格类别 k ∈ {0..3}^256, 标记 N, 源场景种子,
成绩 σ, 入池步, depth, generation}.

- 存类别不存位图 (抽出时重渲, 腐蚀重抽); 标记 N 只在池元数据里, 永不进前向 (A4: 本模块不 import 模型).
- 入池 (§6.4 + C4.3): 两条路径 (场景起链 gen=0 / 池起链 gen=亲本+1) 同机制, 均为 append + FIFO,
  逐出按年龄不按成绩 (A23: 不存在因成绩删除条目的代码路径; 亲本在产物入池后原样保留);
  初值 σ = 池内当前成绩中位数; 容量 C_pool, 满则 FIFO 逐出最老.
- 抽取 (§6.5 + C4.4): 按 N 分层 (阶段 1 逐 N / 阶段 2 按 N 带), 跨层均匀配额, 层内
  P(i) = (1−ε)·softmax(σ/T_pool) + ε/|层|; 抽出并读过后 σ 用当次 w_t 加权六任务均值 EMA 更新 (C4.5,
  加权在调用方算好传入).
- 留出防火墙 (§6.6): 池只收场景/池链闭环笔记, 场景不含留出 N ⇒ 池中无留出 N (A3 在 checks 里核对).
- 换代动力学读数 (C10.7/C10.8): gen 直方图、逐 gen σ、年龄分布、池内 N 直方图、逐格类别成对汉明距
  (码族多样性; 塌陷即向单一码锁定).
"""
import torch

from .. import geometry as GM

T = GM.T
SIGMA0_EMPTY = 0.5     # [C] 池空时首批条目的初值 (中位数无定义; 全体同值 ⇒ 抽取均匀, 无偏向)


class DataPool:
    def __init__(self, cap):
        assert cap > 0
        self.cap = int(cap)
        self.k = torch.zeros(self.cap, T, dtype=torch.uint8)
        self.n = torch.zeros(self.cap, dtype=torch.long)
        self.zseed = torch.zeros(self.cap, dtype=torch.long)
        self.sigma = torch.zeros(self.cap)
        self.step_in = torch.zeros(self.cap, dtype=torch.long)
        self.depth = torch.zeros(self.cap, dtype=torch.long)   # v3 §3.2: 该纸在链上的深度
        self.gen = torch.zeros(self.cap, dtype=torch.long)     # v3.1 C4.2: generation (场景产物 0, 池链产物 亲本+1)
        self.size = 0
        self.head = 0                 # 下一个写入位 (环形; size==cap 时即最老条目)
        self.n_admit = 0
        self.n_evict = 0

    # ------------------------------------------------------------ 入池
    def median_sigma(self):
        return float(self.sigma[:self.size].median()) if self.size > 0 else SIGMA0_EMPTY

    def admit(self, ks, ns, zseeds, step, mask=None, depths=None, gens=None):
        """ks (m,T) long/uint8 (CPU), ns (m,), zseeds (m,); mask (m,) bool 可选 = 已按 ρ 抽好的入池位;
        depths (m,) 可选 = 链深度 (省略记 0); gens (m,) 可选 = generation (省略记 0, C4.2).
        初值 σ = 当前池中位数 (入池前一次取定, 同批同值). 只 append, 满则 FIFO 逐出最老 (A23)."""
        if depths is None:
            depths = torch.zeros(ks.shape[0], dtype=torch.long)
        if gens is None:
            gens = torch.zeros(ks.shape[0], dtype=torch.long)
        if mask is not None:
            idx = mask.nonzero().squeeze(1)
            if idx.numel() == 0:
                return 0
            ks, ns, zseeds, depths, gens = ks[idx], ns[idx], zseeds[idx], depths[idx], gens[idx]
        m = ks.shape[0]
        if m == 0:
            return 0
        s0 = self.median_sigma()
        for i in range(m):
            j = self.head
            if self.size == self.cap:
                self.n_evict += 1
            self.k[j] = ks[i].to(torch.uint8)
            self.n[j] = int(ns[i])
            self.zseed[j] = int(zseeds[i])
            self.sigma[j] = s0
            self.step_in[j] = int(step)
            self.depth[j] = int(depths[i])
            self.gen[j] = int(gens[i])
            self.head = (self.head + 1) % self.cap
            self.size = min(self.size + 1, self.cap)
        self.n_admit += m
        return m

    # ------------------------------------------------------------ 抽取
    def probs(self, T_pool, eps):
        s = self.sigma[:self.size]
        soft = torch.softmax(s / max(float(T_pool), 1e-8), dim=0)
        return (1.0 - eps) * soft + eps / self.size

    def sample(self, B, T_pool, eps, rng):
        """全池成绩加权抽 B 条 (v2 制, 保留为参照/兼容路). 返回 idx (B,) long."""
        assert self.size > 0
        p = self.probs(T_pool, eps)
        rep = B > self.size
        return torch.multinomial(p, B, replacement=rep, generator=rng)

    def sample_stratified(self, B, T_pool, eps, rng, band_w=1):
        """按 N 分层抽取 (v3.1 C4.4, 无条件生效): 层 = N (band_w=1, 阶段 1) 或 N 带 ⌊(N−1)/band_w⌋
        (阶段 2); 跨层均匀配额 (余数随机层, CRN), 层内 (1−ε)·softmax(σ/T) + ε/|层|; 配额超层容量时
        层内放回. 层数 > B 时均匀抽 B 个层各 1 条. 返回 idx (B,) long."""
        assert self.size > 0
        ns = self.n[:self.size]
        keys = (ns - 1) // max(int(band_w), 1)
        uk = keys.unique()
        nk = uk.numel()
        if nk > B:
            pick = uk[torch.randperm(nk, generator=rng)[:B]]
            quota = {int(u): 1 for u in pick}
        else:
            base, rem = B // nk, B % nk
            quota = {int(u): base for u in uk}
            if rem > 0:
                for u in uk[torch.randperm(nk, generator=rng)[:rem]]:
                    quota[int(u)] += 1
        out = []
        for u, q in quota.items():
            if q == 0:
                continue
            members = (keys == u).nonzero().squeeze(1)
            s = self.sigma[:self.size][members]
            soft = torch.softmax(s / max(float(T_pool), 1e-8), dim=0)
            p = (1.0 - eps) * soft + eps / members.numel()
            sel = torch.multinomial(p, q, replacement=q > members.numel(), generator=rng)
            out.append(members[sel])
        return torch.cat(out)

    def prob_ratio(self, T_pool, eps):
        """最高分与最低分条目的抽取概率比 (§9.2 T_pool 标准: 落在 [3,10]; 全池口径, 分层后作层内近似)."""
        if self.size == 0:
            return float("nan")
        p = self.probs(T_pool, eps)
        return float(p.max() / p.min())

    def update_scores(self, idx, acc, eta):
        """σ_i ← (1−η)σ_i + η·acc (acc = 当次 w_t 加权六任务均值 ∈ [0,1], C4.5, 调用方算好)."""
        idx = idx.cpu()
        a = acc.detach().float().cpu()
        self.sigma[idx] = (1.0 - eta) * self.sigma[idx] + eta * a

    def classes(self, idx):
        return self.k[idx.cpu()].long()

    def labels(self, idx):
        return self.n[idx.cpu()].clone()

    def gens(self, idx):
        return self.gen[idx.cpu()].clone()

    # ------------------------------------------------------------ 仪表 / 状态
    def pair_hamming(self, n_pairs, rng):
        """池内条目成对逐格类别汉明距 (C10.8 码族多样性): 随机 n_pairs 对 (CRN) 的 overall 均值 +
        同 N 对的均值 (码族口径: 同 N 异码才是多样性; 异 N 本应异码). 池 <2 条 → nan."""
        if self.size < 2:
            return dict(overall=float("nan"), same_n=float("nan"), n_same=0)
        i = torch.randint(0, self.size, (n_pairs,), generator=rng)
        j = torch.randint(0, self.size, (n_pairs,), generator=rng)
        ok = i != j
        i, j = i[ok], j[ok]
        ham = (self.k[i] != self.k[j]).float().mean(1)
        same = self.n[i] == self.n[j]
        return dict(overall=round(float(ham.mean()), 4) if ham.numel() else float("nan"),
                    same_n=round(float(ham[same].mean()), 4) if bool(same.any()) else float("nan"),
                    n_same=int(same.sum()))

    def stats(self, T_pool=None, eps=None, step=None, rng=None, n_pairs=512):
        out = dict(size=self.size, cap=self.cap, n_admit=self.n_admit, n_evict=self.n_evict)
        if self.size > 0:
            s = self.sigma[:self.size]
            q = torch.quantile(s, torch.tensor([0.1, 0.5, 0.9]))
            out.update(sig_q10=round(float(q[0]), 4), sig_med=round(float(q[1]), 4),
                       sig_q90=round(float(q[2]), 4), sig_min=round(float(s.min()), 4),
                       sig_max=round(float(s.max()), 4),
                       age_med=int(torch.median(self.step_in[:self.size]).item()),
                       ns=len(set(self.n[:self.size].tolist())))
            dep = self.depth[:self.size]
            out["depth_hist"] = {str(int(d)): int((dep == d).sum()) for d in dep.unique()}   # v3 §11.1
            g = self.gen[:self.size]
            out["gen_hist"] = {str(int(u)): int((g == u).sum()) for u in g.unique()}          # C10.7
            out["gen_max"] = int(g.max())
            out["gen_sigma"] = {str(int(u)): round(float(s[g == u].median()), 4) for u in g.unique()}  # C10.8 逐 gen σ 中位
            nsv = self.n[:self.size]
            out["n_hist"] = {str(int(u)): int((nsv == u).sum()) for u in nsv.unique()}        # C10.7 池内 N 直方图
            if step is not None:                                                              # C10.8 年龄分布
                age = (int(step) - self.step_in[:self.size]).float()
                aq = torch.quantile(age, torch.tensor([0.1, 0.5, 0.9]))
                out["age_q"] = [int(aq[0]), int(aq[1]), int(aq[2])]
            if rng is not None:
                out["pair_ham"] = self.pair_hamming(n_pairs, rng)                             # C10.8 码族多样性
            if T_pool is not None:
                out["prob_ratio"] = round(self.prob_ratio(T_pool, eps), 3)
        return out

    def state(self):
        return dict(cap=self.cap, size=self.size, head=self.head, n_admit=self.n_admit,
                    n_evict=self.n_evict, k=self.k[:self.size].clone(), n=self.n[:self.size].clone(),
                    zseed=self.zseed[:self.size].clone(), sigma=self.sigma[:self.size].clone(),
                    step_in=self.step_in[:self.size].clone(), depth=self.depth[:self.size].clone(),
                    gen=self.gen[:self.size].clone())

    def load_state(self, d):
        assert int(d["cap"]) == self.cap, (d["cap"], self.cap)
        n = int(d["size"])
        self.size, self.head = n, int(d["head"])
        self.n_admit, self.n_evict = int(d["n_admit"]), int(d["n_evict"])
        self.k[:n] = d["k"]
        self.n[:n] = d["n"]
        self.zseed[:n] = d["zseed"]
        self.sigma[:n] = d["sigma"]
        self.step_in[:n] = d["step_in"]
        if "depth" in d:                                   # 旧检查点无此字段 (v2), 保持零
            self.depth[:n] = d["depth"]
        if "gen" in d:                                     # v3.1 前检查点无此字段, 保持零
            self.gen[:n] = d["gen"]
