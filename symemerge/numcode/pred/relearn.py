# symemerge/numcode/pred/relearn.py
"""Iterated relearning: exposure selection, fresh readers, and transfer loss.

新读者 = 与主干同构、参数重新初始化的实例 (复用 Backbone + Heads 类).
学习期 (t_period 步) 边界: 重抽曝光子集 𝒩_exp (E_exp 个, 五训练分带比例分层, 最大余数法配额)
+ 新读者参数重初始化 (fork_rng, 期专用种子) + 其优化器清零. 期内:
  教学流 (只更新新读者): 曝光子集的 场景行 (复用 sb 的 θ/标签) + 纸 (本步链末纸 + 池内同 N 条目重渲,
    θ 新抽), 像素一律 detach; 损失 = LIN 五任务未加权 CE 均值 + sr_mu·直读 CE. t5 不带:
    检索要编码候选 6 张/题, 判据任务 t2/t4/t6 不受影响.
  转移流 (压力, 训练写者): 本步链末纸中 N ∉ 𝒩_exp 的行过新读者, 新读者参数临时 requires_grad=False
    ⇒ 梯度只经像素 → 直通 → 写路径 (b_c/α/Θ_E; W_D1 冻结); 权重 η_tr,
    期首 tr_warm 比例内置 0; η_tr=0 (仪表零臂) 时 no_grad 只测不传.
断言: A30 全关平价 / A31 梯度路由 / A32 曝光防火墙 / A33 换代作为 / A34 隔离 (checks.py).
exposure=0 时本模块不被 import、不被构造 (A30).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import data as D
from .. import geometry as GM
from ..model import D_MODEL
from . import render as R
from . import step as ST
from .data import LIN, build_batch
from .model import Backbone, Heads, PredModel

_PHI = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_MOD = 2 ** 62

BANDS = ((1, 9), (10, 25), (26, 44), (57, 82), (95, 112))   # 五训练分带 (params §5 报表分带)
T246 = (LIN.index("t2"), LIN.index("t4"), LIN.index("t6"))  # 判据任务列 (LIN 序内下标 1,3,4)


def _mix(a, b):
    return (a * _PHI + (b + 1) * _MIX) % _MOD


def band_quotas(e_exp, k):
    """五分带按容量比例配额 (最大余数法). k < K_MAX (阶段 1) ⇒ 单带 = 全训练域 (曝光机制本不激活, 测试用)."""
    ns = D.train_ns(k)
    if k < GM.K_MAX:
        return [list(ns)], [min(int(e_exp), len(ns))]
    groups = [[n for n in ns if lo <= n <= hi] for lo, hi in BANDS]
    sizes = torch.tensor([len(g) for g in groups], dtype=torch.float64)
    exact = sizes / sizes.sum() * float(e_exp)
    base = exact.floor().long()
    rem = int(e_exp) - int(base.sum())
    order = torch.argsort(exact - exact.floor(), descending=True)
    for i in range(rem):
        base[order[i]] += 1
    return groups, [int(x) for x in base]


def draw_exposure(rng, k, e_exp):
    """曝光子集 𝒩_exp: 分带比例分层、带内无放回均匀抽. 返回升序 tuple."""
    groups, quotas = band_quotas(e_exp, k)
    out = []
    for g, q in zip(groups, quotas):
        perm = torch.randperm(len(g), generator=rng)
        out += [g[int(i)] for i in perm[:q]]
    return tuple(sorted(out))


def _theta_targets(ns, k, rng, dev):
    """给一批 N 抽新 θ 并算五线性任务标签 (D.sample_theta/D.targets; CPU 专用流 rng)."""
    ths, tgs = [], []
    for n in ns.tolist():
        th = D.sample_theta(rng, k)
        ths.append(th)
        tgs.append(D.targets(int(n), th))
    th = dict(tau=torch.tensor([t["tau"] - 1 for t in ths], device=dev),
              p=torch.tensor([D.P_CHOICES.index(t["p"]) for t in ths], device=dev),
              m=torch.tensor([D.M_CHOICES.index(t["m"]) for t in ths], device=dev))
    tg = {t: torch.tensor([g[t] for g in tgs], device=dev) for t in LIN}
    return th, tg


class Relearner(nn.Module):
    """新读者: 主干同构 (既有 Backbone + Heads 类), 全新参数. 不带写头/预测件/闸 (只读者)."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.E = Backbone(d)
        self.heads = Heads(d)


def sr_rows(reader, x, ns, th, tg, mu):
    """新读者读一批图像 (场景或纸, 单槽 257 装配): LIN 五任务未加权 CE 均值 + mu·直读 CE.
    返回 (loss 标量, accs (B,5) LIN 序 detach, read_pred (B,) detach)."""
    e = reader.E.enc_seq(reader.E.tokenize(x), None, seg=False)
    tl = reader.heads.task_logits(e["cls"], th)
    loss = 0.0
    accs = []
    for t in LIN:
        loss = loss + F.cross_entropy(tl[t], tg[t])
        accs.append((tl[t].argmax(-1) == tg[t]).float())
    loss = loss / float(len(LIN))
    rl = reader.heads.read(e["cls"])
    loss = loss + float(mu) * F.cross_entropy(rl, ns - 1)
    return loss, torch.stack(accs, dim=1).detach(), (rl.argmax(-1) + 1).detach()


@torch.no_grad()
def exam_with_reader(reader, exp, model, es, cfg, w, dev, rng_eval, extra, chunk=64):
    """考试共用体 (甲案考官 = 孪生新读者; 乙案考官 = 主脑自身). 主模型给评测场景写纸 (chain_side,
    消耗 rng_eval — 同配置同耗 = 可配对), reader 读纸与场景; 未曝/曝 × 纸/场景 四块分列."""
    def _mask(ns):
        t = torch.as_tensor(exp, device=ns.device)
        return (ns.unsqueeze(1) == t.unsqueeze(0)).any(1)
    A_nt, A_sc, RD_nt, RD_sc, NS = [], [], [], [], []
    items = es["groups"]
    for i in range(0, len(items), chunk):
        sb = build_batch(items[i:i + chunk], dev)
        draw = R.draw_channel(sb["B"], cfg.s, rng_eval, dev, cfg.occ_k)
        tpl = R.templates(draw.u, draw.s)
        out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
        tg5 = {t: sb["tg"][t] for t in LIN}
        for x, al, rl_ in ((out["writes"][-1]["x"], A_nt, RD_nt), (sb["x1"], A_sc, RD_sc)):
            _, accs, rd = sr_rows(reader, x, sb["ns"], sb["th"], tg5, 0.0)
            al.append(accs.cpu())
            rl_.append(rd.cpu())
        NS.append(sb["ns"].cpu())
    a_nt, a_sc = torch.cat(A_nt), torch.cat(A_sc)
    r_nt, r_sc = torch.cat(RD_nt), torch.cat(RD_sc)
    ns = torch.cat(NS)
    um = ~_mask(ns)

    def block(a, rd, m):
        if int(m.sum()) == 0:
            return None
        acc = {t: round(float(a[m][:, i].mean()), 4) for i, t in enumerate(LIN)}
        t246 = round(float(a[m][:, list(T246)].mean()), 4)
        err = (rd[m] - ns[m]).abs()
        tol = [round(float((err <= e).float().mean()), 4) for e in (0, 1, 3, 5)]
        bands = {}
        for lo, hi in BANDS:
            bm = m & (ns >= lo) & (ns <= hi)
            if int(bm.sum()) > 0:
                bands[f"{lo}-{hi}"] = round(float(a[bm].mean()), 4)
        return dict(acc=acc, mean=round(float(a[m].mean()), 4), t246=t246, tol=tol,
                    bands=bands, n=int(m.sum()))

    return dict(**extra, exp=list(exp),
                unexp=block(a_nt, r_nt, um), exp_set=block(a_nt, r_nt, ~um),
                scene_unexp=block(a_sc, r_sc, um), scene_exp=block(a_sc, r_sc, ~um))


class Relearn:
    """甲案机械. exposure=0 时不构造 (A30); 阶段 2 才激活 (tick/teach/transfer/exam 由调用方门控)."""

    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev
        self.period = -1                 # 尚未开期
        self.steps_in = 0
        self.exp = ()                    # 𝒩_exp (升序 tuple)
        self.k = None                    # 当前课程档
        self.reader = None
        self.opt = None
        self.rng = torch.Generator().manual_seed(_mix(int(cfg.seed) + 9002, 0))   # 教学抽样/θ/信道 专用流
        self.last_teach_ns = None        # 断言用 (A32)
        self.last_tr_ns = None

    # ---------------------------------------------------------- 期界
    def rollover(self, k, step1):
        """开新学习期: 期号+1, 重抽 𝒩_exp, 新读者重初始化 (fork_rng 期专用种子, CPU 上构造后搬 dev),
        优化器全新 (状态清零). 返回登记行 dict."""
        self.period += 1
        self.steps_in = 0
        self.k = int(k)
        sub = torch.Generator().manual_seed(_mix(int(self.cfg.seed) + 9001, self.period))
        self.exp = draw_exposure(sub, self.k, int(self.cfg.e_exp))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_mix(int(self.cfg.seed) + 9003, self.period))
            self.reader = Relearner().to(self.dev)
        self.opt = torch.optim.AdamW(self.reader.parameters(), lr=float(self.cfg.sr_lr),
                                     weight_decay=float(self.cfg.wd))
        return dict(relearn_period=self.period, exp=list(self.exp), k=self.k)

    def tick(self, k, step1):
        """每步开头调用 (仅阶段 2): 未开期 / 课程档变 / 到期界 ⇒ rollover 并返回登记行; 否则 None."""
        if self.reader is None or self.k != int(k) or self.steps_in >= int(self.cfg.t_period):
            return self.rollover(k, step1)
        return None

    @property
    def eta_now(self):
        """期首豁免: 期内前 tr_warm 比例步 η_tr = 0 (新读者尚无知)."""
        if self.steps_in < float(self.cfg.tr_warm) * int(self.cfg.t_period):
            return 0.0
        return float(self.cfg.eta_tr)

    def exp_mask(self, ns):
        t = torch.as_tensor(self.exp, device=ns.device)
        return (ns.unsqueeze(1) == t.unsqueeze(0)).any(1)

    # ---------------------------------------------------------- 转移流 (压力; chain_train_step 内调用)
    def transfer(self, x_notes, sb):
        """N∉𝒩_exp 的链末纸 (attached) 过新读者. 返回 (η·loss | None, 计量 dict).
        新读者参数前向期间 requires_grad=False ⇒ 梯度只到像素 (A31); η=0 ⇒ no_grad 只测 (仪表零臂)."""
        mask = ~self.exp_mask(sb["ns"])
        n = int(mask.sum())
        if n == 0:
            self.last_tr_ns = torch.empty(0, dtype=torch.long)
            return None, dict(n_tr=0)
        th = {kk: v[mask] for kk, v in sb["th"].items()}
        tg = {t: sb["tg"][t][mask] for t in LIN}
        ns = sb["ns"][mask]
        self.last_tr_ns = ns.detach().cpu()
        eta = self.eta_now
        if eta == 0.0:
            with torch.no_grad():
                l, accs, _ = sr_rows(self.reader, x_notes[mask], ns, th, tg, float(self.cfg.sr_mu))
            return None, dict(n_tr=n, l_tr=float(l), tr_acc=float(accs.mean()),
                              tr_acc246=float(accs[:, list(T246)].mean()))
        req = [p.requires_grad for p in self.reader.parameters()]
        for p in self.reader.parameters():
            p.requires_grad_(False)
        l, accs, _ = sr_rows(self.reader, x_notes[mask], ns, th, tg, float(self.cfg.sr_mu))
        for p, r in zip(self.reader.parameters(), req):
            p.requires_grad_(r)
        return eta * l, dict(n_tr=n, l_tr=float(l), tr_acc=float(accs.mean()),
                             tr_acc246=float(accs[:, list(T246)].mean()))

    # ---------------------------------------------------------- 教学流 (只更新新读者; run() 主步后调用)
    def teach(self, sb, notes_det, pool):
        """曝光子集教材: 场景行 (≤b_teach_sc, 复用 sb θ/标签) + 本步链末纸曝光行 (≤b_teach, 复用 sb θ/标签)
        + 池内同 N 条目重渲补足 (θ 新抽). 像素全 detach; n_s 步 AdamW. 返回计量 dict; steps_in += 1."""
        cfg = self.cfg
        mask = self.exp_mask(sb["ns"])
        idx = mask.nonzero(as_tuple=True)[0]
        xs, ns_l, th_l, tg_l = [], [], [], []
        n_pool = 0
        if idx.numel() > 0:
            i_sc = idx[: int(cfg.b_teach_sc)]
            xs.append(sb["x1"][i_sc].detach())
            ns_l.append(sb["ns"][i_sc])
            th_l.append({kk: v[i_sc] for kk, v in sb["th"].items()})
            tg_l.append({t: sb["tg"][t][i_sc] for t in LIN})
            i_nt = idx[: int(cfg.b_teach)]
            xs.append(notes_det[i_nt])
            ns_l.append(sb["ns"][i_nt])
            th_l.append({kk: v[i_nt] for kk, v in sb["th"].items()})
            tg_l.append({t: sb["tg"][t][i_nt] for t in LIN})
        top = int(cfg.b_teach) - (int(idx.numel()) if idx.numel() < int(cfg.b_teach) else int(cfg.b_teach))
        if top > 0 and pool is not None and pool.size > 0:
            cand = torch.isin(pool.n[: pool.size], torch.as_tensor(self.exp)).nonzero(as_tuple=True)[0]
            if cand.numel() > 0:
                pick = cand[torch.randint(0, int(cand.numel()), (int(top),), generator=self.rng)]
                n_pool = int(pick.numel())
                kc = pool.classes(pick).to(self.dev)
                drawp = R.draw_channel(kc.shape[0], cfg.s, self.rng, self.dev, cfg.occ_k)
                with torch.no_grad():
                    xp = R.channel(R.render_classes(kc, drawp), drawp)
                np_ = pool.labels(pick).to(self.dev)
                thp, tgp = _theta_targets(np_, self.k, self.rng, self.dev)
                xs.append(xp)
                ns_l.append(np_)
                th_l.append(thp)
                tg_l.append(tgp)
        self.steps_in += 1
        if not xs:
            self.last_teach_ns = torch.empty(0, dtype=torch.long)
            return dict(n_teach=0, n_teach_pool=0, rl_period=self.period, rl_steps_in=self.steps_in)
        x = torch.cat(xs)
        ns = torch.cat(ns_l)
        th = {kk: torch.cat([d[kk] for d in th_l]) for kk in ("tau", "p", "m")}
        tg = {t: torch.cat([d[t] for d in tg_l]) for t in LIN}
        self.last_teach_ns = ns.detach().cpu()
        l = accs = None
        for _ in range(int(cfg.n_s)):
            l, accs, _ = sr_rows(self.reader, x, ns, th, tg, float(cfg.sr_mu))
            self.opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.reader.parameters(), float(cfg.clip))
            self.opt.step()
        return dict(n_teach=int(ns.numel()), n_teach_pool=n_pool, sr_loss=float(l),
                    sr_acc=float(accs.mean()), rl_period=self.period, rl_steps_in=self.steps_in)

    # ---------------------------------------------------------- 考试 (逐评; 只读)
    @torch.no_grad()
    def exam(self, model, es, cfg, w, dev, rng_eval, chunk=64):
        """甲案考试: 考官 = 孪生新读者 (exam_with_reader 共用体)."""
        return exam_with_reader(self.reader, self.exp, model, es, cfg, w, dev, rng_eval,
                                dict(period=self.period, steps_in=self.steps_in, eta_now=self.eta_now),
                                chunk=chunk)

    # ---------------------------------------------------------- 检查点状态
    def state(self):
        return dict(period=self.period, steps_in=self.steps_in, exp=list(self.exp), k=self.k,
                    reader={kk: v.cpu() for kk, v in self.reader.state_dict().items()},
                    opt=self.opt.state_dict()) if self.reader is not None else \
            dict(period=self.period, steps_in=self.steps_in, exp=list(self.exp), k=self.k)

    def load_state(self, st):
        self.period = int(st["period"])
        self.steps_in = int(st["steps_in"])
        self.exp = tuple(st["exp"])
        self.k = st["k"]
        if "reader" in st:
            self.reader = Relearner().to(self.dev)
            self.reader.load_state_dict(st["reader"])
            self.opt = torch.optim.AdamW(self.reader.parameters(), lr=float(self.cfg.sr_lr),
                                         weight_decay=float(self.cfg.wd))
            self.opt.load_state_dict(st["opt"])


# ================================================================ 乙案 (换代重学; spec §2.2 乙执行细则)
def gen_scene_bank(exp, per_n, seed):
    """曝光子集场景银行 (乙案模仿/传承输入; CPU 张量). 期专用种子 ⇒ resume 可重建;
    同一池条目跨批配多种新场景 ([U] 2026-08-24 落地配对)."""
    g = torch.Generator().manual_seed(int(seed))
    return {int(n): torch.stack([D.render_scene(g, int(n)) for _ in range(int(per_n))]) for n in exp}


class GenRelearn:
    """乙案机械 ([U] 2026-08-24「既然已经澄清，那就根据讨论试一试乙吧」). gen_relearn=0 时不构造 (A35).
    代 = 期界重置的学习期: 主脑参数重初始化 (E.cnn 按 gen_keep_cnn 保留 = 眼部承袭; D1.lin 恒保留 =
    解剖件) + 主优化器清零 (调用方重建) + 曝光子集重抽 (甲案同流 seed+9001 ⇒ 期 0 子集同式) + 场景银行
    重建; 池不清空 (A37). 本代号 = period+1 (种子池/换代前旧脑条目 gen=0).
    流 (chain_train_step 内, 与主损失同一反传): 模仿 = 池上一代 (gen<本代, 每 N 取最大 gen) 曝光条目 ×
    银行新抽场景 → 写路径逐格 CE × eta_im (A38); 池读传承 = 池曝光条目 (任意代) 重渲+信道 → 主读者
    LIN 五任务 + μ 直读 CE. 标签限域在 chain_train_step (A36)."""
    BANK_PER_N = 32

    def __init__(self, cfg, dev):
        self.cfg = cfg
        self.dev = dev
        self.period = -1
        self.steps_in = 0
        self.exp = ()
        self.k = None
        self.bank = None
        self.rng = torch.Generator().manual_seed(_mix(int(cfg.seed) + 9102, 0))  # 模仿/传承抽样+θ+信道 专用流
        self.last_im_ns = None
        self.last_pr_ns = None

    # ---------------------------------------------------------- 期界
    def needs_rollover(self, k):
        return self.bank is None or self.k != int(k) or self.steps_in >= int(self.cfg.t_gen)

    def rollover(self, model, k, step1):
        self.period += 1
        self.steps_in = 0
        self.k = int(k)
        sub = torch.Generator().manual_seed(_mix(int(self.cfg.seed) + 9001, self.period))
        self.exp = draw_exposure(sub, self.k, int(self.cfg.e_exp))
        info = self.reset_model_(model)
        self.bank = gen_scene_bank(self.exp, self.BANK_PER_N, _mix(int(self.cfg.seed) + 9104, self.period))
        return dict(gen_period=self.period, gen=self.period + 1, exp=list(self.exp), k=self.k, reset=info)

    def reset_model_(self, model):
        """主脑重置: 全量重初始化, 例外 = D1.lin (恒) + E.cnn (gen_keep_cnn=1). in-place copy_ ⇒
        参数张量身份不变 (params 列表仍有效; 优化器清零由调用方重建)."""
        cfg = self.cfg
        keep_cnn = bool(int(getattr(cfg, "gen_keep_cnn", 1)))
        d1_ref = [p.detach().clone() for p in model.D1.lin.parameters()]
        cnn_ref = {kk: v.detach().clone() for kk, v in model.E.cnn.state_dict().items()}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_mix(int(cfg.seed) + 9103, self.period))
            fresh = PredModel(gate_a0=cfg.gate_a0, gate_b0=cfg.gate_b0,
                              d_bottleneck=int(cfg.d_bottleneck),
                              d1_cm=bool(int(getattr(cfg, "d1_cm", 0)))).state_dict()
        sd = model.state_dict()
        n_new = n_keep = 0
        with torch.no_grad():
            for kk in sd:
                if kk.startswith("D1.lin.") or (keep_cnn and kk.startswith("E.cnn.")):
                    n_keep += 1
                    continue
                sd[kk].copy_(fresh[kk])
                n_new += 1
        for p, r in zip(model.D1.lin.parameters(), d1_ref):
            assert torch.equal(p.detach(), r), "A37: W_D1 期界逐位不动"
        if keep_cnn:
            for kk, v in model.E.cnn.state_dict().items():
                assert torch.equal(v, cnn_ref[kk]), "A37: gen_keep_cnn=1 ⇒ E.cnn 期界逐位不动"
        return dict(n_new=n_new, n_keep=n_keep, keep_cnn=int(keep_cnn))

    @property
    def in_warm(self):
        """期首豁免带 (gen_warm × t_gen 步内): 调用方执行 禁入池 + λ 置 0. steps_in 在 streams()
        内自增 ⇒ 调用方须在训练步前取值一次 (gen_warm_now), 全步共用."""
        return self.steps_in < float(getattr(self.cfg, "gen_warm", 0.0)) * int(self.cfg.t_gen)

    def exp_mask(self, ns):
        t = torch.as_tensor(self.exp, device=ns.device)
        return (ns.unsqueeze(1) == t.unsqueeze(0)).any(1)

    # ---------------------------------------------------------- 池候选
    def _pool_exp_idx(self, pool):
        if pool is None or pool.size == 0 or not self.exp:
            return torch.empty(0, dtype=torch.long)
        m = torch.isin(pool.n[: pool.size], torch.as_tensor(self.exp))
        return m.nonzero(as_tuple=True)[0]

    def _prev_gen_idx(self, pool):
        idx = self._pool_exp_idx(pool)
        if idx.numel() == 0:
            return idx
        idx = idx[pool.gen[idx] < self.period + 1]
        if idx.numel() == 0:
            return idx
        keep = torch.zeros(idx.numel(), dtype=torch.bool)
        ns = pool.n[idx]
        for n in ns.unique().tolist():
            sel = ns == n
            gmax = pool.gen[idx[sel]].max()
            keep |= sel & (pool.gen[idx] == gmax)
        return idx[keep]

    # ---------------------------------------------------------- 两流 (chain_train_step 内; 主反传合流)
    def streams(self, model, pool):
        cfg = self.cfg
        self.steps_in += 1
        m = dict(gen_period=self.period, gen_steps_in=self.steps_in)
        l_im = l_pr = None
        cand = self._prev_gen_idx(pool)
        if cand.numel() > 0 and int(cfg.b_im) > 0:
            pick = cand[torch.randint(0, int(cand.numel()), (int(cfg.b_im),), generator=self.rng)]
            tgt = pool.classes(pick).long().to(self.dev)
            ns_im = pool.labels(pick)
            j = torch.randint(0, self.BANK_PER_N, (int(pick.numel()),), generator=self.rng)
            xs = torch.stack([self.bank[int(n)][int(jj)] for n, jj in zip(ns_im, j)]).to(self.dev)
            e = model.E.enc_seq(model.E.tokenize(xs), None, seg=False)   # chain_side k=0 写路径同式
            lg = ST.write_logits(model, e["tokens"], cfg)
            wk = float(getattr(cfg, "im_wink", 1.0))
            if wk != 1.0:                                  # 墨格权重: 抄墨不抄空 (空格 ~92%, 逐格平均会奖励空纸)
                flat = tgt.reshape(-1)
                ce = F.cross_entropy(lg.reshape(-1, R.NCLS), flat, reduction="none")
                cw = torch.ones_like(ce)
                cw[flat > 0] = wk
                l_im = (ce * cw).sum() / cw.sum()
            else:
                l_im = F.cross_entropy(lg.reshape(-1, R.NCLS), tgt.reshape(-1))
            with torch.no_grad():
                m["im_agree"] = float((lg.argmax(-1) == tgt).float().mean())
            m["n_im"] = int(pick.numel())
            self.last_im_ns = ns_im.clone()
        else:
            m["n_im"] = 0
            self.last_im_ns = torch.empty(0, dtype=torch.long)
        candr = self._pool_exp_idx(pool)
        if candr.numel() > 0 and int(cfg.b_poolread) > 0:
            pick = candr[torch.randint(0, int(candr.numel()), (int(cfg.b_poolread),), generator=self.rng)]
            kc = pool.classes(pick).to(self.dev)
            drawp = R.draw_channel(kc.shape[0], cfg.s, self.rng, self.dev, cfg.occ_k)
            with torch.no_grad():
                xp = R.channel(R.render_classes(kc, drawp), drawp)
            np_ = pool.labels(pick).to(self.dev)
            thp, tgp = _theta_targets(np_, self.k, self.rng, self.dev)
            l_pr, accs, _ = sr_rows(model, xp, np_, thp, tgp, float(cfg.mu))
            m["pr_acc"] = float(accs.mean())
            m["n_pr"] = int(pick.numel())
            self.last_pr_ns = np_.detach().cpu()
        else:
            m["n_pr"] = 0
            self.last_pr_ns = torch.empty(0, dtype=torch.long)
        return l_im, l_pr, m

    # ---------------------------------------------------------- 考试 (逐评; 只读; 考官 = 主脑自身)
    def exam(self, model, es, cfg, w, dev, rng_eval, chunk=64):
        return exam_with_reader(model, self.exp, model, es, cfg, w, dev, rng_eval,
                                dict(period=self.period, steps_in=self.steps_in), chunk=chunk)

    # ---------------------------------------------------------- 检查点状态 (银行按期种子重建; rng 流不入册 = 既有各流同例)
    def state(self):
        return dict(period=self.period, steps_in=self.steps_in, exp=list(self.exp), k=self.k)

    def load_state(self, st):
        self.period = int(st["period"])
        self.steps_in = int(st["steps_in"])
        self.exp = tuple(st["exp"])
        self.k = st["k"]
        if self.k is not None and self.exp:
            self.bank = gen_scene_bank(self.exp, self.BANK_PER_N,
                                       _mix(int(self.cfg.seed) + 9104, self.period))
