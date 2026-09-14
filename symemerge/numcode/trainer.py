# symemerge/numcode/trainer.py
"""训练器 (spec §7): Phase 1 感知与画布素养 / Phase 2 K=4 全损失 + GRPO 自举.

职责: 前向路径与梯度路由的装配 (spec §2.7), 数据流 (worker 预取), 评测仪器
(冻结判据探针 / 空白画布对照 §10.1 / 像素探针 §9.2 / 码表 §11.1 / R6 配对可视化),
检查点. 机制本体在 model/losses/grpo/data/geometry, 此处不重复.

本文件的 [C] 实例化决策 (spec 未定死处), 理由登记在 params.md §8:
  P1-COUNT-BAND  Phase 1 计数在全训练域 (k=128) 上训, 之后各 Phase 持续 --
                 "给定精确数量感知"是前提, 感知先于写者; 课程 K 只门控任务/RL 侧.
  MASK-KEEP      BERT 遮蔽配方 (Devlin et al. 2019) 的本分支实例: 选中格 15-40%,
                 其中 85% 换 [MASK] / 15% 保留原 token, 损失在全部选中格.
                 均匀画布上纯遮蔽位的准确率天花板是贝叶斯后验 (~0.61, 与盲猜"空"
                 几乎重合) -- 素养冻结门用保留桶识别准确率 (天花板 1), 遮蔽位与
                 实测 oracle (mask_oracle) 同报.
  P2-BETA0       Phase 2 KL 系数 β=0 (纯组内中心化 REINFORCE, on-policy 单步);
                 spec §8 的 β=编码突变率语义在自举段无锚可依 (无 SFT 参考策略),
                 稳定性由 lr + 梯度裁剪承担, 失稳再升 β (机制保留).
  P2-READER-ALL  读者 (任务/直读损失) 在全部 G 条 rollout 上训 -- 自举要求读者
                 从探索噪声里学到偶然规律, 奖励才能分化 (spec §7 Phase 2 机制);
                 贪心条在其中充当锚 (spec §6.5).
  预注册判读操作化 (spec §11.2-1, 跑前写死): 一元码 = 各 N 章数中位数严格递增
                 且跨相邻 N 的同索引配对严格递增占比 >= 0.9, 零假设 = 置换
                 N 标签重算 (table_stats 输出 mono_p).
"""
import dataclasses
import json
import math
import os
import shutil
import time
from collections import Counter

import torch
import torch.nn.functional as F

from . import align as AL
from . import data as D
from . import geometry as GM
from . import grpo as GR
from . import losses as L
from . import probe as PB
from .model import ABSTAIN, NumCodeModel, Writer

_PHI = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_MOD = 2 ** 62


@dataclasses.dataclass
class TrainCfg:
    phase: int = 1
    k: int = GM.K_MAX            # phase1: 128 (P1-COUNT-BAND); phase2: 4
    steps: int = 30000
    batch_scenes: int = 64       # 计数流 batch (全域, 两 Phase 通用)
    batch_mask: int = 64         # 掩码素养流 batch
    s: float = 0.05              # 腐蚀强度 s_w (params §4 预登记校准值)
    lr: float = 3e-4
    wd: float = 0.01
    clip: float = 1.0
    workers: int = 16
    # ---- phase 2
    n_groups: int = 32           # B (spec §6.4 建议 >=32)
    g: int = 8                   # rollout/组
    n_greedy: int = 2            # spec §6.5: 2 贪心 + 6 采样
    temp: float = 1.0
    lam: float = 0.0             # λ 每章代价 (Phase 2 预注册 λ=0)
    beta: float = 0.0            # KL 系数 (P2-BETA0)
    mu: float = 0.05             # 直读损失权重 (量具, 取小; spec §5.5)
    conf_gate: float = 0.9       # 早期 rollout 过滤 (spec §8)
    c_hi: float = math.log(128.0)  # 弃权代价从高退火到中 (spec §8)
    c_lo: float = math.log(8.0)
    c_anneal_steps: int = 3000
    keep_frac: float = 0.15      # MASK-KEEP 保留桶占选中格比例
    accum: int = 1               # 组的微批数 (显存不够时 >1)
    # ---- s110 修复轮 (params.md §9, 用户六裁定; 全 0 = 首跑旧制)
    soft_gate: int = 0           # 1 = 优势×置信度 (裁定5); 0 = 旧硬门 conf>=conf_gate
    occ_k: int = 0               # >0 = 信道恒遮恰 k 格 (裁定1); 0 = 旧概率遮挡
    reader_lag: float = 0.0      # >0 = 滞后读者 EMA 衰减 (裁定2); 奖励+β参考共用影子
    c_delta: float = 0.0         # >0 = 弃权下限改 log(k)-c_delta (裁定6); 0 = 旧制 c_lo
    l_max_override: int = 0      # >0 = 程序长度上限改此值 (s110 追加: K=4 档 28); 0 = 公式
    # ---- s111 第二修复轮 (params.md §10, 用户四改动; 默认值 = p2b 旧制)
    anchor: int = 0              # 1 = KL 参考改固定锚状态机 AnchorCtl (改3, [U] 推荐方案)
    reward_m: int = 1            # 奖励对 m 次独立腐蚀取均值 (改4); 1 = 旧制单抽
    # ---- s112 (params.md §10c, [U] p2d 判决后)
    lam0: float = 0.0            # 空白惩罚: 全空画布加收此值 (给章数投影的零点定价,
    #                              破 p2d 零墨码字角落; λ0 > λ 使一章严格优于零章)
    beta_adapt: int = 1          # 0 = β 固定 (不校准不自适应; 闸的回滚加压照旧)
    # ---- s113 (params.md §10d, [U] K 扩张 + 四仪表)
    ood_bot: int = 0             # >0 = 每步 n 张结构性非法画布 -> ⊥ 直读监督 (改3,
    #                              取代空白→⊥: 空白在单章码制度下不是 OOD, p2e 实测)
    k_expand: int = 0            # 1 = 稳定触发的 K 扩张 (改4: 成绩连续两评 >=0.95 且 v<=2;
    #                              步长见 EXPAND_STEP, s114 C5 +4→+2)
    # ---- s114 (params.md §10e, [U] 探针奖励 v2)
    alpha: float = 0.0           # r_probe 系数 (C2: 0.10); 0 = 双探针整套不建,
    #                              奖励/日志/检查点回 s113 旧制
    reset_every: int = 2500      # C3b 重置读头仪表节律 (须为 eval_every 整倍;
    #                              M5 口径复核后为记录项, 不进任何判据)
    c9: int = 0                  # [U] p3d: 退化组守卫 (div 触发 forced 臂); 0 = 不建
    # ---- v4.1 对齐最小化 (align.py; 全默认 = 旧制逐位不变)
    holdout_frac: float = 0.0    # α 滚动留出比例 (§4.1; 0 = 不留出, 对齐机构整套不建)
    t_rotate: int = 200          # T_rotate 留出轮换周期 (步)
    beta_align: float = 0.0      # β 对齐强度 (§5.2 u_S 修正; 0 = 算而不 apply = T1 只读臂)
    kappa: float = 0.0           # κ 奖励调制 (§6.3 c 通道; 0 = c 算而不进 r)
    delta_mult: float = 1.0      # δ = delta_mult × 上一步 AdamW 实际位移 (§10, GA2 定标)
    l_star: float = 0.0          # L* 水平门 (§5.5; 0 = 未定, 门不作用; T2 须由 GA5 定死)
    zeta: float = 0.999          # 名义字段: 旧 G_n 缓冲 EMA 衰减 ([U] 2026-08-17 改动 4 后 G_n 为
    #                              每评窗均值, ζ 不再参与计算)
    align_every: int = 50        # 半批 𝒮 / 打乱零假设 τ_null,c_null / 联合余弦 的节律
    expand_step: int = 0         # 0 = 常数 EXPAND_STEP (+2); v4.1 §4.4 = 1 (前沿逐个引入, 跳过空洞)
    metric: str = "adam"         # 度规源: adam (v̂) | efisher (当场估) | identity (退化, 全文标注)
    u_ema: float = 0.0           # u_S 施加方向的 EMA 衰减 (0 = 无; GA4 补救「延长累积窗」; 两路由同用)
    init_opt: int = 0            # --init 时同时载入检查点的优化器状态 (F⁻¹ 度规需要 v̂)
    m1_eval: int = 0             # 1 = holdout_frac=0 时仍建**只评用** AlignCtx (S 恒空): 每评
    #                              M1 R 谱/对照列/M7-V, 终报 R 史; 训练路径 align=None 逐位旧制
    #                              (B 臂 α=0 复跑的仪表, [C] s123)
    # ---- [U] 2026-08-17 正式跑五改动 (改动 2/4 无旗标: 计数流全带 apply / G_n 窗均值 恒开)
    beta_sc: float = 0.0         # 改动 1: 场景路由 (Θ_E ∪ count_head, L_count) 的 u_S 对齐强度
    #                              (0 = 算而不 apply); 夹取同 β_cv: β_sc‖u_S‖_{F⁻¹} ≤ BETA_CAP
    expand_every: int = 0        # 改动 3: >0 = 距上次推进满此步数亦推进 (OR 稳定触发); 0 = 仅稳定触发
    # ---- [U] 2026-08-17 第四阶段 (接续 F5-C 停跑态)
    c_mode: str = "proj"         # c 定义: proj = 投影率 (只在 S 行, V 子空间正交投影, 无中心化; 改动 2)
    #                              | cos = 旧余弦 (F5-A..C 与 T1 的口径)
    k_force: int = 0             # >0 = --resume 后把课程档强制设为此值 (前沿退回, 改动 1; 训练域/评测集/
    #                              留出/守卫随之; 闩与并轨/定居计数重置; 记 retreat)
    stop_count: float = 0.80     # 停跑: count 训练带 exact 连续两评 < 此值 (第四阶段 .90)
    stop_dmin: int = 0           # >0 = dmin ≤ 此值即停 (第四阶段 1: 「dmin 在 F=12 内跌到 ≤1 ⇒ 停」)
    stop_cnull: int = 0          # 1 = 当评 c 窗均不高于其零假设即停 (第四阶段「κ 通道无效」)
    # ---- 评测节律
    eval_every: int = 500
    probe_every: int = 2000
    blank_every: int = 200
    ckpt_every: int = 5000
    viz_every: int = 2000
    seed: int = 0
    out: str = "outputs/nc_p1"


def anneal_c(cfg, step):
    """弃权代价 c: c_hi -> 下限 线性退火 (spec §8 必须从高退火到中).
    裁定6: c_delta>0 时下限随课程 = log(k) - c_delta, 弃权才可能优于知先验瞎蒙."""
    lo = (math.log(cfg.k) - cfg.c_delta) if cfg.c_delta > 0 else cfg.c_lo
    if step >= cfg.c_anneal_steps:
        return lo
    f = step / cfg.c_anneal_steps
    return cfg.c_hi + (lo - cfg.c_hi) * f


def l_max_of(cfg):
    """程序长度上限: 覆写与课程公式取大 -- 覆写是给低档留头寸的地板 (s110 K=4 档
    28, s111 起 40), K>=32 后公式接管, 覆写不得压穿 spec §3.2 的 l_max >= K (s113)."""
    if cfg.l_max_override > 0:
        return max(cfg.l_max_override, GM.l_max(cfg.k))
    return GM.l_max(cfg.k)


def modal_templates(tab):
    """码表 -> 每 N 众数模板 (list 长 T, 值 0..S, 0=空). dmin 与位移仪表共用."""
    cfgs = {}
    for n, ps in tab.items():
        cnt = [Counter() for _ in range(GM.T)]
        for p in ps:
            seen = {a // GM.S: a % GM.S + 1 for a in p}
            for cell in range(GM.T):
                cnt[cell][seen.get(cell, 0)] += 1
        cfgs[n] = [c.most_common(1)[0][0] for c in cnt]
    return cfgs


def min_pair_hamming(tab):
    """码表 -> 逐对众数模板汉明距与最小值 (s110 头号仪表). 判读 (用户裁定):
    升稳 >~12 = 冗余被买回来; 个位数+准确率高 = 压力不够升 k; 个位数+准确率低 =
    读写协调未稳, 动滞后读者/β 侧, 别动腐蚀. 配合章数曲线: 距离升+章数升 = 买在
    长度维; 距离升+章数平 = 买在章型维."""
    cfgs = modal_templates(tab)
    ns = sorted(cfgs)
    pairs = {}
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            pairs[f"{a}-{b}"] = sum(1 for x, y in zip(cfgs[a], cfgs[b])
                                    if x != y)
    return dict(min=min(pairs.values()), pairs=pairs)


def template_shift(prev, cur):
    """改5 (§10) 头号监控: 两次评测间逐 N 众数模板的汉明位移 (格数), 报逐 N +
    中位 + 最大. 判读 (用户裁定三分支): 中位 ≲2 且成绩稳 = 锚+均值生效; 不降但
    成绩稳 = 读者追赶能力提高 (好但脆); 不降且塌方继续 = 锚定方式不对, 升级为
    对写者做参数级 EMA. prev/cur = modal_templates 输出."""
    per = {n: sum(1 for a, c in zip(prev[n], cur[n]) if a != c)
           for n in sorted(cur) if n in prev}
    if not per:
        return None
    vals = sorted(per.values())
    mid = len(vals) // 2
    med = (float(vals[mid]) if len(vals) % 2 else
           0.5 * (vals[mid - 1] + vals[mid]))
    return dict(per_n=per, med=med, max=vals[-1])


def gate_weights(conf, cfg, g):
    """rollout 过滤权重, 逐 rollout: 软门 = 组置信度本身 (裁定5), 硬门 = 0/1 截断."""
    w = conf.clamp(0.0, 1.0) if cfg.soft_gate else (conf >= cfg.conf_gate).float()
    return w.repeat_interleave(g)


@torch.no_grad()
def ema_update(lag, model, decay):
    """影子模型 EMA (裁定2/3): lag <- decay*lag + (1-decay)*model. 只动浮点张量."""
    msd = model.state_dict()
    for k, v in lag.state_dict().items():
        if v.is_floating_point():
            v.mul_(decay).add_(msd[k], alpha=1.0 - decay)


def make_anchor(model):
    """改3 (§10): KL 参考 = 写者头 D1 的冻结副本 (固定锚, 不 EMA 跟随).
    KL 惩罚"离锚点多远"而非"离上一步多远" -- 直接约束累积位移.
    只经 AnchorCtl 更新; 读侧滞后影子照旧管奖励."""
    a = Writer().to(next(model.parameters()).device)
    a.load_state_dict(model.writer.state_dict())
    for p in a.parameters():
        p.requires_grad_(False)
    a.eval()
    return a


# 固定锚状态机常数 (params.md §10, [U] 2026-08-12 推荐方案 + s111b 五条闸逻辑修正;
# 改动需重登记). p2c 死循环教训: 闸不看成绩会把 λ 跳变引发的一次性合法重构当成病.
ANCHOR_TMIN = 2000        # 刷新最小间隔 (步)
ANCHOR_TMAX = 5000        # 超此间隔强制刷新 -- 刷向历史最优检查点, 不是当前策略
ANCHOR_SCORE_FLOOR = 0.90  # 刷新条件1: 六任务均值绝对下限 (不用"≥锚点成绩":
#                            成绩由当前读者测, 锚点成绩随读者训练而变, 相对比较循环依赖)
ANCHOR_DMIN_FLOOR = 8     # 刷新条件2: 最小模板间距 (锚要有裕度)
ANCHOR_V_MAX = 3.0        # 刷新条件3a: 瞬时位移中位 v (策略当前静止)
ANCHOR_D_MAX = 12.0       # 刷新条件3b: 对锚累积位移中位 D (锚没被拖远)
ANCHOR_STREAK = 2         # 三条件需连续成立的评测次数
ANCHOR_D_RESET = 20.0     # 快速闸位移项: 单评 D 超此值
ANCHOR_RESET_SCORE = 0.85  # s111b#1 合取项: 且 (成绩 < 0.85 或 dmin < 6) 才回滚 --
ANCHOR_RESET_DMIN = 6     #   位移大而质量好的重构不回滚 (p2c 评499: 0.9948/dmin8 被误滚)
ANCHOR_FALL = 0.15        # 兜底闸: 成绩 < 锚点成绩 − 此值
ANCHOR_FALL_STREAK = 3    # 兜底连续评测数
ANCHOR_RESET_CAP = 2      # s111b#5: 同一锚连续回滚上限; 第三次闸触发改为接受当前策略为新锚
ANCHOR_BOOST = 3.0        # 回滚后 β 峰值倍率
ANCHOR_BOOST_STEPS = 1000  # s111b#3: 加压改乘性衰减, 此为 3→1 的衰减尺度 (无悬崖)
BOOST_DECAY = (1.0 / ANCHOR_BOOST) ** (1.0 / ANCHOR_BOOST_STEPS)
ANCHOR_GRACE = 2000       # s111b#2: λ 变更后宽限期 (步), 判据全停; 期末稳定点设为新锚
ANCHOR_COOLDOWN = 200     # 回滚后过渡期 (步): 期内评测不参与任何判据 (防震荡)
BETA_MIN, BETA_MAX = 1e-3, 0.32   # β 自适应夹取 ([C] 工程护栏)
BETA_CALIB_STEPS = 2000   # 校准窗长: KL_tgt = "v≤2" 档实测 KL; 窗起点 = 宽限期末
#                           (宽限期内的 KL 是对旧锚的冲击数据, 不代表受控运行域)


class AnchorCtl:
    """固定锚 + 慢刷新 + 反向闸 + β 自适应的状态机 (§10 [U] 推荐方案 + s111b 修正).
    两个位移指标分开记 (缺一不可): v = 相对上次评测的逐 N 位移中位 (瞬时速度,
    判"是否正在搬家"); D = 相对当前锚的逐 N 位移中位 (累积位移, KL 真正约束的量).
    v 为零不代表位置正确 -- 写者可以漂远后停下.
    锚的模板表/成绩在刷新时缓存 (锚点成绩是定格标量, 无循环依赖)."""

    def __init__(self, model, beta0):
        self.ref = make_anchor(model)          # KL 参考 (冻结写者副本)
        self.tpl = None                        # 锚的贪心模板表 (刷新时缓存)
        self.score = None                      # 锚点成绩 (刷新时缓存)
        self.beta = beta0
        self.kl_tgt = None                     # 校准窗收口后生效
        self.last_refresh = 0
        self.last_reset = -10 ** 9
        self.streak_ok = 0
        self.streak_below = 0
        self.boost_from = -10 ** 9             # 回滚时刻; β 加压自此乘性衰减 3→1
        self.grace_done = False                # λ 变更宽限期是否已收口建锚
        self.grace_until = ANCHOR_GRACE        # 宽限期终点 (发射 = 常数; 扩张时前移)
        self.resets_since_refresh = 0          # 同锚连续回滚计数 (s111b#5)
        self.calib = []                        # 校准窗 (v, kl) 散点
        self.best = None                       # (writer_sd_cpu, tpl, score) 历史最优

    def beta_eff(self, step):
        b = ANCHOR_BOOST * BOOST_DECAY ** max(0, step - self.boost_from)
        return self.beta * max(1.0, b)

    @torch.no_grad()
    def note_best(self, model, tpl, score):
        """ckpt_best 落盘时同步缓存写者快照 -- 强制刷新的目标 (刷向历史最优,
        否则死锁解除时锚会被漂移中的策略污染)."""
        sd = {k: v.detach().cpu().clone()
              for k, v in model.writer.state_dict().items()}
        self.best = (sd, tpl, score)

    @torch.no_grad()
    def _refresh(self, model, tpl, score, s1, forced=False):
        if forced and self.best is not None:
            sd, btpl, bscore = self.best
            self.ref.load_state_dict(sd)
            self.tpl, self.score = btpl, bscore
        else:
            self.ref.load_state_dict(model.writer.state_dict())
            self.tpl, self.score = tpl, score
        self.last_refresh = s1
        self.streak_ok = 0
        self.resets_since_refresh = 0          # 换锚 = 连续回滚计数清零 (s111b#5)
        return "best" if (forced and self.best is not None) else "current"

    @torch.no_grad()
    def _reset(self, model, s1, opt=None):
        """反向闸: 写者硬回锚点; β 加压 3 倍起乘性衰减 (s111b#3, 无悬崖);
        读者不重置 (被动追赶方, 几百步能追回, 比重学快); 计数器清零. 同时清
        写者参数的优化器一二阶矩 (评审: exp_avg_sq 半衰期 ~700 步, 不清则回滚
        后仍带着漂移方向的动量, "硬回锚点"就不干净)."""
        model.writer.load_state_dict(self.ref.state_dict())
        if opt is not None:
            for p in model.writer.parameters():
                opt.state.pop(p, None)
        self.boost_from = s1
        self.last_reset = s1
        self.streak_below = 0
        self.streak_ok = 0

    def _gate_fire(self, model, tpl, score, s1, opt, out, kind):
        """闸触发统一出口 (s111b#5): 同一锚连续回滚至多 ANCHOR_RESET_CAP 次;
        再触发不回滚, 改为接受当前策略为新锚 (闸连输三次 = 锚错了, 不是策略错了)."""
        if self.resets_since_refresh >= ANCHOR_RESET_CAP:
            self._refresh(model, tpl, score, s1)
            out["event"] = "accept_current"
            return out
        self._reset(model, s1, opt)
        self.resets_since_refresh += 1
        prior = out.get("event")
        out["event"] = (prior + "+" if prior else "") + "reset_" + kind
        return out

    def on_expand(self, s1):
        """K 扩张 = 制度变更 (s113 改4): 复用 λ 变更宽限机制 (s111b#2 同款同值) --
        判据全停到 s1+ANCHOR_GRACE, 期末稳定点立新锚. 历史最优快照作废 (跨档成绩
        语义不可比, 强制刷新在新档内重新积累目标)."""
        self.grace_until = s1 + ANCHOR_GRACE
        self.grace_done = False
        self.best = None
        self.streak_ok = 0
        self.streak_below = 0
        self.kl_tgt = None                     # 新档重校准 (评审 Important#5)
        self.calib = []

    def on_eval(self, model, s1, score, dmin_min, v_med, tpl, kl_win,
                opt=None):
        """每次评测喂 (成绩, dmin, v, 当前模板表, 窗内 KL 均值), 返回入册记录."""
        out = dict(beta=round(self.beta, 5))
        if self.tpl is None:                   # 无锚档 (未做发射前建档的兜底)
            self.tpl, self.score = tpl, score
            out["event"] = "init"
            return out
        if not self.grace_done:                # s111b#2: λ 变更宽限期
            if s1 < self.grace_until:
                out["D"] = template_shift(self.tpl, tpl)
                out["grace"] = True            # 判据全停, 只记仪表
                return out
            self._refresh(model, tpl, score, s1)   # 期末稳定点设为新锚
            self.grace_done = True
            out["event"] = "grace_anchor"
            out["anchor_score"] = round(score, 4)
            return out
        d = template_shift(self.tpl, tpl)
        out["D"] = d
        if s1 - self.last_reset < ANCHOR_COOLDOWN:
            out["transition"] = True           # 过渡期: 不参与任何判据
            return out
        if s1 - self.last_refresh > ANCHOR_TMAX:
            # s111b#4: 强制刷新移出回滚路径 -- 最先查, 回滚不再挡它 (p2c 死锁之一)
            out["forced_to"] = self._refresh(model, tpl, score, s1,
                                             forced=True)
            out["event"] = "refresh_forced"    # forced_to=current 表示无 best 可刷
            d = template_shift(self.tpl, tpl)  # 对新锚重算
            out["D"] = d
        if kl_win is not None:                 # s111b#3: 校准/自适应始终运行
            if self.kl_tgt is None:
                if v_med is not None:
                    self.calib.append((float(v_med), float(kl_win)))
                if s1 >= self.grace_until + BETA_CALIB_STEPS and self.calib:
                    ok = [k for vv, k in self.calib if vv <= 2.0]
                    self.kl_tgt = max(ok) if ok else min(
                        k for _, k in self.calib)
                    out["kl_tgt"] = round(self.kl_tgt, 6)
            elif kl_win > 1.5 * self.kl_tgt:
                self.beta = min(self.beta * 2.0, BETA_MAX)
                out["beta"] = round(self.beta, 5)
            elif kl_win < self.kl_tgt / 1.5:
                self.beta = max(self.beta / 2.0, BETA_MIN)
                out["beta"] = round(self.beta, 5)
        bad = (score < ANCHOR_RESET_SCORE or dmin_min < ANCHOR_RESET_DMIN)
        if d is not None and d["med"] > ANCHOR_D_RESET and bad:
            # s111b#1 合取快速闸: 位移大且质量真差才回滚; 位移大而质量好 = 合法
            # 大重构, 不回滚也不追认 (留给慢刷新/强制刷新按各自节律收编)
            return self._gate_fire(model, tpl, score, s1, opt, out, "fast")
        if self.score is not None and score < self.score - ANCHOR_FALL:
            self.streak_below += 1
        else:
            self.streak_below = 0
        if self.streak_below >= ANCHOR_FALL_STREAK:
            return self._gate_fire(model, tpl, score, s1, opt, out, "fall")
        ok = (score >= ANCHOR_SCORE_FLOOR and dmin_min >= ANCHOR_DMIN_FLOOR
              and v_med is not None and v_med <= ANCHOR_V_MAX
              and d is not None and d["med"] <= ANCHOR_D_MAX)
        self.streak_ok = self.streak_ok + 1 if ok else 0
        if (self.streak_ok >= ANCHOR_STREAK
                and s1 - self.last_refresh >= ANCHOR_TMIN):
            self._refresh(model, tpl, score, s1)
            out["event"] = "refresh"
        return out

    def state(self):
        return dict(sd=self.ref.state_dict(), tpl=self.tpl, score=self.score,
                    beta=self.beta, kl_tgt=self.kl_tgt,
                    last_refresh=self.last_refresh, last_reset=self.last_reset,
                    streak_ok=self.streak_ok, streak_below=self.streak_below,
                    boost_from=self.boost_from, grace_done=self.grace_done,
                    resets_since_refresh=self.resets_since_refresh,
                    calib=self.calib, best=self.best,
                    grace_until=self.grace_until)

    def load_state(self, d):
        self.ref.load_state_dict(d["sd"])
        for k in ("tpl", "score", "beta", "kl_tgt", "last_refresh",
                  "last_reset", "streak_ok", "streak_below", "boost_from",
                  "grace_done", "resets_since_refresh", "calib", "best",
                  "grace_until"):
            if k in d:                         # 旧检查点缺新键时保默认
                setattr(self, k, d[k])


def ood_bot_loss(model, cfg, rng, dev):
    """s113 改3 (§10d): 结构性非法画布 -> ⊥ 直读监督, 取代空白→⊥ (p2e 判决:
    单章码字的遮挡像≈空白, 空白落在合法读音区内, 仪器与码结构性冲突; 稠密非法
    画布则处在任何合法码字腐蚀轨道之外). 画布走训练信道全套.
    梯度只进直读头, 主干 detach (p3a 实测教训, §10d 修正案): 稠密画布是满负荷
    激活输入, 空白时代校准的 μ×3% 权重经共享主干的梯度冲击大几十倍 -- p3a 500
    步内读者先坏 (步 400 四任务 0.78-0.82 而写者未动) 写者随崩 (章数 5.5→0.8,
    模板 2/3/4 全并轨), 对照 p2e 同价格同热启动首评满分. 主干的稠密表征由素养
    流保障 (掩码画布上限 200 章), 读头单独可分. 任务头/奖励/素养流结构上不可达.
    调用方乘 μ·(ood_bot/(B·g))."""
    progs = [GM.sample_illegal_prog(rng, l_max_of(cfg), cfg.occ_k)
             for _ in range(cfg.ood_bot)]
    x = GM.render_channel(progs, cfg.s, rng, cfg.occ_k).to(dev)
    logits = model.heads.read(model.encode_canvas(x)["cls"].detach())
    tgt = torch.full((cfg.ood_bot,), ABSTAIN, dtype=torch.long, device=dev)
    return F.cross_entropy(logits, tgt)


@torch.no_grad()
def ood_abstain(model, es, cfg, dev):
    """P4 新仪 (§10d 改3): 固定非法程序集经评测信道 (CRN seed+777) 的直读弃权率.
    判据 ≥0.9 (best+终局双点), 零假设 = 发射建档实测 ood0."""
    if not es.get("ood_progs"):
        return None                            # 容量护栏停表 (评审 Critical#1)
    rng = torch.Generator().manual_seed(cfg.seed + 777)
    x = GM.render_channel(es["ood_progs"], cfg.s, rng, cfg.occ_k).to(dev)
    rp = model.heads.read(model.encode_canvas(x)["cls"])
    return round(float((rp.argmax(-1) == ABSTAIN).float().mean()), 4)


def ensure_ood_progs(es, cfg, cache=None):
    """s113 改3: 评测集补非法程序集 (旧磁盘缓存的迁移路径; 确定性 seed+777,
    其余字段 CRN 不动). 幂等."""
    if cfg.phase != 2 or "ood_progs" in es:
        return es
    if not GM.illegal_capacity_ok(l_max_of(cfg), cfg.occ_k):
        es["ood_progs"] = []   # 容量护栏: P4 停表 (评审 Critical#1, §10d)
        print("[ood] 容量护栏: l_max+occ_k 顶画布容量, OOD 监督/探针停摆", flush=True)
        return es
    rng = torch.Generator().manual_seed(cfg.seed + 777)
    es["ood_progs"] = [GM.sample_illegal_prog(rng, l_max_of(cfg), cfg.occ_k)
                       for _ in range(128)]
    if cache:
        torch.save(es, cache)
    return es


COLLAPSE_DMIN = 1      # s113 改1 ([U]): dmin <= 1 且成绩 < 0.85 -> 只记录不动作
COLLAPSE_SCORE = 0.85  # 与快速闸成绩项同一常数 (p2c 实测健康重构 0.9948/塌方 0.7487 之间)
COLLAPSE_STREAK = 2    # s114 C7 ([U]): 连续两评成立才置位 -- p3b 26/26 评全响 =
#                        学习凹陷内饱和, 分不出凹陷与塌方, 升级动作前必须加制度限定


def collapse_flag(dmin_min, score):
    """并轨记录器瞬时条件 (§10d 改1, [U] 先只记录): 间距塌到并轨临界且成绩已崩."""
    return dmin_min <= COLLAPSE_DMIN and score < COLLAPSE_SCORE


class CollapseCtl:
    """并轨记录器限定版 (s114 C7): 瞬时条件连续 COLLAPSE_STREAK 评成立才置位.
    扩张时清零 (跨档成绩语义不可比, 受扰时刻不作证据 -- 与扩张 streak 同理)."""

    def __init__(self):
        self.streak = 0

    def offer(self, dmin_min, score):
        self.streak = self.streak + 1 if collapse_flag(dmin_min, score) else 0
        return self.streak >= COLLAPSE_STREAK

    def reset(self):
        self.streak = 0


class BestLatch:
    """s113 改2 ([U]): 最优闩改稳定窗 -- 闩值 = 连续两评的较小者, 超过现闩才锁存
    (单点尖峰按构造占不了闩位; 首个合格对子即锁存 -- 恒定成绩平台上对子最小值
    不再超越自身, 若首对只垫底则整段平台零存档, fin.best 无对应检查点).
    锁存检查点 = 第二评时刻权重. K 扩张时 reset (跨档成绩不可比, best 按档内重记)."""

    def __init__(self, best=-1.0):
        self.best, self.step, self.prev = best, 0, None

    def offer(self, score, step):
        pair = min(score, self.prev) if self.prev is not None else None
        self.prev = score
        if pair is not None and pair > self.best:
            self.best, self.step = pair, step
            return True
        return False

    def reset(self):
        self.best, self.prev = -1.0, None


EXPAND_SCORE = 0.95   # s113 改4 ([U]): 成绩连续两评 >= 0.95 且 v <= 2 才扩张,
EXPAND_V = 2.0        #   否则撞进塌方谷; 成绩口径 = 固定集六任务均值 (与闸同)
EXPAND_STREAK = 2


class ExpandCtl:
    """K 扩张触发器 (§10d 改4): 按稳定性而非固定步数. 回滚评/过渡评清零 streak
    (制度受扰时刻的成绩不作稳定性证据); 触发后清零重计.
    [U] 2026-08-17 改动 3: 推进条件 = 稳定性触发 OR 距上次推进已满 every 步 (every>0 时);
    offer 返回 None (不推进) / "stable" / "timer" (稳定触发优先记为 stable)."""

    def __init__(self, every=0):
        self.streak = 0
        self.every = int(every)

    def offer(self, score, v_med, disturbed, step=None, last_expand=None):
        ok = (score >= EXPAND_SCORE and v_med is not None
              and v_med <= EXPAND_V and not disturbed)
        self.streak = self.streak + 1 if ok else 0
        if self.streak >= EXPAND_STREAK:
            self.streak = 0
            return "stable"
        if (self.every > 0 and step is not None and last_expand is not None
                and step - last_expand >= self.every):
            self.streak = 0
            return "timer"
        return None


# ---- [U] 2026-08-17 正式跑停跑规则 (指令原文: 「六任务闩峰连续三评低于 .85 ⇒ 停」「count 训练带
# exact 连续两评低于 .80 ⇒ 停」「并轨记录器置位 ⇒ 停」; 「章数中位在提价后不降反升」由人工按评判)
STOP_SCORE, STOP_SCORE_STREAK = 0.85, 3
STOP_COUNT, STOP_COUNT_STREAK = 0.80, 2


class StopCtl:
    """停跑规则计数器: 六任务成绩 < STOP_SCORE 连续 STOP_SCORE_STREAK 评 / count 训练带 exact
    < stop_count 连续 STOP_COUNT_STREAK 评 / 并轨记录器置位; 第四阶段 ([U] 2026-08-17) 另加
    dmin ≤ stop_dmin 即停、当评 c 窗均不高于其零假设即停 (stop_cnull). offer 返回触发规则名列表.
    [C] 「闩峰连续三评」按连续三评成绩口径 (闩在扩张时重置, 不能直接读); 单评下探不停."""

    def __init__(self, stop_count=STOP_COUNT, stop_dmin=0, stop_cnull=0):
        self.s_score = 0
        self.s_count = 0
        self.stop_count = float(stop_count)
        self.stop_dmin = int(stop_dmin)
        self.stop_cnull = int(stop_cnull)

    def offer(self, score, count_train_exact, collapse, dmin=None, c_mean=None, c_null=None):
        self.s_score = self.s_score + 1 if score < STOP_SCORE else 0
        self.s_count = self.s_count + 1 if count_train_exact < self.stop_count else 0
        hits = []
        if self.s_score >= STOP_SCORE_STREAK:
            hits.append(f"score<{STOP_SCORE}x{STOP_SCORE_STREAK}")
        if self.s_count >= STOP_COUNT_STREAK:
            hits.append(f"count_train<{self.stop_count}x{STOP_COUNT_STREAK}")
        if collapse:
            hits.append("collapse_flag")
        if self.stop_dmin > 0 and dmin is not None and dmin <= self.stop_dmin:
            hits.append(f"dmin<={self.stop_dmin}")
        if self.stop_cnull and c_mean is not None and c_null is not None and c_mean <= c_null:
            hits.append("c<=c_null")
        return hits

    def state(self):
        return dict(s_score=self.s_score, s_count=self.s_count)

    def load_state(self, d):
        self.s_score = int(d.get("s_score", 0))
        self.s_count = int(d.get("s_count", 0))


def eval_disturbed(anchor_out):
    """扩张 streak 的受扰判定 (评审 Important#4): 回滚评/过渡评/收编评的成绩
    不作稳定性证据 -- accept_current 同样是锚被替换的制度扰动时刻."""
    an = anchor_out or {}
    ev = an.get("event") or ""
    return bool(an.get("transition")) or "reset" in ev or "accept_current" in ev


@torch.no_grad()
def measure_extrap0(model, cfg, new_ns, dev, seed, m=32):
    """扩张瞬间基线 (§10d 改4, [U]): 新 N 在任何针对性训练之前的写-读直读准确率,
    附章数中位 (码是否顺势延伸). 专用种子流, 不占训练/评测流. 调用时 cfg.k 已是
    新档 (l_max 按新档取)."""
    rng = torch.Generator().manual_seed(seed)
    out = {}
    for n in new_ns:
        sc = torch.stack([D.render_scene(rng, n) for _ in range(m)])
        st = model.encode_scene(sc.to(dev))["tokens"]
        progs, _, _ = model.writer.rollout(
            st, l_max_of(cfg), torch.zeros(m, device=dev))
        x2 = GM.render_channel(progs, cfg.s, rng, cfg.occ_k).to(dev)
        rp = model.heads.read(model.encode_canvas(x2)["cls"])
        acc = float((rp[:, :GM.K_MAX].argmax(-1) + 1 == n).float().mean())
        stmp = torch.tensor([len(p) for p in progs], dtype=torch.float32)
        out[n] = dict(read=round(acc, 4), stamps_med=float(stmp.median()))
    accs = [v["read"] for v in out.values()]
    return dict(per_n=out, mean=round(sum(accs) / max(len(accs), 1), 4))


EXPAND_STEP = 2       # s114 C5 ([U]): 进档步长 +4 -> +2 (4→6→8) -- p3b 翻倍一次
#                       扩四个新 N, 点火速率 ~1 数量/4-5k 步下同时自举四个太贪
SETTLE_STREAK = 3     # [U] p3c-alpha2: 连续三评贪心码表汉明变动量=0 记 settled

# C9 退化组守卫常数 ([U] p3d 给定; p3e 一行修正: 触发/退出改自然臂口径 div_nat
# 同线 0.15 -- 旧退出线 0.25 恰 = 2/8 受迫臂机械占比, 采纳成功 ⇒ div 回受迫底
# ⇒ 结构上退不出, p3d 四守卫全程 active 实测. 撤除离散化与观察粒度为 [C] 实例化)
C9_DIV_LO = 0.15      # 触发线: div_nat < 此值 (idle 全 8 臂自然, 1/8=0.125 仍触发)
C9_TRIG = 50          # 连续观察步数 (N 缺席的步冻结不计)
C9_DIV_HI = 0.15      # 退出线: div_nat > 此值 (active 6 自然臂, 1/6=0.167 即越线)
C9_EXIT = 200         # 退出持续观察步数
C9_RAMP = 200         # 线性撤除窗 (臂数 ceil(2·(1−t/窗)): 前半 2 后半 1, 满窗归零)
C9_ARMS = 2           # forced 臂数 = 每组末 2 条采样臂 (贪心锚不动)


class C9Guard:
    """退化组守卫 (C9, [U] p3d; p3e 改口径): div_nat = 本步该 N 各组「与首条
    贪心臂程序不同」的**自然臂**(非受迫) rollout 占比均值 (程序按动作序列精确
    比较; 受迫臂不入分子分母 -- 守卫测的是策略自己长出的多样性, 不是自己垫进去
    的). 吸收态实测背景: N5 采样 512 条零落墨、逐 token 熵 1e-4 -- 组内比较付
    不了没被采样的程序, 守卫在检测到零多样性时垫进正常质量的短程序候选 (随机
    位置禁众数动作, 其余 T=1 照常).
    状态机: idle --div_nat<LO×TRIG--> active --div_nat>HI×EXIT--> ramp
    --RAMP 步--> idle; ramp 期再触发回 active. 采纳记录 = forced 臂门后优势
    > 0 的计数."""

    def __init__(self, ns):
        self.state = {n: "idle" for n in ns}
        self.below = {n: 0 for n in ns}
        self.above = {n: 0 for n in ns}
        self.ramp_t = {n: 0 for n in ns}
        self.since = {n: 0 for n in ns}
        self.adopt_pos = {n: 0 for n in ns}
        self.adopt_tot = {n: 0 for n in ns}
        self.events = []

    def arms(self, n):
        st = self.state.get(n)
        if st == "active":
            return C9_ARMS
        if st == "ramp":
            return math.ceil(C9_ARMS * (C9_RAMP - self.ramp_t[n]) / C9_RAMP)
        return 0

    def observe(self, div_by_n, step):
        """喂本步逐 N div (缺席 N 不在 dict = streak 冻结). 返回本步事件."""
        evs = []
        for n, d in div_by_n.items():
            st = self.state[n]
            if st != "idle":
                self.since[n] += 1
            if st == "idle":
                self.below[n] = self.below[n] + 1 if d < C9_DIV_LO else 0
                if self.below[n] >= C9_TRIG:
                    self.state[n] = "active"
                    self.below[n] = self.above[n] = self.since[n] = 0
                    evs.append(dict(n=n, event="trigger", step=step))
            elif st == "active":
                self.above[n] = self.above[n] + 1 if d > C9_DIV_HI else 0
                if self.above[n] >= C9_EXIT:
                    self.state[n] = "ramp"
                    self.ramp_t[n] = 0
                    evs.append(dict(n=n, event="ramp", step=step,
                                    dur=self.since[n]))
            elif st == "ramp":
                self.ramp_t[n] += 1
                self.below[n] = self.below[n] + 1 if d < C9_DIV_LO else 0
                if self.below[n] >= C9_TRIG:
                    self.state[n] = "active"
                    self.below[n] = self.above[n] = 0
                    evs.append(dict(n=n, event="retrigger", step=step))
                elif self.ramp_t[n] >= C9_RAMP:
                    self.state[n] = "idle"
                    self.below[n] = self.above[n] = 0
                    evs.append(dict(n=n, event="off", step=step,
                                    dur=self.since[n], adopt=self._adopt(n)))
        self.events += evs
        return evs

    def _adopt(self, n):
        t = self.adopt_tot[n]
        return dict(pos=self.adopt_pos[n], tot=t,
                    frac=round(self.adopt_pos[n] / t, 4) if t else None)

    def note_adoption(self, n, advs):
        self.adopt_tot[n] += len(advs)
        self.adopt_pos[n] += sum(1 for a in advs if a > 0)

    def summary(self):
        return {n: dict(state=st, since=self.since[n], adopt=self._adopt(n))
                for n, st in self.state.items()
                if st != "idle" or self.adopt_tot[n] > 0}

    def state_dict(self):
        return dict(state=self.state, below=self.below, above=self.above,
                    ramp_t=self.ramp_t, since=self.since,
                    adopt_pos=self.adopt_pos, adopt_tot=self.adopt_tot,
                    events=self.events)

    def load_state(self, d):
        for k in ("state", "below", "above", "ramp_t", "since",
                  "adopt_pos", "adopt_tot", "events"):
            if k in d:
                setattr(self, k, d[k])


class SettleCtl:
    """churn/定居仪表 ([U] p3c-alpha2): 逐评贪心模板表相对上评的汉明变动总格数
    (= shift.per_n 之和); 连续 SETTLE_STREAK 评为 0 记 settled -- Q5「不定居的
    终局不入账」. 扩张时清零 (跨档模板不可比)."""

    def __init__(self):
        self.streak = 0

    def offer(self, churn):
        self.streak = self.streak + 1 if churn == 0 else 0
        return self.streak >= SETTLE_STREAK

    def reset(self):
        self.streak = 0


# ================================================================ v4.1 对齐机构
class AlignCtx:
    """对齐最小化 v4.1 的训练期状态 (align.py 的机构在训练器里的装配):
    度规 / 留出调度 / c 通道缓冲 / u_S EMA / 首见登记表 / 逐步仪表窗 / R 史 / β 水平门.
    holdout_frac=0 时不建 (旧制逐位不变)."""

    def __init__(self, model, opt, cfg, dev, ns):
        self.cfg = cfg
        self.dev = dev
        self.cvp = AL.cv_params(model)
        self.scp = AL.sc_params(model)
        self.n_bb = sum(p.numel() for p in AL.bb_params(model))   # 白化扁平向量前段 = 主干
        self.metric = AL.Metric(opt, cfg.metric)
        self.opt = opt
        self.hold = AL.HoldoutSched(ns, cfg.holdout_frac, cfg.t_rotate, cfg.seed)
        self.cchan = AL.CChannel(model, cfg.zeta, dev)
        self.u_ema = None                  # u_S 的 EMA 方向 (cfg.u_ema>0 时), 画布路由
        self.last_u = None                 # 上一步 u_S (相邻步方向余弦仪表, §12.6), 画布路由
        self.u_ema_sc = None               # 同上, 场景路由 (改动 1)
        self.last_u_sc = None
        self.first_seen = []               # M6 首见登记表 (永不改写)
        self.win = []                      # 评测间逐步仪表
        self.c_by_n = {}                   # 评测间逐数量 c 累加 (sum, cnt)
        self.c_hist = torch.zeros(40, dtype=torch.long)   # c 直方 [-1,1] 40 箱
        self.R_hist = []                   # (step, F, R_cv) 史
        self.last_ell = {}                 # 上评 M1 逐数量 ℓ_n (M6 地板用)
        self.last_ell_count = {}
        self.last_acc = {}                 # 上评 M1 逐数量六任务均值 (改动 3: 地板取已达标数量)
        self.beta_gate = True              # L* 水平门 (§5.5)
        self.hold_log = None
        self.capped = 0
        self.capped_sc = 0
        self.steps = 0
        self.last_expand = 0               # 改动 3 定时器: 上次前沿推进步 (发射步 = 0 起算)
        self.expands = []                  # 扩张记录 (跨恢复累计, 终报用)

    def S(self, step):
        S = self.hold.current(step)
        key = (self.hold.rot_step, tuple(self.hold.S))
        if key != self.hold_log:
            self.hold_log = key
            return S, True
        return S, False

    def note_c(self, c, ns):
        for n in sorted(set(int(x) for x in ns.tolist())):
            m = ns == n
            s, k = self.c_by_n.get(n, (0.0, 0))
            self.c_by_n[n] = (s + float(c[m].sum()), k + int(m.sum()))
        b = ((c.clamp(-1, 1) + 1.0) * 20).long().clamp(max=39).cpu()
        self.c_hist += torch.bincount(b, minlength=40)

    def flush(self):
        """评测时汇总窗内仪表 (均值), 清窗."""
        out = {}
        if self.win:
            keys = set().union(*[w.keys() for w in self.win])
            for k in keys:
                vals = [w[k] for w in self.win if k in w and w[k] is not None]
                if vals:
                    out[k] = round(sum(vals) / len(vals), 5)
            out["n_steps"] = len(self.win)
        out["c_by_n"] = {str(n): round(s / k, 4) for n, (s, k) in
                         sorted(self.c_by_n.items()) if k > 0}
        out["c_hist"] = self.c_hist.tolist()
        out["capped"] = self.capped
        out["capped_sc"] = self.capped_sc
        self.win, self.c_by_n, self.capped, self.capped_sc = [], {}, 0, 0
        self.c_hist.zero_()
        return out

    def state(self):
        return dict(hold=self.hold.state(), cchan=self.cchan.state(),
                    first_seen=self.first_seen, R_hist=self.R_hist,
                    beta_gate=self.beta_gate,
                    u_ema=([u.cpu() for u in self.u_ema] if self.u_ema is not None
                           else None),
                    u_ema_sc=([u.cpu() for u in self.u_ema_sc] if self.u_ema_sc is not None
                              else None),
                    last_ell=self.last_ell, last_ell_count=self.last_ell_count,
                    last_acc=self.last_acc, last_expand=self.last_expand,
                    expands=self.expands)

    def load_state(self, d):
        self.hold.load_state(d["hold"])
        self.cchan.load_state(d["cchan"])
        self.first_seen = list(d.get("first_seen", []))
        self.R_hist = list(d.get("R_hist", []))
        # L* 水平门 (§5.5): l_star>0 才有门; l_star=0 ⇒ 门恒开 (检查点里的门态不继承 -- 第四阶段步骤 2
        # 实测: 门态「关」随检查点带入, β_cv 整窗置零, 与「β_cv 定法同 β_sc, 不受门」不符)
        self.beta_gate = bool(d.get("beta_gate", True)) if self.cfg.l_star > 0.0 else True
        if d.get("u_ema") is not None:
            self.u_ema = [u.to(self.dev) for u in d["u_ema"]]
        if d.get("u_ema_sc") is not None:
            self.u_ema_sc = [u.to(self.dev) for u in d["u_ema_sc"]]
        self.last_ell = {int(k): v for k, v in d.get("last_ell", {}).items()}
        self.last_ell_count = {int(k): v for k, v in d.get("last_ell_count", {}).items()}
        self.last_acc = {int(k): v for k, v in d.get("last_acc", {}).items()}
        self.last_expand = int(d.get("last_expand", 0))
        self.expands = list(d.get("expands", []))


def _rows_of(gidx, g):
    return [i * g + j for i in gidx for j in range(g)]


def _sub_batch(b, gidx, dev):
    """build_group_batch 输出按组子集切片 (逐 rollout 字段按行, 候选按组)."""
    g = b["g"]
    rows = torch.tensor(_rows_of(gidx, g), dtype=torch.long, device=dev)
    gi = torch.tensor(gidx, dtype=torch.long, device=dev)
    return dict(rows=rows,
                th={k: v[rows] for k, v in b["th"].items()},
                tg={k: v[rows] for k, v in b["tg"].items()},
                truth5=b["truth5"][rows], ns=b["ns_rep"][rows],
                cands=b["cands"][gi])


def _route_correction(align, params, den, g_V, g_S, recompute, beta, cfg, opt, route):
    """一条路由 (route ∈ {"cv","sc"}) 的交叉项与施压 (改动 1 两路由同定法):
    u_S = [g_S(Θ+δF⁻¹ĝ_V) − g_S(Θ)]/(δ‖g_S‖_{F⁻¹}); 施加量 = −β·‖g_V‖_{F⁻¹}·u (u 可为 EMA 方向),
    β 逐步夹取使 β‖u‖_{F⁻¹} ≤ BETA_CAP (‖修正‖ ≤ 0.1‖g_V‖); β=0 ⇒ 算而不 apply.
    返回仪表 dict (键无后缀; 调用方按路由加 _sc)."""
    u, d = AL.cross_term(params, den, g_V, g_S, recompute, delta_mult=cfg.delta_mult, opt=opt)
    diag = dict(u=d["u"], delta=d["delta"], gV=d["gV"], gS=d["gS"],
                degenerate=float(bool(d.get("degenerate"))))
    last_attr, ema_attr, cap_attr = (("last_u", "u_ema", "capped") if route == "cv"
                                     else ("last_u_sc", "u_ema_sc", "capped_sc"))
    if not d.get("degenerate"):                      # §12.6 稳定性仪表: 相邻步 u 方向余弦
        last = getattr(align, last_attr)
        if last is not None:
            diag["u_cos_prev"] = float(AL.Metric.cos(u, last, den))
        setattr(align, last_attr, [x.detach().clone() for x in u])
    if cfg.u_ema > 0.0:
        with torch.no_grad():
            ema = getattr(align, ema_attr)
            if ema is None:
                setattr(align, ema_attr, [x.clone() for x in u])
            else:
                for e, x in zip(ema, u):
                    e.mul_(cfg.u_ema).add_(x, alpha=1.0 - cfg.u_ema)
        u_app = getattr(align, ema_attr)
    else:
        u_app = u
    if beta > 0.0 and not d.get("degenerate"):
        nu = float(AL.Metric.norm(u_app, den))
        if beta * nu > AL.BETA_CAP:
            beta = AL.BETA_CAP / max(nu, 1e-30)
            setattr(align, cap_attr, getattr(align, cap_attr) + 1)
        scale = beta * d["gV"]
        with torch.no_grad():
            for p, x in zip(params, u_app):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(x, alpha=-scale)
        diag["beta_eff"] = beta
        diag["corr_frac"] = float(scale * nu / max(d["gV"], 1e-30))   # ‖修正‖/‖g_V‖
    else:
        diag["beta_eff"] = 0.0
    return diag


def apply_align_grads(model, align, out, cfg, opt, dev):
    """把 phase2_losses 的对齐产物装进 .grad (调用方已对 l_grpo+l_count 反传):
    ① g_V^cv 加入 .grad (V 的画布路由精确梯度 = 旧制照常 apply);
    ② 画布路由 u_S (§5.2): 算而不 apply 除非 β_align>0 且水平门开;
    ③ 场景路由 u_S ([U] 2026-08-17 改动 1, Θ_sc = Θ_E ∪ count_head, L_count 分 V/S): 算而不
       apply 除非 β_sc>0 (同一水平门); 两路由各自独立夹取.
    S 的画布路由精确梯度 g_S^cv 在任何路径上都不进 .grad (X1, 范围 = L_task/L_read; 计数流
    S 样本按改动 2 经 l_count 照常 apply). 返回仪表 dict (场景路由键带 _sc 后缀)."""
    A = out["align"]
    cvp = align.cvp
    den = A["den"]
    with torch.no_grad():
        for p, g in zip(cvp, A["g_V"]):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.add_(g)
    diag = dict()
    if A["g_S"] is not None:
        beta = cfg.beta_align if align.beta_gate else 0.0
        diag.update(_route_correction(align, cvp, den, A["g_V"], A["g_S"], A["recompute"],
                                      beta, cfg, opt, "cv"))
    if A.get("g_S_sc") is not None:
        # [C] 2026-08-17: L* 水平门 (§5.5) 只管画布路由 -- 它防的是「读者对 S 稳定弃权」(读头弃权类
        # 把 S 损失停在弃权地板附近); 场景路由 (计数头, 无弃权类) 无此退化解, β_sc 不受该门.
        # 阶段 A 实测两评 L_S 1.77/1.96 > L* 1.7175 而 abst_S ≈ 0 -- 门关是 S 未学会, 不是弃权停车.
        dsc = _route_correction(align, align.scp, A["den_sc"], A["g_V_sc"], A["g_S_sc"],
                                A["recompute_sc"], cfg.beta_sc, cfg, opt, "sc")
        diag.update({k + "_sc": v for k, v in dsc.items()})
    return diag


def next_k(old_k, step):
    """下一课程档: step=1 时前沿 +1 **个训练带数量** (跨空洞段直接跳到下一个训练带 N, 首见
    跳距 >1 自动产生 -- v4.1 §4.4 「跨越空洞段时自动产生跳距 >1 的首见事件」; 空洞内的 k
    值不产生新数量, 不作为档); step>1 保持旧制 +step (p3e 制度, +2)."""
    if step != 1:
        return min(old_k + step, GM.K_MAX)
    nk = old_k + 1
    while nk < GM.K_MAX and D.band_of(nk) != "train":
        nk += 1
    return min(nk, GM.K_MAX)


def do_expand(model, cfg, ctl, latch, s1, dev, why="stable"):
    """K 扩张动作 (§10d 改4 [U]; 步长 s114 C5 改 +EXPAND_STEP; v4.1 §4.4 逐个引入
    经 cfg.expand_step=1). 顺序是登记的
    防火墙 -- extrap@0 必须在任何针对新 N 的训练之前测 (调用点在评测钩内, 本步
    梯度已按旧档结束; 新装载器由调用方在返回后重建, 旧装载器预取样本一并废弃).
    旧档 best 检查点另存 ckpt_best_k{旧档}.pt (双点口径跨档保留), 闩重置.
    锚宽限期: 只在稳定性触发 (why="stable") 时进 ([U] s113 改4「K 扩张 = 制度变更」的原语境);
    [C] 2026-08-17 改动 3 定时推进 (why="timer", 每 1500 步) 不重启锚宽限 -- 否则宽限 2000 步
    永远盖过推进间隔, 锚永不刷新、KL 参考冻结在发射态, 与 B 臂制度 (grace@2000/forced@7500)
    不同; 定时推进只重置闩/并轨/定居/守卫 (跨档成绩与模板语义不可比, 同稳定触发)."""
    old_k = cfg.k
    step = cfg.expand_step if cfg.expand_step > 0 else EXPAND_STEP
    new_k = next_k(old_k, step)
    old_ns = set(D.train_ns(old_k))
    new_ns = [n for n in D.train_ns(new_k) if n not in old_ns]
    cfg.k = new_k
    e0 = measure_extrap0(model, cfg, new_ns, dev, cfg.seed + 888 + new_k)
    bp = os.path.join(cfg.out, "ckpt_best.pt")
    if os.path.exists(bp):
        shutil.copyfile(bp, os.path.join(cfg.out, f"ckpt_best_k{old_k}.pt"))
    prev = dict(score=round(latch.best, 4), step=latch.step)
    latch.reset()
    if ctl is not None and why == "stable":
        ctl.on_expand(s1)
    return dict(k=[old_k, new_k], extrap0=e0, prev_best=prev, why=why)


# ================================================================ 遮蔽/保留桶
def split_mask_keep(selected, rng, keep_frac):
    """选中格 (B,T) bool -> (replace, keep). keep 桶按 keep_frac 逐格独立抽;
    [MASK] 只换 replace 桶, 损失在全部选中格 (MASK-KEEP)."""
    if keep_frac <= 0.0:
        return selected.clone(), torch.zeros_like(selected)
    u = torch.rand(selected.shape, generator=rng)
    keep = selected & (u < keep_frac)
    return selected & ~keep, keep


def mask_posterior_q(truth, replace, lmax):
    """均匀画布上被遮格的贝叶斯后验 P(有章 | 可见): 先验 L ~ U{0..lmax}
    (geometry.sample_uniform_prog 的生成过程), 似然为超几何. truth (B,T) ∈ {0..S}
    为腐蚀前真值, replace (B,T) bool 为 [MASK] 位. 被遮格可交换 -> 每画布一个 q."""
    B, T = truth.shape
    Ls = torch.arange(0, lmax + 1, dtype=torch.float64)

    def lcomb(n, k):
        return (torch.lgamma(n + 1) - torch.lgamma(k + 1)
                - torch.lgamma(n - k + 1))

    qs = torch.zeros(B, dtype=torch.float64)
    for b in range(B):
        vis = ~replace[b]
        V = int(vis.sum())
        mv = int(((truth[b] > 0) & vis).sum())
        M = T - V
        if M == 0:
            continue
        ok = (Ls >= mv) & (T - Ls >= V - mv) & (Ls <= lmax)
        ll = torch.full_like(Ls, float("-inf"))
        Lok = Ls[ok]
        ll[ok] = (lcomb(Lok, torch.tensor(float(mv)))
                  + lcomb(torch.tensor(float(T)) - Lok,
                          torch.tensor(float(V - mv))))
        post = torch.softmax(ll, 0)
        qs[b] = (post * (Ls - mv).clamp(min=0) / M).sum()
    return qs.clamp(0.0, 1.0)


def mask_oracle(truth, replace, lmax):
    """遮蔽位的贝叶斯 oracle 期望准确率与 CE (MEASURED-NULL: 素养门的天花板在
    当前评测画布上实测). 章型不可预测 -> 章命中按期望 1/3 计."""
    q = mask_posterior_q(truth, replace, lmax)
    acc_n, ce_n, n_tot = 0.0, 0.0, 0
    for b in range(truth.shape[0]):
        cells = replace[b]
        m = int(cells.sum())
        if m == 0:
            continue
        emp = float((truth[b][cells] == 0).sum())
        stp = m - emp
        qb = float(q[b])
        if 1.0 - qb >= qb / 3.0:
            acc_n += emp
        else:
            acc_n += stp / 3.0
        eps = 1e-12
        ce_n += -(emp * math.log(max(1.0 - qb, eps))
                  + stp * math.log(max(qb / 3.0, eps)))
        n_tot += m
    return dict(acc=acc_n / max(n_tot, 1), ce=ce_n / max(n_tot, 1))


# ================================================================ 空白零假设
def blank_nulls(k):
    """六任务的画布盲最优准确率, 对 N ~ U(train_ns(k)) 与 θ 抽样分布精确枚举
    (spec §10.1 的"随机水平"按 MEASURED-NULL 落地). t5 给盲上界: 两条硬负共享
    同一 Z (风格成对可识别), 盲读者可剔除 -> 1/(N_CAND - N_HARD)."""
    ns = torch.tensor(D.train_ns(k), dtype=torch.float64)
    W = ns.shape[0]
    pn = 1.0 / W
    a1 = a3 = 0.0
    for tau in range(1, k):
        pgt = float((ns > tau).sum()) / W
        a1 += max(pgt, 1.0 - pgt)
        best_pt = pn if bool((ns > tau).any()) else 0.0
        a3 += max(float((ns <= tau).sum()) / W, best_pt)
    a1 /= (k - 1)
    a3 /= (k - 1)
    a2 = max(float((ns % 2 == 0).sum()), float((ns % 2 == 1).sum())) / W
    a4 = 0.0
    for p in D.P_CHOICES:
        r = ns.long() % p
        a4 += float(torch.bincount(r, minlength=p).max()) / W
    a4 /= len(D.P_CHOICES)
    return dict(t1=a1, t2=a2, t3=a3, t4=a4,
                t5=1.0 / (D.N_CAND - D.N_HARD), t6=pn)


# ================================================================ 数据流
class SceneStream(torch.utils.data.IterableDataset):
    """计数流: 无限 (scene, n), N ~ U(train_ns(k)), Z 全新 (防火墙由 data 保证)."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = torch.Generator().manual_seed(
            (self.seed * _PHI + (wid + 1) * _MIX) % _MOD)
        while True:
            n = D.sample_n(rng, self.k)
            yield D.render_scene(rng, n), n


class GroupStream(torch.utils.data.IterableDataset):
    """GRPO 组流: 无限 sample_group dict (共享 x1 / 组内冻结 θ / 候选集)."""

    def __init__(self, k, seed):
        self.k, self.seed = k, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = torch.Generator().manual_seed(
            (self.seed * _PHI + (wid + 7) * _MIX) % _MOD)
        while True:
            yield D.sample_group(rng, self.k)


RESUME_SALT = 104729   # --resume 时数据流/信道噪声流种子按恢复步加盐 (质数; 不加盐则重放流开头)


def make_group_loader(cfg, salt=0):
    """组流装载器 (s113 抽出): K 扩张时重建. 种子按课程档加盐 -- 新档流不与旧档
    重放同一场景种子序列; salt = RESUME_SALT × 恢复步 (分阶段 --resume 不重放旧段的组序列)."""
    return torch.utils.data.DataLoader(
        GroupStream(cfg.k, cfg.seed + 1 + 1000 * cfg.k + salt),
        batch_size=cfg.n_groups, num_workers=cfg.workers,
        collate_fn=lambda x: x, persistent_workers=cfg.workers > 0,
        prefetch_factor=2 if cfg.workers > 0 else None)


def make_loaders(cfg, salt=0):
    sl = torch.utils.data.DataLoader(
        SceneStream(GM.K_MAX, cfg.seed + salt), batch_size=cfg.batch_scenes,
        num_workers=cfg.workers, persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None)
    gl = make_group_loader(cfg, salt) if cfg.phase == 2 else None
    return sl, gl


# ================================================================ 前向装配
def mask_forward(model, cfg, rng, dev, ret_metrics=False):
    """掩码素养流一步: 均匀程序 -> 腐蚀画布 -> [MASK] 替换 replace 桶 ->
    损失在全部选中格 (MASK-KEEP). 画布不经写者 (spec §2.6)."""
    # 素养通道维持旧配方 (s110 接线裁定: 素养是"看得见图章", 不跟 occ_k 上调 --
    # 否则主干把容量花在补全被遮格上, 与 15-40% 遮蔽任务叠加过头)
    progs = [GM.sample_uniform_prog(rng) for _ in range(cfg.batch_mask)]
    x = GM.channel(GM.raster(progs, cfg.s, rng), cfg.s, rng)
    truth = GM.cell_truth(progs)
    sel = L.sample_mask_cells(cfg.batch_mask, rng)
    rep, keep = split_mask_keep(sel, rng, cfg.keep_frac)
    enc = model.encode_canvas(x.to(dev), mask_cells=rep.to(dev))
    logits = model.heads.cell(enc["tokens"])
    loss = L.mask_loss(logits, truth.to(dev), sel.to(dev))
    if not ret_metrics:
        return loss
    with torch.no_grad():
        pred = logits.argmax(-1).cpu()
        tf = truth.view(-1, GM.T)
        m_acc = float((pred[rep] == tf[rep]).float().mean())
        k_acc = float((pred[keep] == tf[keep]).float().mean()) if bool(
            keep.any()) else float("nan")
    return loss, dict(mask_acc=m_acc, kept_acc=k_acc)


def build_group_batch(groups, g, dev):
    """B 组静态部分 -> 训练张量. 逐 rollout 展开的字段重复 g 次 (组内共享,
    spec §6.4); x1/cands 不重复 (编码后再展开)."""
    B = len(groups)
    rep = lambda t: t.repeat_interleave(g)
    x1 = torch.stack([gr["scene"] for gr in groups]).to(dev)
    ns = torch.tensor([gr["n"] for gr in groups], dtype=torch.long, device=dev)
    th = dict(
        tau=torch.tensor([gr["theta"]["tau"] - 1 for gr in groups],
                         device=dev),
        p=torch.tensor([D.P_CHOICES.index(gr["theta"]["p"]) for gr in groups],
                       device=dev),
        m=torch.tensor([D.M_CHOICES.index(gr["theta"]["m"]) for gr in groups],
                       device=dev))
    tg = {t: torch.tensor([gr["targets"][t] for gr in groups], device=dev)
          for t in ("t1", "t2", "t3", "t4", "t6")}
    truth5 = torch.tensor([gr["truth5"] for gr in groups], device=dev)
    cands = torch.stack([gr["cands"] for gr in groups]).to(dev)
    return dict(x1=x1, ns=ns, ns_rep=rep(ns),
                th={k: rep(v) for k, v in th.items()},
                tg={k: rep(v) for k, v in tg.items()},
                truth5=rep(truth5), cands=cands, B=B, g=g)


def phase2_losses(model, cfg, groups, count_scenes, count_ns, dev,
                  rng_cpu, rng_dev, step, tw, ref_writer=None, lag=None,
                  beta=None, probe=None, guard=None, align=None, hold=None,
                  s_scale=None):
    """一个微批的 Phase 2 全损失 (调用方 backward). 路由 (spec §2.7):
    写者输入 no_grad 场景 token; 监督损失走独立带梯度前向.
    lag (裁定2/3): 影子模型 -- 奖励/任务权重读影子 (搬家立刻扣分); β 参考默认取
    影子写者, ref_writer (固定锚) 给定时用锚. beta 覆写 cfg.beta (锚状态机的
    自适应值 + 重置加压). probe (s114 §10e): 双探针 -- r_probe 入奖励按
    先拟合后入册次序 (当步样本不进当步拟合), probe_out 只训不酬.
    guard (C9 §10e-补2): 退化组守卫 -- 先按现态定 forced 臂再 rollout, 后喂
    div 推进状态机 (每次调用 = 一个观察步; accum>1 时观察粒度为微批, 已登记).
    align/hold (v4.1 §4-§6): hold = 本步留出集 S; S 组的画布路由损失 (L_task+μL_read)
    **算而不 apply** (精确梯度经 out['align'] 交给 apply_align_grads 只作交叉项),
    GRPO 照常发钱; V/S 各走独立前向 (图分离). 计数流 ([U] 2026-08-17 改动 2): L_count 在
    整个训练带全程 apply, 不受 S 影响 (l_count_S 仍记账); 场景路由 S/V 精确梯度 (改动 1)
    每步另算: D^sc_V = 计数流 n∉S 样本 (= 实际 apply 的 V 方向), D^sc_S = 计数流 n∈S 样本 ∪
    本步 GRPO 组场景 x1 中 n∈S 者 ([C] 实例化: |S|=2 对计数流 88 数量均匀抽样每步期望 1.45
    个 S 样本、23% 步为零 -- 并入同 S 数量的组场景 (~5/步) 使交叉项非退化, 组场景的 L_count
    只算不 apply). c 通道 (§6) 由影子读者算 c^(i), κ>0 且窗均值就绪时入奖励. s_scale: X1
    判别测试用 -- 对 S 逐行画布路由损失与场景路由 S 损失乘以此因子 (β=0 下 .grad 必须逐位
    不变). align=None ⇒ 旧路径逐位不变."""
    b = build_group_batch(groups, cfg.g, dev)
    B, g = b["B"], cfg.g
    with torch.no_grad():
        enc = model.encode_scene(b["x1"])
        conf = model.heads.count(enc["cls"]).softmax(-1).max(-1).values
        st = enc["tokens"].repeat_interleave(g, 0)
    temps = torch.tensor([0.0] * cfg.n_greedy
                         + [cfg.temp] * (g - cfg.n_greedy),
                         device=dev).repeat(B)
    ban = None
    if guard is not None:
        ban = torch.zeros(B * g, dtype=torch.bool, device=dev)
        for i, gr in enumerate(groups):
            for j in range(guard.arms(gr["n"])):
                ban[i * g + (g - 1 - j)] = True    # 末 k 条采样臂, 贪心锚不动
    progs, acts, lens = model.writer.rollout(st, l_max_of(cfg), temps,
                                             rng=rng_dev, ban_mode=ban)
    c9_out = None
    if guard is not None:
        # div_nat ([U] p3e): 自然臂口径 -- 受迫臂不入分子分母, 守卫测策略自己
        # 的多样性; div_mix (全臂口径, 旧 div) 仍记录, 差值 = 受迫臂实际偏离量
        nat_by_n, mix_by_n = {}, {}
        for i, gr in enumerate(groups):
            ref = progs[i * g]                     # 首条贪心臂为参照 (恒自然)
            nat = [j for j in range(g) if not bool(ban[i * g + j])]
            nat_by_n.setdefault(gr["n"], []).append(
                sum(1 for j in nat if progs[i * g + j] != ref) / len(nat))
            mix_by_n.setdefault(gr["n"], []).append(
                sum(1 for j in range(g) if progs[i * g + j] != ref) / g)
        nat_by_n = {n: sum(v) / len(v) for n, v in nat_by_n.items()}
        c9_out = dict(div={str(n): round(v, 3)
                           for n, v in sorted(nat_by_n.items())},
                      div_mix={str(n): round(sum(v) / len(v), 3)
                               for n, v in sorted(mix_by_n.items())},
                      evs=guard.observe(nat_by_n, step))
    # 改4 (§10): 奖励对 m 次独立腐蚀取均值 (一次抽样 = 重渲几何抖动 + 信道全套,
    # Phase 0 噪声定义"同内容独立腐蚀重渲"); 信道方差降 1/m, 期望不变. 第 0 抽的
    # 张量与在线监督共源 (接线一: 读者在腐蚀画布上训练, 奖励走同一信道).
    m_r = max(1, cfg.reward_m)
    x2s = [GM.render_channel(progs, cfg.s, rng_cpu, cfg.occ_k).to(dev)
           for _ in range(m_r)]
    x2 = x2s[0]
    c_read = anneal_c(cfg, step)
    S = set(hold) if hold else set()
    isS = torch.tensor([gr["n"] in S for gr in groups], device=dev)
    gV = [i for i, gr in enumerate(groups) if gr["n"] not in S]
    gS = [i for i, gr in enumerate(groups) if gr["n"] in S]
    outV = outS = None
    if align is None:
        # 旧路径 (逐位不变): 全行一次前向
        enc2 = model.encode_canvas(x2)
        h = enc2["cls"]
        tl = model.heads.task_logits(h, b["th"])
        hc = model.encode_scene(
            b["cands"].view(B * D.N_CAND, *b["cands"].shape[2:]))["cls"]
        hc_rep = hc.view(B, D.N_CAND, -1).repeat_interleave(g, 0)
        t5 = model.heads.t5_scores(h, hc_rep)
        accs_on = GR.task_accs(tl, t5, b["tg"], b["truth5"]).detach()
    else:
        # v4.1: V / S 分路前向 (图分离; S 的精确梯度只作交叉项, 不 apply)
        subV = _sub_batch(b, gV, dev) if gV else None
        subS = _sub_batch(b, gS, dev) if gS else None
        if subV is not None:
            outV = AL.canvas_route(model, x2[subV["rows"]], subV["th"], subV["tg"],
                                   subV["truth5"], subV["cands"], subV["ns"],
                                   c_read, cfg.mu)
        if subS is not None:
            outS = AL.canvas_route(model, x2[subS["rows"]], subS["th"], subS["tg"],
                                   subS["truth5"], subS["cands"], subS["ns"],
                                   c_read, cfg.mu)
        accs_on = torch.zeros(B * g, 6, device=dev)
        rp_all = torch.zeros(B * g, GM.K_MAX + 1, device=dev)
        for sub, o in ((subV, outV), (subS, outS)):
            if o is not None:
                accs_on[sub["rows"]] = o["accs"]
                rp_all[sub["rows"]] = o["read"].detach()
    with torch.no_grad():
        rdr = lag if lag is not None else model
        if lag is None and align is None:
            hc_l = hc_rep                  # 旧制无影子: 候选表征复用在线前向
        else:
            hc_l = rdr.encode_scene(
                b["cands"].view(B * D.N_CAND, *b["cands"].shape[2:]))["cls"]
            hc_l = hc_l.view(B, D.N_CAND, -1).repeat_interleave(g, 0)
        w = tw.weights(dev)
        draws, cs, parts_all = [], [], []
        c_bases, c_diags = [], []                # 投影率 c 的 V 子空间/秩 (c_mode=proj)
        dom = list(D.train_ns(cfg.k))
        for i, x2i in enumerate(x2s):
            if lag is None and i == 0 and align is None:
                draws.append(accs_on)      # 旧制无影子: draw0 直接复用在线命中
                continue
            h_i = rdr.encode_canvas(x2i)["cls"]
            tl_i = rdr.heads.task_logits(h_i, b["th"])
            t5_i = rdr.heads.t5_scores(h_i, hc_l)
            draws.append(GR.task_accs(tl_i, t5_i, b["tg"], b["truth5"]))
            if align is not None:
                den_h = align.metric.denom(align.cchan.hp)
                parts = align.cchan.parts(rdr.heads, h_i, tl_i, t5_i, hc_l, b["tg"],
                                          b["truth5"], b["th"], w)
                if cfg.c_mode == "proj":
                    # [U] 2026-08-17 第四阶段 改动 2: 投影率 c, 只在 n∈S 行算 (V 行 0), 子空间 =
                    # 本步 V 行按数量的 mean G^(i) (无中心化); κ 项对 V 组自然为 0
                    isS_row = isS.repeat_interleave(g)
                    c_i, basis_i, cdiag = align.cchan.c_proj(parts, b["ns_rep"], den_h, isS_row)
                    cs.append(c_i)
                    c_bases.append(basis_i)
                    c_diags.append(cdiag)
                else:
                    cs.append(align.cchan.c_of(parts, b["ns_rep"], den_h, dom))
                parts_all.append(parts)
        accs = torch.stack(draws).mean(0)
        n_st = torch.tensor([len(p) for p in progs], device=dev)
        ph = (probe.reward_margin(x2s, b["ns_rep"], cfg.k)
              if probe is not None else None)
        c_al = torch.stack(cs).mean(0) if cs else None
        if align is not None and cfg.c_mode == "proj":
            kappa_on = cfg.kappa > 0.0 and bool(isS.any()) and any(
                b_ is not None for b_ in c_bases)
        else:
            kappa_on = (align is not None and cfg.kappa > 0.0
                        and align.cchan.ready(dom))
        r = GR.reward(accs, w, n_st, cfg.lam, cfg.lam0, probe_hits=ph,
                      alpha=cfg.alpha, c_align=(c_al if kappa_on else None),
                      kappa=cfg.kappa)
        adv = GR.advantages(r, B, g)
        keepg = gate_weights(conf, cfg, g)
        adv = adv * keepg
        if guard is not None and ban is not None and bool(ban.any()):
            for i, gr in enumerate(groups):        # forced 臂门后优势采纳记录
                rows = [float(adv[i * g + j]) for j in range(g)
                        if bool(ban[i * g + j])]
                if rows:
                    guard.note_adoption(gr["n"], rows)
        c_null = None
        c_rank = c_kV = None
        if align is not None and parts_all:
            for parts in parts_all:                # 入册 (打分之后; 全 rollout 含 S; 窗均值仪表)
                align.cchan.update(parts, b["ns_rep"])
            if cfg.c_mode == "proj":
                isS_row = isS.repeat_interleave(g)
                if bool(isS_row.any()):
                    align.note_c(c_al[isS_row], b["ns_rep"][isS_row])   # 只记 S 行
                if c_diags:
                    c_rank = sum(d_["rank"] for d_ in c_diags) / len(c_diags)
                    c_kV = sum(d_["kV"] for d_ in c_diags) / len(c_diags)
                if step % cfg.align_every == 0 and bool(isS_row.any()) and c_bases[0] is not None:
                    # 硬约束 (c): 零假设 = 打乱 S 行的数量标签 (按行 θ 重算标签, t5 换错候选),
                    # V 子空间不变, 同法投影率; 每窗多次 ⇒ 每评有当评重算值
                    rS = isS_row.nonzero().squeeze(1)
                    thS = {k_: v_[rS] for k_, v_ in b["th"].items()}
                    tgn, t5n, nsn = AL.null_targets(b["ns_rep"][rS], thS, b["truth5"][rS],
                                                    cfg.k, rng_cpu, dev)
                    h_0 = rdr.encode_canvas(x2s[0][rS])["cls"]
                    tl_0 = rdr.heads.task_logits(h_0, thS)
                    t5_0 = rdr.heads.t5_scores(h_0, hc_l[rS])
                    pn = align.cchan.parts(rdr.heads, h_0, tl_0, t5_0, hc_l[rS], tgn, t5n,
                                           thS, w)
                    c_null = float(align.cchan.c_from_basis(
                        c_bases[0], pn, torch.arange(rS.numel(), device=dev)).mean())
            else:
                align.note_c(c_al, b["ns_rep"])
                if step % cfg.align_every == 0:    # X2 打乱数量标签零假设 (c, 余弦口径)
                    tgn, t5n, nsn = AL.null_targets(b["ns_rep"], b["th"], b["truth5"],
                                                    cfg.k, rng_cpu, dev)
                    h_0 = rdr.encode_canvas(x2s[0])["cls"]
                    tl_0 = rdr.heads.task_logits(h_0, b["th"])
                    t5_0 = rdr.heads.t5_scores(h_0, hc_l)
                    pn = align.cchan.parts(rdr.heads, h_0, tl_0, t5_0, hc_l, tgn, t5n,
                                           b["th"], w)
                    c_null = float(align.cchan.c_of(pn, nsn, den_h, dom).mean())
    logp, lp, keep = model.writer.logp_of(acts, lens, st)
    ref = ref_writer if ref_writer is not None else (
        lag.writer if lag is not None else None)
    bt = cfg.beta if beta is None else beta
    if bt > 0.0 and ref is not None:
        with torch.no_grad():
            _, lp_ref, _ = ref.logp_of(acts, lens, st)
        kl = GR.k3_kl(lp, lp_ref, keep)
    else:
        kl = torch.zeros((), device=dev)
    l_grpo = GR.grpo_loss(logp, adv, kl, bt)
    csc = count_scenes.to(dev)
    encg = model.encode_scene(csc)
    cn = count_ns.to(dev)
    lc_row = F.cross_entropy(model.heads.count(encg["cls"]), cn - 1, reduction="none")
    al_out = None
    if align is None:
        l_task = L.task_loss(tl, t5, b["tg"], b["truth5"])
        l_read = L.read_loss(model.heads.read(h), b["ns_rep"], c_read)
        l_count = lc_row.mean()
        l_count_S = None
    else:
        # 改动 2 ([U] 2026-08-17): 计数流全带 apply (S 不再过滤); l_count_S 只记账
        mS = torch.tensor([int(n) in S for n in cn.tolist()], device=dev)
        l_count = lc_row.mean()
        l_count_S = lc_row[mS].mean() if bool(mS.any()) else None
        l_task = outV["l_task"].detach() if outV is not None else torch.zeros((), device=dev)
        l_read = outV["l_read"].detach() if outV is not None else torch.zeros((), device=dev)
        # 改动 1: 场景路由 S 数据 = 计数流 S 样本 ∪ 组场景 S (每组一张 x1)
        xS_sc = torch.cat([csc[mS], b["x1"][isS]]) if (bool(mS.any()) or bool(isS.any())) \
            else None
        nS_sc = torch.cat([cn[mS], b["ns"][isS]]) if xS_sc is not None else None
        al_out = _align_grads(model, cfg, align, outV, outS, subV if gV else None,
                              subS if gS else None, x2, c_read, step, rng_cpu, dev,
                              lc_row, mS, s_scale, xS_sc, nS_sc)
    tw.update(accs)
    if probe is not None:
        probe.step_update(x2, b["ns_rep"])     # 打分之后入册 (C2 界外打分次序)
    with torch.no_grad():
        samp = temps > 0
        ent = float(-(lp * keep)[samp].sum() / keep[samp].sum().clamp(min=1))
        if align is None:
            rp = model.heads.read(h).softmax(-1)
        else:
            rp = rp_all
        met = dict(
            r_mean=float(r.mean()),
            r_std=float(r.view(B, g).std(dim=1).mean()),   # 组内分化 = 点火仪表
            stamps_mean=float(n_st.float().mean()),
            stamps_greedy=float(n_st.view(B, g)[:, :cfg.n_greedy]
                                .float().mean()),
            acc=[round(float(a), 4) for a in accs.mean(0)],
            conf_keep=float(keepg.mean()), ent_tok=ent,
            abstain=float((rp.argmax(-1) == ABSTAIN).float().mean()),
            kl=float(kl))
        if ph is not None:
            met["r_probe"] = float(ph.mean())
        if lag is not None or align is not None:
            met["acc_on"] = [round(float(a), 4) for a in accs_on.mean(0)]
        if align is not None:
            isS_row = isS.repeat_interleave(g)
            ab = (rp.argmax(-1) == ABSTAIN).float()
            met["abst_S"] = float(ab[isS_row].mean()) if bool(isS_row.any()) else None
            met["abst_V"] = float(ab[~isS_row].mean()) if bool((~isS_row).any()) else None
            if cfg.c_mode == "proj":               # 投影率: c 只在 S 行有定义 (c_mean = S 均)
                met["c_mean"] = (float(c_al[isS_row].mean()) if c_al is not None
                                 and bool(isS_row.any()) else None)
                met["c_S"] = met["c_mean"]
                met["c_V"] = None
                met["c_rank"] = c_rank
                met["c_kV"] = c_kV
            else:
                met["c_mean"] = float(c_al.mean()) if c_al is not None else None
                met["c_S"] = (float(c_al[isS_row].mean()) if c_al is not None
                              and bool(isS_row.any()) else None)
                met["c_V"] = (float(c_al[~isS_row].mean()) if c_al is not None
                              and bool((~isS_row).any()) else None)
            met["c_null"] = c_null
            met["kappa_on"] = float(kappa_on)
            met["l_count_S"] = float(l_count_S) if l_count_S is not None else None
            met["n_S_groups"] = len(gS)
            met["n_S_count"] = int(mS.sum())                 # 计数流落在 S 的样本数
            met["n_S_sc"] = int(nS_sc.numel()) if nS_sc is not None else 0   # 场景路由 S 样本数
    return dict(l_grpo=l_grpo, l_task=l_task, l_read=l_read, l_count=l_count,
                metrics=met, c9=c9_out, align=al_out)


def _align_grads(model, cfg, align, outV, outS, subV, subS, x2, c_read, step,
                 rng_cpu, dev, lc_row, mS, s_scale, xS_sc=None, nS_sc=None):
    """v4.1 §5 的逐步梯度产物 (调用方 apply_align_grads 装配):
    g_V^cv = ∇_{Θ_cv}[L_task+μL_read]|_V (要 apply), g_S^cv (只作交叉项),
    recompute() 在当前参数值重算 S 路由梯度 (有限差分用), 度规 den, 与逐步仪表:
    cos_cv, τ (§7.3 M4), L_S/L_V; 每 align_every 步另算: 半批 𝒮̂ (M3),
    τ_null (X2 打乱标签), 联合余弦 (§5.1).
    改动 1 ([U] 2026-08-17): 场景路由每步 -- g_V^sc = ∇_{Θ_sc} L_count|计数流 V (retain, 主反传
    照常 apply 该图), g_S^sc = ∇_{Θ_sc} L_count|D^sc_S (xS_sc/nS_sc 独立前向, 只作交叉项),
    recompute_sc(), den_sc, 与 cos_sc / tau_sc 每步 (τ 与 cos 分两路报, 场景路由为头条)."""
    cvp, metric = align.cvp, align.metric
    den = metric.denom(cvp)
    extra = (step % cfg.align_every == 0)
    out = dict(den=den, g_S=None, recompute=None, L_V=None, L_S=None,
               den_sc=None, g_V_sc=None, g_S_sc=None, recompute_sc=None, L_S_sc=None)
    diag = {}
    # ---- 场景路由 (改动 1): 每步
    scp = align.scp
    den_sc = metric.denom(scp)
    out["den_sc"] = den_sc
    if bool((~mS).any()):
        gsV = AL.grads_of(lc_row[~mS].mean(), scp, retain=True)
        out["g_V_sc"] = gsV
        diag["gV_sc_raw"] = float(AL.Metric.norm(gsV, den_sc))
    else:
        gsV = None
    if xS_sc is not None and gsV is not None:
        sS = (s_scale if s_scale is not None else 1.0)

        def sc_loss():
            enc = model.encode_scene(xS_sc)
            return F.cross_entropy(model.heads.count(enc["cls"]), nS_sc - 1) * sS

        L_S_sc = sc_loss()
        out["L_S_sc"] = float(L_S_sc) / sS if sS else None
        gsS = AL.grads_of(L_S_sc, scp)
        out["g_S_sc"] = gsS
        out["recompute_sc"] = lambda: AL.grads_of(sc_loss(), scp)
        diag["cos_sc"] = float(AL.Metric.cos(gsS, gsV, den_sc))
        ss_sc = float(AL.Metric.inner(gsS, gsS, den_sc))
        diag["tau_sc"] = (float(AL.Metric.inner(gsS, gsV, den_sc)) / ss_sc
                          if ss_sc > 0 else None)
    if outV is not None:
        L_V = outV["lrow"].mean()
        out["L_V"] = float(L_V)
        g_V = AL.grads_of(L_V, cvp, retain=extra)
    else:
        g_V = [torch.zeros_like(p) for p in cvp]
    out["g_V"] = g_V
    if outS is not None:
        lrowS = outS["lrow"] * (s_scale if s_scale is not None else 1.0)
        L_S = lrowS.mean()
        out["L_S"] = float(outS["lrow"].mean())
        g_S = AL.grads_of(L_S, cvp, retain=extra)
        out["g_S"] = g_S
        diag["cos_cv"] = float(AL.Metric.cos(g_S, g_V, den))
        ss = float(AL.Metric.inner(g_S, g_S, den))
        diag["tau"] = float(AL.Metric.inner(g_S, g_V, den)) / ss if ss > 0 else None
        xs, th_s, tg_s, t5_s, cd_s, ns_s = (x2[subS["rows"]], subS["th"], subS["tg"],
                                            subS["truth5"], subS["cands"], subS["ns"])

        def recompute():
            o = AL.canvas_route(model, xs, th_s, tg_s, t5_s, cd_s, ns_s, c_read, cfg.mu)
            return AL.grads_of(o["lrow"].mean() * (s_scale if s_scale is not None
                                                   else 1.0), cvp)
        out["recompute"] = recompute
        if extra:
            nS = lrowS.shape[0]
            if nS >= 2:                              # M3 半批无偏 𝒮̂_S
                h1 = torch.arange(nS, device=dev) % 2 == 0
                g1 = AL.grads_of(lrowS[h1].mean(), cvp, retain=True)
                g2 = AL.grads_of(lrowS[~h1].mean(), cvp, retain=True)
                diag["S_S"] = float(AL.Metric.inner(g1, g2, den))
                diag["S_S_full"] = ss
            tgn, t5n, nsn = AL.null_targets(subS["ns"], subS["th"], subS["truth5"],
                                            cfg.k, rng_cpu, dev)
            parts = [F.cross_entropy(outS["tl"][t], tgn[t], reduction="none")
                     for t in AL.TASKS_LIN]
            parts.append(F.cross_entropy(outS["t5"], t5n, reduction="none"))
            logp = F.log_softmax(outS["read"], dim=-1)
            a = logp[:, ABSTAIN].exp()
            lpn = logp.gather(1, (nsn - 1).unsqueeze(1)).squeeze(1)
            rr = (1.0 - a) * (torch.log1p(-a.clamp(max=1 - 1e-6)) - lpn) + a * c_read
            L_null = (sum(parts) / math.sqrt(6.0) + cfg.mu * rr).mean()
            g_null = AL.grads_of(L_null, cvp, retain=False)
            sn = float(AL.Metric.inner(g_null, g_null, den))
            diag["tau_null"] = (float(AL.Metric.inner(g_null, g_V, den)) / sn
                                if sn > 0 else None)
            diag["cos_null"] = float(AL.Metric.cos(g_null, g_V, den))
    if extra and outV is not None:
        nV = outV["lrow"].shape[0]
        if nV >= 2:                                  # 𝒮̂_V 半批 (分箱差值用)
            h1 = torch.arange(nV, device=dev) % 2 == 0
            g1 = AL.grads_of(outV["lrow"][h1].mean(), cvp, retain=True)
            g2 = AL.grads_of(outV["lrow"][~h1].mean(), cvp, retain=False)
            diag["S_V"] = float(AL.Metric.inner(g1, g2, den))
        # 联合余弦 (§5.1 参考数): Θ_cv ∪ Θ_sc, 共享 E 求和 (两路由 S/V 梯度都在时)
        if out["g_S"] is not None and out["g_S_sc"] is not None:
            gsV, gsS = out["g_V_sc"], out["g_S_sc"]
            idx_sc = {id(p): i for i, p in enumerate(scp)}
            jV, jS, jd = [], [], []
            for p, gv, gs_, d in zip(cvp, g_V, out["g_S"], den):
                i = idx_sc.get(id(p))
                jV.append(gv + (gsV[i] if i is not None else 0.0))
                jS.append(gs_ + (gsS[i] if i is not None else 0.0))
                jd.append(d)
            for i, p in enumerate(scp):
                if not any(p is q for q in cvp):
                    jV.append(gsV[i]); jS.append(gsS[i]); jd.append(den_sc[i])
            diag["cos_joint"] = float(AL.Metric.cos(jS, jV, jd))
    out["diag"] = diag
    return out


# ================================================================ 评测仪器
@torch.no_grad()
def eval_count(model, scenes, ns, dev, bs=128):
    """计数头逐带评测: exact + 容差曲线 ±1/±3/±5 (TOLERANCE-CURVE 法则)."""
    preds = []
    for i in range(0, scenes.shape[0], bs):
        lg = model.heads.count(model.encode_scene(scenes[i:i + bs].to(dev))
                               ["cls"])
        preds.append(lg.argmax(-1).cpu() + 1)
    pred = torch.cat(preds)
    err = (pred - ns).abs()
    bands = [D.band_of(int(n)) for n in ns]
    out = {}
    for band in sorted(set(bands)):
        m = torch.tensor([b == band for b in bands])
        e = err[m]
        out[band] = {"n": int(m.sum()),
                     "exact": round(float((e == 0).float().mean()), 4),
                     **{f"tol{t}": round(float((e <= t).float().mean()), 4)
                        for t in (1, 3, 5)}}
    return out


def fit_probe(model, scenes, ns, dev, steps=400, bs=256):
    """冻结 g 上的线性探针读 N vs 计数头 (spec §2.3 监控项, Phase 1 冻结判据).
    探针参数是局部的, 不碰模型."""
    feats = []
    with torch.no_grad():
        for i in range(0, scenes.shape[0], bs):
            feats.append(model.encode_scene(scenes[i:i + bs].to(dev))["cls"])
    g = torch.cat(feats)
    n_tr = int(0.75 * g.shape[0])
    y = (ns - 1).to(dev)
    probe = torch.nn.Linear(g.shape[1], GM.K_MAX).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(g[:n_tr]), y[:n_tr]).backward()
        opt.step()
    with torch.no_grad():
        pa = float((probe(g[n_tr:]).argmax(-1) == y[n_tr:]).float().mean())
        ha = float((model.heads.count(g[n_tr:]).argmax(-1)
                    == y[n_tr:]).float().mean())
    return dict(probe=round(pa, 4), head=round(ha, 4), gap=round(ha - pa, 4))


@torch.no_grad()
def blank_control(model, k, theta_ns, dev, eval_groups=None):
    """空白画布对照 (spec §10.1): 六任务在全空画布上必须掉到盲最优水平.
    theta_ns: 固定 (n, θ) 列表. 超过 null + 4σ 记入 alarm."""
    nulls = blank_nulls(k)
    h1 = model.encode_canvas(torch.zeros(1, GM.SIDE, GM.SIDE,
                                         device=dev))["cls"]
    n_s = len(theta_ns)
    h = h1.expand(n_s, -1)
    th = dict(
        tau=torch.tensor([t["tau"] - 1 for _, t in theta_ns], device=dev),
        p=torch.tensor([D.P_CHOICES.index(t["p"]) for _, t in theta_ns],
                       device=dev),
        m=torch.tensor([D.M_CHOICES.index(t["m"]) for _, t in theta_ns],
                       device=dev))
    tl = model.heads.task_logits(h, th)
    accs, alarm = {}, []
    for t in ("t1", "t2", "t3", "t4", "t6"):
        tgt = torch.tensor([D.targets(n, th_)[t] for n, th_ in theta_ns],
                           device=dev)
        accs[t] = round(float((tl[t].argmax(-1) == tgt).float().mean()), 4)
    if eval_groups is not None:
        hits, tot = 0, 0
        for gr in eval_groups:
            hc = model.encode_scene(gr["cands"].to(dev))["cls"]
            sc = model.heads.t5_scores(h1, hc.unsqueeze(0))
            hits += int(sc.argmax(-1)[0] == gr["truth5"])
            tot += 1
        accs["t5"] = round(hits / tot, 4)
    for t, a in accs.items():
        nu = nulls[t]
        n_eff = n_s if t != "t5" else (len(eval_groups) if eval_groups else 1)
        if a > nu + 4.0 * math.sqrt(nu * (1 - nu) / n_eff):
            alarm.append(t)
    rd_ab = float((model.heads.read(h1).argmax(-1) == ABSTAIN).float().mean())
    return dict(acc=accs, null={t: round(v, 4) for t, v in nulls.items()},
                alarm=alarm, read_abstain_blank=round(rd_ab, 4))


@torch.no_grad()
def eval_groups_tasks(model, groups, cfg, dev, rng_cpu):
    """固定评测组上的贪心 rollout -> 六任务准确率 + 直读 (Phase 2 主评测)."""
    accs, reads, absts, stamps = [], [], [], []
    for i in range(0, len(groups), cfg.n_groups):
        chunk = groups[i:i + cfg.n_groups]
        b = build_group_batch(chunk, 1, dev)
        enc = model.encode_scene(b["x1"])
        progs, _, _ = model.writer.rollout(
            enc["tokens"], l_max_of(cfg),
            torch.zeros(len(chunk), device=dev))
        x2 = GM.channel(GM.raster(progs, cfg.s, rng_cpu), cfg.s,
                        rng_cpu, cfg.occ_k).to(dev)
        h = model.encode_canvas(x2)["cls"]
        tl = model.heads.task_logits(h, b["th"])
        hc = model.encode_scene(
            b["cands"].view(-1, *b["cands"].shape[2:]))["cls"]
        t5 = model.heads.t5_scores(h, hc.view(len(chunk), D.N_CAND, -1))
        accs.append(GR.task_accs(tl, t5, b["tg"], b["truth5"]))
        rp = model.heads.read(h)
        reads.append((rp[:, :GM.K_MAX].argmax(-1) + 1 == b["ns"]).float())
        absts.append((rp.argmax(-1) == ABSTAIN).float())
        stamps += [len(p) for p in progs]
    a = torch.cat(accs).mean(0)
    return dict(acc={t: round(float(a[i]), 4)
                     for i, t in enumerate(GR.TASK_KEYS)},
                read=round(float(torch.cat(reads).mean()), 4),
                abstain=round(float(torch.cat(absts).mean()), 4),
                stamps_mean=round(sum(stamps) / len(stamps), 2))


@torch.no_grad()
def code_table(model, table_scenes, k, dev, lmax=None):
    """码表 (spec §11.1): 每个 N 的贪心程序. table_scenes: dict n -> (M,side,side)."""
    lm = lmax if lmax else GM.l_max(k)
    out = {}
    for n, sc in sorted(table_scenes.items()):
        st = model.encode_scene(sc.to(dev))["tokens"]
        progs, _, _ = model.writer.rollout(
            st, lm, torch.zeros(sc.shape[0], device=dev))
        out[n] = progs
    return out


def _midrank(v):
    """(n,) -> 平局取中位秩的秩向量 (Spearman 用)."""
    order = torch.argsort(v)
    sv = v[order]
    ranks = torch.arange(1, v.shape[0] + 1, dtype=torch.float32)
    out = torch.empty_like(ranks)
    i = 0
    while i < v.shape[0]:
        j = i
        while j + 1 < v.shape[0] and float(sv[j + 1]) == float(sv[i]):
            j += 1
        out[order[i:j + 1]] = ranks[i:j + 1].mean()
        i = j + 1
    return out


def table_stats(counts_by_n, n_perm=2000, seed=0):
    """章数-N 关系的预注册统计: 中位数, 严格递增配对占比 (同索引跨相邻 N),
    置换零假设的 p 值 (置换 N 标签 = **整行互换** -- 同 N 各场景章数强相关,
    行才是可交换单元; 修正前打散全表单元格, 在行内近常数的表上把 p 压到 0:
    p3d 终局表 slope .138/r2 .0023 报 mono_p .000, 整行置换实测 .335 -- p3e
    前修正), 线性拟合斜率/R², Spearman ρ(N, 章数) (F3 报数项, 平局中位秩)."""
    ns = sorted(counts_by_n)
    med = {n: float(torch.tensor(counts_by_n[n], dtype=torch.float32)
                    .median()) for n in ns}
    mat = torch.tensor([counts_by_n[n] for n in ns], dtype=torch.float32)

    def mono_frac(m):
        return float((m[1:] > m[:-1]).float().mean()) if m.shape[0] > 1 else 0.0

    obs = mono_frac(mat)
    rng = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(n_perm):
        if mono_frac(mat[torch.randperm(mat.shape[0],
                                        generator=rng)]) >= obs - 1e-12:
            hits += 1
    xs = torch.tensor([float(n) for n in ns
                       for _ in counts_by_n[n]])
    ys = mat.flatten()
    xm, ym = xs.mean(), ys.mean()
    vx = ((xs - xm) ** 2).sum()
    slope = float(((xs - xm) * (ys - ym)).sum() / vx) if float(vx) > 0 else 0.0
    ss_res = ((ys - ym) - slope * (xs - xm)) ** 2
    ss_tot = ((ys - ym) ** 2).sum()
    r2 = float(1.0 - ss_res.sum() / ss_tot) if float(ss_tot) > 0 else 0.0
    rx, ry = _midrank(xs), _midrank(ys)
    sx = float(rx.std(correction=0))
    sy = float(ry.std(correction=0))
    rho = (round(float(((rx - rx.mean()) * (ry - ry.mean())).mean()) / (sx * sy),
                 4)
           if sx > 0 and sy > 0 else None)   # 常数表 ρ 无定义, 报 None 不报 0
    return dict(median=med, mono_frac=round(obs, 4),
                mono_p=round(hits / n_perm, 4), slope=round(slope, 4),
                r2=round(r2, 4), rho=rho)


def pixel_probe(progs_by_n, k, s, rng, dev, steps=300, occ_k=0):
    """容量受限探针 (spec §9.2): 原始像素 (4x 平均池化) 线性读 N, 每次重训.
    信息在不在画布里, 独立于主干状态."""
    xs, ys = [], []
    ns = sorted(progs_by_n)
    for i, n in enumerate(ns):
        x = GM.channel(GM.raster(progs_by_n[n], s, rng), s, rng, occ_k)
        xs.append(F.avg_pool2d(x.unsqueeze(1), 4).flatten(1))
        ys += [i] * len(progs_by_n[n])
    X = torch.cat(xs)
    y = torch.tensor(ys)
    perm = torch.randperm(X.shape[0], generator=rng)   # CPU 上洗牌再上卡
    X, y = X[perm].to(dev), y[perm].to(dev)
    n_tr = int(0.75 * X.shape[0])
    lin = torch.nn.Linear(X.shape[1], len(ns)).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(X[:n_tr]), y[:n_tr]).backward()
        opt.step()
    with torch.no_grad():
        acc = float((lin(X[n_tr:]).argmax(-1) == y[n_tr:]).float().mean())
    return dict(acc=round(acc, 4), chance=round(1.0 / len(ns), 4))


@torch.no_grad()
def read_per_n(model, tab, cfg, rng, dev):
    """逐 N 直读准确率 (s114 v2, P1-P4 判读仪表): 贪心码表独立渲染一次经在位
    读头, 评测口径受限 argmax. 每 N = 码表张数 (32, σ≈0.09@p=0.5), 趋势跨评判读."""
    out = {}
    for n, ps in sorted(tab.items()):
        x = GM.render_channel(ps, cfg.s, rng, cfg.occ_k).to(dev)
        rp = model.heads.read(model.encode_canvas(x)["cls"])
        out[n] = round(float((rp[:, :GM.K_MAX].argmax(-1) + 1 == n)
                             .float().mean()), 4)
    return out


def reset_head_probe(model, tab, cfg, dev, seed, steps=200, r_train=6,
                     r_test=2, bs=512):
    """C3b 常设主干仪表 (s114 v2 [U]; M5 口径复核后 = 记录项, 不进判据):
    冻结主干上重训全新同构直读头 (独立种子) 至 plateau, 报主干表征的可提取
    上限. 探针参数局部, 不动模型. M5 实测三方持平 (探针 0.7866 / 在位读头
    0.7861 / 重置头 0.7851, 同信道 n=2048) -- 此仪表跟踪该持平是否被训练打破."""
    progs, ys = [], []
    for n, ps in sorted(tab.items()):
        progs += ps
        ys += [n] * len(ps)
    y = torch.tensor(ys)
    rng = torch.Generator().manual_seed(seed)
    feats, labs = [], []
    with torch.no_grad():
        for _ in range(r_train + r_test):
            x = GM.render_channel(progs, cfg.s, rng, cfg.occ_k)
            fs = [model.encode_canvas(x[j:j + 256].to(dev))["cls"]
                  for j in range(0, x.shape[0], 256)]
            feats.append(torch.cat(fs))
            labs.append(y)
    ntr = r_train * len(progs)
    X = torch.cat(feats)
    Y = torch.cat(labs).to(dev)
    Xtr, ytr, Xte, yte = X[:ntr], Y[:ntr] - 1, X[ntr:], Y[ntr:]
    d = X.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)          # 独立种子, 不扰训练流
        head = torch.nn.Sequential(
            torch.nn.Linear(d, 2 * d), torch.nn.GELU(),
            torch.nn.Linear(2 * d, GM.K_MAX + 1))
    head = head.to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 2)
    accs = []
    for t in range(1, steps + 1):
        idx = torch.randint(0, Xtr.shape[0], (bs,), generator=g).to(dev)
        opt.zero_grad()
        F.cross_entropy(head(Xtr[idx]), ytr[idx]).backward()
        opt.step()
        if t % 25 == 0:
            with torch.no_grad():
                a = float((head(Xte)[:, :GM.K_MAX].argmax(-1) + 1
                           == yte).float().mean())
            accs.append(round(a, 4))
    tail = accs[-3:]
    return dict(plateau=round(sum(tail) / len(tail), 4), best=max(accs),
                steps=steps, curve=accs)


# ================================================================ v4.1 探针 (§7.3)
M1_POOL_N = 32          # 逐数量 CRN 池: 每数量 32 组 (场景+θ+候选), 与码表张数同值
SEED_M1_POOL = 1414     # 池场景种子偏移 (与 555/556/557/558/777/888+k/999 分离)
SEED_M1_CHAN = 1415     # 池评测信道种子偏移 (每评重置 = CRN 跨评可比)


def ensure_m1_pool(es, cfg, cache=None):
    """评测集补逐数量池 (v4.1 M1/M6/CAL 用): 当前档训练域每数量 M1_POOL_N 个指定
    数量的组 (D.sample_group(n=n)); 幂等, 旧缓存迁移后回写."""
    if cfg.phase != 2:
        return es
    have = es.get("m1_pool") or {}
    need = [n for n in D.train_ns(cfg.k) if n not in have]
    if not need:
        return es
    rng = torch.Generator().manual_seed(cfg.seed + SEED_M1_POOL + 100000 * cfg.k)
    t0 = time.time()
    for n in D.train_ns(cfg.k):
        if n in have:
            continue
        have[n] = [D.sample_group(rng, cfg.k, n=n) for _ in range(M1_POOL_N)]
    es["m1_pool"] = have
    print(f"[m1_pool] built {len(need)} numbers in {time.time() - t0:.0f}s", flush=True)
    if cache:
        torch.save(es, cache)
    return es


def pool_batch(groups_n, dev):
    """逐数量池 -> 张量 (每组 1 条 rollout 的 build_group_batch)."""
    return build_group_batch(groups_n, 1, dev)


def per_number_grads(model, cfg, align, es, dev, progs_by_n=None, rng_seed=None,
                     with_sc=True):
    """M1 主体: 每数量 n 的 ḡ_n = ∇_{Θ_cv}[L_task+μL_read]|_{D_n} (逐数量一次反传,
    CRN 池 + CRN 信道), ℓ_n 损失值, 六任务命中; progs_by_n 给定 (定标码) 则不经写者.
    with_sc: 同时算场景路由 g^sc_n = ∇_{Θ_sc} L_count 与表征向量 (CLS g 均值,
    场景 token 池化均值). 返回 dict(ns, cv (F,D) 白化, sc, ell, ell_count, acc, rep_cls, rep_tok)."""
    ns = list(D.train_ns(cfg.k))
    rng = torch.Generator().manual_seed(cfg.seed + SEED_M1_CHAN if rng_seed is None
                                       else rng_seed)
    den = align.metric.denom(align.cvp)
    den_sc = align.metric.denom(align.scp) if with_sc else None
    c_read = anneal_c(cfg, cfg.c_anneal_steps)          # 退火终值 (跨评同尺)
    cv, sc, ell, ellc, accs, rc, rt = [], [], [], [], [], [], []
    for n in ns:
        gsn = es["m1_pool"][n]
        b = pool_batch(gsn, dev)
        if progs_by_n is None:
            with torch.no_grad():
                st = model.encode_scene(b["x1"])["tokens"]
                progs, _, _ = model.writer.rollout(
                    st, l_max_of(cfg), torch.zeros(len(gsn), device=dev))
        else:
            progs = progs_by_n[n]
            if len(progs) == 1:
                progs = [list(progs[0]) for _ in range(len(gsn))]
        x2 = GM.render_channel(progs, cfg.s, rng, cfg.occ_k).to(dev)
        o = AL.canvas_route(model, x2, b["th"], b["tg"], b["truth5"], b["cands"],
                            b["ns_rep"], c_read, cfg.mu)
        Ln = o["lrow"].mean()
        g = AL.grads_of(Ln, align.cvp)
        cv.append(AL.Metric.whiten_flat(g, den))
        ell.append(float(Ln))
        accs.append([round(float(a), 4) for a in o["accs"].mean(0)])
        if with_sc:
            enc = model.encode_scene(b["x1"])
            lc = F.cross_entropy(model.heads.count(enc["cls"]), b["ns_rep"] - 1)
            gs_ = AL.grads_of(lc, align.scp)
            sc.append(AL.Metric.whiten_flat(gs_, den_sc))
            ellc.append(float(lc))
            with torch.no_grad():
                rc.append(enc["cls"].mean(0))
                rt.append(enc["tokens"].mean((0, 1)))
    out = dict(ns=ns, cv=torch.stack(cv), ell=ell, acc=accs)
    if with_sc:
        out.update(sc=torch.stack(sc), ell_count=ellc, rep_cls=torch.stack(rc),
                   rep_tok=torch.stack(rt))
    return out


def R_report(vecs, ns, ell=None, n_bb=None):
    """M1 报数: R + 完整本征谱 + A + R(f) 截断曲线 (+ 主干/头分块 R)."""
    gr = AL.gram_R(vecs)
    out = dict(R=round(gr["R"], 4), eig=[round(e, 6) for e in gr["eig"]])
    if ell is not None:
        out["A"] = round(AL.alignment_A(gr["eig"], gr["evecs"], ell), 4)
    out["R_f"] = [(f, round(r, 4)) for f, r in AL.R_truncation(vecs, ns)]
    if n_bb is not None and 0 < n_bb < vecs.shape[1]:
        out["R_bb"] = round(AL.gram_R(vecs[:, :n_bb])["R"], 4)
        out["R_head"] = round(AL.gram_R(vecs[:, n_bb:])["R"], 4)
    return out


def rep_R(vecs):
    """表征侧参与比 (双侧 R 的表征版, [C]): 逐数量均值向量去均值 Gram 的参与比."""
    return round(AL.gram_R(vecs.float())["R"], 4)


def mask_control_R(model, cfg, align, es, dev, F_bins):
    """M1 对照列 (i): 与数量无关的任务族 (L_mask) 上按同法算 R -- 评测掩码画布分 F 箱,
    逐箱梯度 (Θ_E ∪ 格位头), 同度规白化. 那边也掉 = 表示整体塌缩."""
    params = AL.bb_params(model) + [model.heads.cell.weight, model.heads.cell.bias]
    den = align.metric.denom(params)
    n = es["mask_x"].shape[0]
    per = max(2, n // max(F_bins, 1))
    vecs = []
    for i in range(0, per * F_bins, per):
        x = es["mask_x"][i:i + per].to(dev)
        rep = es["mask_rep"][i:i + per].to(dev)
        enc = model.encode_canvas(x, mask_cells=rep)
        logits = model.heads.cell(enc["tokens"])
        loss = L.mask_loss(logits, es["mask_truth"][i:i + per].to(dev),
                           es["mask_sel"][i:i + per].to(dev))
        g = AL.grads_of(loss, params)
        vecs.append(AL.Metric.whiten_flat(g, den))
    if len(vecs) < 2:
        return None
    return round(AL.gram_R(torch.stack(vecs))["R"], 4)


def curvature_S(model, cfg, align, es, dev, S, seed):
    """M7: tr(H_S) 单次 Hutchinson (double backward), S = 留出集; 用固定评测组里
    数量 ∈ S 的组 (贪心写 → 评测信道). 同法给 V 一个参考值."""
    out = {}
    rng = torch.Generator().manual_seed(seed)
    c_read = anneal_c(cfg, cfg.c_anneal_steps)
    for tag, pred in (("S", lambda n: n in S), ("V", lambda n: n not in S)):
        gs = [gr for gr in es["groups"] if pred(gr["n"])][:16]
        if len(gs) < 2:
            out[tag] = None
            continue
        b = build_group_batch(gs, 1, dev)
        with torch.no_grad():
            st = model.encode_scene(b["x1"])["tokens"]
            progs, _, _ = model.writer.rollout(st, l_max_of(cfg),
                                               torch.zeros(len(gs), device=dev))
        x2 = GM.render_channel(progs, cfg.s, rng, cfg.occ_k).to(dev)

        def loss_fn():
            o = AL.canvas_route(model, x2, b["th"], b["tg"], b["truth5"], b["cands"],
                                b["ns_rep"], c_read, cfg.mu)
            return o["lrow"].mean()
        gen = torch.Generator().manual_seed(seed + 1)
        out[tag] = round(AL.hutchinson_trace(loss_fn, align.cvp, gen), 4)
    return out


def align_eval(model, cfg, align, es, dev, s1, S, v_med):
    """每评 v4.1 仪表 (§7.3 M1-M8): M1 R 谱/A/分块/截断曲线 + 对照列 (掩码 R,
    场景侧 R, 表征双侧 R), M7 曲率, M3/M4/M5/M8 窗汇总, L* 水平门, X4 缓冲重建.
    返回入册 dict."""
    ev = dict(F=len(D.train_ns(cfg.k)), S=sorted(int(n) for n in S),
              metric=cfg.metric)
    fl = align.flush()
    if not fl.get("n_steps"):          # 无逐步窗 (只评 AlignCtx, S 恒空): c 直方/逐数量 c 不入册
        fl = {k: v for k, v in fl.items()
              if k not in ("c_by_n", "c_hist", "capped", "capped_sc")}
    ev.update(fl)
    pn = per_number_grads(model, cfg, align, es, dev)
    ev["M1"] = R_report(pn["cv"], pn["ns"], pn["ell"], align.n_bb)
    ev["M1"]["ell"] = {str(n): round(l, 4) for n, l in zip(pn["ns"], pn["ell"])}
    ev["M1"]["acc"] = {str(n): a for n, a in zip(pn["ns"], pn["acc"])}
    ev["M1"]["R_sc"] = round(AL.gram_R(pn["sc"])["R"], 4)
    ev["M1"]["R_rep_cls"] = rep_R(pn["rep_cls"])
    ev["M1"]["R_rep_tok"] = rep_R(pn["rep_tok"])
    ev["M1"]["R_ctrl_mask"] = mask_control_R(model, cfg, align, es, dev, len(pn["ns"]))
    align.last_ell = {n: l for n, l in zip(pn["ns"], pn["ell"])}
    align.last_ell_count = {n: l for n, l in zip(pn["ns"], pn["ell_count"])}
    align.last_acc = {n: float(sum(a) / len(a)) for n, a in zip(pn["ns"], pn["acc"])}
    align.R_hist.append((s1, len(pn["ns"]), ev["M1"]["R"]))
    ev["M7"] = curvature_S(model, cfg, align, es, dev, S, cfg.seed + 1416)
    if cfg.l_star > 0.0 and ev.get("L_S") is not None:
        align.beta_gate = bool(ev["L_S"] <= cfg.l_star)   # §5.5: L_S > L* ⇒ β 当评置零
        ev["beta_gate"] = align.beta_gate
    # 改动 4 ([U] 2026-08-17): Ḡ 每评从本窗 rollout 重算 (窗均值整体换新, 无 EMA / 无 X4 重建)
    ev["cchan_rolled"] = align.cchan.roll()
    ev["cchan_filled"] = int((align.cchan.n_bar[list(D.train_ns(cfg.k))] > 0).sum())
    return ev


def align_final(align):
    """终报汇总 (§7.3 M2/M6, §8.2 A1/A2 的记账项): 首见登记表 + L_preq(F) 累积
    (跳距 =1 / >1 分列) + R(F) 史 (每档取最后一评) + dR/dF + 二阶差分."""
    fs = list(align.first_seen)
    Lp, Lpc = 0.0, 0.0
    rows1, rows2 = [], []
    for r in fs:
        (rows1 if r["jump"] == 1 else rows2).append(r)
    preq = []
    for r in rows1:
        if r.get("l_ex") is not None:
            Lp += r["l_ex"]
        if r.get("l_ex_count") is not None:
            Lpc += r["l_ex_count"]
        preq.append(dict(n=r["n"], L_preq=round(Lp, 4), L_preq_count=round(Lpc, 4)))
    lastR = {}
    for s, Fv, R in align.R_hist:
        lastR[Fv] = (s, R)
    Fs = sorted(lastR)
    RF = [(Fv, lastR[Fv][1]) for Fv in Fs]
    dR = [(Fs[i], round((RF[i][1] - RF[i - 1][1]) / (Fs[i] - Fs[i - 1]), 4))
          for i in range(1, len(Fs))]
    d2 = [(dR[i][0], round(dR[i][1] - dR[i - 1][1], 4)) for i in range(1, len(dR))]
    return dict(first_seen=fs, first_seen_jump_gt1=rows2, L_preq=preq,
                R_F=[(f, round(r, 4)) for f, r in RF], dR_dF=dR, d2R=d2,
                R_hist=[(s, f, round(r, 4)) for s, f, r in align.R_hist],
                cchan_rolls=getattr(align.cchan, "rolls", None))


FLOOR_ACC = EXPAND_SCORE   # 改动 3 配套: 「已达标数量」= 上评 M1 池六任务均值 ≥ 此线 ([C] 取扩张线 .95)


def _median(vals):
    v = sorted(vals)
    return v[len(v) // 2] if v else None


def first_seen_rows(model, cfg, align, es, dev, new_ns, old_max):
    """M6 首见登记 (§7.3): 前沿 +1 引入 n 时, 在任何针对它的训练之前测
    ℓ_note(n) = [L_task+μL_read] 于 n 的新 CRN 池 (贪心写→评测信道), ℓ_ex = ℓ_note − 地板;
    地板 ([U] 2026-08-17 改动 3 配套) = 同评**已达标数量** (上评 M1 池六任务均值 ≥ FLOOR_ACC)
    的 ℓ_n 中位 (align.last_ell/last_acc); 无已达标数量时退化为全部已引入数量的中位并标
    floor_kind="all" (原地板口径并报 floor_all); 含 L_count 版本并报 (不干净, §4.3);
    跳距 = n − 前沿旧最大值. 行永不改写."""
    rows = []
    if not new_ns:
        return rows
    ok_ns = [n for n, l in align.last_ell.items() if align.last_acc.get(n, 0.0) >= FLOOR_ACC]
    floor_all = _median(align.last_ell.values())
    floor_ok = _median([align.last_ell[n] for n in ok_ns])
    floor = floor_ok if floor_ok is not None else floor_all
    floor_kind = "qualified" if floor_ok is not None else "all"
    floor_c = _median([align.last_ell_count[n] for n in ok_ns
                       if n in align.last_ell_count]) if ok_ns else None
    if floor_c is None:
        floor_c = _median(align.last_ell_count.values())
    rng = torch.Generator().manual_seed(cfg.seed + SEED_M1_CHAN)
    c_read = anneal_c(cfg, cfg.c_anneal_steps)
    for n in new_ns:
        gsn = es["m1_pool"][n]
        b = pool_batch(gsn, dev)
        with torch.no_grad():
            enc = model.encode_scene(b["x1"])
            progs, _, _ = model.writer.rollout(
                enc["tokens"], l_max_of(cfg), torch.zeros(len(gsn), device=dev))
            x2 = GM.render_channel(progs, cfg.s, rng, cfg.occ_k).to(dev)
            o = AL.canvas_route(model, x2, b["th"], b["tg"], b["truth5"], b["cands"],
                                b["ns_rep"], c_read, cfg.mu)
            ln = float(o["lrow"].mean())
            lc = float(F.cross_entropy(model.heads.count(enc["cls"]), b["ns_rep"] - 1))
        rows.append(dict(n=int(n), jump=int(n - old_max), l_note=round(ln, 4),
                         floor=(round(floor, 4) if floor is not None else None),
                         l_ex=(round(ln - floor, 4) if floor is not None else None),
                         floor_kind=floor_kind, n_qualified=len(ok_ns),
                         floor_all=(round(floor_all, 4) if floor_all is not None else None),
                         l_count=round(lc, 4),
                         floor_count=(round(floor_c, 4) if floor_c is not None else None),
                         l_ex_count=(round(lc - floor_c, 4) if floor_c is not None
                                     else None),
                         stamps_med=float(torch.tensor([len(p) for p in progs],
                                                       dtype=torch.float32).median())))
    return rows


def r6_viz(model, table_scenes, k, dev, path, lmax=None):
    """R6 配对可视化 (法则, 2026-07-09): 行 = N, 列 = [场景 | 写完的画布],
    黑底白墨 cmap='gray'. 每行取该 N 的第一张场景."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lm = lmax if lmax else GM.l_max(k)
    ns = sorted(table_scenes)[:16]
    fig, axes = plt.subplots(len(ns), 2, figsize=(5, 2.4 * len(ns)))
    if len(ns) == 1:
        axes = axes.reshape(1, 2)
    with torch.no_grad():
        for r, n in enumerate(ns):
            sc = table_scenes[n][:1]
            st = model.encode_scene(sc.to(dev))["tokens"]
            progs, _, _ = model.writer.rollout(
                st, lm, torch.zeros(1, device=dev))
            canv = GM.raster(progs)
            axes[r, 0].imshow(sc[0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, 0].set_ylabel(f"N={n}", color="w")
            axes[r, 1].imshow(canv[0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, 1].set_title(f"{len(progs[0])} stamps", color="w",
                                 fontsize=9)
            for c in (0, 1):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


# ================================================================ 评测集 (CRN)
def _eval_cache_path(cfg):
    """缓存键必须含 k (评审 Critical#1: 扩档否则静默复用旧评测集); occ_k>0 的新腐蚀
    制度再补 s/occ_k (mask_x 是按渲染时的信道烘进缓存的, 换制度必须换键)."""
    tag = f"_sw{cfg.s}_ok{cfg.occ_k}" if cfg.occ_k else ""
    return os.path.join(os.path.dirname(cfg.out.rstrip("/")) or ".",
                        f"nc_evalsets_s{cfg.seed}_p{cfg.phase}_k{cfg.k}{tag}.pt")


def build_eval_sets(cfg):
    """启动时一次性渲染的固定评测集 (跨步/跨 Phase 可比). 磁盘缓存按种子命名."""
    cache = _eval_cache_path(cfg)
    if os.path.exists(cache):
        es = torch.load(cache, weights_only=True)
        print(f"[eval] cache hit {cache}", flush=True)
    else:
        t0 = time.time()
        rng = torch.Generator().manual_seed(cfg.seed + 999)
        es = {}
        ns, bands = [], {"train": 512, "hole": 256, "extrap": 256}
        pool = {"train": D.train_ns(GM.K_MAX), "hole": D.hole_ns(),
                "extrap": D.extrap_ns()}
        for band, cnt in bands.items():
            ns += [pool[band][int(torch.randint(0, len(pool[band]), (1,),
                                                generator=rng))]
                   for _ in range(cnt)]
        es["count_ns"] = torch.tensor(ns, dtype=torch.long)
        es["count_scenes"] = torch.stack(
            [D.render_scene(rng, int(n)) for n in ns])
        pn, psc = [], []
        for _ in range(3072):
            n = D.sample_n(rng, GM.K_MAX)
            pn.append(n)
            psc.append(D.render_scene(rng, n))
        es["probe_ns"] = torch.tensor(pn, dtype=torch.long)
        es["probe_scenes"] = torch.stack(psc)
        mp = [GM.sample_uniform_prog(rng) for _ in range(256)]
        es["mask_progs"] = mp
        es["mask_x"] = GM.channel(GM.raster(mp, cfg.s, rng), cfg.s, rng)
        es["mask_truth"] = GM.cell_truth(mp).view(256, GM.T)
        sel = L.sample_mask_cells(256, rng)
        es["mask_rep"], es["mask_keep"] = split_mask_keep(sel, rng,
                                                         cfg.keep_frac)
        es["mask_sel"] = sel
        es["theta_ns"] = [(D.sample_n(rng, cfg.k), D.sample_theta(rng, cfg.k))
                          for _ in range(512)]
        if cfg.phase == 2:
            es["groups"] = [D.sample_group(rng, cfg.k) for _ in range(128)]
            es["table_scenes"] = {
                n: torch.stack([D.render_scene(rng, n) for _ in range(32)])
                for n in D.train_ns(cfg.k)}
        torch.save(es, cache)
        print(f"[eval] built in {time.time() - t0:.0f}s -> {cache}",
              flush=True)
    if cfg.phase == 2 and "groups" not in es:
        raise RuntimeError("eval cache 缺 Phase 2 字段, 删缓存重建: " + cache)
    ensure_ood_progs(es, cfg, cache)
    if cfg.holdout_frac > 0.0 or cfg.m1_eval:
        ensure_m1_pool(es, cfg, cache)
    return es


@torch.no_grad()
def eval_mask(model, es, dev, bs=64):
    """素养评测: 保留桶识别 (冻结门, 天花板 1) + 遮蔽位 vs 实测 oracle."""
    preds = []
    for i in range(0, es["mask_x"].shape[0], bs):
        enc = model.encode_canvas(es["mask_x"][i:i + bs].to(dev),
                                  mask_cells=es["mask_rep"][i:i + bs].to(dev))
        preds.append(model.heads.cell(enc["tokens"]).argmax(-1).cpu())
    pred = torch.cat(preds)
    t, rep, keep = es["mask_truth"], es["mask_rep"], es["mask_keep"]
    orc = mask_oracle(t, rep, GM.l_max(GM.K_MAX))
    return dict(
        kept_acc=round(float((pred[keep] == t[keep]).float().mean()), 4),
        mask_acc=round(float((pred[rep] == t[rep]).float().mean()), 4),
        oracle_acc=round(orc["acc"], 4))


# ================================================================ 检查点
def save_ckpt(path, model, opt, tw, step, cfg, best, anchor=None, probe=None,
              c9=None, align=None, lag=None, misc=None):
    d = dict(model=model.state_dict(), opt=opt.state_dict(),
             tw=tw.ema, step=step, cfg=dataclasses.asdict(cfg), best=best)
    if anchor is not None:
        d["anchor"] = anchor   # AnchorCtl.state() dict (恢复保刷新史/β/计数器)
    if probe is not None:
        d["probe"] = probe.state()   # 双探针 (缓冲+probe_out+opt, s114 §10e)
    if c9 is not None:
        d["c9"] = c9           # C9Guard.state_dict() (§10e-补2)
    if align is not None:
        d["align"] = align.state()   # v4.1: 留出调度/G_n 窗均值/首见登记表/R 史/扩张记录
    if lag is not None:
        d["lag"] = lag.state_dict()  # 影子读者 (恢复保真: 分阶段 --resume 时影子不重置为在线模型)
    if misc is not None:
        d["misc"] = misc       # tpl_hist / 各 streak / stop 计数 (恢复保真)
    torch.save(d, path)


def load_ckpt(path, model, opt=None, tw=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if tw is not None:
        tw.ema = ck["tw"]
    return dict(step=ck["step"], best=ck["best"], cfg=ck["cfg"],
                anchor_sd=ck.get("anchor"), probe_sd=ck.get("probe"),
                c9_sd=ck.get("c9"), align_sd=ck.get("align"),
                lag_sd=ck.get("lag"), misc=ck.get("misc"))


# ================================================================ 器官体检
def newborn_battery(model, cfg, dev, rng_cpu, rng_dev):
    """器官动力学门 (预发射, ORGAN-VITALS A): 新生儿 rollout 长度分布 /
    去重格数 / STOP 可达 / 头饱和审计. 表达力: 屏蔽只禁非法, 任何合法程序
    概率 > 0 (结构保证), 此处实测采样多样性."""
    sc, ns = D.sample_scene_batch(rng_cpu, cfg.k, 8)
    lm = l_max_of(cfg)
    with torch.no_grad():
        enc = model.encode_scene(sc.to(dev))
        st = enc["tokens"].repeat_interleave(8, 0)
        temps = torch.ones(64, device=dev)
        progs, _, _ = model.writer.rollout(st, lm, temps,
                                           rng=rng_dev)
        lens = torch.tensor([len(p) for p in progs], dtype=torch.float32)
        cells = set()
        for p in progs:
            cells |= {a // GM.S for a in p}
        cl = model.heads.count(enc["cls"])
        x2 = GM.raster(progs[:16])
        tok = model.encode_canvas(x2.to(dev))["tokens"]
        cel = model.heads.cell(tok)
    return dict(
        len_min=int(lens.min()), len_med=float(lens.median()),
        len_max=int(lens.max()), lmax=lm,
        p_stop0=float((lens == 0).float().mean()),
        p_cut=float((lens == lm).float().mean()),
        distinct_cells=len(cells),
        count_logit_absmax=round(float(cl.abs().max()), 2),
        cell_logit_absmax=round(float(cel.abs().max()), 2))


# ================================================================ 主循环
def _merge_metrics(ms):
    """微批仪表跨块平均 (评审 Important#3: 覆盖写只留最后一块). 各块等大."""
    if len(ms) == 1:
        return ms[0]
    out = {}
    for k in ms[0]:
        if isinstance(ms[0][k], list):
            out[k] = [round(sum(m[k][i] for m in ms) / len(ms), 4)
                      for i in range(len(ms[0][k]))]
        else:
            out[k] = round(sum(m[k] for m in ms) / len(ms), 4)
    return out


def _log(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def run(cfg, probe_only=False, init=None, resume=None, device=None):
    # 双装载器 x16 worker 的组样本含多个张量, fd 传递策略在软上限 1024 下必炸
    # (实测 'received 0 items of ancdata' 崩于 Phase 2 首步); file_system 策略
    # 走 /dev/shm (pod 上 28G), 消除逐张量 fd 占用
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError):
        pass
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model = NumCodeModel().to(dev)
    ck_init = None
    if init:
        ck_init = torch.load(init, map_location="cpu", weights_only=True)
        model.load_state_dict(ck_init["model"])
        print(f"[init] weights from {init} (step {ck_init['step']})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.wd)
    if ck_init is not None and cfg.init_opt and ck_init.get("opt"):
        # v4.1: F⁻¹ 度规要 v̂ -- 只载优化器状态 (一二阶矩+步计数), 其余仪表照 --init 冷启
        opt.load_state_dict(ck_init["opt"])
        for g_ in opt.param_groups:
            g_["lr"], g_["weight_decay"] = cfg.lr, cfg.wd
        print("[init] optimizer state loaded from init ckpt (init_opt=1)", flush=True)
    tw = GR.TaskWeights()
    step0 = 0
    latch = BestLatch()
    meta = None
    retreat_from = None
    if resume:
        meta = load_ckpt(resume, model, opt, tw)
        step0 = meta["step"]
        b = meta["best"]
        if isinstance(b, dict):
            latch = BestLatch(best=b.get("score", -1.0))
            latch.step = int(b.get("step", 0))
        else:                                  # 旧检查点标量 (评审 Important#6)
            latch = BestLatch(best=b)
        ck_k = (meta.get("cfg") or {}).get("k")
        if ck_k and ck_k != cfg.k:
            cfg.k = ck_k          # 恢复: 课程档随检查点走, 无条件 (评审 Important#2:
            #                        漏传 --k_expand 的 resume 会静默拉回 CLI 默认档)
            print(f"[resume] 课程档 k={cfg.k} (从检查点)", flush=True)
        if cfg.k_force > 0 and cfg.k_force != cfg.k:
            # [U] 2026-08-17 第四阶段 改动 1: 前沿退回 (权重不动, 训练域/评测集/GRPO 组回到 1..k_force)
            retreat_from, cfg.k = cfg.k, cfg.k_force
            latch = BestLatch()                    # 跨档成绩不可比, 闩重记
            print(f"[resume] 前沿退回 k {retreat_from} -> {cfg.k} (k_force)", flush=True)
        print(f"[resume] {resume} at step {step0}", flush=True)
    ctl = None
    if cfg.phase == 2 and cfg.anchor > 0 and cfg.beta > 0.0:
        ctl = AnchorCtl(model, cfg.beta)     # 锚起点 = 当前权重 (init/resume 之后)
        if meta is not None and meta.get("anchor_sd"):
            ctl.load_state(meta["anchor_sd"])
            print("[anchor] 从检查点恢复锚状态机", flush=True)
        print(f"[anchor] 固定锚开启 (宽限 {ANCHOR_GRACE} 步期末建锚; 合取快速闸 "
              f"D>{ANCHOR_D_RESET} 且 (score<{ANCHOR_RESET_SCORE} 或 "
              f"dmin<{ANCHOR_RESET_DMIN}); 同锚回滚上限 {ANCHOR_RESET_CAP}; "
              f"β0={cfg.beta} 校准窗 = 宽限后 {BETA_CALIB_STEPS} 步)", flush=True)
    xctl = (ExpandCtl(cfg.expand_every) if (cfg.phase == 2 and cfg.k_expand
                                             and not probe_only) else None)
    if xctl is not None:
        print(f"[expand] K 扩张开启: 成绩连续 {EXPAND_STREAK} 评 >= {EXPAND_SCORE} "
              f"且 v <= {EXPAND_V} 触发 +{cfg.expand_step or EXPAND_STEP} 进档 "
              f"(现档 k={cfg.k}); 定时推进 every={cfg.expand_every} "
              f"(0=关; >0 = 距上次推进满此步数亦推进, 改动 3)", flush=True)
    stopc = (StopCtl(cfg.stop_count, cfg.stop_dmin, cfg.stop_cnull)
             if (cfg.phase == 2 and not probe_only) else None)
    # --resume: 训练流 (计数流/组流) 与信道噪声流按恢复步加盐, 不重放旧段开头 (评测种子不动, CRN 照旧)
    salt = RESUME_SALT * step0 if (resume and step0 > 0) else 0
    if salt:
        print(f"[resume] 训练流/噪声流种子加盐 (step0={step0})", flush=True)
    rng_cpu = torch.Generator().manual_seed(cfg.seed + 1 + salt)
    rng_dev = (torch.Generator(device=dev).manual_seed(cfg.seed + 2 + salt)
               if dev.type == "cuda" else torch.Generator().manual_seed(
                   cfg.seed + 2 + salt))
    if probe_only:
        bat = newborn_battery(model, cfg, dev, rng_cpu, rng_dev)
        print("[battery]", json.dumps(bat), flush=True)
        assert bat["p_stop0"] < 0.9, "新生儿几乎必然立即 STOP -- 结构死门"
        assert bat["distinct_cells"] > GM.T // 4, "落格多样性塌缩"
    lag = None
    if cfg.phase == 2 and cfg.reader_lag > 0.0:
        lag = NumCodeModel().to(dev)
        if meta is not None and meta.get("lag_sd"):
            lag.load_state_dict(meta["lag_sd"])    # 恢复保真: 影子接续, 不重置为在线模型
            print("[lag] 从检查点恢复影子模型", flush=True)
        else:
            lag.load_state_dict(model.state_dict())
        for p in lag.parameters():
            p.requires_grad_(False)
        lag.eval()
        print(f"[lag] 影子模型开启 decay={cfg.reader_lag} (奖励读者 + β 参考)",
              flush=True)
    probe = None
    if cfg.phase == 2 and cfg.alpha > 0.0:
        probe = PB.ProbePair(cfg, dev)
        if meta is not None and meta.get("probe_sd"):
            probe.load_state(meta["probe_sd"])
            print("[probe] 从检查点恢复双探针缓冲/权重", flush=True)
        print(f"[probe] 双探针开启 α={cfg.alpha} (probe_in 岭闭式每步重训入环, "
              f"probe_out 单隐层体外; 缓冲 {PB.CAP} 拟合下限 {PB.MIN_FIT})",
              flush=True)
    guard = None
    if cfg.phase == 2 and cfg.c9 > 0:
        guard = C9Guard(list(D.train_ns(cfg.k)))
        if meta is not None and meta.get("c9_sd"):
            guard.load_state(meta["c9_sd"])
            print("[c9] 从检查点恢复守卫状态机", flush=True)
        print(f"[c9] 退化组守卫开启 (div_nat<{C9_DIV_LO}×{C9_TRIG}步触发 -> 末"
              f"{C9_ARMS}采样臂随机位置禁众数动作; div_nat>{C9_DIV_HI}×"
              f"{C9_EXIT}步 -> {C9_RAMP}步线性撤除)", flush=True)
    align = None
    if cfg.phase == 2 and cfg.holdout_frac > 0.0:
        assert cfg.accum == 1, "v4.1 对齐机构只支持 accum=1 (逐步梯度装配)"
        assert cfg.reader_lag > 0.0, \
            "对齐机构要求影子读者 (v4.1 §6.4: c 必须用滞后影子读者算, T1 的 c 基线亦然)"
        align = AlignCtx(model, opt, cfg, dev, list(D.train_ns(cfg.k)))
        if meta is not None and meta.get("align_sd"):
            align.load_state(meta["align_sd"])
            print("[align] 从检查点恢复对齐机构状态", flush=True)
        print(f"[align] v4.1 对齐机构开启: α={cfg.holdout_frac} T_rotate={cfg.t_rotate} "
              f"|S|={align.hold.size()} β_align(cv)={cfg.beta_align} β_sc={cfg.beta_sc} "
              f"κ={cfg.kappa} δ×{cfg.delta_mult} L*={cfg.l_star} 度规={cfg.metric} "
              f"扩张步长={cfg.expand_step or EXPAND_STEP} u_ema={cfg.u_ema}; "
              f"[U] 2026-08-17 改动 1/2/4: 场景路由每步算 u_S (β_sc>0 施压), 计数流全带 apply, "
              f"G_n 每评窗均值 (无 EMA/无 X4); c_mode={cfg.c_mode} (proj = 投影率, 只在 S 行, κ 只对 S 组); "
              f"停跑 count<{cfg.stop_count}×2 / dmin<={cfg.stop_dmin or '关'} / c<=null:{cfg.stop_cnull}",
              flush=True)
    # 只评用 AlignCtx (B 臂 α=0, [C] s123): S 恒空, 训练路径仍 align=None (旧制逐位不变);
    # 每评 M1 R 谱/对照列/M7-V + 扩张时 M6 首见行 + 终报 R 史 (与 T1 同仪同池同种子).
    align_ev = align
    if align is None and cfg.phase == 2 and cfg.m1_eval and not probe_only:
        align_ev = AlignCtx(model, opt, cfg, dev, list(D.train_ns(cfg.k)))
        assert align_ev.hold.size() == 0
        if meta is not None and meta.get("align_sd"):
            align_ev.load_state(meta["align_sd"])
        print(f"[align] m1_eval=1: 只评仪表开启 (M1 R 谱/对照列/M7-V/M6 首见, S 恒空, "
              f"度规={cfg.metric}); 训练路径 align=None 旧制", flush=True)
    cctl = CollapseCtl()
    sctl = SettleCtl()
    es = build_eval_sets(cfg)
    sl, gl = make_loaders(cfg, salt)
    sit = iter(sl)
    git = iter(gl) if gl is not None else None
    fh = open(os.path.join(cfg.out, "log.jsonl"), "a")
    n_steps = 10 if probe_only else cfg.steps
    t_data = t_step = 0.0
    tpl_hist = []                # 位移仪表: 最近 8 次评测的众数模板 (改5)
    kl_win, watch_acc, watch_rstd = [], [], []   # 评测间窗口累加器 (锚/优势监控)
    expands = []
    if meta is not None and meta.get("misc"):     # 恢复保真: 模板史/各 streak/扩张记录接续
        ms_ = meta["misc"]
        tpl_hist = [{int(k): v for k, v in t.items()} for t in ms_.get("tpl_hist", [])]
        expands = list(ms_.get("expands", []))
        cctl.streak = int(ms_.get("cctl", 0))
        sctl.streak = int(ms_.get("sctl", 0))
        if xctl is not None:
            xctl.streak = int(ms_.get("xctl", 0))
        if stopc is not None and ms_.get("stop"):
            stopc.load_state(ms_["stop"])
        print(f"[resume] misc 恢复: tpl_hist {len(tpl_hist)} / expands {len(expands)} / "
              f"streaks c{cctl.streak} s{sctl.streak}", flush=True)
    if align_ev is not None and align_ev.expands and not expands:
        expands = list(align_ev.expands)
    if retreat_from is not None:                  # 前沿退回落地 (改动 1): 留出槽/守卫/计数器/记录
        keep = list(D.train_ns(cfg.k))
        if align_ev is not None:
            align_ev.hold.retreat(keep)
            align_ev.hold_log = None
        if guard is not None:
            evs_hist = guard.events
            guard = C9Guard(keep)
            guard.events = evs_hist
        cctl.reset()
        sctl.reset()
        if xctl is not None:
            xctl.streak = 0
        rec_ = dict(step=step0, k=[retreat_from, cfg.k], why="retreat", extrap0=None, prev_best=None)
        expands.append(rec_)
        if align_ev is not None:
            align_ev.expands = list(expands)
            align_ev.last_expand = step0
        _log(fh, dict(step=step0, retreat=rec_))
        print(f"[retreat] 前沿 {retreat_from} -> {cfg.k}: 留出集 {sorted(align_ev.hold.ns) if align_ev else None}, "
              f"守卫重建, 闩/并轨/定居重置", flush=True)

    def _misc():
        return dict(tpl_hist=[{str(k): v for k, v in t.items()} for t in tpl_hist],
                    expands=expands, cctl=cctl.streak, sctl=sctl.streak,
                    xctl=(xctl.streak if xctl is not None else 0),
                    stop=(stopc.state() if stopc is not None else None))

    def _save(path, step_):
        save_ckpt(path, model, opt, tw, step_, cfg, dict(score=latch.best, step=latch.step),
                  anchor=ctl.state() if ctl is not None else None, probe=probe,
                  c9=guard.state_dict() if guard is not None else None,
                  align=align_ev, lag=lag, misc=_misc())
    if ctl is not None and ctl.tpl is None and not probe_only:
        # 发射建档: 锚的模板表与锚点成绩按当前权重实测 (精确, 非首评近似).
        # 不入 best 闩 (步 0 评测按登记规则不参与最优闩).
        tab0 = code_table(model, es["table_scenes"], cfg.k, dev,
                          lmax=l_max_of(cfg))
        ctl.tpl = modal_templates(tab0)
        t0 = eval_groups_tasks(model, es["groups"], cfg, dev,
                               torch.Generator().manual_seed(cfg.seed + 555))
        ctl.score = sum(t0["acc"].values()) / 6.0
        _log(fh, dict(step=step0, anchor_init=dict(
            score=round(ctl.score, 4), tasks=t0["acc"])))
        print(f"[anchor] 建档: 锚点成绩 {ctl.score:.4f}", flush=True)
    if cfg.phase == 2 and not probe_only and step0 == 0:
        o0 = ood_abstain(model, es, cfg, dev)
        _log(fh, dict(step=step0, ood0=o0))
        print(f"[ood] 建档: 非法画布弃权率 {o0} (P4 零假设, §10d)", flush=True)
    print(f"[run] phase={cfg.phase} k={cfg.k} dev={dev} steps={n_steps} "
          f"λ={cfg.lam} λ0={cfg.lam0} α={cfg.alpha} "
          f"β={cfg.beta}(adapt={cfg.beta_adapt}) out={cfg.out}", flush=True)
    if not probe_only:                           # 分阶段 --resume 的阶段标记 (报表按此归段)
        _log(fh, dict(step=step0, run_start=dict(
            steps=n_steps, k=cfg.k, lam=cfg.lam, lam0=cfg.lam0, beta_align=cfg.beta_align,
            beta_sc=cfg.beta_sc, kappa=cfg.kappa, expand_every=cfg.expand_every,
            holdout_frac=cfg.holdout_frac, u_ema=cfg.u_ema, resume=bool(resume),
            init=bool(init), c_mode=cfg.c_mode, k_force=cfg.k_force, k_expand=cfg.k_expand,
            stop_count=cfg.stop_count, stop_dmin=cfg.stop_dmin, stop_cnull=cfg.stop_cnull)))
    stop_hits = []
    for step in range(step0, n_steps):
        t0 = time.time()
        sc, ns = next(sit)
        groups = None
        if cfg.phase == 2:
            groups = next(git)
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        logline = dict(step=step)
        if cfg.phase == 1:
            encg = model.encode_scene(sc.to(dev))
            l_count = L.count_loss(model.heads.count(encg["cls"]),
                                   ns.to(dev))
            l_count.backward()
            l_mask = mask_forward(model, cfg, rng_cpu, dev)
            l_mask.backward()
            logline.update(l_count=round(float(l_count), 4),
                           l_mask=round(float(l_mask), 4))
        else:
            mb = max(1, len(groups) // cfg.accum)
            csz = max(1, sc.shape[0] // cfg.accum)
            bt = ctl.beta_eff(step) if ctl is not None else None
            agg, mets, c9s = {}, [], []
            hold = None
            if align is not None:
                hold, rotated = align.S(step)
                if rotated:
                    _log(fh, dict(step=step, holdout=sorted(hold)))
            for a in range(cfg.accum):
                gs = groups[a * mb:(a + 1) * mb]
                if not gs:
                    continue
                out = phase2_losses(
                    model, cfg, gs, sc[a * csz:(a + 1) * csz],
                    ns[a * csz:(a + 1) * csz], dev, rng_cpu, rng_dev,
                    step, tw, ref_writer=(ctl.ref if ctl is not None else None),
                    lag=lag, beta=bt, probe=probe, guard=guard,
                    align=align, hold=hold)
                if align is None:
                    tot = (out["l_grpo"] + out["l_task"] + out["l_count"]
                           + cfg.mu * out["l_read"]) / cfg.accum
                    tot.backward()
                else:
                    # v4.1: 画布路由 V 梯度经 autograd.grad 提取后手工装配 (S 不 apply)
                    (out["l_grpo"] + out["l_count"]).backward()
                    ad = apply_align_grads(model, align, out, cfg, opt, dev)
                    A = out["align"]
                    rec = dict(A["diag"], L_S=A["L_S"], L_V=A["L_V"], L_S_sc=A["L_S_sc"],
                               **ad)
                    for kk in ("abst_S", "abst_V", "c_mean", "c_S", "c_V", "c_null",
                               "c_rank", "c_kV", "kappa_on", "l_count_S", "n_S_count", "n_S_sc"):
                        rec[kk] = out["metrics"].get(kk)
                    align.win.append({k_: v_ for k_, v_ in rec.items()
                                      if isinstance(v_, (int, float))})
                    logline["al"] = {k_: (round(v_, 5) if isinstance(v_, float) else v_)
                                     for k_, v_ in rec.items() if v_ is not None
                                     and k_ in ("cos_cv", "tau", "u", "beta_eff",
                                                "L_S", "L_V", "S_S", "S_V",
                                                "tau_null", "cos_joint",
                                                "c_null", "corr_frac", "delta",
                                                "u_cos_prev",
                                                # 场景路由 (改动 1, 头条)
                                                "cos_sc", "tau_sc", "u_sc", "beta_eff_sc",
                                                "corr_frac_sc", "delta_sc", "u_cos_prev_sc",
                                                "L_S_sc", "n_S_sc", "degenerate_sc",
                                                # 投影率 c (第四阶段 改动 2)
                                                "c_mean", "c_null", "c_rank", "c_kV", "kappa_on")}
                    for kk in ("abst_S", "abst_V", "c_mean", "c_S", "c_V",
                               "c_null", "c_rank", "c_kV", "kappa_on", "l_count_S", "n_S_groups",
                               "n_S_count", "n_S_sc"):
                        out["metrics"].pop(kk, None)
                for kk in ("l_grpo", "l_task", "l_count", "l_read"):
                    agg[kk] = agg.get(kk, 0.0) + float(out[kk]) / cfg.accum
                mets.append(out["metrics"])
                if out.get("c9"):
                    c9s.append(out["c9"])
            l_mask = mask_forward(model, cfg, rng_cpu, dev)
            l_mask.backward()
            if cfg.ood_bot > 0 and GM.illegal_capacity_ok(l_max_of(cfg), cfg.occ_k):
                l_ood = ood_bot_loss(model, cfg, rng_cpu, dev)
                (cfg.mu * cfg.ood_bot / (cfg.n_groups * cfg.g)
                 * l_ood).backward()
                logline["l_ood"] = round(float(l_ood), 4)
            logline.update({k: round(v, 4) for k, v in agg.items()},
                           l_mask=round(float(l_mask), 4),
                           **_merge_metrics(mets))
            if c9s:
                logline["div"] = c9s[-1]["div"]    # accum>1 取末微批 (已登记)
                logline["div_mix"] = c9s[-1]["div_mix"]
                evs = [e for c in c9s for e in c["evs"]]
                if evs:
                    logline["c9_ev"] = evs
                    print(f"[c9] {json.dumps(evs)}", flush=True)
            if bt is not None:
                logline["beta"] = round(bt, 5)
            kl_win.append(logline.get("kl", 0.0))
            watch_acc.append(sum(logline["acc"]) / len(logline["acc"]))
            watch_rstd.append(logline.get("r_std", 0.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        if lag is not None:
            ema_update(lag, model, cfg.reader_lag)
        t_data += t1 - t0
        t_step += time.time() - t1
        if probe_only or step % 50 == 0:
            logline["t_data"] = round(t_data / max(step - step0 + 1, 1), 3)
            logline["t_step"] = round(t_step / max(step - step0 + 1, 1), 3)
            print(json.dumps(logline, ensure_ascii=False), flush=True)
        _log(fh, logline)
        s1 = step + 1
        if not probe_only and cfg.phase == 2 and s1 % cfg.blank_every == 0:
            bc = blank_control(model, cfg.k, es["theta_ns"], dev,
                               es["groups"][:64])
            _log(fh, dict(step=step, blank=bc))
            if bc["alarm"]:
                print(f"[ALARM] blank-canvas leak? {bc['alarm']} "
                      f"acc={bc['acc']}", flush=True)
        if not probe_only and s1 % cfg.eval_every == 0:
            ev = dict(step=step, count=eval_count(
                model, es["count_scenes"], es["count_ns"], dev),
                mask=eval_mask(model, es, dev))
            score = ev["count"]["train"]["exact"]
            v_med, tpl = None, None
            if cfg.phase == 2:
                ev["k"] = cfg.k
                # 评测用独立固定种子的腐蚀流: 不消耗训练流, 跨步实现值相同可比
                rng_e = torch.Generator().manual_seed(cfg.seed + 555)
                ev["tasks"] = eval_groups_tasks(model, es["groups"], cfg,
                                                dev, rng_e)
                tab = code_table(model, es["table_scenes"], cfg.k, dev,
                                 lmax=l_max_of(cfg))
                counts = {n: [len(p) for p in ps] for n, ps in tab.items()}
                ev["table"] = table_stats(counts)
                ev["dmin"] = min_pair_hamming(tab)   # s110 头号仪表
                tpl = modal_templates(tab)           # s111 头号监控 (改5)
                if tpl_hist:
                    ev["shift"] = template_shift(tpl_hist[-1], tpl)
                if len(tpl_hist) >= 7:               # 3.5k 窗, 与首跑满分窗同尺
                    ev["shift7"] = template_shift(tpl_hist[-7], tpl)
                tpl_hist.append(tpl)
                if len(tpl_hist) > 8:
                    tpl_hist.pop(0)
                sh0 = ev.get("shift")                # [U] p3c-alpha2 churn/定居仪表
                churn = int(sum(sh0["per_n"].values())) if sh0 else None
                ev["churn"] = churn
                ev["settled"] = sctl.offer(churn if churn is not None else -1)
                ev["pixel"] = pixel_probe(tab, cfg.k, cfg.s, rng_e, dev,
                                          occ_k=cfg.occ_k)
                ev["ood"] = dict(abstain=ood_abstain(model, es, cfg, dev))
                if probe is not None:                # s114 C3: 双探针在环状态各报一次
                    rng_p = torch.Generator().manual_seed(cfg.seed + 556)
                    ev["probe2"] = probe.eval_acc(tab, cfg, rng_p, dev)
                ev["read_n"] = read_per_n(           # s114 v2: 逐 N 直读 (P1-P4 判读)
                    model, tab, cfg,
                    torch.Generator().manual_seed(cfg.seed + 557), dev)
                if s1 % cfg.reset_every == 0:        # s114 v2 C3b (记录项, 不进判据)
                    ev["reset_head"] = reset_head_probe(model, tab, cfg, dev,
                                                        cfg.seed + 558)
                score = sum(ev["tasks"]["acc"].values()) / 6.0
                if cctl.offer(ev["dmin"]["min"], score):
                    ev["collapse_flag"] = 1          # s114 C7: 连续两评成立才置位
                na = max(len(watch_acc), 1)
                at = sum(watch_acc) / na
                # 新失败模式监控 (§10 [U]): 锚钉住写者 -> 影子追上 -> 训练批回
                # 0.96-0.99 -> 优势重塌零. 训练批与固定集差距 -> 0 且 r_std -> 0
                # 是"完全不动"的危险信号, 不是好消息; 出现则缩短影子时距.
                ev["watch"] = dict(
                    acc_train=round(at, 4), gap=round(at - score, 4),
                    r_std=round(sum(watch_rstd) / max(len(watch_rstd), 1), 4))
                sh = ev.get("shift")
                v_med = sh["med"] if sh else None
                if align_ev is not None:             # v4.1 §7.3 每评仪表 M1-M8 (只评态: M1/M7-V)
                    ev["align"] = align_eval(model, cfg, align_ev, es, dev, s1,
                                             align_ev.hold.current(step), v_med)
                if ctl is not None:
                    # beta_adapt=0 (s112): 喂 None 使校准/自适应整块跳过, β 恒基准
                    kw = (sum(kl_win) / max(len(kl_win), 1)
                          if cfg.beta_adapt else None)
                    ev["anchor"] = ctl.on_eval(model, s1, score,
                                               ev["dmin"]["min"], v_med, tpl,
                                               kw, opt=opt)
                kl_win.clear()
                watch_acc.clear()
                watch_rstd.clear()
            if guard is not None:
                ev["c9"] = guard.summary()           # 守卫现态 + 采纳累计
            if latch.offer(score, s1):               # s113 改2: 稳定窗锁存
                ev["best_latch"] = round(latch.best, 4)
                _save(os.path.join(cfg.out, "ckpt_best.pt"), s1)
                if ctl is not None:
                    ctl.note_best(model, tpl, score)
            stop_hits = []
            if stopc is not None and cfg.phase == 2:  # [U] 2026-08-17 停跑规则 (评后即判)
                al_ev = ev.get("align") or {}
                stop_hits = stopc.offer(score, ev["count"]["train"]["exact"],
                                        bool(ev.get("collapse_flag")),
                                        dmin=ev["dmin"]["min"], c_mean=al_ev.get("c_mean"),
                                        c_null=al_ev.get("c_null"))
                if stop_hits:
                    ev["stop"] = stop_hits
            if xctl is not None and cfg.k < GM.K_MAX and not stop_hits:
                disturbed = eval_disturbed(ev.get("anchor"))
                last_x = align_ev.last_expand if align_ev is not None else (
                    expands[-1]["step"] if expands else 0)
                why = xctl.offer(score, v_med, disturbed, step=s1, last_expand=last_x)
                if why:
                    old_ns = set(D.train_ns(cfg.k))
                    ex = do_expand(model, cfg, ctl, latch, s1, dev, why=why)
                    cctl.reset()                     # s114 C7: 扩张评不作塌方证据
                    sctl.reset()                     # 跨档模板不可比, 定居重计
                    if guard is not None:            # 新档重建守卫 (含新 N;
                        evs_hist = guard.events      #  事件史跨档累计入终报)
                        guard = C9Guard(list(D.train_ns(cfg.k)))
                        guard.events = evs_hist
                    ev["expand"] = ex
                    expands.append(dict(step=s1, **ex))
                    if align_ev is not None:
                        align_ev.last_expand = s1
                        align_ev.expands = list(expands)
                    del git, gl
                    gl = make_group_loader(cfg, salt)
                    git = iter(gl)
                    es = build_eval_sets(cfg)
                    if align_ev is not None:         # v4.1 M6 首见登记 (训练新 N 之前)
                        new_ns = [n for n in D.train_ns(cfg.k) if n not in old_ns]
                        rows = first_seen_rows(model, cfg, align_ev, es, dev, new_ns,
                                               max(old_ns))
                        for r_ in rows:
                            r_["step"] = s1
                            align_ev.first_seen.append(r_)
                        ev["first_seen"] = rows
                        _log(fh, dict(step=step, first_seen=rows))
                        align_ev.hold.on_expand(new_ns)
                    print(f"[expand] K {ex['k'][0]}->{ex['k'][1]} @ {s1} ({why}) "
                          f"extrap0={ex['extrap0']['mean']}", flush=True)
            print(json.dumps(ev, ensure_ascii=False), flush=True)
            _log(fh, ev)
            if stop_hits:
                print(f"[STOP] 停跑规则触发 @ {s1}: {stop_hits} -- 存检查点后收尾", flush=True)
                _save(os.path.join(cfg.out, "ckpt_last.pt"), s1)
                cfg.steps = s1                       # 终报以停跑步为终评步
                break
        if not probe_only and s1 % cfg.probe_every == 0:
            pr = fit_probe(model, es["probe_scenes"], es["probe_ns"], dev)
            print(json.dumps(dict(step=step, probe=pr)), flush=True)
            _log(fh, dict(step=step, probe=pr))
        if not probe_only and s1 % cfg.viz_every == 0 and cfg.phase == 2:
            r6_viz(model, es["table_scenes"], cfg.k, dev,
                   os.path.join(cfg.out, f"r6_{s1}.png"),
                   lmax=l_max_of(cfg))
        if not probe_only and s1 % cfg.ckpt_every == 0:
            _save(os.path.join(cfg.out, f"ckpt_{s1}.pt"), s1)
        save_last = (not probe_only and s1 % cfg.eval_every == 0)
        if save_last:
            _save(os.path.join(cfg.out, "ckpt_last.pt"), s1)
    if probe_only:
        if dev.type == "cuda":
            print(f"[mem] max_alloc={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB",
                  flush=True)
        print("[probe] done", flush=True)
        return
    _save(os.path.join(cfg.out, "ckpt_last.pt"), cfg.steps)
    fin = dict(step=cfg.steps, final=True, k_final=cfg.k,
               # 双点口径 (s112 [U]): 结论必须报"最优检查点 + 终局", 不许只报终局.
               # s113 改2: best 为当前档内稳定窗闩值 (扩张时重置; 旧档 best 在
               # expands[].prev_best 与 ckpt_best_k*.pt)
               best=dict(step=latch.step, score=round(latch.best, 4)),
               count=eval_count(model, es["count_scenes"], es["count_ns"],
                                dev),
               mask=eval_mask(model, es, dev),
               probe=fit_probe(model, es["probe_scenes"], es["probe_ns"],
                               dev))
    if cfg.phase == 2:
        rng_e = torch.Generator().manual_seed(cfg.seed + 555)
        fin["tasks"] = eval_groups_tasks(model, es["groups"], cfg, dev,
                                         rng_e)
        tab = code_table(model, es["table_scenes"], cfg.k, dev,
                         lmax=l_max_of(cfg))
        fin["table"] = table_stats({n: [len(p) for p in ps]
                                    for n, ps in tab.items()})
        fin["dmin"] = min_pair_hamming(tab)
        fin["pixel"] = pixel_probe(tab, cfg.k, cfg.s, rng_e, dev,
                                   occ_k=cfg.occ_k)
        fin["blank"] = blank_control(model, cfg.k, es["theta_ns"], dev,
                                     es["groups"][:64])
        fin["ood"] = dict(abstain=ood_abstain(model, es, cfg, dev))
        if probe is not None:
            rng_p = torch.Generator().manual_seed(cfg.seed + 556)
            fin["probe2"] = probe.eval_acc(tab, cfg, rng_p, dev)
        fin["read_n"] = read_per_n(
            model, tab, cfg, torch.Generator().manual_seed(cfg.seed + 557),
            dev)
        fin["reset_head"] = reset_head_probe(model, tab, cfg, dev,
                                             cfg.seed + 558)
        fin["settled"] = sctl.streak >= SETTLE_STREAK   # Q5 (末评流的定居态)
        if guard is not None:
            fin["c9"] = dict(summary=guard.summary(), events=guard.events)
        fin["expands"] = expands
        fin["stop"] = stop_hits                      # 停跑规则触发记录 (空 = 跑满)
        if align_ev is not None:                     # v4.1 终报: 首见表/L_preq/R(F) 史/dR/dF
            fin["align"] = align_final(align_ev)
        torch.save(tab, os.path.join(cfg.out, "code_table.pt"))
        r6_viz(model, es["table_scenes"], cfg.k, dev,
               os.path.join(cfg.out, "r6_final.png"),
               lmax=l_max_of(cfg))
    print("[final]", json.dumps(fin, ensure_ascii=False), flush=True)
    _log(fh, fin)
    fh.close()
