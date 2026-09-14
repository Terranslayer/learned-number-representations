# symemerge/numcode/pred/trainer.py
"""训练器 (spec-v3 §6-§12, 在 v2 之上): 三流批配额 (场景**链**/池抽/随机画布) · K 跳链 + 闸定入池
(depth 字段) · 常数预测器 (§6.4 零假设仪器) · 两阶段课程 (切换看 深度0 + k* 侧) · 逐深度评测入册 (§11.1)
· 最优闩 (双点口径, 闩看 k* 侧) · 停跑 (v2 两条改 k* 侧 + [U] 平台期+零入流) · 检查点/恢复 · 码表与
R6 配对可视化 (链版多列). β/u_S/水位闸挂起 (§8): 单次反传, 读数移到逐评 (eval_readings_chain).

机制本体在 render/model/pool/data/step; 此处只做编排与仪表. [C] 实例化决策登记在 params.md §13.
"""
import dataclasses
import json
import math
import os
import time

import torch

from .. import data as D
from .. import geometry as GM
from ..grpo import TASK_KEYS, TaskWeights
from ..losses import sample_mask_cells
from . import render as R
from . import step as ST
from .data import LIN, KitMaker, build_batch, make_add_loader, make_scene_loader
from .model import PredModel
from .pool import DataPool

RESUME_SALT = 104729
_PHI = 0x9E3779B97F4A7C15
_MIX = 0xBF58476D1CE4E5B9
_MOD = 2 ** 62


@dataclasses.dataclass
class TrainCfg:
    steps: int = 30000
    # ---- 批配额 (§6.7, 定死; v3 §9.3 登记 B_scene 从 4 起, 探针实测能放大再放大 — K=1 实测 8 可行, params §13)
    b_scene: int = 4             # B_scene: 每步场景链条数 (每条 K+1 个状态)
    b_pool: int = 32             # B_pool ≥ B_scene (§9.2)
    b_mask: int = 64             # B_mask
    # ---- 损失系数
    lam: float = 0.0             # λ 介质代价 (§5.2 绝对尺度直接生效; 步 1 取 §9.2 下界 = 0: 探针实测 λ=1e-4 使 p_空 176 步内
    #                              升至 .92 (饱和死解), λ=0 停在 .44; 步 5 起按升法上调, 须实测标定)
    lam_ladder: str = ""         # [U] 2026-08-20 步3: 起征后 λ 级距表 (逗号分隔, 如 "5e-4,2e-3,8e-3"); 空 = 无时程, lam 恒定
    lam_onset_stamps: float = 5.0    # 起征触发阈: 评测章数均 ≥ 此值 (首跑点火后 50 步章数已 45, 阈宽松)
    lam_onset_evals: int = 2         # 连续过阈评数 (=2: 单评尖峰不触发)
    lam_onset_buffer: int = 500      # 触发评之后的缓冲步数, 起征点 = 触发评步 + 此值
    lam_level_steps: int = 2000      # 每级步数, 末级保持到跑完
    # ---- [U] 2026-08-21 写头共模扣除 + λ PI 伺服 + 只跑阶段 1 (params §13「共模写头 K=1」块)
    d1_cm: int = 0               # 1 = 写头 ℓ = W_D1(z − α z̄) + b_c (α 初 1 可学, b_c 初 0 可学, W_D1 冻结); 0 = 旧式 W_D1 z + b 逐位不变
    # ---- [U] 2026-08-23「W_D1 解冻，行归一化，lr = 0.1 × 主 lr」(params §13 W_D1 解冻块)
    d1_learn: int = 0            # 1 = 解冻 D1.lin.weight: 独立参数组 (lr = d1_lr_mult × lr, wd=0), 主参数组/裁剪列表不含;
    #                              0 = 冻结逐位同旧 (A1). 旧冻结偏置恒不解冻 (cm 路本就不参与). 解冻检查点 (opt 2 组) 用
    #                              d1_learn=0 恢复会在 load_state_dict 报参数组数不符 — 属显式拒绝, 不静默丢状态.
    d1_lr_mult: float = 0.1      # W_D1 参数组学习率倍率 ([U] 0.1 × 主 lr)
    d1_rownorm: int = 1          # 1 = W_D1 行 L2 归一化 ([C] 语义 = 单位范数: 解冻/载入时立即归一 + 每步 post-hook 归一;
    #                              初始行范数 ≈ 1/√3, 首次单位化 = 逐行 ×√3 量级的尺度跳变, 起点复评量化)
    lam_servo: int = 0           # 1 = λ 不设定值, 每步按训练侧章数 EMA 对目标章数做 PI 伺服 (log λ 域); 与 lam_ladder 互斥
    lam_target: float = 5.0      # 目标章数 s* ([C] = E[N], N~U{1..9} 一元码均值; 用户未给数值, 待改定)
    lam_kp: float = 0.05         # PI 比例增益 (作用于误差增量, log λ 域)
    lam_ki: float = 1e-3         # PI 积分增益 (每步, log λ 域; [C] 按 λ 时程跑实测响应滞后 ≤500 步定: 满误差下 λ 每 ~350 步翻倍)
    lam_ema: float = 0.99        # 被控量 = 章数均的 EMA 动量 (≈100 步窗)
    lam_init: float = 1e-4       # 伺服起点 λ₀ (λ 探针在册量级; 由伺服接管后起点只影响前几百步)
    lam_lo: float = 1e-6         # λ 下夹 (log 域下界, 实质零)
    lam_hi: float = 0.1          # λ 上夹 (lam02 定值 .02 一步压死的 5×, 安全顶)
    lam_eclip: float = 2.0       # 误差夹取 |e| ≤ 此, e = log((ŝ+1)/(s*+1))
    stage1_only: int = 0         # 1 = 只跑阶段 1 (N∈{1..9}): 课程永不切换 (boot 与 T_stage1 均不触发), 阶段 1 全程豁免停跑
    switch_at_resume: int = 0    # [U] 2026-08-22「延用 B 臂设置，切阶段 2」: 1 = --resume 时若检查点仍在阶段 1, 续跑第一步前当场切大数量阶段
    #                              (why='resume'; 状态改写同 boot/forced 切换: k=K_MAX / 收敛水平 = 末两评 note_1 均 / 闩重置 / switch_step = 续跑起点),
    #                              与 stage1_only 互斥; 已在阶段 2 的检查点 = 无动作
    alpha_freeze: int = 0        # [U] 2026-08-22「…立刻冻结 α 重启」: 1 = 共模写头 α 不更新 (反传后 α.grad 置 None ⇒ AdamW 跳过: 无动量/无权重衰减,
    #                              α 逐位保持检查点值; b_c 与其余参数照常); 要求 d1_cm=1
    task_wmul: str = ""          # [U] 2026-08-23「将 t4 权重设置为 3 倍续跑 3000 步」: 逐任务损失权重倍率, 6 个逗号分隔正数 (t1..t6 序, 如 "1,1,1,3,1,1");
    #                              损失权重 w = 自动归一 w × 倍率 (倍率只乘进损失; 池成绩 σ 的 w 加权均值仍用归一 w, C4.5); 空 = 不乘, 逐位同旧
    # ---- [U] 2026-08-24「先跑甲，写头d1保持冻结」迭代重学甲案 (spec 2026-08-24-iterated-relearning-phase-spec.md §2)
    exposure: int = 0            # 1 = 学习期 + 曝光子集 + 孪生新读者 (阶段 2 激活, 阶段 1 挂空转); 0 = 逐位同旧 (A30)
    e_exp: int = 20              # 曝光子集大小 |𝒩_exp| (点评例值; 敏感性旋钮; 五分带比例配额 2/4/4/6/4)
    t_period: int = 2000         # 学习期长 (步; 步 1 后按新读者收敛实测重定)
    eta_tr: float = 0.3          # 转移损失权重 η_tr (0 = 仪表零臂: 教学照跑、转移只测不传; 步 2 探针按梯度范数比回填)
    tr_warm: float = 0.25        # 期首豁免比例 (期内前此比例步 η_tr 置 0)
    sr_lr: float = 3e-4          # 新读者 lr (主训练配方逐字; wd 沿 cfg.wd, clip 沿 cfg.clip)
    n_s: int = 1                 # 每训练步新读者教学步数
    b_teach: int = 16            # 教学批纸张数上限 (本步链末纸曝光行 + 池内同 N 条目补足)
    b_teach_sc: int = 8          # 教学批场景张数上限 (sb 曝光行)
    sr_mu: float = 0.05          # 新读者直读 CE 权重 (主 μ 同值)
    # ---- [U] 2026-08-24「既然已经澄清，那就根据讨论试一试乙吧」迭代重学乙案 (spec §2.2 乙执行细则)
    gen_relearn: int = 0         # 1 = 换代重学: 主脑期界重置 + 标签限域 + 写侧模仿 + 池读传承 (阶段 2 激活); 0 = 逐位同旧 (A35)
    t_gen: int = 3000            # 代长 (步; 乙2 探针实测重定)
    eta_im: float = 1.0          # 写侧模仿 CE 权重 (乙2 探针按梯度范数比回填)
    b_im: int = 16               # 模仿批行数 (池上一代曝光条目 × 银行新抽场景)
    b_poolread: int = 16         # 池读传承批行数 (池曝光条目重渲 + 标签, 任意代)
    gen_keep_cnn: int = 1        # 1 = 期界保留 E.cnn (眼部承袭); 0 = 全量重置
    # ---- [U] 2026-08-24 乙案空纸塌缩三修 (期首禁入池 + 期首 λ 豁免 + 模仿墨格权重)
    im_wink: float = 1.0         # 模仿 CE 非空格权重 (类不平衡矫正: 抄纸=抄它的墨不是抄空; 1 = 逐位同旧)
    gen_warm: float = 0.0        # 期首比例: 本代前此比例步 禁入池 + λ 置 0 (婴儿涂鸦不进谱系/不因写墨挨罚; 0 = 逐位同旧)
    # ---- [U] 2026-08-27 加法流 (「输入两个scene，分别为N和m个物体，m = 1 - 3，输出笔记。除了输入外，流程和主流程内其他任务相同，不新造器官」)
    add_on: int = 0              # 1 = 每步另跑 B_add 条加法链: 深度 0 两槽装配 (场景 N | 场景 m, 现成段嵌入路, 零新参数) → 写头 → 纸 → 单槽读, 标签 s=N+m;
    #                              纸按 ρ_admit 入池 (标签 s, 深度 K); 加法评测集 + R6 配对图; 0 = 逐位同旧 (A18 平价路不变)
    b_add: int = 16              # 每步加法链条数 ([C] = 发射配方的 B_scene 16 一比一; 类默认 b_scene 4 只是 v3 登记起点, 发射时二者都显式给; 探针 16 = 峰值 16.0 GiB)
    add_workers: int = 10        # 加法流工作进程 (每项渲 2 张场景 + 6 候选; 独立于场景流 ⇒ 主流程输入与基线逐项同)
    add_admit: int = 1           # 1 = 加法纸按 ρ_admit 入池 (流程同场景链); 0 = 不入池 (诊断臂: 隔离「池被加法纸占半」)
    add_wpop: int = 1            # 1 = 加法纸命中并入 w_t 人口 (流程同场景末纸); 0 = 不并入 (诊断臂)
    add_sc_weight: float = 1.0   # 两槽 CLS (加法深度 0) 项权重 (乘在 scene_ds_weight 之上; 0 = 只训加法纸, 诊断臂)
    add_note_weight: float = 1.0  # 加法纸项权重 (0 = 只训两槽 CLS, 诊断臂)
    add_start: int = 0           # 加法流从此步起激活 (之前不抽加法批、不算加法项; 0 = 从零)
    mu: float = 0.05             # μ 直读损失权重 (小, 量具不是压力)
    # ---- 信道 (登记值 geometry §4, 单旋钮 s; 遮挡 occ_k>0 = 恒开恰 k 格)
    s: float = 0.25
    occ_k: int = 48
    keep_frac: float = 0.15      # MASK-KEEP 保留桶比例
    # ---- 数据池 (§6.3-6.5)
    pool_cap: int = 20000        # C_pool
    rho_admit: float = 0.25      # ρ_admit
    t_pool: float = 0.5          # T_pool (σ∈[0,1] 全幅时 softmax 比 e² ≈ 7.4 落 [3,10]; 300 步探针 T=.3 时比 19.9 越界)
    eps_pool: float = 0.1        # ε 探索底
    eta_pool: float = 0.5        # σ EMA 更新率 ([C])
    # ---- 课程 (§8)
    k1: int = 9                  # 小 N 阶段训练域 {1..9}
    theta_boot: float = 0.95     # θ_boot (起用 0.95, 阶段 1 首次收敛时当场重推)
    n_warm: int = 4000           # N_warm 冷启动豁免期
    t_stage1: int = 12000        # T_stage1 ≥ 3×N_warm, 到则强制切换
    # ---- 更新规则 (§7)
    beta: float = 0.0            # 绝对 β (夹取 β‖u_S‖_{F⁻¹} ≤ 0.1); beta_frac>0 时被忽略
    beta_frac: float = 0.0       # [C] 2026-08-18 固定剂量臂: f = β/β_ceil, 每步 β = f·0.1/‖u_S‖_{F⁻¹} (修正项 F⁻¹ 范数 = f·0.1·‖g_V‖); 0 = 不用
    u_mode: str = "exact"        # u_S 算法: exact = 显式场精确 vjp Jᵀw ([U] 2026-08-18 有限差分停用) / fd = 旧有限差分 (只为旧诊断脚本)
    delta_mult: float = 1.0      # δ = delta_mult × 上一步 AdamW 位移 (只在 u_mode=fd 下有意义)
    water_gate: int = 1          # 水位闸 (§7.5, [U] 2026-08-18 G1 归一化口径): 1 = 每评实测 r = L_S/地板 (同批/同 w_t/同 θ_t, ≥3 抽中位),
    #                              r > 0.5 ⇒ β 当评置零, 下评重测; 0 = 门恒开 (只为探针). 不再有绝对阈值 L* (旧 l_star 字段废止)
    gate_scenes: int = 32        # [C] 2026-08-18 闸量批 = 最近 ≥ 此数张场景 (B_scene<32 时拼接最近几批; = 首跑 B_scene 32 时的单批量级, 使 r 的
    #                              批噪声与首跑同级: T-PROBE B=8 单批 r 散布 .20–.47 对 B=32 .366)
    perm_seed: int = 0           # >0 = 写头输出固定格置换 (读者恢复时间实测的「人为切换编码」; 只在定标窗用)
    switch_grace: int = 0        # [U] 2026-08-18: 阶段切换后新豁免期 (步), 生效值 = max(此值, 池跨度 C_pool/(ρ·B_scene)); 0 = 恰一个池跨度
    read_every: int = 25         # 读数节律 (v3 §8: β 挂起, 读数移到逐评; 本旋钮只被 v2 诊断路使用)
    # ---- v3/v3.1 链 + 预测闸 (spec-v3 + spec-v3.1; [C] 实例化与继承常数复核 = params §13)
    chain_k: int = 2             # K 链深 (§9.2: 2 = 能表达「为下一步铺路」的最小单位; 步1 平价跑用 1)
    origin_cat: int = 0          # v3.1 C1: origin 拼接撤销, 恒 0 (置 1 即链装配断言报错; seg 代码路留 model)
    origin_write: int = 0        # [U] 2026-08-23「写路径加 origin，读路径不动」+「写第一张和第二张笔记都能看原题 origin … 读笔记 … 只能读笔记自身」:
    #                              1 = 场景链每个写步 k 的写路径另过一遍两槽编码 [CLS]⊕tok(当前槽)+seg_cur⊕tok(x0)+seg_org (长 513),
    #                              写头 D1 吃其当前槽 256 个 token (step.write_tokens_org); 当前槽 k=0 = 空白纸 ([U] 裁定; step.blank_note =
    #                              全空类硬渲染过同链信道), k≥1 = 纸 x_k; 读路径 (场景/各纸 六任务/直读/h_k) 仍单槽 257 不动; 段嵌入
    #                              seg_cur/seg_org (零初始化, 本系检查点从未受训) 由此首次受训; 与 use_flag=1 兼容路互斥; 0 = 逐位同旧
    use_flag: int = 0            # A18 兼容路: 1 = v2 装配 (场景/候选带 flag, 无段嵌入). v3 训练恒 0 (§3.1 删 flag)
    pred_on: int = 0             # V̂ + 前向模型 g + L_pred (v3.1 C5/C6)
    gate_on: int = 0             # 闸 + L_gate + 入池 k* (§4.5/§5.4; v3.1 C3 本阶段恒 0 = 恒跑满 K)
    pred_to_backbone: int = 0    # C8: 0 = sg[h], sg[φ]; 1 = L_pred(BCE+L_fwd) 进 Θ_E ([U] 2026-08-21 主臂 = 1 先跑)
    chain_from_pool: int = 1     # v3.1 C4.1: 池抽笔记起链 (0 = 单态读出兼容路, A18/消融用)
    scene_ds_weight: float = 1.0  # C2.3: L_chain 的场景 k=0 项权重 (0 旁支臂登记, 本跑不开)
    rho_admit_gen: float = 0.03125  # C4.3 ρ_admit_gen ≤ ρ_admit·B_scene/B_pool (默认档 = .25·4/32 自洽;
    #                                 B_scene 探针实测后按上式回填, 发射显式给值)
    g_max: int = 0               # C4.3: 允许入池的最大 generation; 0 = 不限 (None), 谱系由 FIFO 自然淘汰
    strat_band: int = 16         # C4.4 [C]: 阶段 2 分层抽取的 N 带宽 (阶段 1 恒逐 N); 8 带覆盖 {1..127}
    eta_fwd: float = 1.0         # C6.5: L_fwd 系数 (起始 1.0, 首跑实测 BCE/L_fwd 两项梯度范数后回填)
    s_ema_m: float = 0.99        # C5.2: s[t] EMA 动量
    s_clamp: float = 1e-3        # C5.2: s[t] clamp 下界
    s_init_steps: int = 100      # C5.2: 初值 = 首 100 步均值 (自举期累计均值, [C])
    d_bottleneck: int = 64       # C6.2: 前向模型 g 的瓶颈宽 (MLP d→64→d, 末层零初始化)
    eta_pred: float = 0.1        # L_pred 系数 (§9.2 步 3 起用; ptb=1 臂下进 Θ_E, 继承值复核见 params §13 v3.1)
    eta_gate: float = 1.0        # L_gate 系数 (§9.2: 只影响 2 个标量的学习速度)
    c_step: float = 0.0          # ponder 成本 (§9.2: 从 0 起; 时程表待步 4 定, 定值一律不可用 — λ=0.02 教训)
    kappa_p: float = 0.01        # KL 防饱和正则 (§5.4, 小; 步 4 定)
    p_g: float = 0.382           # Geom_K 先验 (§9.2: K=2 时先验期望深度 ≈ 1 = K/2; 步 4 定)
    eps_write: float = 0.0       # v3.1 C2.5 退役 (场景链 k=0 恒写, 入流结构性非零); 旋钮留给休眠对照臂, 恒 0
    gate_a0: float = 0.5413248546129181  # 闸初值 a₀ (§9.2 定于步 4; 默认 = softplus⁻¹(1) 步 0 原值; 步 4 发射 92.0 = Δ̂ 终评量程 [−.026,+.024] 定标, params §13)
    gate_b0: float = 0.0                 # 闸初值 b₀ (§9.2: 0)
    c_ladder: str = ""                   # c_step 时程级距表 (§9.2 步 4「从 0 起, 定值一律不可用」; 逗号分隔; 空 = 恒 cfg.c_step, 测试/断言用)
    c_level_steps: int = 2000            # 每级步数, 末级保持 (λ 时程同构)
    c_onset_stamps: float = 5.0          # 起征触发阈: 任一深度章数均 ≥ 此值 (λ 时程同构)
    c_onset_evals: int = 2               # 连续过阈评数
    c_onset_buffer: int = 500            # 触发评之后的缓冲步数
    gate_mode: str = "learned"           # §11.2 对照臂: learned | const (恒停率臂) | rand (均匀随机闸臂)
    gate_const_lam: float = 0.5          # const 臂恒定停止率 (发射时由主臂终评 E[k*] 经 const_rate_match 反解)
    gate_rand_hi: float = 1.0            # rand 臂 λ ~ U(lo, hi) 上界
    gate_rand_lo: float = 0.0            # rand 臂下界 (均值对齐 λ̄>.5 时 lo=2λ̄−1, hi=1 — U(0,·) 均值上限 .5)
    w_plat: int = 5              # 平台期窗长 W_plat ([U] 2026-08-20 = 5 评)
    eps_plat: float = 0.01       # 提升判定余量 (§9.2: 登记 0.01 起, 首跑实测单评噪声后回填)
    # ---- 优化
    lr: float = 3e-4
    wd: float = 0.01
    clip: float = 1.0
    workers: int = 14            # 场景流工作进程 (大 N 阶段 32×7 渲染/步 ≈ 4.5 s CPU, 需 ≥ 8)
    kit_workers: int = 10        # 池候选包进程 (32×6 渲染/步)
    kit_chunk: int = 2           # 候选包每任务条数 (并行度 = B_pool/kit_chunk; 旧值 8 ⇒ 32 条只 4 任务并行, 大 N 段 t_data ≈ 1.3 s/步 = 瓶颈)
    # ---- 节律
    eval_every: int = 500
    log_every: int = 25
    viz_every: int = 2000
    ckpt_every: int = 5000
    eval_per_n: int = 8          # 评测集: 每数量组数
    table_per_n: int = 16        # 码表: 每数量场景数
    seed: int = 0
    out: str = "outputs/nc_pred2"


def build_model(cfg):
    """cfg → 模型 (闸初值旋钮 + v3.1 C6.2 前向模型瓶颈宽 + [U] 2026-08-21 写头共模扣除 d1_cm 穿线).
    Gate/FwdModel/α/b_c 初值不耗额外 RNG 序 ⇒ 默认值下与 PredModel() 同种子逐位同."""
    return PredModel(gate_a0=cfg.gate_a0, gate_b0=cfg.gate_b0, d_bottleneck=int(cfg.d_bottleneck),
                     d1_cm=bool(int(getattr(cfg, "d1_cm", 0))))


# ================================================================ 小部件
class BestLatch:
    """最优闩 (§11 双点口径): 连续两评到线才锁存, 闩值取两评较小者 (单点尖峰占不了闩位)."""

    def __init__(self):
        self.best, self.step, self.prev = -1.0, 0, None

    def offer(self, score, step):
        pair = min(score, self.prev) if self.prev is not None else None
        self.prev = score
        if pair is not None and pair > self.best:
            self.best, self.step = pair, step
            return True
        return False

    def reset(self):
        self.best, self.step, self.prev = -1.0, 0, None

    def state(self):
        return dict(best=self.best, step=self.step, prev=self.prev)

    def load_state(self, d):
        self.best, self.step, self.prev = d["best"], d["step"], d["prev"]


class WinAgg:
    """步间窗口累加器: 数值取均值 (list 逐元素均值), None 跳过."""

    def __init__(self):
        self.d = {}
        self.n = 0

    def add(self, m):
        self.n += 1
        for k, v in m.items():
            if v is None or isinstance(v, torch.Tensor):
                continue
            if isinstance(v, dict):
                sub = self.d.setdefault(k, WinAgg())
                sub.add(v)
            elif isinstance(v, list):
                acc = self.d.setdefault(k, [0.0] * len(v))
                if len(acc) == len(v):
                    for i, x in enumerate(v):
                        acc[i] += float(x)
                    self.d[k + "__n"] = self.d.get(k + "__n", 0) + 1
            elif isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                self.d[k] = self.d.get(k, 0.0) + float(v)
                self.d[k + "__n"] = self.d.get(k + "__n", 0) + 1

    def mean(self, digits=5):
        out = {}
        for k, v in self.d.items():
            if k.endswith("__n"):
                continue
            if isinstance(v, WinAgg):
                out[k] = v.mean(digits)
            elif isinstance(v, list):
                n = self.d.get(k + "__n", 1)
                out[k] = [round(x / n, digits) for x in v]
            else:
                n = self.d.get(k + "__n", 1)
                out[k] = round(v / n, digits)
        return out

    def reset(self):
        self.d, self.n = {}, 0


def _log(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def _seed_mix(a, b):
    return (a * _PHI + (b + 1) * _MIX) % _MOD


# ================================================================ 常数预测器 (v3 §6.4, 零参数零假设仪器)
class ConstPred:
    """逐 (t, θ_t) 最近实际正确率 EMA, 直接当预测输出 — 完全不看纸只看任务参数. 与主前向同批更新,
    无梯度; H-C 的那条线: V̂ 打不过它 ⇒ 预测实质不依赖 h. η_const = [C] 0.02 (只读仪器).
    链状态维在 update 处边缘化 (V̂ 在全部深度被评, 常数器吃同一分布)."""
    SIZES = dict(t1=GM.K_MAX - 1, t2=1, t3=GM.K_MAX - 1, t4=3, t5=1, t6=3)

    def __init__(self, eta=0.02):
        self.eta = float(eta)
        self.val = {t: torch.full((n,), 0.5) for t, n in self.SIZES.items()}

    def _idx(self, th, t, B):
        if t in ("t1", "t3"):
            return th["tau"].detach().cpu()
        if t == "t4":
            return th["p"].detach().cpu()
        if t == "t6":
            return th["m"].detach().cpu()
        return torch.zeros(B, dtype=torch.long)

    def update(self, th, accs):
        """accs (B,6) 或 (B,K+1,6); 逐 θ 用该 θ 的本批均值走一步 EMA."""
        a = accs.detach().float().cpu()
        if a.dim() == 3:
            a = a.mean(1)
        B = a.shape[0]
        for j, t in enumerate(TASK_KEYS):
            idx = self._idx(th, t, B)
            for u in idx.unique():
                m = idx == u
                self.val[t][u] = (1.0 - self.eta) * self.val[t][u] + self.eta * a[m, j].mean()

    def predict(self, th, B):
        """(B,6) CPU: V̂_const_t = EMA[t, θ_t]."""
        return torch.stack([self.val[t][self._idx(th, t, B)] for t in TASK_KEYS], dim=1)

    def state(self):
        return {t: v.clone() for t, v in self.val.items()}

    def load_state(self, d):
        for t in self.val:
            if t in d:
                self.val[t] = d[t].clone()


# ================================================================ 校准读数 (v3 §11.1)
def brier(p, y):
    return float(((p - y) ** 2).mean()) if p.numel() else float("nan")


def ece10(p, y):
    """10 箱 ECE."""
    p, y = p.reshape(-1), y.reshape(-1)
    n = p.numel()
    if n == 0:
        return float("nan")
    e = 0.0
    for i in range(10):
        hi = (i + 1) / 10 + (1e-9 if i == 9 else 0.0)
        m = (p >= i / 10) & (p < hi)
        if bool(m.any()):
            e += float(m.float().sum() / n * (p[m].mean() - y[m].mean()).abs())
    return e


def auc_rank(p, y):
    """成对比较 AUC (平局记 0.5); 单类 → nan."""
    p, y = p.reshape(-1), y.reshape(-1)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    d = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float(((d > 0).float() + 0.5 * (d == 0).float()).mean())


# ================================================================ 三分解读数 (v3.1 C10.1, 本跑核心)
def _rankdata(x):
    """平均并列名次 (Spearman 用; 真 Δ 在 1/6 网格上并列多, 不平均会偏)."""
    x = x.double()
    order = x.argsort()
    r = torch.empty_like(x)
    r[order] = torch.arange(x.numel(), dtype=torch.float64)
    u, inv = torch.unique(x, return_inverse=True)
    sums = torch.zeros(u.numel(), dtype=torch.float64).scatter_add_(0, inv, r)
    cnts = torch.zeros(u.numel(), dtype=torch.float64).scatter_add_(0, inv, torch.ones_like(r))
    return (sums / cnts)[inv]


def _pearson(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    if a.numel() < 2:
        return float("nan")
    sa, sb = a.std(unbiased=False), b.std(unbiased=False)
    if float(sa) == 0.0 or float(sb) == 0.0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def pair_metrics(x, y, sign_mask=None):
    """一对 Δ 序列的 C10.1 四读数 + 费雪显著界: r_thresh = 1.96/√(n−3) = 同行零假设 (r=0, 当评样本量
    下的 95% 界, MEASURED-NULL). 符号一致率在 sign_mask (真 Δ≠0) 上算; 无掩码 = 全样本."""
    n = x.numel()
    out = dict(n=int(n), r=round(_pearson(x, y), 4),
               rho=round(_pearson(_rankdata(x), _rankdata(y)), 4),
               mae=round(float((x - y).abs().mean()), 5),
               r_thresh=round(1.96 / math.sqrt(max(n - 3, 1)), 4))
    m = sign_mask if sign_mask is not None else torch.ones(n, dtype=torch.bool)
    out["sign"] = round(float(((x > 0) == (y > 0))[m].float().mean()), 4) if bool(m.any()) else float("nan")
    return out


def dtrue_stats(d):
    """[U] 2026-08-23 主判据 Δ_true[1] = 纸 2 六任务 − 纸 1 六任务, 「连续 5 评显著为正」的 [C] 读法: d (M,) = 评测集逐链
    相邻深度六任务命中均值之差 (配对样本); 入册 mean / se (样本 sd/√M) / lo95 = mean − 1.96·se (配对正态近似; 同行零假设 =
    均值 0, 即第二张纸不比第一张好) / t = mean/se / 正、负链占比 / n. 「显著为正」⇔ lo95 > 0."""
    n = int(d.numel())
    mean = float(d.mean()) if n else float("nan")
    sd = float(d.std(unbiased=True)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=round(mean, 5), se=round(se, 5), lo95=round(mean - 1.96 * se, 5),
                t=(round(mean / se, 3) if se > 0 else None),
                pos_frac=round(float((d > 0).float().mean()), 4) if n else None,
                neg_frac=round(float((d < 0).float().mean()), 4) if n else None, n=n)


def corr3_block(dh, do_, realD):
    """三分解 (C10.1): Δ̂ 对 Δ_true (dt, 判决对) / Δ_oracle 对 Δ_true (ot, gap 归因 V̂) /
    Δ̂ 对 Δ_oracle (do, gap 归因 g). 池化 (M·K) + 逐 k 的 Pearson. 硬标的两对的符号一致率
    在 真Δ≠0 上算 (同 dsign 口径)."""
    K = dh.shape[1]
    nz = (realD != 0).reshape(-1)
    f_dh, f_do, f_rt = dh.reshape(-1), do_.reshape(-1), realD.reshape(-1)
    out = dict(dt=pair_metrics(f_dh, f_rt, nz), ot=pair_metrics(f_do, f_rt, nz),
               do=pair_metrics(f_dh, f_do))
    out["per_k"] = [dict(dt_r=round(_pearson(dh[:, k], realD[:, k]), 4),
                         ot_r=round(_pearson(do_[:, k], realD[:, k]), 4),
                         do_r=round(_pearson(dh[:, k], do_[:, k]), 4)) for k in range(K)]
    return out


# ================================================================ 平台期 + 零入流停跑 ([U] 2026-08-20, v3 §11.3)
def plateau_zero_inflow(st, cfg, score_k, inflow_win, exempt):
    """停 ⇔ 平台期 (最近 W_plat 评 k* 侧均值均未超过窗前历史最优 + ε_plat) ∧ 同窗入池率 ≡ 0.
    两条须同时成立, 任一单独成立只入册不停; 豁免期 (冷启动/阶段1/切换后) 与斜率条件 (连续两评上升)
    与其它水平判据同辖. 返回 dict(plateau, zero_inflow, hit, prev_best)."""
    h = st.setdefault("hist_kside", [])
    a = st.setdefault("inflow_hist", [])
    h.append(float(score_k))
    a.append(float(inflow_win))
    W = int(cfg.w_plat)
    out = dict(plateau=False, zero_inflow=False, hit=False, prev_best=None)
    if len(h) > W:
        out["zero_inflow"] = max(a[-W:]) == 0.0
        out["prev_best"] = max(h[:-W])
        if not exempt and not rising2(h[-3:]):
            out["plateau"] = all(s <= out["prev_best"] + float(cfg.eps_plat) for s in h[-W:])
            out["hit"] = out["plateau"] and out["zero_inflow"]
    return out


# ================================================================ 评测集 (CRN, 缓存)
def eval_cache_path(cfg, k):
    root = os.path.dirname(cfg.out.rstrip("/")) or "."
    return os.path.join(root, f"nc_pred2_evalsets_s{cfg.seed}_k{k}_g{cfg.eval_per_n}_t{cfg.table_per_n}.pt")


def build_eval_sets(cfg, k):
    """固定评测集 (每档一份, 磁盘缓存): groups (每数量 eval_per_n 项, 含候选/θ/标签), table_scenes
    (每数量 table_per_n 张场景, 码表/可视化用), mask_k + 遮蔽选格 (随机画布素养评测)."""
    path = eval_cache_path(cfg, k)
    if os.path.exists(path):
        es = torch.load(path, weights_only=True)
        print(f"[eval] cache hit {path}", flush=True)
        return es
    t0 = time.time()
    rng = torch.Generator().manual_seed(cfg.seed + 999 + k)
    groups = []
    for n in D.train_ns(k):
        groups += [D.sample_group(rng, k, n=n) for _ in range(cfg.eval_per_n)]
    table = {n: torch.stack([D.render_scene(rng, n) for _ in range(cfg.table_per_n)])
             for n in D.train_ns(k)}
    mk = R.random_canvas_classes(256, rng)
    sel = sample_mask_cells(256, rng)
    rep, keep = ST.split_mask_keep(sel, rng, cfg.keep_frac)
    es = dict(k=k, groups=groups, table_scenes=table, mask_k=mk, mask_sel=sel, mask_rep=rep,
              mask_keep=keep)
    torch.save(es, path)
    print(f"[eval] built k={k} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return es


def add_eval_cache_path(cfg, k):
    root = os.path.dirname(cfg.out.rstrip("/")) or "."
    return os.path.join(root, f"nc_pred2_addsets_s{cfg.seed}_k{k}_g{cfg.eval_per_n}.pt")


def build_add_eval_sets(cfg, k):
    """加法流固定评测集 ([U] 2026-08-27; 每档一份, 磁盘缓存): 每个和 s ∈ add_sums(k) 各 eval_per_n 项, m 在该 s 的合法 m 中均匀抽,
    N = s − m; 项 = sample_add_group (两张场景 + s 的候选包/θ/标签). CRN: seed + 1999 + k."""
    path = add_eval_cache_path(cfg, k)
    if os.path.exists(path):
        es = torch.load(path, weights_only=True)
        print(f"[eval] cache hit {path}", flush=True)
        return es
    t0 = time.time()
    rng = torch.Generator().manual_seed(cfg.seed + 1999 + k)
    pairs = D.add_pairs(k)
    items = []
    for s in D.add_sums(k):
        ms = sorted({m for n, m in pairs if n + m == s})
        for _ in range(cfg.eval_per_n):
            m = ms[int(torch.randint(0, len(ms), (1,), generator=rng))]
            items.append(D.sample_add_group(rng, k, n_a=s - m, m=m))
    es = dict(k=k, groups=items)
    torch.save(es, path)
    print(f"[eval] built add k={k} ({len(items)} items) in {time.time() - t0:.0f}s -> {path}", flush=True)
    return es


# ================================================================ 评测
def _acc_dict(accs):
    a = accs.mean(0)
    return {t: round(float(a[i]), 4) for i, t in enumerate(TASK_KEYS)}


def _tol(pred, ns):
    e = (pred - ns).abs()
    return {f"tol{t}": round(float((e <= t).float().mean()), 4) for t in (0, 1, 3, 5)}


@torch.no_grad()
def eval_mask(model, es, cfg, dev, rng_eval, bs=64):
    """素养评测: 固定随机画布 (评测信道固定 rng), 保留桶识别 + 遮蔽位准确率. 编码走 v3 单图式
    (与训练掩码路同装配; 兼容路 = v2 裸装配)."""
    mk = es["mask_k"]
    preds = []
    for i in range(0, mk.shape[0], bs):
        k = mk[i:i + bs].to(dev)
        draw = R.draw_channel(k.shape[0], cfg.s, rng_eval, dev, occ_k=0)
        x = R.channel(R.render_classes(k, draw), draw)
        enc = ST.enc_v3(model, x, cfg, flagged=False, mask_cells=es["mask_rep"][i:i + bs].to(dev))
        preds.append(model.heads.cell(enc["tokens"]).argmax(-1).cpu())
    pred = torch.cat(preds)
    rep, keep = es["mask_rep"], es["mask_keep"]
    return dict(kept_acc=round(float((pred[keep] == mk[keep]).float().mean()), 4),
                mask_acc=round(float((pred[rep] == mk[rep]).float().mean()), 4))


# ================================================================ v3.1 链评测 (spec-v3 §11.1 + v3.1 C10, 逐深度分列)
@torch.no_grad()
def eval_groups_chain(model, es, cfg, dev, w, rng_eval, const=None, sscale=None, chunk=64):
    """场景 (深度0, 感知上界量尺 C2.2) + 逐深度纸 + k* 侧 (交付) 的六任务/直读/章数/对位率; pred_on 时
    另出 v3.1 C10 读数: 三分解 corr3 (C10.1: Δ̂/Δ_oracle/Δ_true 的 Pearson/Spearman/符号一致/MAE +
    费雪界), cos(ĥ,h) 对 g≡0 基线 (C10.2), V̂ 校准 (C10.3: 对软标签 Brier 主口径 + 硬口径续册 /
    对硬 ECE / AUC / 对常数预测器差 软硬双口径), var_batch(h) 逐深度 (C10.4), x1-x2 汉明距 (C10.6).
    闸关 (C3) ⇒ k* ≡ K; 开闸时场景链 k* 判据从 k=1 起 (A22). 腐蚀 = 每 chunk 单 draw 整链重放 (C7)."""
    rng_pol = torch.Generator().manual_seed(int(cfg.seed) + 4243)   # 对照臂评测 k* 抽样流 (CRN)
    items = es["groups"]
    K = int(cfg.chain_k)
    pred_on = bool(cfg.pred_on)
    A = [[] for _ in range(K + 1)]
    CE = [[] for _ in range(K + 1)]
    RD = [[] for _ in range(K + 1)]
    LS = [[] for _ in range(K + 1)]
    MG = [[] for _ in range(K + 1)]
    HV = [[] for _ in range(K + 1)]
    STm = [[] for _ in range(K)]
    PE = [[] for _ in range(K)]
    IC = [[] for _ in range(K)]
    CF = [[] for _ in range(K)]
    CB = [[] for _ in range(K)]
    HAM = []
    NS, KS, DH, DO, PH, Vp, Wp, Cp = [], [], [], [], [], [], [], []
    for i in range(0, len(items), chunk):
        sb = build_batch(items[i:i + chunk], dev)
        B = sb["B"]
        draw = R.draw_channel(B, cfg.s, rng_eval, dev, cfg.occ_k)
        tpl = R.templates(draw.u, draw.s)
        out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
        for d, stt in enumerate(out["states"]):
            A[d].append(stt["accs"].cpu())
            CE[d].append(stt["ce"].cpu())
            RD[d].append(stt["read_pred"].cpu())
            LS[d].append(stt["rows"].cpu())
            MG[d].append(stt["marg"].cpu())
            HV[d].append(stt["h"].var(0).mean().cpu())
        for d, wr in enumerate(out["writes"]):
            STm[d].append(R.stamp_count(wr["k"]).cpu().float())
            PE[d].append(wr["p"][:, :, R.EMPTY].mean(1).cpu())
            IC[d].append(iconic_rate(wr["k"], sb["x1"]).cpu())
        if K >= 2:
            HAM.append((out["writes"][0]["k"] != out["writes"][1]["k"]).float().mean(1).cpu())
        NS.append(sb["ns"].cpu())
        if pred_on:
            pg = ST.pred_gate(model, out["states"], sb, cfg, k0_gated=False)
            Vp.append(pg["V"].cpu())
            Wp.append(pg["W"].cpu())
            DH.append(pg["dhat"].cpu())
            DO.append((pg["V"][:, 1:].mean(-1) - pg["V"][:, :K].mean(-1)).cpu())   # Δ_oracle (C10.1)
            PH.append(pg["p_halt"].cpu())
            for kk in range(K):
                CF[kk].append(torch.nn.functional.cosine_similarity(
                    pg["hhat"][kk], out["states"][kk + 1]["h"], dim=-1).cpu())
                CB[kk].append(torch.nn.functional.cosine_similarity(
                    out["states"][kk]["h"], out["states"][kk + 1]["h"], dim=-1).cpu())
            if const is not None:
                Cp.append(const.predict(sb["th"], B))
            if cfg.gate_on:
                mode = str(getattr(cfg, "gate_mode", "learned"))
                if mode == "learned":
                    ks = ST.kstar_det(pg["dhat"].cpu(), cfg.c_step, k0_gated=False)
                else:
                    ks = ST.kstar_sample(ST.policy_lam(pg["lam"].detach().cpu(), cfg, rng_pol), rng_pol,
                                         k0_gated=False)
            else:
                ks = torch.full((B,), K, dtype=torch.long)
        else:
            ks = torch.full((B,), K, dtype=torch.long)
        KS.append(ks)
    Ast = torch.stack([torch.cat(x) for x in A], dim=1)            # (M,K+1,6)
    CEst = torch.stack([torch.cat(x) for x in CE], dim=1)          # (M,K+1,6) 逐任务未加权 CE
    Rst = torch.stack([torch.cat(x) for x in RD], dim=1)           # (M,K+1)
    Lst = torch.stack([torch.cat(x) for x in LS], dim=1)           # (M,K+1)
    NS = torch.cat(NS)
    ks = torch.cat(KS)
    M = NS.shape[0]
    ar = torch.arange(M)
    Aks, Rks, Lks = Ast[ar, ks], Rst[ar, ks], Lst[ar, ks]
    CEks = CEst[ar, ks]
    STt = torch.stack([torch.cat(x) for x in STm], dim=1)          # (M,K)
    PEt = torch.stack([torch.cat(x) for x in PE], dim=1)
    ICt = torch.stack([torch.cat(x) for x in IC], dim=1)

    def _icm(v):
        v = v[~v.isnan()]
        return round(float(v.mean()), 4) if v.numel() else None

    ink = ks >= 1
    stamps_ks = STt[ar[ink], (ks[ink] - 1)] if bool(ink.any()) else torch.zeros(0)
    depths = []
    for d in range(1, K + 1):
        depths.append(dict(
            depth=d, acc=_acc_dict(Ast[:, d]), ce=_acc_dict(CEst[:, d]), score=round(float(Ast[:, d].mean()), 4),
            read=round(float((Rst[:, d] == NS).float().mean()), 4), L_ds=round(float(Lst[:, d].mean()), 4),
            stamps_mean=round(float(STt[:, d - 1].mean()), 2), stamps_med=float(STt[:, d - 1].median()),
            p_empty=round(float(PEt[:, d - 1].mean()), 5), iconic=_icm(ICt[:, d - 1])))
    per_n = {}
    for n in sorted(set(NS.tolist())):
        m = NS == n
        pn = dict(score=round(float(Aks[m].mean()), 4), score_sc=round(float(Ast[m, 0].mean()), 4),
                  read=round(float((Rks[m] == n).float().mean()), 4), kstar_mean=round(float(ks[m].float().mean()), 3))
        mi = m & ink
        pn["stamps_med"] = float(STt[mi.nonzero().squeeze(1), ks[mi] - 1].median()) if bool(mi.any()) else None
        per_n[str(n)] = pn
    ev = dict(
        scene=dict(acc=_acc_dict(Ast[:, 0]), ce=_acc_dict(CEst[:, 0]), score=round(float(Ast[:, 0].mean()), 4),
                   read=_tol(Rst[:, 0], NS), L_ds=round(float(Lst[:, 0].mean()), 4)),
        depths=depths,
        notes=dict(acc=_acc_dict(Aks), ce=_acc_dict(CEks), score=round(float(Aks.mean()), 4), read=_tol(Rks, NS),
                   L_ds=round(float(Lks.mean()), 4),
                   stamps_mean=(round(float(stamps_ks.mean()), 2) if stamps_ks.numel() else 0.0),
                   stamps_med=(float(stamps_ks.median()) if stamps_ks.numel() else 0.0),
                   iconic=_icm(ICt[ar[ink], ks[ink] - 1]) if bool(ink.any()) else None),
        per_n=per_n, n_items=int(M),
        gate=dict(E_kstar=round(float(ks.float().mean()), 3), inflow=round(float(ink.float().mean()), 4),
                  kstar_hist={str(d): int((ks == d).sum()) for d in range(K + 1)}))
    if cfg.gate_on:
        lb, Pm = ST.const_rate_match(float(ks.float().mean()), K)
        mix = sum(Pm[d] * float(Ast[:, d].mean()) for d in range(K + 1))
        ev["gate"]["hd_mix"] = dict(lam_bar=round(lb, 4), P=[round(x, 4) for x in Pm],
                                    mix_score=round(mix, 4),
                                    delta=round(float(Aks.mean()) - mix, 4))
    # [U] 2026-08-23 主判据 Δ_true[1] = 纸 2 六任务 − 纸 1 六任务: 逐链相邻深度六任务均值之差的配对统计, 长 K, 不依赖 pred_on, 顶层入册
    realD_all = Ast[:, 1:].mean(-1) - Ast[:, :-1].mean(-1)        # (M,K) 真 Δ_k (下标 0 = 纸 1 − 场景)
    ev["dtrue"] = [dtrue_stats(realD_all[:, kk]) for kk in range(K)]
    if HAM:                                                        # [U] 2026-08-23 K=2 主判据 ham12: 不依赖 pred_on, 顶层入册
        hm = torch.cat(HAM)                                        # 逐链 x1≠x2 格占比 (硬类别图, 256 格)
        qs3 = torch.tensor([0.1, 0.5, 0.9])
        ev["ham12"] = dict(mean=round(float(hm.mean()), 4), med=round(float(hm.median()), 4),
                           q=[round(float(x), 4) for x in torch.quantile(hm, qs3)],
                           nz_frac=round(float((hm > 0).float().mean()), 4),   # ham12>0 的链占比 (0 = 全部照抄/幂等)
                           n=int(hm.numel()))
    if pred_on:
        V, W = torch.cat(Vp), torch.cat(Wp)                        # (M,K+1,6), (M,K,6)
        dh, do_, ph = torch.cat(DH), torch.cat(DO), torch.cat(PH)
        Y = Ast
        Msoft = torch.stack([torch.cat(x) for x in MG], dim=1)     # (M,K+1,6) 边距
        ys = ST.y_soft_of(Msoft, sscale) if sscale is not None else None
        realD = Y[:, 1:].mean(-1) - Y[:, :-1].mean(-1)             # (M,K) 真 Δ_k
        nz = realD != 0
        sign_ok = float(((dh > 0) == (realD > 0))[nz].float().mean()) if bool(nz.any()) else float("nan")
        pos_frac = float((realD > 0)[nz].float().mean()) if bool(nz.any()) else float("nan")
        cf = torch.stack([torch.cat(x) for x in CF], dim=1).reshape(-1)
        cb = torch.stack([torch.cat(x) for x in CB], dim=1).reshape(-1)
        qs = torch.tensor([0.1, 0.5, 0.9])
        pr = dict(brier_V_hard=round(brier(V, Y), 5), brier_W_hard=round(brier(W, Y[:, 1:]), 5),
                  ece_V=round(ece10(V, Y), 5),
                  auc_V={t: round(auc_rank(V[..., j], Y[..., j]), 4) for j, t in enumerate(TASK_KEYS)},
                  dhat_q=[round(float(x), 5) for x in torch.quantile(dh.reshape(-1), qs)],
                  dsign=round(sign_ok, 4), dcorr=round(_pearson(dh[nz], realD[nz]), 4),
                  d_nz_frac=round(float(nz.float().mean()), 4), d_pos_frac=round(pos_frac, 4),
                  corr3=corr3_block(dh, do_, realD),
                  cos_fwd=dict(mean=round(float(cf.mean()), 5),
                               q=[round(float(x), 5) for x in torch.quantile(cf, qs)]),
                  cos_base=dict(mean=round(float(cb.mean()), 5),
                                q=[round(float(x), 5) for x in torch.quantile(cb, qs)]),
                  var_h=[round(float(torch.stack(x).mean()), 5) for x in HV],
                  top1_kside=round(float(ST.pb_top1(V[ar, ks]).float().mean()), 3))
        if ys is not None:
            pr["brier_V"] = round(brier(V, ys), 5)
            pr["s_scale"] = [round(float(x), 4) for x in sscale.scale()]
        if HAM:
            pr["ham12"] = ev["ham12"]                              # 旧 schema 位置保留 (同一字典)
        if Cp:
            C = torch.cat(Cp).unsqueeze(1).expand_as(V)
            pr["brier_const_hard"] = round(brier(C, Y), 5)
            pr["d_const_hard"] = round(pr["brier_const_hard"] - pr["brier_V_hard"], 5)
            if ys is not None:
                pr["brier_const"] = round(brier(C, ys), 5)
                pr["d_const"] = round(pr["brier_const"] - pr["brier_V"], 5)   # C10.3 主口径 = 对软标签
        ev["pred"] = pr
        ev["gate"].update(p_halt_mean=[round(float(x), 4) for x in ph.mean(0)],
                          slope_a=round(model.gate.slope(), 5), b=round(float(model.gate.b.detach()), 5),
                          b_over_a=round(float(model.gate.b.detach()) / max(model.gate.slope(), 1e-9), 5),
                          c_step=float(cfg.c_step))
    return ev


@torch.no_grad()
def eval_add_chain(model, esa, cfg, dev, w, rng_eval, chunk=64):
    """加法流评测 ([U] 2026-08-27): 两槽场景侧 (深度 0) 与纸侧 (深度 K = 交付) 的六任务/直读/章数/对位率 (对场景 N), 标签 = s = N+m.
    同批两条零假设 (同一读者、同一信道实现 CRN, 当评重算): null_single = 只看场景 N 单槽写出的纸对 s 标签打分 (「忽略 m」零假设);
    纸对 N 标签重打分 = 五线性任务均值 acc5_vs_N 与直读容差 read_vs_N (「纸只写了 N」对照; t5 无 N 候选不计, 同口径 acc5 对 s 标签并列).
    逐 m / 逐 s 分列 (per_m / per_n)."""
    items = esa["groups"]
    K = int(cfg.chain_k)
    A0, AK, AN, A5N, R0, RK, RN, L0, LK, STm, PE, IC, NS, NA, MM, HM = ([] for _ in range(16))
    for i in range(0, len(items), chunk):
        chunk_items = items[i:i + chunk]
        sb = build_batch(chunk_items, dev)
        B = sb["B"]
        draw = R.draw_channel(B, cfg.s, rng_eval, dev, cfg.occ_k)
        tpl = R.templates(draw.u, draw.s)
        out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
        sb1 = {kk: v for kk, v in sb.items() if kk not in ("x_aux", "n_a", "m_add")}
        out1 = ST.chain_side(model, sb1, w, cfg.mu, cfg, draw, tpl)      # 零假设: 单槽 (只看场景 N) 写纸, 同 draw
        s0, sK, s1 = out["states"][0], out["states"][K], out1["states"][K]
        tgN = {t: torch.tensor([D.targets(int(n), it["theta"])[t] for n, it in zip(sb["n_a"].tolist(), chunk_items)], device=dev)
               for t in LIN}
        A5N.append(torch.stack([(sK["tl"][t].argmax(-1) == tgN[t]).float() for t in LIN], 1).cpu())
        A0.append(s0["accs"].cpu())
        AK.append(sK["accs"].cpu())
        AN.append(s1["accs"].cpu())
        R0.append(s0["read_pred"].cpu())
        RK.append(sK["read_pred"].cpu())
        RN.append(s1["read_pred"].cpu())
        L0.append(s0["rows"].cpu())
        LK.append(sK["rows"].cpu())
        wr = out["writes"][K - 1]
        STm.append(R.stamp_count(wr["k"]).cpu().float())
        PE.append(wr["p"][:, :, R.EMPTY].mean(1).cpu())
        IC.append(iconic_rate(wr["k"], sb["x1"]).cpu())
        HM.append((wr["k"] != out1["writes"][K - 1]["k"]).float().mean(1).cpu())   # 加法纸 对 单场景纸 逐格类别差异率 (看到 m 改了多少格)
        NS.append(sb["ns"].cpu())
        NA.append(sb["n_a"].cpu())
        MM.append(sb["m_add"].cpu())
    A0, AK, AN, A5N = (torch.cat(x) for x in (A0, AK, AN, A5N))
    R0, RK, RN, NS, NA, MM = (torch.cat(x) for x in (R0, RK, RN, NS, NA, MM))
    STt, PEt, ICt, HMt = torch.cat(STm), torch.cat(PE), torch.cat(IC), torch.cat(HM)
    lin_idx = [TASK_KEYS.index(t) for t in LIN]

    def _icm(v):
        v = v[~v.isnan()]
        return round(float(v.mean()), 4) if v.numel() else None

    out = dict(
        scene=dict(acc=_acc_dict(A0), score=round(float(A0.mean()), 4), read=_tol(R0, NS),
                   L_ds=round(float(torch.cat(L0).mean()), 4)),
        notes=dict(acc=_acc_dict(AK), score=round(float(AK.mean()), 4), read=_tol(RK, NS), read_vs_N=_tol(RK, NA),
                   acc5=round(float(AK[:, lin_idx].mean()), 4), acc5_vs_N=round(float(A5N.mean()), 4),
                   L_ds=round(float(torch.cat(LK).mean()), 4), stamps_mean=round(float(STt.mean()), 2),
                   stamps_med=float(STt.median()), p_empty=round(float(PEt.mean()), 5), iconic=_icm(ICt),
                   ham_vs_single=dict(mean=round(float(HMt.mean()), 4), med=round(float(HMt.median()), 4),
                                      nz_frac=round(float((HMt > 0).float().mean()), 4))),   # 0 = 两张纸逐格全同 (m 未改纸)
        null_single=dict(acc=_acc_dict(AN), score=round(float(AN.mean()), 4), read=_tol(RN, NS), read_vs_N=_tol(RN, NA)),
        per_m={}, per_n={}, n_items=int(NS.shape[0]))
    for m in sorted(set(MM.tolist())):
        sel = MM == m
        out["per_m"][str(m)] = dict(score=round(float(AK[sel].mean()), 4),
                                    read0=round(float((RK[sel] == NS[sel]).float().mean()), 4),
                                    read0_vs_N=round(float((RK[sel] == NA[sel]).float().mean()), 4),
                                    null_single=round(float(AN[sel].mean()), 4), n=int(sel.sum()))
    for s in sorted(set(NS.tolist())):
        sel = NS == s
        out["per_n"][str(s)] = dict(score=round(float(AK[sel].mean()), 4), score_sc=round(float(A0[sel].mean()), 4),
                                    read=round(float((RK[sel] == s).float().mean()), 4),
                                    read_vs_N=round(float((RK[sel] == NA[sel]).float().mean()), 4),
                                    null_single=round(float(AN[sel].mean()), 4), stamps_med=float(STt[sel].median()))
    return out


@torch.no_grad()
def add_viz(model, esa, dev, cfg, path, n_rows=16):
    """加法流 R6 配对图 (法则: 每张纸挨着它写自的题): 行 = (N, m), 列 = [场景 N | 场景 m | 纸 (干净探针: 信道 s=0, 模板零抖动, 与码表同约定)],
    黑底白墨. 行 = 评测集中每个和 s 的首项, 再等距抽 n_rows 行."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = {}
    for it in esa["groups"]:
        first.setdefault(it["n"], it)
    ss = sorted(first)
    if len(ss) > n_rows:
        ss = [ss[int(round(i * (len(ss) - 1) / (n_rows - 1)))] for i in range(n_rows)]
    rows = [first[s] for s in ss]
    sb = build_batch(rows, dev)
    tpl0 = R.templates(torch.zeros(sb["B"], GM.T, 4, device=dev), 0.0)
    e = model.E.enc_seq(model.E.tokenize(sb["x1"]), model.E.tokenize(sb["x_aux"]), seg=True)
    kmap = ST.write_logits(model, e["tokens"], cfg).argmax(-1)
    canv = R.render_hard(kmap, tpl0).cpu()
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)))
    axes = axes.reshape(len(rows), 3)
    for r, it in enumerate(rows):
        axes[r, 0].imshow(it["scene"], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"N={it['n_a']}", color="w")
        axes[r, 1].imshow(it["scene_m"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"m={it['m']}", color="w", fontsize=9)
        axes[r, 2].imshow(canv[r], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(f"s={it['n']}: {int(R.stamp_count(kmap[r:r + 1])[0])} stamps", color="w", fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    return path


@torch.no_grad()
def eval_pool_chain(model, pool, cfg, dev, w, rng_eval, kit_fn, const=None, sscale=None, band_w=1, n=128):
    """池抽笔记侧单态读出 ([C] v3.1: 三分解核心读数在场景链评测集上算, 池侧保 V̂/常数器对照, 软硬双口径).
    抽样口径与训练同 = 按 N 分层 (C4.4, band_w 随课程阶段); 不记账."""
    if pool.size == 0:
        return None
    n = min(n, pool.size)
    idx = pool.sample_stratified(n, cfg.t_pool, cfg.eps_pool, rng_eval, band_w=band_w)
    kits = kit_fn(pool.labels(idx).tolist())
    pb = ST.render_pool_batch(pool, idx, kits, cfg, rng_eval, dev)
    po = ST.pool_side_v3(model, pb, w, cfg.mu, cfg)
    out = dict(acc=_acc_dict(po["accs"]), ce=_acc_dict(po["ce"]), score=round(float(po["accs"].mean()), 4),
               read=_tol(po["read_pred"].cpu(), pb["ns"].cpu()), L_ds=round(float(po["rows"].mean()), 4),
               n=n, gen_mean=round(float(pool.gens(idx).float().mean()), 3))
    if cfg.pred_on:
        vlog = model.pred.v_logits(model.heads, po["h"].detach(), pb["th"], detach_emb=True)
        vp = torch.sigmoid(vlog).cpu()
        out["brier_V_hard"] = round(brier(vp, po["accs"].cpu()), 5)
        ysp = ST.y_soft_of(po["marg"].cpu(), sscale) if sscale is not None else None
        if ysp is not None:
            out["brier_V"] = round(brier(vp, ysp), 5)
        if const is not None:
            cp = const.predict(pb["th"], pb["B"])
            out["brier_const_hard"] = round(brier(cp, po["accs"].cpu()), 5)
            out["d_const_hard"] = round(out["brier_const_hard"] - out["brier_V_hard"], 5)
            if ysp is not None:
                out["brier_const"] = round(brier(cp, ysp), 5)
                out["d_const"] = round(out["brier_const"] - out["brier_V"], 5)
    return out


@torch.no_grad()
def code_table_chain(model, es, dev, cfg, bs=64):
    """码表 (链版): 每数量 table_per_n 张场景 → 各深度类别图 (干净探针: 信道 s=0 恒等, 模板零抖动 —
    与 v2 码表同约定, 腐蚀是增广不进探针). v3.1 C1: 单槽装配. 返回 tab {depth: {n: (M,T) uint8}}, med."""
    K = int(cfg.chain_k)
    tab = {d: {} for d in range(1, K + 1)}
    med = {d: {} for d in range(1, K + 1)}
    for n, sc in sorted(es["table_scenes"].items()):
        per_d = {d: [] for d in range(1, K + 1)}
        for i in range(0, sc.shape[0], bs):
            x = sc[i:i + bs].to(dev)
            tpl0 = R.templates(torch.zeros(x.shape[0], GM.T, 4, device=dev), 0.0)
            e = model.E.enc_seq(model.E.tokenize(x), None, seg=False)
            z = e["tokens"]
            for d in range(1, K + 1):
                kmap = ST.write_logits(model, z, cfg).argmax(-1)
                per_d[d].append(kmap.cpu())
                if d < K:
                    x1 = R.render_hard(kmap, tpl0)
                    e = model.E.enc_seq(model.E.tokenize(x1), None, seg=False)
                    z = e["tokens"]
        for d in range(1, K + 1):
            kk = torch.cat(per_d[d])
            tab[d][n] = kk.to(torch.uint8)
            med[d][str(n)] = float(R.stamp_count(kk).float().median())
    return tab, med


def r6_viz_chain(es, tab, path_prefix, per_page=16):
    """R6 配对可视化 (法则): 行 = N, 列 = [场景 | 深度1 画布 | … | 深度K 画布], 黑底白墨, 分页存图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = sorted(tab)
    ns = sorted(tab[depths[0]])
    ncol = 1 + len(depths)
    tpl0 = R.templates(torch.zeros(1, GM.T, 4), 0.0)
    paths = []
    for p0 in range(0, len(ns), per_page):
        page = ns[p0:p0 + per_page]
        fig, axes = plt.subplots(len(page), ncol, figsize=(2.5 * ncol, 2.4 * len(page)))
        axes = axes.reshape(len(page), ncol)
        for r, n in enumerate(page):
            axes[r, 0].imshow(es["table_scenes"][n][0], cmap="gray", vmin=0, vmax=1)
            axes[r, 0].set_ylabel(f"N={n}", color="w")
            for ci, d in enumerate(depths):
                k = tab[d][n][:1].long()
                axes[r, 1 + ci].imshow(R.render_hard(k, tpl0)[0], cmap="gray", vmin=0, vmax=1)
                axes[r, 1 + ci].set_title(f"d{d}: {int(R.stamp_count(k)[0])} stamps", color="w", fontsize=9)
            for c in range(ncol):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        fig.patch.set_facecolor("black")
        fig.tight_layout()
        path = f"{path_prefix}_{page[0]}_{page[-1]}.png"
        fig.savefig(path, dpi=110, facecolor="black")
        plt.close(fig)
        paths.append(path)
    return paths


# ================================================================ 检查点
def save_ckpt(path, model, opt, tw, step, cfg, st, pool, const=None, sscale=None):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), tw=tw.ema, step=step,
                    cfg=dataclasses.asdict(cfg), st=st, pool=pool.state(),
                    const=(const.state() if const is not None else None),
                    sscale=(sscale.state() if sscale is not None else None)), path)


def load_ckpt(path, model, opt=None, tw=None, pool=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if tw is not None:
        tw.ema = ck["tw"]
    if pool is not None and ck.get("pool") is not None:
        pool.load_state(ck["pool"])
    return ck


def d1_rownorm_(model):
    """W_D1 行 L2 归一化 (单位范数, in-place, no_grad; [U] 2026-08-23「行归一化」)."""
    with torch.no_grad():
        w = model.D1.lin.weight
        w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-12))


def d1_learn_setup(model, opt, cfg):
    """[U] 2026-08-23「W_D1 解冻，行归一化，lr = 0.1 × 主 lr」: D1.lin.weight requires_grad 置 True,
    进独立参数组 (lr = d1_lr_mult × 主 lr; wd=0 [C] 行归一化下衰减只余径向拉扯, 归一后无净效应);
    d1_rownorm=1 时挂 optimizer post-step hook 每步行归一. 主参数组与裁剪列表在解冻前构建, 不含 W_D1.
    返回 hook 句柄 (无 hook 时 None). 载入时归一由调用方在模型载入后另行执行 (d1_rownorm_)."""
    W = model.D1.lin.weight
    W.requires_grad_(True)
    opt.add_param_group(dict(params=[W], lr=cfg.lr * float(cfg.d1_lr_mult), weight_decay=0.0))
    if int(getattr(cfg, "d1_rownorm", 1)):
        return opt.register_step_post_hook(lambda o, a, k: d1_rownorm_(model))
    return None


def new_state(cfg):
    return dict(stage=1, k=cfg.k1, boot_streak=0, switch_step=None, switch_why=None,
                stage1_level=None, latch=BestLatch().state(), s_notes=0, s_d1=0,
                beta_gate=True, evals=0, last_d1_scores=[], hist_S=[], hist_d1=[], lam_onset=None, lam_streak=0,
                c_onset=None, c_streak=0,
                hist_kside=[], inflow_hist=[], low_inflow=0, last_n_admit=0)


STOP_NOTES, STOP_NOTES_STREAK = 0.85, 3
STOP_D1_DROP, STOP_D1_STREAK = 0.10, 2


def lam_ladder_list(cfg):
    return [float(x) for x in cfg.lam_ladder.split(",") if x.strip()] if cfg.lam_ladder else []


def task_wmul_list(cfg):
    """[U] 2026-08-23 逐任务损失权重倍率: task_wmul 空 ⇒ None (不乘); 非空 ⇒ 恰 6 个正数 (TaskWeights 内断言)."""
    s = str(getattr(cfg, "task_wmul", "") or "")
    return [float(x) for x in s.split(",") if x.strip()] if s.strip() else None


def lam_at(cfg, st, step1):
    """时程表 λ ([U] 2026-08-20 步3): lam_ladder 空 ⇒ 恒 cfg.lam; 非空 ⇒ 起征点 (st['lam_onset']) 前 0,
    起征后每 lam_level_steps 升一级, 末级保持."""
    lad = lam_ladder_list(cfg)
    if not lad:
        return cfg.lam
    on = st.get("lam_onset")
    if on is None or step1 < on:
        return 0.0
    lvl = min((step1 - on) // max(int(cfg.lam_level_steps), 1), len(lad) - 1)
    return lad[int(lvl)]


def lam_trigger(cfg, st, step1, stamps_mean):
    """起征触发 (评测时调, st 就地更新): 章数均 ≥ lam_onset_stamps 连续 lam_onset_evals 评 ⇒ 定起征点
    = 当评步 + lam_onset_buffer. 返回 True 当且仅当本评完成确认. 已起征后恒 False."""
    if not lam_ladder_list(cfg) or st.get("lam_onset") is not None:
        return False
    st["lam_streak"] = (st.get("lam_streak", 0) + 1) if stamps_mean >= cfg.lam_onset_stamps else 0
    if st["lam_streak"] >= cfg.lam_onset_evals:
        st["lam_onset"] = step1 + int(cfg.lam_onset_buffer)
        return True
    return False


class LamServo:
    """λ PI 伺服 ([U] 2026-08-21「λ 用 PI 伺服在目标章数上，不设定值」): 被控量 = 训练侧每步章数均 (场景链末写) 的
    EMA ŝ; 误差 e = clip(log((ŝ+1)/(s*+1)), ±eclip); 控制量 = log λ (乘性: 章数对 λ 的响应跨量级, 线性域步长无法同时
    兼顾带内/带外); 速度型 PI: Δlog λ = k_i·e + k_p·(e − e_prev), 夹取 [log lo, log hi] (夹取即抗积分饱和). 无梯度;
    状态入 st 随检查点续跑. 已有做法 → 为什么不用: PID-Lagrangian (Stooke/Achiam/Abbeel, ICML 2020, arXiv 2007.03964;
    位置型、线性 λ) 与 ControlVAE (Shao+, ICML 2020, arXiv 2004.05988; PI 调 β 钉 KL) 控制量皆在线性域, 本问题 λ 有效带
    5e-4–8e-3 (λ 时程跑实测) 而起点 1e-4/顶 .1 跨三量级 ⇒ 取 log 域 (GECO, Rezende & Viola 2018, arXiv 1810.00597 的
    乘性乘子同理); 十行实现, 无维护库, 内联. 增益 [C] 推导见 params §13「共模写头 K=1」块."""

    def __init__(self, cfg):
        self.target = float(cfg.lam_target)
        self.kp, self.ki, self.m = float(cfg.lam_kp), float(cfg.lam_ki), float(cfg.lam_ema)
        self.lo, self.hi = math.log(float(cfg.lam_lo)), math.log(float(cfg.lam_hi))
        self.eclip = float(cfg.lam_eclip)
        self.loglam = min(max(math.log(float(cfg.lam_init)), self.lo), self.hi)
        self.s_ema, self.e_prev, self.n = None, 0.0, 0

    @property
    def lam(self):
        return math.exp(self.loglam)

    def error(self):
        if self.s_ema is None:
            return 0.0
        e = math.log((self.s_ema + 1.0) / (self.target + 1.0))
        return max(-self.eclip, min(self.eclip, e))

    def update(self, stamps):
        """一训练步一更新: 先 EMA (首步自举) 后 PI; 返回下一步 λ."""
        s = float(stamps)
        self.s_ema = s if self.s_ema is None else self.m * self.s_ema + (1.0 - self.m) * s
        e = self.error()
        self.loglam = min(max(self.loglam + self.ki * e + self.kp * (e - self.e_prev), self.lo), self.hi)
        self.e_prev = e
        self.n += 1
        return self.lam

    def readings(self):
        return dict(lam=self.lam, s_ema=self.s_ema, e=self.error(), loglam=self.loglam, n=self.n)

    def state(self):
        return dict(loglam=self.loglam, s_ema=self.s_ema, e_prev=self.e_prev, n=self.n)

    def load_state(self, d):
        self.loglam, self.s_ema, self.e_prev, self.n = float(d["loglam"]), d["s_ema"], float(d["e_prev"]), int(d["n"])


def c_ladder_list(cfg):
    return [float(x) for x in cfg.c_ladder.split(",") if x.strip()] if cfg.c_ladder else []


def c_at(cfg, st, step1):
    """时程表 c_step (§9.2: 从 0 起,「定值一律不可用」): c_ladder 空 ⇒ 恒 cfg.c_step (测试/断言用);
    非空 ⇒ 起征点 (st['c_onset']) 前 0, 起征后每 c_level_steps 升一级, 末级保持. v3.1 闸关不启用."""
    lad = c_ladder_list(cfg)
    if not lad:
        return cfg.c_step
    on = st.get("c_onset")
    if on is None or step1 < on:
        return 0.0
    lvl = min((step1 - on) // max(int(cfg.c_level_steps), 1), len(lad) - 1)
    return lad[int(lvl)]


def c_trigger(cfg, st, step1, stamps_mean):
    """c_step 起征触发 (λ 时程同构; 评测时调, st 就地更新): 任一深度章数均 ≥ c_onset_stamps 连续
    c_onset_evals 评 ⇒ 起征点 = 当评步 + c_onset_buffer. 已起征后恒 False."""
    if not c_ladder_list(cfg) or st.get("c_onset") is not None:
        return False
    st["c_streak"] = (st.get("c_streak", 0) + 1) if stamps_mean >= cfg.c_onset_stamps else 0
    if st["c_streak"] >= cfg.c_onset_evals:
        st["c_onset"] = step1 + int(cfg.c_onset_buffer)
        return True
    return False


def iconic_rate(k, x1, thresh=1.0):
    """对位率 ([C] 2026-08-20 预登记仪表, 只读): 逐样本 = 落章格中「对应场景格有物体墨」的比例; 全空样本 NaN.
    场景格墨 = x1 该 8×8 格像素和 > thresh (量程 [0,64]; 物体为亮斑, 空格仅噪声 σ≈.04). k (B,T) long; x1 (B,SIDE,SIDE)."""
    B = k.shape[0]
    cell = x1.view(B, GM.G, GM.P, GM.G, GM.P).sum(dim=(2, 4)).reshape(B, GM.T)
    ink = k != R.EMPTY
    n_ink = ink.sum(-1)
    out = (ink & (cell > thresh)).sum(-1).float() / n_ink.clamp(min=1).float()
    out[n_ink == 0] = float("nan")
    return out


def rising2(hist):
    """斜率条件 ([U] 2026-08-18): 最近三评连续两次上升 (h[-3] < h[-2] < h[-1])."""
    return len(hist) >= 3 and hist[-3] < hist[-2] < hist[-1]


def update_stops(st, score_S, score_d1, exempt):
    """停跑规则 (spec §11 + [U] 2026-08-18 斜率/豁免 + v3.1 C2.2/A21): 水平判据只吃 k* 侧与 note_1 侧
    — 深度 0 (场景侧) 读出退出判定, 原「场景侧跌落」判据重基到 note_1 侧, stage1_level 同步改记
    note_1 收敛水平 ([C] 实例化, 与 C2.4 课程切换同基). 非豁免期: k* 侧 < .85 连续三评 /
    note_1 侧 < 阶段 1 收敛水平 − .10 连续两评; 连续两评上升的轨迹不计入 (斜率条件)."""
    st.setdefault("hist_S", []).append(float(score_S))
    st.setdefault("hist_d1", []).append(float(score_d1))
    st["hist_S"], st["hist_d1"] = st["hist_S"][-3:], st["hist_d1"][-3:]
    if exempt:
        st["s_notes"], st["s_d1"] = 0, 0
        return []
    low_S = score_S < STOP_NOTES and not rising2(st["hist_S"])
    st["s_notes"] = st["s_notes"] + 1 if low_S else 0
    lvl = st.get("stage1_level")
    low_d1 = lvl is not None and score_d1 < lvl - STOP_D1_DROP and not rising2(st["hist_d1"])
    st["s_d1"] = st.get("s_d1", 0) + 1 if low_d1 else 0
    hits = []
    if st["s_notes"] >= STOP_NOTES_STREAK:
        hits.append(f"notes_score<{STOP_NOTES}x{STOP_NOTES_STREAK}")
    if st["s_d1"] >= STOP_D1_STREAK:
        hits.append(f"d1_score<level-{STOP_D1_DROP}x{STOP_D1_STREAK}")
    return hits


def is_exempt(st, cfg, step1, span):
    """冷启动豁免 (spec §11) + 切换后豁免 ([U] 2026-08-18: ≥ 一个池跨度)."""
    if step1 < cfg.n_warm or st["stage"] == 1:
        return True
    grace = max(int(cfg.switch_grace), int(math.ceil(span)))
    return st.get("switch_step") is not None and step1 < st["switch_step"] + grace


def apply_switch(st, latch, step1, why):
    """课程切换的状态改写 (C2.4; boot / forced / resume 三路共用): stage 2 / k=K_MAX / 收敛水平 = 末两评 note_1 均 /
    分数史与平台窗史清零 / 闩重置. loader 重建、存档、日志等副作用由调用方执行."""
    st["stage"], st["k"] = 2, GM.K_MAX
    st["switch_step"], st["switch_why"] = step1, why
    lds = st.get("last_d1_scores") or []
    st["stage1_level"] = round(sum(lds) / len(lds), 4) if lds else None
    st["hist_S"], st["hist_d1"], st["s_notes"], st["s_d1"] = [], [], 0, 0
    st["hist_kside"], st["inflow_hist"] = [], []
    latch.reset()
    st["latch"] = latch.state()


def eval_decisions(st, cfg, latch, kside, d1, step1, span, inflow_win):
    """闩 / 课程切换 / 停跑的全部判定 (v3.1 C2.2/C2.4/A21): 输入只有 k* 侧 (kside) 与 note_1 侧 (d1)
    — 深度 0 (场景侧) 读出不出现在任何判定输入里 (checks A21 以签名 + 双调一致性验证).
    课程切换 (C2.4): note_1 与 k* 侧双双连续两评 ≥ θ_boot, 或 T_stage1 强制 (cfg.stage1_only=1 时二者皆不触发,
    [U] 2026-08-21 只跑阶段 1); 切换的状态改写 (stage/k/收敛水平/分数史清零/闩重置) 就地完成, loader 重建与存档等
    副作用由 run() 按返回执行.
    返回 dict(latched, want_switch, why, hits, exempt, pz)."""
    out = dict(latched=None, want_switch=False, why=None)
    if latch.offer(kside, step1):
        out["latched"] = latch.best
    st["latch"] = latch.state()
    if st["stage"] == 1:
        ok = d1 >= cfg.theta_boot and kside >= cfg.theta_boot
        st["boot_streak"] = st["boot_streak"] + 1 if ok else 0
        st["last_d1_scores"] = (st.get("last_d1_scores", []) + [float(d1)])[-2:]
        why = None
        if int(getattr(cfg, "stage1_only", 0)):
            why = None                    # [U] 2026-08-21 只跑阶段 1: boot/forced 均不切换 (阶段 1 全程豁免, 跑满 steps)
        elif st["boot_streak"] >= 2:
            why = "boot"
        elif step1 >= cfg.t_stage1:
            why = "forced"
        if why:
            apply_switch(st, latch, step1, why)
            out["want_switch"], out["why"] = True, why
    exempt = is_exempt(st, cfg, step1, span)
    hits = update_stops(st, kside, d1, exempt)
    pz = plateau_zero_inflow(st, cfg, kside, inflow_win, exempt)
    if pz["hit"]:
        hits.append("plateau+zero_inflow")
    out.update(hits=hits, exempt=exempt, pz=pz)
    return out


# ================================================================ 主循环 (v3.1 链路径; β/u_S/水位闸挂起, spec-v3 §8)
def run(cfg, device=None, resume=None, init=None, max_steps=None):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    assert not cfg.origin_cat, "v3.1 C1: origin 拼接已撤销 (origin_cat 恒 0)"
    assert not (int(getattr(cfg, "origin_write", 0)) and int(getattr(cfg, "use_flag", 0))), \
        "origin_write=1 与 A18 兼容路 (use_flag=1) 互斥"
    assert not (cfg.gate_on and not cfg.pred_on), "gate_on 要求 pred_on=1 (闸吃 Δ̂)"
    add_on = bool(int(getattr(cfg, "add_on", 0)))
    if add_on:                                                     # [U] 2026-08-27 加法流
        assert not int(getattr(cfg, "use_flag", 0)) and not int(getattr(cfg, "origin_write", 0)), \
            "加法流两槽装配与 use_flag 兼容路 / origin_write 互斥"
        assert not int(getattr(cfg, "exposure", 0)) and not int(getattr(cfg, "gen_relearn", 0)), "加法流与迭代重学互斥 (本轮不并用)"
        assert not cfg.gate_on, "加法纸入池恒取最深写步 (无 k* 早停抽样); gate_on=1 与 add_on=1 同开需另接 (评审 #2)"
        assert int(cfg.b_add) > 0
    if cfg.beta or cfg.beta_frac:
        print("[§8] β/u_S/水位闸挂起: beta/beta_frac 强制 0 (旋钮保留, 训练路径不走)", flush=True)
        cfg.beta, cfg.beta_frac = 0.0, 0.0
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kits = KitMaker(cfg.kit_workers, cfg.kit_chunk)  # 在 CUDA 初始化前建进程池 (fork 安全)
    model = build_model(cfg).to(dev)
    params = model.trainable_params()
    if int(getattr(cfg, "alpha_freeze", 0)):                       # [U] 2026-08-22 冻结 α: 反传后 α.grad 置 None ⇒ 裁剪与 AdamW 均跳过
        assert int(getattr(cfg, "d1_cm", 0)), "alpha_freeze 要求 d1_cm=1 (α 只在共模写头里)"
        model.D1.alpha.register_post_accumulate_grad_hook(lambda p: setattr(p, "grad", None))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    tw = TaskWeights(mul=task_wmul_list(cfg))                      # [U] 2026-08-23 逐任务损失倍率 (缺省不乘)
    pool = DataPool(cfg.pool_cap)
    const = ConstPred()
    sscale = ST.SoftScale(cfg.s_ema_m, cfg.s_clamp, cfg.s_init_steps)
    st = new_state(cfg)
    latch = BestLatch()
    servo = LamServo(cfg) if int(getattr(cfg, "lam_servo", 0)) else None     # [U] 2026-08-21 λ PI 伺服
    assert not (servo is not None and lam_ladder_list(cfg)), "lam_servo 与 lam_ladder 互斥 (λ 不设定值)"
    rl = None
    if int(getattr(cfg, "exposure", 0)):
        from . import relearn as RL
        assert not int(getattr(cfg, "use_flag", 0)), "exposure 与 A18 兼容路 (use_flag) 互斥"
        rl = RL.Relearn(cfg, dev)
    grl = None
    if int(getattr(cfg, "gen_relearn", 0)):
        from . import relearn as RLG
        assert not int(getattr(cfg, "exposure", 0)), "gen_relearn 与 exposure (甲案) 互斥"
        assert not int(getattr(cfg, "use_flag", 0)), "gen_relearn 与 A18 兼容路互斥"
        assert not int(getattr(cfg, "chain_from_pool", 0)), "乙案要求 chain_from_pool=0 (池 gen 字段 = 哪一代写的)"
        assert not int(getattr(cfg, "d1_learn", 0)), "乙案 W_D1 冻结 ([U] 2026-08-24)"
        assert not int(getattr(cfg, "origin_write", 0)), "乙案模仿流 = 单槽写路径; origin_write=1 (两槽) 与 gen_relearn 互斥 (评审 #2)"
        grl = RLG.GenRelearn(cfg, dev)
    step0 = 0
    if init:
        ck = load_ckpt(init, model)
        print(f"[init] weights from {init} (step {ck['step']})", flush=True)
    d1_hook = None
    if resume:
        if int(getattr(cfg, "d1_learn", 0)):                       # 解冻检查点 (opt 已带 W_D1 组): 先建组再载, 动量随载
            _og = torch.load(resume, map_location="cpu", weights_only=True).get("opt")
            if _og and len(_og["param_groups"]) == len(opt.param_groups) + 1:
                d1_hook = d1_learn_setup(model, opt, cfg)
            del _og
        ck = load_ckpt(resume, model, opt, tw, pool)
        st = ck["st"]
        if ck.get("const"):
            const.load_state(ck["const"])
        if ck.get("sscale"):
            sscale.load_state(ck["sscale"])
        if servo is not None and st.get("lam_servo"):
            servo.load_state(st["lam_servo"])
        latch.load_state(st["latch"])
        if rl is not None and st.get("relearn"):
            rl.load_state(st["relearn"])
            print(f"[rl] resume 期 {rl.period} steps_in {rl.steps_in} 𝒩_exp={list(rl.exp)}", flush=True)
        if grl is not None and st.get("genrl"):
            grl.load_state(st["genrl"])
            print(f"[gen] resume 代 {grl.period + 1} steps_in {grl.steps_in} 𝒩_exp={list(grl.exp)}", flush=True)
        step0 = int(ck["step"])
        print(f"[resume] {resume} @ step {step0} stage {st['stage']} k {st['k']} pool {pool.size}", flush=True)
    d1_ref = None
    if int(getattr(cfg, "d1_learn", 0)):                           # [U] 2026-08-23 W_D1 解冻 (新解冻/旧检查点: 载后建组)
        if d1_hook is None:
            d1_hook = d1_learn_setup(model, opt, cfg)
        if int(getattr(cfg, "d1_rownorm", 1)):
            d1_rownorm_(model)                                     # 载入后归一 (首次 = 单位化尺度跳变, 起点复评量化; 续跑幂等)
        d1_ref = model.D1.lin.weight.detach().clone()              # 本进程漂移基准 (逐评 1−cos 入册)
    switched_at_resume = False
    if int(getattr(cfg, "switch_at_resume", 0)):                   # [U] 2026-08-22「延用 B 臂设置，切阶段 2」: 续跑第一步前当场切换
        assert not int(getattr(cfg, "stage1_only", 0)), "switch_at_resume 与 stage1_only 互斥"
        if resume and st["stage"] == 1:
            apply_switch(st, latch, step0, "resume")
            switched_at_resume = True
            print(f"[stage] 切大 N 阶段 @ {step0} (resume); note_1 侧收敛水平 {st['stage1_level']}", flush=True)
    salt = RESUME_SALT * step0 if step0 else 0
    es_by_k = {cfg.k1: build_eval_sets(cfg, cfg.k1), GM.K_MAX: build_eval_sets(cfg, GM.K_MAX)}
    loader = make_scene_loader(st["k"], cfg.seed + 1 + 1000 * st["k"] + salt, cfg.b_scene, cfg.workers)
    it = iter(loader)
    esa_by_k = it_add = None
    if add_on:                                                     # 加法流: 自己的评测集 / 装载器 / 随机流 (主流程各随机流不动)
        esa_by_k = {cfg.k1: build_add_eval_sets(cfg, cfg.k1), GM.K_MAX: build_add_eval_sets(cfg, GM.K_MAX)}
        loader_add = make_add_loader(st["k"], cfg.seed + 17 + 1000 * st["k"] + salt, cfg.b_add, cfg.add_workers)
        it_add = iter(loader_add)
    rng_add = torch.Generator().manual_seed(cfg.seed + 63 + salt)      # 加法链信道 + 加法纸入池 (add_on=0 时不耗用)
    rng = torch.Generator().manual_seed(cfg.seed + 31 + salt)          # 信道抽样 (训练; C7 每链一份)
    rng_pool = torch.Generator().manual_seed(cfg.seed + 77 + salt)     # 池抽取 / k* 抽样 / 入池
    rng_mask = torch.Generator().manual_seed(cfg.seed + 55 + salt)     # 随机画布
    rng_c = torch.Generator().manual_seed(cfg.seed + 91 + salt)        # 逐评读数 (零假设当评重算)
    log = open(os.path.join(cfg.out, "log.jsonl"), "a")
    _log(log, dict(step=step0, run_start=dict(cfg=dataclasses.asdict(cfg), resume=bool(resume),
                                                init=bool(init), stage=st["stage"], k=st["k"])))
    if switched_at_resume:
        save_ckpt(os.path.join(cfg.out, "ckpt_stage1_end.pt"), model, opt, tw, step0, cfg, st, pool, const, sscale)
        _log(log, dict(step=step0, resume_switch=True,
                       switch=dict(why="resume", stage1_level=st["stage1_level"], k=st["k"])))
    print(f"[run] dev={dev} steps={cfg.steps} stage={st['stage']} k={st['k']} K={cfg.chain_k} "
          f"origin_write={int(getattr(cfg, 'origin_write', 0))} "
          f"pred={cfg.pred_on} ptb={cfg.pred_to_backbone} cfp={cfg.chain_from_pool} gate={cfg.gate_on} "
          f"λ={cfg.lam} s={cfg.s} occ_k={cfg.occ_k} B={cfg.b_scene}/{cfg.b_pool}/{cfg.b_mask} "
          f"pool cap={cfg.pool_cap} ρ={cfg.rho_admit} ρ_gen={cfg.rho_admit_gen} out={cfg.out}"
          + (f" add_on=1 B_add={cfg.b_add} add_workers={cfg.add_workers}" if add_on else ""), flush=True)
    if (servo is not None or int(getattr(cfg, "d1_cm", 0)) or int(getattr(cfg, "stage1_only", 0))
            or int(getattr(cfg, "switch_at_resume", 0)) or int(getattr(cfg, "alpha_freeze", 0))
            or int(getattr(cfg, "d1_learn", 0))):
        print(f"[cm] d1_cm={cfg.d1_cm} (写头共模扣除: α 初 {float(model.D1.alpha) if cfg.d1_cm else None}) "
              f"lam_servo={cfg.lam_servo} (目标章数 {cfg.lam_target}, k_p {cfg.lam_kp}, k_i {cfg.lam_ki}, EMA {cfg.lam_ema}, "
              f"λ₀ {servo.lam if servo is not None else None}, 夹 [{cfg.lam_lo},{cfg.lam_hi}]) stage1_only={cfg.stage1_only} "
              f"switch_at_resume={cfg.switch_at_resume} alpha_freeze={cfg.alpha_freeze}",
              flush=True)
    if d1_ref is not None:
        print(f"[d1] d1_learn=1 lr={cfg.lr * float(cfg.d1_lr_mult):.2e} (= {float(cfg.d1_lr_mult)} × 主 lr {cfg.lr:.2e}) "
              f"rownorm={int(getattr(cfg, 'd1_rownorm', 1))} 行范数={[round(float(x), 6) for x in model.D1.lin.weight.detach().norm(dim=1).cpu()]} "
              f"参数组={len(opt.param_groups)}", flush=True)
    if tw.mul is not None:
        print(f"[tw] task_wmul={[float(x) for x in tw.mul]} (损失权重 = 归一 w × 倍率; 池成绩 σ 仍用归一 w) "
              f"起步 w={[round(float(x), 4) for x in tw.weights()]} w_norm={[round(float(x), 4) for x in tw.weights_norm()]}",
              flush=True)
    rho_g = float(cfg.rho_admit_gen) if cfg.chain_from_pool else 0.0
    rho_a = cfg.rho_admit * cfg.b_add if add_on else 0.0           # 加法纸入流 (同 ρ_admit)
    span = cfg.pool_cap / max(cfg.rho_admit * cfg.b_scene + rho_g * cfg.b_pool + rho_a, 1e-9)
    print(f"[pool] 时间跨度 ≥ C_pool/(ρ·B_scene + ρ_gen·B_pool{' + ρ·B_add' if add_on else ''}) = {span:.0f} 步 "
          f"(C4.3 {'三' if add_on else '两'}流合计; 实测值逐评入册)", flush=True)
    if rl is not None:
        print(f"[rl] exposure=1 (甲案孪生新读者; [U] 2026-08-24) e_exp={cfg.e_exp} t_period={cfg.t_period} "
              f"η_tr={cfg.eta_tr} warm={cfg.tr_warm} sr_lr={cfg.sr_lr} n_s={cfg.n_s} "
              f"b_teach={cfg.b_teach}+{cfg.b_teach_sc}sc (阶段 2 激活; 阶段 1 挂空转)", flush=True)
    if grl is not None:
        print(f"[gen] gen_relearn=1 (乙案换代重学; [U] 2026-08-24) t_gen={cfg.t_gen} e_exp={cfg.e_exp} "
              f"eta_im={cfg.eta_im} b_im={cfg.b_im} b_poolread={cfg.b_poolread} keep_cnn={cfg.gen_keep_cnn} "
              f"(阶段 2 激活; 期界 = 重置(眼承袭)+优化器清零+子集重抽; 池不清空)", flush=True)

    def kit_sync(ns):
        return kits.make_sync(ns, st["k"], _seed_mix(cfg.seed + 5, st["evals"] + 12345))

    win = WinAgg()
    ewin = WinAgg()
    pending_idx = None
    end = cfg.steps if max_steps is None else min(cfg.steps, step0 + max_steps)
    stop = None
    last_sb = last_pb = None
    step = step0 - 1
    K = int(cfg.chain_k)
    for step in range(step0, end):
        t0 = time.time()
        if rl is not None and st["stage"] == 2:
            row_rl = rl.tick(st["k"], step + 1)
            if row_rl is not None:
                _log(log, dict(step=step + 1, relearn=row_rl))
                print(f"[rl] 期 {row_rl['relearn_period']} @ {step + 1} 𝒩_exp={row_rl['exp']}", flush=True)
        if grl is not None and st["stage"] == 2 and grl.needs_rollover(st["k"]):
            if grl.period >= 0:                            # 上一代收官检查点 (§6.3乙 判据轴 1 逐代码样本)
                save_ckpt(os.path.join(cfg.out, f"ckpt_gen{grl.period + 1}_end.pt"),
                          model, opt, tw, step + 1, cfg, st, pool, const, sscale)
            row_g = grl.rollover(model, st["k"], step + 1)
            opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)   # 期界优化器清零
            st["genrl"] = grl.state()
            _log(log, dict(step=step + 1, genrl=row_g))
            print(f"[gen] 代 {row_g['gen']} @ {step + 1} 𝒩_exp={row_g['exp']} reset={row_g['reset']}", flush=True)
        items = next(it)
        sb = build_batch(items, dev)
        ab = build_batch(next(it_add), dev) if (add_on and step + 1 > int(cfg.add_start)) else None   # 加法流批 (x1 = 场景 N, x_aux = 场景 m, ns = s); add_start 前不抽
        pb = None
        if pending_idx is not None:
            kit_list = kits.collect()
            draw_p = R.draw_channel(pending_idx.shape[0], cfg.s, rng, dev, cfg.occ_k)   # C7: 池链整链一份 (含 x0 重渲)
            pb = ST.render_pool_batch(pool, pending_idx, kit_list, cfg, rng, dev, draw=draw_p)
            pb["draw"], pb["tpl"] = draw_p, R.templates(draw_p.u, draw_p.s)
            pb["pgen"] = pool.gens(pending_idx)                        # 亲本 generation (逐出前取定)
            pb["pzseed"] = pool.zseed[pending_idx.cpu()].clone()       # 亲本源场景种子 (谱系)
        mb = ST.make_mask_batch(cfg.b_mask, cfg, rng_mask, dev)
        t_data = time.time() - t0
        w = tw.weights(dev)                            # 损失权重 (含倍率); 池成绩 σ 用归一 w (倍率缺省时二者同一张量)
        w_pool = w if tw.mul is None else tw.weights_norm(dev)
        cfg.lam = lam_at(cfg, st, step + 1)            # 时程表 λ (lam_ladder 空时 = 自赋值, 无副作用)
        if servo is not None:
            cfg.lam = servo.lam                        # [U] 2026-08-21 伺服 λ (上一步更新后的值; 首步 = λ₀)
        cfg.c_step = c_at(cfg, st, step + 1)           # 时程表 c_step (闸关不启用; 空表 = 自赋值)
        gen_warm_now = bool(grl is not None and st["stage"] == 2 and grl.bank is not None and grl.in_warm)
        lam_true = cfg.lam                         # lam_at 无阶梯 = 自赋值恒等 ⇒ 置 0 会经时程永久粘住; 豁免只限本步
        if gen_warm_now:
            cfg.lam = 0.0                          # 乙案期首 λ 豁免 (gen_warm; 婴儿期不因写墨挨罚)
        m = ST.chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, const=const, sscale=sscale,
                                w_pool=w_pool,
                                relearn=(rl if (rl is not None and st["stage"] == 2) else None),
                                genrl=(grl if (grl is not None and st["stage"] == 2) else None), pool=pool,
                                ab=ab, rng_add=rng_add)
        if gen_warm_now:
            cfg.lam = lam_true                     # 步后还原 (防自赋值时程粘 0; 探针核证 = 豁免带出口 L_lam > 0)
        if servo is not None:
            m["lam"] = cfg.lam                         # 本步实际所用 λ (窗均 5 位; 精确值见评测行 servo 块)
            m["log10_lam"] = math.log10(cfg.lam)
            servo.update(m["stamps"])                  # 被控量 = 本步场景链末写章数均; 更新下一步 λ
            st["lam_servo"] = servo.state()
            m["stamps_ema"] = servo.s_ema
        if int(getattr(cfg, "d1_cm", 0)):
            m["alpha"] = float(model.D1.alpha.detach())              # 共模写头 5 参数轨迹 (窗均入日志)
            m["bias_c"] = [float(x) for x in model.D1.bias_c.detach().cpu()]
        if d1_ref is not None:                                       # W_D1 解冻: 步梯度范数 (窗均入日志; 步后 grad 尚在)
            _gw = model.D1.lin.weight.grad
            m["d1_gnorm"] = float(_gw.norm()) if _gw is not None else 0.0
        ks_maps = m.pop("ks")
        ks_maps_pool = m.pop("ks_pool")
        ks_maps_add = m.pop("ks_add")
        lam_rows = m.pop("lam_rows")
        lam_rows_pool = m.pop("lam_rows_pool")
        notes_x = m.pop("notes_x", None)
        if not math.isfinite(m["L"]):
            stop = "nan_or_diverge"
            _log(log, dict(step=step, stop=stop, L=m["L"]))
            print(f"[STOP] {stop} @ {step}", flush=True)
            break
        tw.update(m["accs_notes"])                                  # w_t ∝ 1−acc̄_t (场景末纸 + 池笔记本体)
        # ---- 池记账: 先更新已抽条目成绩 (w_t 加权, C4.5), 再两流入池 (C4.3)
        if pb is not None:
            pool.update_scores(pb["idx"], m["pool_accs"], cfg.eta_pool)
        B = sb["B"]
        if cfg.gate_on:
            lam_pol = ST.policy_lam(lam_rows, cfg, rng_pool)        # 对照臂策略 (learned 透传, 不耗 rng)
            ks_tr = ST.kstar_sample(lam_pol, rng_pool, k0_gated=False)
        else:
            ks_tr = torch.full((B,), K, dtype=torch.long)           # 闸关 = 恒继续 (C3)
        m["inflow"] = float((ks_tr >= 1).float().mean())            # 场景侧入池率 Pr[k*≥1] (C2 下恒 1)
        m["kstar_train"] = float(ks_tr.float().mean())
        adm = (torch.rand(B, generator=rng_pool) < cfg.rho_admit) & (ks_tr >= 1)
        if gen_warm_now:
            adm = torch.zeros_like(adm)            # 乙案期首禁入池 (婴儿涂鸦不进谱系)
            m["gen_warm_now"] = 1.0
        kstack = torch.stack(ks_maps)                               # (K,B,T) cpu
        sel = kstack[(ks_tr - 1).clamp(min=0), torch.arange(B)]
        gtag = None
        if grl is not None and st["stage"] == 2 and grl.bank is not None:
            gtag = torch.full((B,), grl.period + 1, dtype=torch.long)
        pool.admit(sel, sb["ns"].cpu(), sb["zseed"], step, mask=adm, depths=ks_tr, gens=gtag)
        if ab is not None and ks_maps_add is not None:              # 加法纸入池 ([U] 流程同主流程): 同 ρ_admit, 标签 s=N+m, 深度 K, gen 0
            Ba = ab["B"]
            adm_a = torch.rand(Ba, generator=rng_add) < cfg.rho_admit
            if gen_warm_now or not int(getattr(cfg, "add_admit", 1)):
                adm_a = torch.zeros_like(adm_a)
            pool.admit(ks_maps_add[-1], ab["ns"].cpu(), ab["zseed"], step, mask=adm_a,
                       depths=torch.full((Ba,), K, dtype=torch.long))
            m["inflow_add"] = float(adm_a.float().mean())
        m["inflow_gen"] = 0.0
        if pb is not None and ks_maps_pool is not None:             # C4.3b: 池链产物, gen = 亲本+1
            Bp = pb["B"]
            if cfg.gate_on:
                lam_pp = ST.policy_lam(lam_rows_pool, cfg, rng_pool)
                ks_pp = ST.kstar_sample(lam_pp, rng_pool, k0_gated=True)
            else:
                ks_pp = torch.full((Bp,), K, dtype=torch.long)
            gens_child = pb["pgen"] + 1
            adm_p = (torch.rand(Bp, generator=rng_pool) < cfg.rho_admit_gen) & (ks_pp >= 1)
            if int(cfg.g_max) > 0:
                adm_p &= gens_child <= int(cfg.g_max)
            kstack_p = torch.stack(ks_maps_pool)
            sel_p = kstack_p[(ks_pp - 1).clamp(min=0), torch.arange(Bp)]
            pool.admit(sel_p, pb["ns"].cpu(), pb["pzseed"], step, mask=adm_p, depths=ks_pp, gens=gens_child)
            m["inflow_gen"] = float(adm_p.float().mean())
            m["kstar_pool_train"] = float(ks_pp.float().mean())
        if rl is not None and st["stage"] == 2 and notes_x is not None:
            m.update(rl.teach(sb, notes_x, pool))
        pending_idx = None
        if pool.size >= cfg.b_pool:
            band_w = 1 if st["stage"] == 1 else int(cfg.strat_band)
            pending_idx = pool.sample_stratified(cfg.b_pool, cfg.t_pool, cfg.eps_pool, rng_pool, band_w=band_w)
            kits.submit(pool.labels(pending_idx).tolist(), st["k"], _seed_mix(cfg.seed + 3, step + salt))
        m["pool_size"] = pool.size
        m["t_step"] = time.time() - t0
        m["t_data"] = t_data
        win.add(m)
        ewin.add(m)
        last_sb, last_pb = sb, pb
        if (step + 1) % cfg.log_every == 0:
            row = dict(step=step + 1, stage=st["stage"], k=st["k"], **win.mean())
            row["w"] = [round(float(x), 4) for x in tw.weights()]
            if tw.mul is not None:
                row["w_norm"] = [round(float(x), 4) for x in tw.weights_norm()]
            _log(log, row)
            win.reset()
        # ---- 评测 (v3 §11 + v3.1 C10)
        if (step + 1) % cfg.eval_every == 0 or step + 1 == end:
            es = es_by_k[st["k"]]
            rng_eval = torch.Generator().manual_seed(cfg.seed + 4242)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            model.eval()
            ev = dict(step=step + 1, stage=st["stage"], k=st["k"], eval=True)
            ev.update(eval_groups_chain(model, es, cfg, dev, w, rng_eval, const=const, sscale=sscale))
            band_w = 1 if st["stage"] == 1 else int(cfg.strat_band)
            ev["pool"] = eval_pool_chain(model, pool, cfg, dev, w, rng_eval, kit_sync, const=const,
                                         sscale=sscale, band_w=band_w)
            ev["mask"] = eval_mask(model, es, cfg, dev, rng_eval)
            if add_on:                                             # 加法流评测 (自己的 CRN 流, 主评测随机流不动)
                ev["add"] = eval_add_chain(model, esa_by_k[st["k"]], cfg, dev, w,
                                           torch.Generator().manual_seed(cfg.seed + 4244))
            tab, med = code_table_chain(model, es, dev, cfg)
            ev["table_med"] = med
            ev["window"] = ewin.mean()
            ewin.reset()
            if last_sb is not None:                                # §8 只读读数, 逐深度分列
                ev["readings"] = ST.eval_readings_chain(model, opt, params, last_sb, last_pb, cfg, w, dev,
                                                        rng_c, st["k"])
            model.train()
            ev["pool_stats"] = pool.stats(cfg.t_pool, cfg.eps_pool, step=step + 1, rng=rng_eval)
            ev["pool_span"] = span
            adm_d = pool.n_admit - st.get("last_n_admit", 0)       # C10.8 池跨度实测 (两流合计入池率)
            st["last_n_admit"] = pool.n_admit
            ev["adm_window"] = int(adm_d)
            ev["span_meas"] = round(cfg.pool_cap / max(adm_d / max(cfg.eval_every, 1), 1e-9))
            if rl is not None and st["stage"] == 2 and rl.reader is not None:
                ev["tr"] = rl.exam(model, es, cfg, w, dev, rng_eval)
                st["relearn"] = rl.state()
            if grl is not None and st["stage"] == 2 and grl.bank is not None:
                ev["gen"] = grl.exam(model, es, cfg, w, dev, rng_eval)
                st["genrl"] = grl.state()
            ev["w"] = [round(float(x), 4) for x in tw.weights()]
            if tw.mul is not None:
                ev["w_norm"] = [round(float(x), 4) for x in tw.weights_norm()]
                ev["task_wmul"] = [float(x) for x in tw.mul]
            if d1_ref is not None:                                 # W_D1 逐评漂移: 逐行 1−cos(当前, 本进程起点) + 行范数
                _W = model.D1.lin.weight.detach()
                _cos = (_W * d1_ref).sum(1) / (_W.norm(dim=1) * d1_ref.norm(dim=1)).clamp_min(1e-12)
                ev["d1"] = dict(drift=[round(float(1.0 - c), 6) for c in _cos.cpu()],
                                wnorm=[round(float(x), 6) for x in _W.norm(dim=1).cpu()])
            st["evals"] += 1
            score_S = ev["notes"]["score"]                          # k* 侧 = 交付 (闩看的量, C2.2)
            score_d1 = ev["depths"][0]["score"]                     # note_1 侧 (课程/停跑的另一半, C2.4)
            score_sc = ev["scene"]["score"]                         # 场景侧: 只入册/打印 (感知上界量尺), 不进判定 (A21)
            stamps_any = max(d["stamps_mean"] for d in ev["depths"])
            if lam_trigger(cfg, st, step + 1, stamps_any):
                ev["lam_onset"] = st["lam_onset"]
                print(f"[lam] 起征确认 @ {step + 1}; λ 时程自 {st['lam_onset']} 起 (级距 {cfg.lam_ladder})", flush=True)
            ev["lam_now"] = cfg.lam
            if servo is not None:
                ev["servo"] = servo.readings()                         # 全精度: lam / s_ema / e / loglam / n
            if int(getattr(cfg, "d1_cm", 0)):
                ev["cm"] = dict(alpha=float(model.D1.alpha.detach()),
                                bias_c=[float(x) for x in model.D1.bias_c.detach().cpu()],
                                frozen=bool(int(getattr(cfg, "alpha_freeze", 0))))
            if c_trigger(cfg, st, step + 1, stamps_any):
                ev["c_onset"] = st["c_onset"]
                print(f"[c_step] 起征确认 @ {step + 1}; c_step 时程自 {st['c_onset']} 起 (级距 {cfg.c_ladder})", flush=True)
            ev["c_step_now"] = cfg.c_step
            # ---- 闩 / 课程 / 停跑判定 (A21: 只吃 k* 侧 + note_1 侧)
            inflow_win = ev["window"].get("inflow", 1.0)
            dec = eval_decisions(st, cfg, latch, score_S, score_d1, step + 1, span, inflow_win)
            if dec["latched"] is not None:
                ev["latched"] = dec["latched"]
                save_ckpt(os.path.join(cfg.out, "ckpt_best.pt"), model, opt, tw, step + 1, cfg, st, pool,
                          const, sscale)
            if dec["want_switch"]:
                save_ckpt(os.path.join(cfg.out, "ckpt_stage1_end.pt"), model, opt, tw, step + 1, cfg, st,
                          pool, const, sscale)
                ev["switch"] = dict(why=dec["why"], stage1_level=st["stage1_level"], k=st["k"])
                del it
                loader = make_scene_loader(st["k"], cfg.seed + 1 + 1000 * st["k"] + salt + step,
                                           cfg.b_scene, cfg.workers)
                it = iter(loader)
                if add_on:
                    del it_add
                    loader_add = make_add_loader(st["k"], cfg.seed + 17 + 1000 * st["k"] + salt + step,
                                                 cfg.b_add, cfg.add_workers)
                    it_add = iter(loader_add)
                print(f"[stage] 切大 N 阶段 @ {step + 1} ({dec['why']}); note_1 侧收敛水平 {st['stage1_level']}", flush=True)
            hits = dec["hits"]
            ev["plateau"] = dec["pz"]
            # ---- 入池率报警 (§6.2: 连续两评 <0.05 报警入册, 不自动动作; C2 下场景流恒 1, 结构性不触发)
            st["low_inflow"] = st.get("low_inflow", 0) + 1 if inflow_win < 0.05 else 0
            if st["low_inflow"] >= 2:
                ev["alarm_low_inflow"] = True
            ev["stop_hits"] = hits
            ev["exempt"] = dec["exempt"]
            ev["stop_state"] = dict(s_notes=st["s_notes"], s_d1=st["s_d1"], hist_S=st["hist_S"],
                                    hist_d1=st["hist_d1"], plateau=dec["pz"]["plateau"],
                                    zero_inflow=dec["pz"]["zero_inflow"])
            _log(log, ev)
            dsc = "/".join(f"{d['score']:.3f}" for d in ev["depths"])
            pr = ev.get("pred") or {}
            c3 = (pr.get("corr3") or {}).get("dt") or {}
            print(f"[eval] step {step + 1} stage {st['stage']} k {st['k']} 场景 {score_sc} 逐深度 {dsc} "
                  f"k*侧 {score_S} 池 {ev['pool'] and ev['pool']['score']} 章数均 {ev['notes']['stamps_mean']} "
                  f"d_const {pr.get('d_const')} Δ̂-Δ_true r {c3.get('r')}/界 {c3.get('r_thresh')} "
                  f"ρ(秩) {c3.get('rho')} 符号 {c3.get('sign')} cosĝ {pr.get('cos_fwd', {}).get('mean')} "
                  f"基线 {pr.get('cos_base', {}).get('mean')} ham12 {ev['ham12']['med'] if ev.get('ham12') else None}"
                  + (f"(均 {ev['ham12']['mean']} 非零链 {ev['ham12']['nz_frac']}) " if ev.get('ham12') else " ")
                  + "Δtrue " + "/".join(f"{d['mean']:+.4f}±{d['se']:.4f}(lo95 {d['lo95']:+.4f})" for d in ev["dtrue"]) + " "
                  + f"gen最大 {ev['pool_stats'].get('gen_max')} 闩 {latch.best}@{latch.step}"
                  + (f" λ {cfg.lam:.3g} ŝ {servo.s_ema:.2f} e {servo.error():+.2f}" if servo is not None and servo.s_ema is not None else "")
                  + (f" α {float(model.D1.alpha):.4f} b_c {[round(float(x), 3) for x in model.D1.bias_c.detach().cpu()]}"
                     if int(getattr(cfg, 'd1_cm', 0)) else "")
                  + (f" t4CE 场景 {ev['scene']['ce']['t4']} 纸 {ev['notes']['ce']['t4']} 池 {ev['pool'] and ev['pool']['ce']['t4']}"
                     f" t4acc 场景 {ev['scene']['acc']['t4']} 纸 {ev['notes']['acc']['t4']} w {ev['w']}"
                     if tw.mul is not None else "")
                  + (f" tr期{ev['tr']['period']} 未曝纸 {ev['tr']['unexp'] and ev['tr']['unexp']['mean']}"
                     f"/t246 {ev['tr']['unexp'] and ev['tr']['unexp']['t246']}"
                     f" 曝纸 {ev['tr']['exp_set'] and ev['tr']['exp_set']['mean']}"
                     f" 未曝场景 {ev['tr']['scene_unexp'] and ev['tr']['scene_unexp']['mean']}"
                     if ev.get("tr") else "")
                  + (f" | 加法 场景 {ev['add']['scene']['score']} 纸 {ev['add']['notes']['score']}"
                     f" 单场景零假设 {ev['add']['null_single']['score']}"
                     f" 直读s/N {ev['add']['notes']['read']['tol0']}/{ev['add']['notes']['read_vs_N']['tol0']}"
                     f" 章数 {ev['add']['notes']['stamps_mean']}"
                     if ev.get("add") else ""), flush=True)
            if ev.get("d1") is not None:
                print(f"[d1] step {step + 1} drift {ev['d1']['drift']} wnorm {ev['d1']['wnorm']} "
                      f"gnorm窗均 {round(ev['window'].get('d1_gnorm', 0.0), 6)}", flush=True)
            final_now = (step + 1 == end) or (bool(hits) and not dec["exempt"])
            if (step + 1) % cfg.viz_every == 0 or final_now:
                torch.save(dict(step=step + 1, k=st["k"], table=tab), os.path.join(cfg.out, "code_table.pt"))
                if final_now:
                    sub = tab
                else:
                    ns16 = sorted(tab[1])[:16]
                    sub = {d: {n: tab[d][n] for n in ns16} for d in tab}
                r6_viz_chain(es, sub, os.path.join(cfg.out, f"notes_{step + 1}"), per_page=16)
                if add_on:                                         # 加法流 R6 配对图 [场景 N | 场景 m | 纸]
                    add_viz(model, esa_by_k[st["k"]], dev, cfg, os.path.join(cfg.out, f"notes_add_{step + 1}.png"))
            save_ckpt(os.path.join(cfg.out, "ckpt_last.pt"), model, opt, tw, step + 1, cfg, st, pool,
                      const, sscale)
            if (step + 1) % cfg.ckpt_every == 0:
                save_ckpt(os.path.join(cfg.out, f"ckpt_{step + 1}.pt"), model, opt, tw, step + 1, cfg, st,
                          pool, const, sscale)
            if hits and not dec["exempt"]:
                stop = ",".join(hits)
                _log(log, dict(step=step + 1, stop=stop))
                print(f"[STOP] {stop} @ {step + 1}", flush=True)
                break
    fin = dict(step=step + 1, final=True, stop=stop, best=latch.state(),
               stage=st["stage"], k=st["k"], switch=dict(step=st["switch_step"], why=st["switch_why"],
                                                          stage1_level=st["stage1_level"]),
               pool=pool.stats(cfg.t_pool, cfg.eps_pool))
    _log(log, fin)
    log.close()
    kits.close()
    print(f"[final] {json.dumps(fin, ensure_ascii=False)}", flush=True)
    return fin
