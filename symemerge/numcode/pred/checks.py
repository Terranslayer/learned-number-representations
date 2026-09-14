# symemerge/numcode/pred/checks.py
"""接线断言 A1–A29 (spec-v2 §10 + spec-v3 §10 + spec-v3.1 C9 + [U] 2026-08-21 共模写头 + [U] 2026-08-23 写路径 origin 槽
+ [U] 2026-08-23 W_D1 解冻, 零成本, 每次实现变更后跑).
每条返回 (PASS/FAIL/N-A, 证据 dict); `scripts/nc_pred2.py --check` 全跑并逐条打印. 也被 tests 调用.

A1  W_D1 任意步前后逐位不变, 且不在 optimizer 参数组内 (d1_learn=0 缺省路; 解冻路的对应断言 = A29)
A2  直通前向恒等: x2 与纯硬渲染逐位相同
A3  留出 N 从不出现在任何 batch, 池中无留出 N 条目
A4  标记 N 不出现在任何前向张量
A5  六任务头与直读头的输入只有 h 与 φ(θ)
A6  随机画布不经 D1、不进池、不走六任务头与直读头
A7  u_S 精确 vjp — spec-v3 §8 挂起: 只验证休眠的 v2 路径原样完好
A10 墨水成本口径 (§5.2'): 全空零成本零梯度; 落章画布梯度只在章格
A11 真实成绩/标记不进任何前向张量: 换标签重跑链, V̂/Ŵ/λ/各深度画布逐位不变
A12 逐深度直通恒等 (v3.1 C7 下与共享 draw 复核 = A24 前半)
A13 随机画布不进链: 不产生 V̂/g/闸/D1 的任何张量
A14 训练期链恒跑满 K; 评测端 k* 截断 (C3 闸位: 场景链最小 1, 池链可 0)
A15 v3.1 C1 撤销 origin 拼接 ⇒ 本跑 N/A (tok(x0) 复用逻辑已删)
A16 预测梯度不进主干 (ptb=0): L_pred = BCE+η_fwd·L_fwd (场景链+池链) 对 Θ_E/任务头/嵌入表逐位零,
    对 V̂ 头与 g 非零
A17 闸梯度只到 (a,b): L_gate 对 (a,b) 非零, 对其余全部参数逐位为零
A18 K=1 平价 (重构回归): 兼容配置 ⇒ 与 v2 train_step 一步逐位相同 (v3.1 交付口径改变 ⇒ 本版为
    新基准, 不与 v3 期锚逐位比对; v2 参照物本身未动)
A19 y_soft 对 Θ_E 与六任务头的梯度逐位为零 (C9 检法: 冻结 V̂、只留 L_pred 跑一步, Θ_E.grad 全零)
A20 L_fwd 对 h[k+1] 一路的梯度逐位为零, 只经 ĥ[k+1] 反传 (ptb=1 下验证: 末态 h 纯目标端 ⇒ 零)
A21 深度 0 读出不出现在闩、停跑判据、课程切换的任何计算里 (eval_decisions 签名无场景参数 + 双调一致)
A22 每条场景链恒执行 k=0 的写; 场景起链不存在 k* = 0 的记录
A23 入池只做 append, 逐出只按年龄; 亲本条目在其产物入池后仍在池中; 池大小恒 ≤ C_pool
A24 同一链内各深度 corruption 参数逐位相同, 不同链之间不同 (步间独立重抽)
A25 不存在独立的 Ŵ 参数; Ŵ 的计算图必经 V̂ 头与 g
A26 ([C] 原步 4 A19 改号) ptb=1 ⇒ L_pred (BCE 与 L_fwd 两项) 梯度到达 Θ_E 与 φ(θ) 嵌入表;
    同配置 A17 仍成立; gate_mode const/rand 臂不产生 L_gate
A27 ([U] 2026-08-21 写头共模扣除) d1_cm=1 ⇒ 初值 α=1/b_c=0 且写头输出逐位 = W_D1(z − z̄) (旧冻结偏置不参与);
    W_D1 与冻结偏置不在 optimizer、一步前后逐位不变 (A1 同口径); α/b_c 在 Θ 内, 一步 L 对二者梯度 |g| 和 > 0
    (非零梯度断言) 且一步后值改变; 同种子 cm=1/0 两模型 W_D1 逐位同; d1_cm=0 模型无 α/b_c 键、写头输出 = 旧式
A28 ([U] 2026-08-23 写路径加 origin, 读路径不动; 第一张当前槽 = 空白纸) origin_write=1 ⇒ 场景读出 h_0 与 origin_write=0 逐位同;
    第一写 logits = write_tokens_org(tok(空白纸), tok(x0)) 逐位同 (空白纸 = 全零画布过同链信道), 换 origin 槽 ⇒ 第一写改变; 各深度读出
    h_k = 单槽 enc_v3(x_k) 逐位同 (读路径不动); 第二写 logits = write_tokens_org(tok(x1), tok(x0)) 逐位同, 换 origin 槽 ⇒ 第二写改变;
    一步 L 对 seg_cur/seg_org 梯度 |g| 和 > 0 (非零梯度断言; origin_write=0 ⇒ 二者 grad None); use_flag=1 互斥 (断言)
A29 ([U] 2026-08-23 W_D1 解冻 + 行归一化 + lr = 0.1 × 主 lr) d1_learn=1 ⇒ W_D1 requires_grad 且在且只在第 2 参数组
    (lr = d1_lr_mult × 主 lr, wd=0), 主参数组不含; 建组归一后行 L2 单位范数; 一步 L 对 W_D1 梯度 |g| 和 > 0 (非零梯度断言);
    一步后 W_D1 值改变且行仍单位范数 (post-hook); 旧冻结偏置仍冻结; 同种子 d1_learn=0 模型 W_D1 = 解冻模型初值逐位
    (解冻不耗 RNG), 一步后逐位不变 (A1 口径)
A30 (迭代重学甲案) exposure=0 ⇒ 逐位同旧: relearn 不构造、chain_train_step(relearn=None) 同种子两跑参数与损失逐位同 (CPU)、无新计量键
A31 (甲案) 梯度路由: 转移损失 → b_c/α/Θ_E 非零、W_D1 与新读者参数零; 教学损失 → 只新读者 (主模型梯度不动)
A32 (甲案) 曝光防火墙: 教学批 N ⊆ 𝒩_exp; 转移批 N ⊆ 训练带∖𝒩_exp
A33 (甲案) 换代作为: 期界重置参数非拷贝、优化器状态清零、子集重抽确定性 + 五分带配额 2/4/4/6/4 (e_exp=20, k=128)
A34 (甲案) 隔离: 新读者参数集与主模型参数集不交; 交付评测路径不经新读者
"""
import torch

from .. import data as D
from .. import geometry as GM
from ..grpo import TaskWeights
from . import render as R
from . import step as ST
from .data import build_batch
from .model import PredModel
from .pool import DataPool

A30_PIN = 9.853376388549805    # HEAD f711e63 pod CPU 采值 (甲案工单§6); A30/A35 全关平价共用锚


def _small_cfg(**kw):
    from .trainer import TrainCfg
    cfg = TrainCfg(b_scene=4, b_pool=4, b_mask=4, workers=0, kit_workers=0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _mini_batches(model, cfg, dev, rng, k=9):
    items = [D.sample_group(rng, k) for _ in range(cfg.b_scene)]
    sb = build_batch(items, dev)
    pool = DataPool(64)
    with torch.no_grad():
        enc = model.encode(sb["x1"], is_scene=True)
        kk = ST.write_logits(model, enc["tokens"], cfg).argmax(-1).cpu()
    pool.admit(kk, sb["ns"].cpu(), sb["zseed"], 0)
    idx = pool.sample(cfg.b_pool, cfg.t_pool, cfg.eps_pool, rng)
    kits = [D.sample_group(rng, k, n=int(n), with_scene=False) for n in pool.labels(idx)]
    draw_p = R.draw_channel(idx.shape[0], cfg.s, rng, dev, cfg.occ_k)
    pb = ST.render_pool_batch(pool, idx, kits, cfg, rng, dev, draw=draw_p)
    pb["draw"], pb["tpl"] = draw_p, R.templates(draw_p.u, draw_p.s)
    pb["pgen"] = pool.gens(idx)
    pb["pzseed"] = pool.zseed[idx.cpu()].clone()
    mb = ST.make_mask_batch(cfg.b_mask, cfg, rng, dev)
    return sb, pb, mb, pool


def run_all(dev=None, seed=0, verbose=True):
    dev = torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    cfg = _small_cfg(beta=1e-3)
    model = PredModel().to(dev)
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    tw = TaskWeights()
    w = tw.weights(dev)
    sb, pb, mb, pool = _mini_batches(model, cfg, dev, rng)
    res = {}

    # ---- A1: D1 冻结
    d1_before = model.d1_weights()
    in_opt = any(any(q is p for q in g["params"]) for p in model.D1.parameters() for g in opt.param_groups)
    for _ in range(2):
        ST.train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, beta=0.0, split=True, want_S=True)
    same = all(torch.equal(a, b) for a, b in zip(d1_before, model.d1_weights()))
    res["A1"] = ("PASS" if same and not in_opt else "FAIL",
                 dict(d1_bitwise_same=same, d1_in_optimizer=in_opt, requires_grad=[p.requires_grad for p in model.D1.parameters()]))

    # ---- A2: 直通前向恒等
    B = sb["B"]
    draw = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    logits = torch.randn(B, GM.T, R.NCLS, device=dev, requires_grad=True)
    r = R.render_ste(logits, tpl)
    hard = R.render_hard(logits.argmax(-1), tpl)
    eq_pre = torch.equal(r["x2"], hard)
    eq_post = torch.equal(R.channel(r["x2"], draw), R.channel(hard, draw))
    g = torch.autograd.grad(R.channel(r["x2"], draw).sum(), logits)[0]
    res["A2"] = ("PASS" if eq_pre and eq_post and float(g.abs().sum()) > 0 else "FAIL",
                 dict(pre_channel_bitwise=eq_pre, post_channel_bitwise=eq_post,
                      max_abs_diff_post=float((R.channel(r["x2"], draw) - R.channel(hard, draw)).abs().max()),
                      grad_abs_sum=float(g.abs().sum())))

    # ---- A3: 留出 N
    ok3 = True
    ev = {}
    for k in (cfg.k1, GM.K_MAX):
        tns = set(D.train_ns(k))
        for _ in range(3):
            it = D.sample_group(rng, k)
            ns_all = [it["n"]] + list(it["cand_ns"])
            if any(n not in tns for n in ns_all):
                ok3 = False
        ev[f"k{k}_bands_seen"] = "train only"
    hold = set(D.hole_ns()) | set(D.extrap_ns())
    pool_ok = not any(int(n) in hold for n in pool.n[:pool.size].tolist())
    res["A3"] = ("PASS" if ok3 and pool_ok else "FAIL", dict(batches_train_only=ok3, pool_no_holdout=pool_ok, **ev))

    # ---- A4: 标记不进前向 (v2 路)
    with torch.no_grad():
        model.eval()
        draw4 = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
        tpl4 = R.templates(draw4.u, draw4.s)
        o1 = ST.scene_side(model, sb, w, cfg.mu, cfg, draw4, tpl4)
        sb2 = dict(sb)
        sb2["ns"] = sb["ns"].roll(1)
        o2 = ST.scene_side(model, sb2, w, cfg.mu, cfg, draw4, tpl4)
        model.train()
    same4 = dict(x2=torch.equal(o1["x2"], o2["x2"]), h2=torch.equal(o1["h2"], o2["h2"]),
                 p=torch.equal(o1["p"], o2["p"]), k=torch.equal(o1["k"], o2["k"]))
    d4 = dict(max_abs_h2=float((o1["h2"] - o2["h2"]).abs().max()), max_abs_x2=float((o1["x2"] - o2["x2"]).abs().max()))
    res["A4"] = ("PASS" if all(same4.values()) else "FAIL", dict(**same4, **d4))

    # ---- A5: 任务头/直读头输入
    caps = {}

    def hook(name):
        def f(mod, inp):
            caps[name] = inp[0].detach().clone()
        return f
    hs = model.heads
    hooks = [getattr(hs, t).register_forward_pre_hook(hook(t)) for t in ("t1", "t2", "t3", "t4", "t5", "t6")]
    hooks.append(hs.read[0].register_forward_pre_hook(hook("read")))
    with torch.no_grad():
        h = torch.randn(B, hs.t2.in_features, device=dev)
        th = sb["th"]
        hc = torch.randn(B, D.N_CAND, hs.t2.in_features, device=dev)
        hs.task_logits(h, th)
        hs.t5_scores(h, hc)
        hs.read(h)
    for hk in hooks:
        hk.remove()
    exp = {"t1": torch.cat([h, hs.emb_tau(th["tau"])], -1), "t2": h,
           "t3": torch.cat([h, hs.emb_tau(th["tau"])], -1),
           "t4": torch.cat([h, hs.emb_p(th["p"])], -1), "t5": h,
           "t6": torch.cat([h, hs.emb_m(th["m"])], -1), "read": h}
    same5 = {t: torch.equal(caps[t], exp[t]) for t in exp}
    res["A5"] = ("PASS" if all(same5.values()) else "FAIL", same5)

    # ---- A6: 随机画布路径 (v2 路)
    cnt = dict(D1=0, task=0, read=0)

    def c_hook(name):
        def f(mod, inp):
            cnt[name] += 1
        return f
    hooks = [model.D1.register_forward_pre_hook(c_hook("D1")), hs.read[0].register_forward_pre_hook(c_hook("read"))]
    hooks += [getattr(hs, t).register_forward_pre_hook(c_hook("task")) for t in ("t1", "t2", "t3", "t4", "t5", "t6")]
    size0 = pool.size
    loss, _ = ST.mask_side(model, mb)
    for hk in hooks:
        hk.remove()
    res["A6"] = ("PASS" if cnt["D1"] == 0 and cnt["task"] == 0 and cnt["read"] == 0 and pool.size == size0 else "FAIL",
                 dict(calls=cnt, pool_size_unchanged=pool.size == size0))

    # ---- A7: u_S 精确 vjp (§8 挂起, 休眠 v2 路径完好性)
    try:
        m = ST.train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, beta=cfg.beta, split=True, want_S=True)
        al = m.get("al", {})
        ok_exact = al.get("u_mode") == "exact" and "u" in al and not al.get("degenerate") and al.get("JTw", 0.0) > 0.0
        drawA = R.draw_channel(B, cfg.s, rng, dev, cfg.occ_k)
        tplA = R.templates(drawA.u, drawA.s)
        with torch.no_grad():
            oa = ST.scene_side(model, sb, w, cfg.mu, cfg, drawA, tplA)
            saved = [p.detach().clone() for p in params]
            for p in params:
                p.add_(torch.randn_like(p) * 1e-3)
            ob = ST.scene_side(model, sb, w, cfg.mu, cfg, drawA, tplA, k_hard=oa["k"])
            for p, s0 in zip(params, saved):
                p.copy_(s0)
        eq7 = torch.equal(oa["hard"], ob["hard"])
        res["A7"] = ("PASS" if eq7 and ok_exact else "FAIL",
                     dict(hard_bitwise_same_under_perturbation=eq7, u_mode=al.get("u_mode"), u=al.get("u"), JTw=al.get("JTw"),
                          beta_eff=al.get("beta_eff"), beta_ceil=al.get("beta_ceil"), degenerate=al.get("degenerate")))
    except AssertionError as e:
        res["A7"] = ("FAIL", dict(error=str(e)))

    # ---- A10: 墨水成本口径 ([U] 2026-08-20 §5.2')
    lam10 = 0.02
    lg10 = torch.randn(B, GM.T, R.NCLS, device=dev) * 0.1
    lg10[..., R.EMPTY] += 5.0
    lg10.requires_grad_(True)
    r10 = R.render_ste(lg10, tpl)
    l10 = ST.lam_cost(lam10, r10["p"], r10["k"])
    g10 = torch.autograd.grad(l10, lg10)[0]
    blank_ok = float(l10) == 0.0 and float(g10.abs().sum()) == 0.0
    lg11 = torch.randn(B, GM.T, R.NCLS, device=dev) * 0.1
    lg11[:, : GM.T // 2, R.EMPTY] += 5.0
    lg11.requires_grad_(True)
    r11 = R.render_ste(lg11, tpl)
    l11 = ST.lam_cost(lam10, r11["p"], r11["k"])
    g11 = torch.autograd.grad(l11, lg11)[0]
    ink11 = r11["k"] != R.EMPTY
    pos_ok = (int(ink11.sum()) > 0 and float(l11) > 0.0 and float(g11[ink11].abs().sum()) > 0.0
              and float(g11[~ink11].abs().sum()) == 0.0)
    res["A10"] = ("PASS" if blank_ok and pos_ok else "FAIL",
                  dict(blank_llam=float(l10), blank_grad_abs=float(g10.abs().sum()), inked_llam=float(l11),
                       ink_grad_abs=float(g11[ink11].abs().sum()), blankcell_grad_abs=float(g11[~ink11].abs().sum()),
                       n_ink=int(ink11.sum())))

    # ================================================================ v3.1 链断言 A11–A25 (spec-v3.1 C9)
    cfgc = _small_cfg(chain_k=2, origin_cat=0, pred_on=1, gate_on=1, chain_from_pool=1, lam=1e-3)
    K = cfgc.chain_k
    rng2 = torch.Generator().manual_seed(seed + 21)
    draw_c = R.draw_channel(B, cfgc.s, rng2, dev, cfgc.occ_k)      # C7: 整链单 draw
    tpl_c = R.templates(draw_c.u, draw_c.s)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def _zero_or_none(g):
        return g is None or float(g.abs().sum()) == 0.0

    # 前向模型末层零初始化 ⇒ Δ̂ ≡ 0, 闸梯度对 a 结构性为零; 梯度类断言 (A16/A17/A20/A25/A26) 在
    # 扰动 fwd 末层的状态下跑 (可达输入上的 pin), 跑完还原.
    l2_saved = [p.detach().clone() for p in model.fwd.l2.parameters()]
    with torch.no_grad():
        for p in model.fwd.l2.parameters():
            p.normal_(0.0, 0.02)

    # ---- A11: 真实成绩/标记不进前向 (换标签重跑链, 前向张量逐位不变)
    with torch.no_grad():
        model.eval()
        o1 = ST.chain_side(model, sb, w, cfg.mu, cfgc, draw_c, tpl_c)
        p1 = ST.pred_gate(model, o1["states"], sb, cfgc, k0_gated=False)
        sb_r = dict(sb)
        sb_r["ns"] = sb["ns"].roll(1)
        sb_r["tg"] = {t: v.roll(1) for t, v in sb["tg"].items()}
        sb_r["truth5"] = sb["truth5"].roll(1)
        o2 = ST.chain_side(model, sb_r, w, cfg.mu, cfgc, draw_c, tpl_c)
        p2 = ST.pred_gate(model, o2["states"], sb_r, cfgc, k0_gated=False)
        model.train()
    same11 = dict(V=torch.equal(p1["V"], p2["V"]), W=torch.equal(p1["W"], p2["W"]),
                  lam=torch.equal(p1["lam"], p2["lam"]),
                  x=all(torch.equal(a["x"], b["x"]) for a, b in zip(o1["writes"], o2["writes"])),
                  h=all(torch.equal(a["h"], b["h"]) for a, b in zip(o1["states"], o2["states"])))
    res["A11"] = ("PASS" if all(same11.values()) else "FAIL", same11)

    # ---- A12: 逐深度直通恒等 (共享 draw 下逐深度重验; 亦 A24 前半的证据)
    ok12 = all(torch.equal(o1["writes"][k12]["x"],
                           R.channel(R.render_hard(o1["writes"][k12]["k"], tpl_c), draw_c))
               for k12 in range(K))
    res["A12"] = ("PASS" if ok12 else "FAIL", dict(depths_bitwise=ok12, K=K))

    # ---- A13: 随机画布不进链 (无 D1/V̂/g 调用; 池不变)
    cnt13 = dict(D1=0, predV=0, fwd=0, task=0, read=0)

    def _h13(name):
        def f(mod, inp):
            cnt13[name] += 1
        return f
    hooks = [model.D1.register_forward_pre_hook(_h13("D1")),
             model.pred.V["t1"].register_forward_pre_hook(_h13("predV")),
             model.fwd.l1.register_forward_pre_hook(_h13("fwd")),
             hs.read[0].register_forward_pre_hook(_h13("read"))]
    hooks += [getattr(hs, t).register_forward_pre_hook(_h13("task")) for t in ("t1", "t2", "t3", "t4", "t5", "t6")]
    size13 = pool.size
    ST.mask_side_v3(model, mb, cfgc)
    for hk in hooks:
        hk.remove()
    ok13 = all(v == 0 for v in cnt13.values()) and pool.size == size13
    res["A13"] = ("PASS" if ok13 else "FAIL", dict(calls=cnt13, pool_size_unchanged=pool.size == size13))

    # ---- A14: 训练期链恒跑满 K (闸逼停仍 K+1 状态); 评测端 k* 截断按 C3 闸位
    b_saved = model.gate.b.detach().clone()
    with torch.no_grad():
        model.gate.b.fill_(-20.0)                       # λ = σ(·+20) ≈ 1 ⇒ 每受闸位皆停
    o14 = ST.chain_side(model, sb, w, cfg.mu, cfgc, draw_c, tpl_c)
    p14 = ST.pred_gate(model, o14["states"], sb, cfgc, k0_gated=False)
    ks14_scene = ST.kstar_det(p14["dhat"], c_step=1.0, k0_gated=False)   # Δ̂ ≤ 1 ⇒ 场景链全停在 1
    ks14_pool = ST.kstar_det(p14["dhat"], c_step=1.0, k0_gated=True)     # 池链可 0
    lam14 = float(p14["lam"][:, 1:].min())
    ph_sum = float((p14["p_halt"].sum(1) - 1.0).abs().max())
    with torch.no_grad():
        model.gate.b.copy_(b_saved)
    ok14 = (len(o14["states"]) == K + 1 and len(o14["writes"]) == K
            and lam14 > 0.99 and bool((ks14_scene == 1).all()) and bool((ks14_pool == 0).all())
            and ph_sum < 1e-6)
    res["A14"] = ("PASS" if ok14 else "FAIL",
                  dict(states=len(o14["states"]), writes=len(o14["writes"]), lam_min_gated=lam14,
                       kstar_scene_all1=bool((ks14_scene == 1).all()),
                       kstar_pool_all0=bool((ks14_pool == 0).all()), p_halt_sum_err=ph_sum))

    # ---- A15: v3.1 C1 ⇒ N/A
    res["A15"] = ("N/A", dict(reason="v3.1 C1: origin 拼接撤销, tok(x0) 缓存复用逻辑已删; 断言随之不适用"))

    # ---- A16: 预测梯度不进主干 (ptb=0; L_pred = BCE + η_fwd·L_fwd, 场景链 + 池链)
    ss16 = ST.SoftScale()
    o16 = ST.chain_side(model, sb, w, cfg.mu, cfgc, draw_c, tpl_c)
    sbp16 = dict(pb)
    sbp16["x1"] = pb["x"]
    o16p = ST.chain_side(model, sbp16, w, cfg.mu, cfgc, pb["draw"], pb["tpl"])
    ss16.update(torch.cat([torch.stack([s_["marg"] for s_ in o16["states"]], dim=1).reshape(-1, 6),
                           torch.stack([s_["marg"] for s_ in o16p["states"]], dim=1).reshape(-1, 6)]))
    p16 = ST.pred_gate(model, o16["states"], sb, cfgc, k0_gated=False)
    p16p = ST.pred_gate(model, o16p["states"], sbp16, cfgc, k0_gated=True)
    L_pred, _c1 = ST.pred_loss(p16, o16["states"], ss16, cfgc.eta_fwd)
    lp2, _c2 = ST.pred_loss(p16p, o16p["states"], ss16, cfgc.eta_fwd)
    L_pred = L_pred + lp2
    g16 = torch.autograd.grad(L_pred, [p for _, p in named], allow_unused=True, retain_graph=True)
    bad16 = [n for (n, _), g in zip(named, g16) if not n.startswith(("pred.", "fwd.")) and not _zero_or_none(g)]
    pred_nz = sum(float(g.abs().sum()) for (n, _), g in zip(named, g16) if n.startswith("pred.") and g is not None)
    fwd_nz = sum(float(g.abs().sum()) for (n, _), g in zip(named, g16) if n.startswith("fwd.") and g is not None)
    res["A16"] = ("PASS" if not bad16 and pred_nz > 0 and fwd_nz > 0 else "FAIL",
                  dict(leaks=bad16[:8], pred_grad_abs_sum=pred_nz, fwd_grad_abs_sum=fwd_nz))

    # ---- A17: 闸梯度只到 (a,b) (扰动 fwd 后 Δ̂ ≠ 0 ⇒ a 有可达梯度)
    L_gate = ST.gate_loss(p16, o16["states"], cfgc) + ST.gate_loss(p16p, o16p["states"], cfgc)
    g17 = torch.autograd.grad(L_gate, [p for _, p in named], allow_unused=True, retain_graph=True)
    bad17 = [n for (n, _), g in zip(named, g17) if (not n.startswith("gate.")) and not _zero_or_none(g)]
    ab_nz = all(g is not None and float(g.abs().sum()) > 0
                for (n, _), g in zip(named, g17) if n.startswith("gate."))
    res["A17"] = ("PASS" if not bad17 and ab_nz else "FAIL", dict(leaks=bad17[:8], ab_grads_nonzero=ab_nz))

    # ---- A19 (v3.1 C9): y_soft 标签零梯度 — 冻结 V̂ 后重建前向、只留 L_pred 跑一步 backward,
    # Θ_E 与六任务头 .grad 全零 (若 y_soft 未 sg, 标签经六任务 logit → 活 h → Θ_E 即非零)
    for p_ in model.pred.parameters():
        p_.requires_grad_(False)
    for _, p_ in named:
        p_.grad = None
    pg19 = ST.pred_gate(model, o16["states"], sb, cfgc, k0_gated=False)   # 冻结后重建 (图不含 V̂ 权重)
    L19, _ = ST.pred_loss(pg19, o16["states"], ss16, cfgc.eta_fwd)
    L19.backward(retain_graph=True)
    badE = [n for (n, p_) in named if (n.startswith("E.") or n.startswith("heads."))
            and p_.grad is not None and float(p_.grad.abs().sum()) != 0.0]
    fwd_got = sum(float(p_.grad.abs().sum()) for n, p_ in named if n.startswith("fwd.") and p_.grad is not None)
    for p_ in model.pred.parameters():
        p_.requires_grad_(True)
    for _, p_ in named:
        p_.grad = None
    res["A19"] = ("PASS" if not badE and fwd_got > 0 else "FAIL",
                  dict(theta_e_or_heads_grads=badE[:8], fwd_grad_abs_sum=fwd_got))

    # ---- A20: L_fwd 只经 ĥ 反传 (ptb=1: 目标端 h[K] 零梯度, 输入端 h[0] 经 g 非零).
    # 审查 Important (2026-08-21) 后改法: 对 pred_loss 真实返回的 L_fwd 张量 (comp["lf_t"]) 取梯度,
    # 不再断言侧自建公式 — pred_loss 内目标端 detach 被误删时本断言即变红 (钉在可达输入上).
    cfg20 = _small_cfg(chain_k=2, origin_cat=0, pred_on=1, pred_to_backbone=1, chain_from_pool=1, lam=1e-3)
    o20 = ST.chain_side(model, sb, w, cfg.mu, cfg20, draw_c, tpl_c)
    p20 = ST.pred_gate(model, o20["states"], sb, cfg20, k0_gated=False)
    ss20 = ST.SoftScale()
    ss20.update(torch.stack([s_["marg"] for s_ in o20["states"]], dim=1))
    _lp20, comp20 = ST.pred_loss(p20, o20["states"], ss20, cfg20.eta_fwd)
    lf20 = comp20["lf_t"]
    hs20 = [stt["h"] for stt in o20["states"]]
    g20 = torch.autograd.grad(lf20, hs20, allow_unused=True, retain_graph=True)
    g20f = torch.autograd.grad(lf20, list(model.fwd.parameters()), allow_unused=True)
    ok20 = _zero_or_none(g20[-1]) and g20[0] is not None and float(g20[0].abs().sum()) > 0 \
        and sum(float(g.abs().sum()) for g in g20f if g is not None) > 0
    res["A20"] = ("PASS" if ok20 else "FAIL",
                  dict(h_last_grad=(float(g20[-1].abs().sum()) if g20[-1] is not None else None),
                       h0_grad_abs=(float(g20[0].abs().sum()) if g20[0] is not None else None),
                       fwd_grad_abs=sum(float(g.abs().sum()) for g in g20f if g is not None)))

    # ---- A21: 深度 0 读出不进闩/停跑/课程 (eval_decisions 签名无场景参数 + 双调一致性)
    import inspect
    from .trainer import BestLatch, TrainCfg, eval_decisions, new_state
    sig = list(inspect.signature(eval_decisions).parameters)
    no_scene = not any("scene" in s or s in ("sc", "score_sc") for s in sig)
    cfgz = TrainCfg()
    runs = []
    for _ in range(2):
        stz = new_state(cfgz)
        stz.update(stage=2, k=GM.K_MAX, stage1_level=0.9, switch_step=0)
        lz = BestLatch()
        seq = [(0.9, 0.88), (0.91, 0.89), (0.5, 0.5), (0.5, 0.5)]
        outs = [eval_decisions(stz, cfgz, lz, ks_, d1_, 100000 + 500 * i, 10.0, 1.0)
                for i, (ks_, d1_) in enumerate(seq)]
        runs.append((outs, {k2: v for k2, v in stz.items()}))
    ok21 = no_scene and runs[0] == runs[1]
    res["A21"] = ("PASS" if ok21 else "FAIL", dict(signature=sig, no_scene_param=no_scene,
                                                   deterministic=runs[0] == runs[1]))

    # ---- A22: 场景链恒执行 k=0 写; 场景起链不存在 k* = 0
    dh_neg = torch.full((5, K), -1.0)
    ks_det = ST.kstar_det(dh_neg, 0.0, k0_gated=False)
    ks_smp = ST.kstar_sample(torch.ones(5, K), torch.Generator().manual_seed(0), k0_gated=False)
    ks_pool0 = ST.kstar_det(dh_neg, 0.0, k0_gated=True)
    ok22 = bool((ks_det >= 1).all()) and bool((ks_smp >= 1).all()) and len(o1["writes"]) == K \
        and bool((ks_pool0 == 0).all())
    res["A22"] = ("PASS" if ok22 else "FAIL",
                  dict(kstar_det_min=int(ks_det.min()), kstar_sample_min=int(ks_smp.min()),
                       writes=len(o1["writes"]), pool_can_direct=bool((ks_pool0 == 0).all())))

    # ---- A23: 入池 append + FIFO 按年龄; 亲本保留; 池 ≤ C_pool
    pl = DataPool(4)
    ks23 = torch.randint(0, 4, (3, GM.T), generator=torch.Generator().manual_seed(3))
    pl.admit(ks23, torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]), 0)
    pl.update_scores(torch.tensor([0]), torch.tensor([0.99]), eta=1.0)       # 最老条目给最高分
    parent_k = pl.k[1].clone()
    pl.admit(ks23[:1], torch.tensor([2]), torch.tensor([9]), 1, gens=torch.tensor([1]))   # 亲本(槽1)的产物入池
    parent_still = torch.equal(pl.k[1], parent_k) and int(pl.gen[3]) == 1
    pl.admit(ks23[:1], torch.tensor([4]), torch.tensor([4]), 2)              # 满 4 → 逐出槽 0 (最老且最高分)
    evict_by_age = pl.n_evict == 1 and int(pl.n[0]) == 4 and pl.size == 4
    no_score_removal = not any(hasattr(pl, mth) for mth in ("remove", "remove_worst", "evict_by_score", "drop"))
    ok23 = parent_still and evict_by_age and pl.size <= pl.cap and no_score_removal
    res["A23"] = ("PASS" if ok23 else "FAIL",
                  dict(parent_kept=parent_still, evicted_oldest_despite_top_sigma=evict_by_age,
                       size_le_cap=pl.size <= pl.cap, no_score_removal_api=no_score_removal))

    # ---- A24: 链内腐蚀 CRN (同链逐深度同参数 = 单 draw 重放, A12 已证逐位; 链间行独立; 步间独立)
    row_diff = bool((draw_c.u[0] != draw_c.u[1]).any())
    draw_next = R.draw_channel(B, cfgc.s, rng2, dev, cfgc.occ_k)
    step_diff = bool((draw_c.u != draw_next.u).any())
    pool_x0_shared = torch.equal(pb["x"], R.channel(R.render_classes(pool.classes(pb["idx"]).to(dev), pb["draw"]), pb["draw"]))
    ok24 = ok12 and row_diff and step_diff and pool_x0_shared
    res["A24"] = ("PASS" if ok24 else "FAIL",
                  dict(within_chain_bitwise=ok12, across_chain_rows_differ=row_diff,
                       across_step_draws_differ=step_diff, pool_x0_same_draw=pool_x0_shared))

    # ---- A25: 无独立 Ŵ 参数; Ŵ 计算图必经 V̂ 头与 g
    no_w = not any(n.startswith("pred.W") for n, _ in model.named_parameters())
    gW = torch.autograd.grad(p16["Wlog"].sum(), list(model.pred.parameters()) + list(model.fwd.parameters()),
                             allow_unused=True)
    v_reach = sum(float(g.abs().sum()) for g in gW[:len(list(model.pred.parameters()))] if g is not None)
    g_reach = sum(float(g.abs().sum()) for g in gW[len(list(model.pred.parameters())):] if g is not None)
    res["A25"] = ("PASS" if no_w and v_reach > 0 and g_reach > 0 else "FAIL",
                  dict(no_independent_W_params=no_w, through_V_abs=v_reach, through_g_abs=g_reach))

    # ---- 还原 fwd 末层零初始化
    with torch.no_grad():
        for p_, s0 in zip(model.fwd.l2.parameters(), l2_saved):
            p_.copy_(s0)

    # ---- A18: K=1 平价 (重构回归, CPU 逐位): 新链路 (兼容配置) ≡ 保留的 v2 train_step
    devc = torch.device("cpu")
    rng18 = torch.Generator().manual_seed(seed + 33)
    items18 = [D.sample_group(rng18, 9) for _ in range(4)]
    sb18 = build_batch(items18, devc)
    pool18 = DataPool(32)
    pool18.admit(torch.randint(0, 4, (4, GM.T), generator=rng18), sb18["ns"], sb18["zseed"], 0)
    idx18 = pool18.sample(4, cfg.t_pool, cfg.eps_pool, rng18)
    kits18 = [D.sample_group(rng18, 9, n=int(n), with_scene=False) for n in pool18.labels(idx18)]
    cfg_v2 = _small_cfg(lam=1e-3)
    cfg_ch = _small_cfg(lam=1e-3, chain_k=1, origin_cat=0, use_flag=1, pred_on=0, gate_on=0,
                        chain_from_pool=0)
    pb18 = ST.render_pool_batch(pool18, idx18, kits18, cfg_v2, rng18, devc)
    mb18 = ST.make_mask_batch(4, cfg_v2, rng18, devc)
    w18 = TaskWeights().weights(devc)

    def _one_step(step_fn, cfg_x):
        torch.manual_seed(seed + 7)
        mm = PredModel().to(devc)
        pp = mm.trainable_params()
        oo = torch.optim.AdamW(pp, lr=cfg_x.lr, weight_decay=cfg_x.wd)
        rr = torch.Generator().manual_seed(88)
        mmx = step_fn(mm, oo, pp, cfg_x, rr)
        gg = {n: (p.grad.detach().clone() if p.grad is not None else None)
              for n, p in mm.named_parameters() if p.requires_grad}
        vv = {n: p.detach().clone() for n, p in mm.named_parameters() if p.requires_grad}
        return gg, vv, mmx

    gA, vA, mA = _one_step(lambda mm, oo, pp, cx, rr: ST.train_step(mm, oo, pp, sb18, pb18, mb18, cx, w18,
                                                                    devc, rr, beta=0.0, split=False), cfg_v2)
    gB, vB, mB = _one_step(lambda mm, oo, pp, cx, rr: ST.chain_train_step(mm, oo, pp, sb18, pb18, mb18, cx,
                                                                          w18, devc, rr), cfg_ch)
    new_pfx = ("pred.", "gate.", "fwd.", "E.seg_")
    shared = [n for n in gA if not n.startswith(new_pfx)]
    diff_g = [n for n in shared if not ((gA[n] is None and gB[n] is None)
                                        or (gA[n] is not None and gB[n] is not None and torch.equal(gA[n], gB[n])))]
    diff_v = [n for n in shared if not torch.equal(vA[n], vB[n])]
    new_used = [n for n in gB if n.startswith(new_pfx) and not _zero_or_none(gB[n])]
    ok18 = not diff_g and not diff_v and not new_used and abs(mA["L"] - mB["L"]) == 0.0
    res["A18"] = ("PASS" if ok18 else "FAIL",
                  dict(grad_diff=diff_g[:8], param_diff=diff_v[:8], new_params_touched=new_used[:8],
                       L_v2=mA["L"], L_chain=mB["L"], n_shared=len(shared)))

    # ---- A26 ([C] 原步 4 A19 改号): ptb=1 ⇒ L_pred 到 Θ_E/φ(θ); 同配置 A17; gate_mode 臂无 L_gate
    res["A26"] = a26(dev, seed=seed)

    # ---- A27 ([U] 2026-08-21 写头共模扣除): 5 参数接线 + 冻结/非零梯度/旧式兼容
    res["A27"] = a27(dev, seed=seed)

    # ---- A28 ([U] 2026-08-23 写路径加 origin, 读路径不动): 读路径逐位不动 + 写路径可达 origin 槽 + 段嵌入非零梯度 (CPU, 逐位)
    res["A28"] = a28(seed=seed)

    # ---- A29 ([U] 2026-08-23 W_D1 解冻 + 行归一化 + lr = 0.1 × 主 lr): 参数组/单位范数/非零梯度/关端平价
    res["A29"] = a29(dev, seed=seed)

    # ---- A30–A34 (迭代重学甲案, spec 2026-08-24-iterated-relearning §5; [U]「先跑甲，写头d1保持冻结」)
    res["A30"] = a30(seed=seed)
    res["A31"] = a31(dev, seed=seed)
    res["A32"] = a32(dev, seed=seed)
    res["A33"] = a33(dev, seed=seed)
    res["A34"] = a34(dev, seed=seed)

    res["A35"] = a35(dev, seed=seed)
    res["A36"] = a36(dev, seed=seed)
    res["A37"] = a37(dev, seed=seed)
    res["A38"] = a38(dev, seed=seed)
    res["A39"] = a39(dev, seed=seed)
    res["A40"] = a40(dev, seed=seed)

    if verbose:
        for k in sorted(res):
            print(f"[{k}] {res[k][0]} {res[k][1]}", flush=True)
    return res


def a26(dev, seed=0):
    """A26 ([C] 原步 4 A19 改号, v3.1 更新): pred_to_backbone=1 ⇒ L_pred (BCE 项经 V̂(h 活), L_fwd 项经
    g(h 活)) 梯度到达 Θ_E 与 φ(θ) 嵌入表 (g.abs().sum()>0, 非零梯度断言口径); 同配置复验 A17 (L_gate
    只到 (a,b)); gate_mode const/rand 臂不产生 L_gate (仅 learned 有). 独立可调."""
    dev = torch.device(dev)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    cfg = _small_cfg(chain_k=2, pred_on=1, gate_on=1, pred_to_backbone=1, chain_from_pool=1, lam=1e-3)
    model = PredModel().to(dev)
    with torch.no_grad():
        for p in model.fwd.l2.parameters():
            p.normal_(0.0, 0.02)                    # Δ̂ ≠ 0 (闸 a 的可达梯度; 零初始化下结构性为零)
    sb, pb, mb, _pool = _mini_batches(model, cfg, dev, rng)
    w = TaskWeights().weights(dev)
    K = cfg.chain_k
    rng2 = torch.Generator().manual_seed(seed + 21)
    draw = R.draw_channel(sb["B"], cfg.s, rng2, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    ss = ST.SoftScale()
    out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    sbp = dict(pb)
    sbp["x1"] = pb["x"]
    outp = ST.chain_side(model, sbp, w, cfg.mu, cfg, pb["draw"], pb["tpl"])
    ss.update(torch.cat([torch.stack([s_["marg"] for s_ in out["states"]], dim=1).reshape(-1, 6),
                         torch.stack([s_["marg"] for s_ in outp["states"]], dim=1).reshape(-1, 6)]))
    pg = ST.pred_gate(model, out["states"], sb, cfg, k0_gated=False)
    pgp = ST.pred_gate(model, outp["states"], sbp, cfg, k0_gated=True)
    L_pred, comp = ST.pred_loss(pg, out["states"], ss, cfg.eta_fwd)
    lp2, _ = ST.pred_loss(pgp, outp["states"], ss, cfg.eta_fwd)
    L_pred = L_pred + lp2

    # ---- 判据 1: L_pred 梯度到达 Θ_E 与 φ(θ) 嵌入表; BCE 与 L_fwd 两项各自到达 Θ_E
    E_params = [p for p in model.E.parameters() if p.requires_grad]
    gE = torch.autograd.grad(L_pred, E_params, allow_unused=True, retain_graph=True)
    e_nz = sum(float(g.abs().sum()) for g in gE if g is not None)
    lf_only = torch.zeros((), device=dev)
    for kf in range(K):
        lf_only = lf_only + (1.0 - torch.nn.functional.cosine_similarity(
            pg["hhat"][kf], out["states"][kf + 1]["h"].detach(), dim=-1)).mean()
    gE_f = torch.autograd.grad(lf_only, E_params, allow_unused=True, retain_graph=True)
    ef_nz = sum(float(g.abs().sum()) for g in gE_f if g is not None)
    emb_tables = dict(emb_tau=model.heads.emb_tau.weight, emb_p=model.heads.emb_p.weight,
                      emb_m=model.heads.emb_m.weight)
    gEmb = torch.autograd.grad(L_pred, list(emb_tables.values()), allow_unused=True, retain_graph=True)
    emb_nz = {n: (float(g.abs().sum()) if g is not None else 0.0) for n, g in zip(emb_tables, gEmb)}
    ok1 = e_nz > 0 and ef_nz > 0 and all(v > 0 for v in emb_nz.values())

    # ---- 判据 2 (同配置 A17 复验): L_gate 只到 (a,b)
    L_gate = ST.gate_loss(pg, out["states"], cfg) + ST.gate_loss(pgp, outp["states"], cfg)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def _zero_or_none(g):
        return g is None or float(g.abs().sum()) == 0.0

    g17 = torch.autograd.grad(L_gate, [p for _, p in named], allow_unused=True)
    bad17 = [n for (n, _), g in zip(named, g17) if (not n.startswith("gate.")) and not _zero_or_none(g)]
    ab_nz = all(g is not None and float(g.abs().sum()) > 0
                for (n, _), g in zip(named, g17) if n.startswith("gate."))
    ok2 = not bad17 and ab_nz

    # ---- 判据 3: gate_mode const/rand 臂不产生 L_gate (learned 才有)
    def _lgate_for(mode):
        torch.manual_seed(seed + 3)
        m3 = PredModel().to(dev)
        p3 = m3.trainable_params()
        opt3 = torch.optim.AdamW(p3, lr=3e-4, weight_decay=0.01)
        cfg3 = _small_cfg(chain_k=2, pred_on=1, gate_on=1, gate_mode=mode, chain_from_pool=1)
        rng3 = torch.Generator().manual_seed(seed + 5)
        sb3, pb3, mb3, _p3 = _mini_batches(m3, cfg3, dev, rng3)
        w3 = TaskWeights().weights(dev)
        mm3 = ST.chain_train_step(m3, opt3, p3, sb3, pb3, mb3, cfg3, w3, dev, rng3, sscale=ST.SoftScale())
        return mm3["L_gate"]

    lg_const = _lgate_for("const")
    lg_rand = _lgate_for("rand")
    lg_learned = _lgate_for("learned")
    ok3 = lg_const is None and lg_rand is None and lg_learned is not None

    status = "PASS" if ok1 and ok2 and ok3 else "FAIL"
    evidence = dict(E_grad_abs_sum=e_nz, E_grad_fwd_only=ef_nz, emb_grad_abs_sum=emb_nz,
                    bce=comp["bce"], fwd=comp["fwd"], gate_leaks=bad17[:8],
                    gate_ab_nonzero=ab_nz, L_gate_const=lg_const, L_gate_rand=lg_rand,
                    L_gate_learned=lg_learned)
    return status, evidence


def a27(dev, seed=0):
    """A27 ([U] 2026-08-21 写头共模扣除, 唯一代码改动 = 5 参数): d1_cm=1 ⇒ (i) 初值 α=1, b_c=0, 写头输出逐位
    = W_D1(z − z̄) (旧冻结偏置不参与); (ii) W_D1 与冻结偏置不在 optimizer, 一步前后逐位不变 (A1 同口径);
    (iii) α、b_c 在 Θ 内, 一步 L 对二者梯度 |g| 和 > 0 (非零梯度断言), AdamW 一步后二者值改变; (iv) 同种子
    cm=1/0 两模型 W_D1 逐位同 (新参数不耗 RNG); (v) d1_cm=0 模型无 α/b_c 键, 写头输出 = 旧式 W_D1 z + b.
    K=1 / pred 关 / cfp 关 的小配置 (本跑形态), 独立可调."""
    dev = torch.device(dev)
    torch.manual_seed(seed)
    m0 = PredModel().to(dev)
    torch.manual_seed(seed)
    m1 = PredModel(d1_cm=True).to(dev)
    same_w = torch.equal(m0.D1.lin.weight, m1.D1.lin.weight) and torch.equal(m0.D1.lin.bias, m1.D1.lin.bias)
    keys0 = set(m0.state_dict())
    keys1 = set(m1.state_dict())
    keys_ok = ("D1.alpha" not in keys0 and "D1.bias_c" not in keys0
               and "D1.alpha" in keys1 and "D1.bias_c" in keys1 and keys1 - keys0 == {"D1.alpha", "D1.bias_c"}
               and keys0 <= keys1)                                   # cm=1 只增不减键 (旧键一个不少)
    init_ok = float(m1.D1.alpha) == 1.0 and float(m1.D1.bias_c.abs().sum()) == 0.0
    rng = torch.Generator().manual_seed(seed + 1)
    cfg = _small_cfg(chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, lam=1e-3, d1_cm=1)
    sb, pb, mb, _pool = _mini_batches(m1, cfg, dev, rng)
    with torch.no_grad():
        z = m1.E.enc_seq(m1.E.tokenize(sb["x1"]), None, seg=False)["tokens"]
        lg1 = m1.D1(z)
        ref1 = torch.nn.functional.linear(z - z.mean(1, keepdim=True), m1.D1.lin.weight)
        lg0 = m0.D1(z)
        ref0 = torch.nn.functional.linear(z, m0.D1.lin.weight, m0.D1.lin.bias)
    form_ok = torch.equal(lg1, ref1) and torch.allclose(lg0, ref0, atol=1e-6)
    params = m1.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    in_opt_frozen = any(any(q is p for q in g["params"]) for p in m1.D1.lin.parameters() for g in opt.param_groups)
    in_opt_new = all(any(any(q is p for q in g["params"]) for g in opt.param_groups)
                     for p in (m1.D1.alpha, m1.D1.bias_c))
    frozen_before = m1.d1_weights()
    a_before, b_before = float(m1.D1.alpha), m1.D1.bias_c.detach().clone()
    w = TaskWeights().weights(dev)
    mm = ST.chain_train_step(m1, opt, params, sb, pb, mb, cfg, w, dev, rng)
    ga = m1.D1.alpha.grad
    gb = m1.D1.bias_c.grad
    g_ok = ga is not None and gb is not None and float(ga.abs().sum()) > 0 and float(gb.abs().sum()) > 0
    frozen_same = all(torch.equal(a, b) for a, b in zip(frozen_before, m1.d1_weights()))
    moved = float(m1.D1.alpha) != a_before and not torch.equal(m1.D1.bias_c.detach(), b_before)
    frozen_rg = not any(p.requires_grad for p in m1.D1.lin.parameters())
    ok = same_w and keys_ok and init_ok and form_ok and (not in_opt_frozen) and in_opt_new and g_ok and frozen_same \
        and moved and frozen_rg
    return ("PASS" if ok else "FAIL",
            dict(same_seed_W_bitwise=same_w, keys_ok=keys_ok, init_ok=init_ok, formula_bitwise=form_ok,
                 frozen_in_optimizer=in_opt_frozen, new_in_optimizer=in_opt_new,
                 grad_alpha_abs_sum=(float(ga.abs().sum()) if ga is not None else None),
                 grad_bias_c_abs_sum=(float(gb.abs().sum()) if gb is not None else None),
                 frozen_bitwise_same_after_step=frozen_same, new_params_moved=moved, frozen_requires_grad=not frozen_rg,
                 alpha_after=float(m1.D1.alpha), bias_c_after=[float(x) for x in m1.D1.bias_c.detach().cpu()],
                 L=mm["L"], L_lam=mm["L_lam"], stamps=mm["stamps"]))


def a28(seed=0):
    """A28 ([U] 2026-08-23「写路径加 origin，读路径不动」+「写第一张和第二张笔记都能看原题」, 第一张当前槽「放空白纸」): origin_write=1 ⇒
    (i) 场景读出 h_0 与 origin_write=0 同模型同 draw 逐位同; (ii) 第一写 logits = write_tokens_org(tok(空白纸), tok(x0)) 两槽装配逐位同,
    空白纸 = 全空类硬渲染过同链信道 (= 全零画布 + 同 draw 信道, 逐位), 换 origin 槽内容 (x0 批内滚动一位) ⇒ 第一写 logits 改变, 对 origin_write=0
    的第一写亦不同; (iii) 各深度读出 h_k = 单槽 enc_v3(x_k) 逐位同 (读路径不动); (iv) 第二写 logits = write_tokens_org(tok(x1), tok(x0)) 逐位同,
    换 origin 槽 ⇒ 第二写改变 (可达输入上的 pin); (v) 一步 chain_train_step 对 seg_cur/seg_org 梯度 |g| 和 > 0 (非零梯度断言), origin_write=0
    下二者 grad None; (vi) use_flag=1 互斥. CPU (逐位比对), K=2 / pred 关 / cfp 关 的小配置 (本跑形态), 独立可调."""
    import dataclasses
    dev = torch.device("cpu")
    torch.manual_seed(seed)
    model = PredModel(d1_cm=True).to(dev)
    rng = torch.Generator().manual_seed(seed + 1)
    cfg0 = _small_cfg(chain_k=2, pred_on=0, gate_on=0, chain_from_pool=0, lam=1e-3, d1_cm=1, origin_write=0)
    cfg1 = dataclasses.replace(cfg0, origin_write=1)
    sb, pb, mb, _pool = _mini_batches(model, cfg0, dev, rng)
    w = TaskWeights().weights(dev)
    draw = R.draw_channel(sb["B"], cfg0.s, rng, dev, cfg0.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out0 = ST.chain_side(model, sb, w, cfg0.mu, cfg0, draw, tpl)
    out1 = ST.chain_side(model, sb, w, cfg1.mu, cfg1, draw, tpl)
    h0_same = torch.equal(out0["states"][0]["h"], out1["states"][0]["h"])
    with torch.no_grad():
        xb = ST.blank_note(sb["B"], draw, tpl)
        blank_ok = torch.equal(xb, R.channel(torch.zeros_like(sb["x1"]), draw))
        tok0 = model.E.tokenize(sb["x1"])
        tok0r = model.E.tokenize(sb["x1"].roll(1, 0))
        l1a = ST.write_logits(model, ST.write_tokens_org(model, model.E.tokenize(xb), tok0), cfg1)
        l1b = ST.write_logits(model, ST.write_tokens_org(model, model.E.tokenize(xb), tok0r), cfg1)
        w1_two_slot = torch.equal(l1a, out1["writes"][0]["logits"].detach())
        org_reach1 = not torch.equal(l1a, l1b)
        w1_diff_vs_off = not torch.equal(out0["writes"][0]["logits"].detach(), out1["writes"][0]["logits"].detach())
        read_single = all(torch.equal(out1["states"][d]["h"].detach(),
                                      ST.enc_v3(model, out1["writes"][d - 1]["x"].detach(), cfg1, flagged=False)["cls"])
                          for d in (1, 2))
        x1 = out1["writes"][0]["x"].detach()
        l2a = ST.write_logits(model, ST.write_tokens_org(model, model.E.tokenize(x1), tok0), cfg1)
        l2b = ST.write_logits(model, ST.write_tokens_org(model, model.E.tokenize(x1), tok0r), cfg1)
        w2_two_slot = torch.equal(l2a, out1["writes"][1]["logits"].detach())
        org_reach2 = not torch.equal(l2a, l2b)
    # (v) 一步训练: 段嵌入非零梯度 (origin_write=1) / grad None (origin_write=0)
    def _seg_grads(cfgx):
        torch.manual_seed(seed + 3)
        mm = PredModel(d1_cm=True).to(dev)
        pp = mm.trainable_params()
        oo = torch.optim.AdamW(pp, lr=cfgx.lr, weight_decay=cfgx.wd)
        rr = torch.Generator().manual_seed(seed + 9)
        ST.chain_train_step(mm, oo, pp, sb, pb, mb, cfgx, w, dev, rr)
        return mm.E.seg_cur.grad, mm.E.seg_org.grad
    gc1, go1 = _seg_grads(cfg1)
    gc0, go0 = _seg_grads(cfg0)
    g_on = gc1 is not None and go1 is not None and float(gc1.abs().sum()) > 0 and float(go1.abs().sum()) > 0
    g_off = gc0 is None and go0 is None
    try:
        ST.chain_side(model, sb, w, cfg1.mu, dataclasses.replace(cfg1, chain_k=1, use_flag=1), draw, tpl)
        excl = False
    except AssertionError:
        excl = True
    ok = (h0_same and blank_ok and w1_two_slot and org_reach1 and w1_diff_vs_off and read_single and w2_two_slot
          and org_reach2 and g_on and g_off and excl)
    return ("PASS" if ok else "FAIL",
            dict(h0_bitwise_same=h0_same, blank_note_is_zero_canvas_through_channel=blank_ok,
                 write1_equals_two_slot_assembly=w1_two_slot, origin_slot_reaches_write1=org_reach1,
                 write1_differs_from_origin_write0=w1_diff_vs_off, read_path_single_slot_bitwise=read_single,
                 write2_equals_two_slot_assembly=w2_two_slot, origin_slot_reaches_write2=org_reach2,
                 grad_seg_cur_abs_sum=(float(gc1.abs().sum()) if gc1 is not None else None),
                 grad_seg_org_abs_sum=(float(go1.abs().sum()) if go1 is not None else None),
                 seg_grads_none_when_off=g_off, use_flag_exclusive=excl))


def a29(dev, seed=0):
    """A29 ([U] 2026-08-23「W_D1 解冻，行归一化，lr = 0.1 × 主 lr」): d1_learn=1 ⇒ (i) W_D1 requires_grad,
    在且只在第 2 参数组 (lr = d1_lr_mult × 主 lr, wd=0), 主参数组不含; (ii) 建组归一后行 L2 单位范数;
    (iii) 一步 L 对 W_D1 梯度 |g| 和 > 0 (非零梯度断言); (iv) 一步后 W_D1 值改变且行仍单位范数 (post-hook);
    (v) 旧冻结偏置仍冻结、一步前后逐位不变; (vi) 同种子 d1_learn=0 模型 W_D1 = 解冻模型初值逐位 (解冻不耗 RNG),
    一步后逐位不变 (A1 口径). K=1 / cm=1 / pred 关 / cfp 关 的小配置 (本跑形态), 独立可调."""
    from .trainer import d1_learn_setup, d1_rownorm_
    dev = torch.device(dev)
    cfg = _small_cfg(chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, lam=1e-3, d1_cm=1,
                     d1_learn=1, d1_lr_mult=0.1, d1_rownorm=1)
    torch.manual_seed(seed)
    m1 = PredModel(d1_cm=True).to(dev)
    w_init = m1.D1.lin.weight.detach().clone()
    b_frozen_before = m1.D1.lin.bias.detach().clone()
    params = m1.trainable_params()                       # 解冻前构建 (W_D1 不在其中 = 主组/裁剪列表口径)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    d1_learn_setup(m1, opt, cfg)
    d1_rownorm_(m1)
    W = m1.D1.lin.weight
    groups_ok = (len(opt.param_groups) == 2 and len(opt.param_groups[1]["params"]) == 1
                 and any(q is W for q in opt.param_groups[1]["params"])
                 and not any(q is W for q in opt.param_groups[0]["params"]))
    lr_ok = (abs(opt.param_groups[1]["lr"] - cfg.lr * cfg.d1_lr_mult) < 1e-12
             and opt.param_groups[1]["weight_decay"] == 0.0)
    ones = torch.ones(W.shape[0], device=dev)
    norm0_ok = torch.allclose(W.detach().norm(dim=1), ones, atol=1e-5)
    w_before = W.detach().clone()
    rng = torch.Generator().manual_seed(seed + 1)
    sb, pb, mb, _pool = _mini_batches(m1, cfg, dev, rng)
    w = TaskWeights().weights(dev)
    mm = ST.chain_train_step(m1, opt, params, sb, pb, mb, cfg, w, dev, rng)
    gw = W.grad
    g_ok = gw is not None and float(gw.abs().sum()) > 0
    moved = not torch.equal(W.detach(), w_before)
    norm1_ok = torch.allclose(W.detach().norm(dim=1), ones, atol=1e-5)
    bias_frozen = (not m1.D1.lin.bias.requires_grad) and torch.equal(m1.D1.lin.bias.detach(), b_frozen_before)
    # (vi) 关端: 同种子 d1_learn=0 ⇒ W_D1 = 解冻模型初值逐位, 一步后不变 (A1 口径)
    torch.manual_seed(seed)
    m0 = PredModel(d1_cm=True).to(dev)
    init_same = torch.equal(m0.D1.lin.weight, w_init)
    cfg0 = _small_cfg(chain_k=1, pred_on=0, gate_on=0, chain_from_pool=0, lam=1e-3, d1_cm=1, d1_learn=0)
    p0 = m0.trainable_params()
    o0 = torch.optim.AdamW(p0, lr=cfg0.lr, weight_decay=cfg0.wd)
    rng0 = torch.Generator().manual_seed(seed + 1)
    sb0, pb0, mb0, _ = _mini_batches(m0, cfg0, dev, rng0)
    ST.chain_train_step(m0, o0, p0, sb0, pb0, mb0, cfg0, w, dev, rng0)
    off_frozen = torch.equal(m0.D1.lin.weight, w_init) and len(o0.param_groups) == 1
    ok = groups_ok and lr_ok and norm0_ok and g_ok and moved and norm1_ok and bias_frozen and init_same and off_frozen
    return ("PASS" if ok else "FAIL",
            dict(groups_ok=groups_ok, lr_wd_ok=lr_ok, rownorm_after_setup=norm0_ok,
                 grad_W_abs_sum=(float(gw.abs().sum()) if gw is not None else None), moved_after_step=moved,
                 rownorm_after_step=norm1_ok, old_bias_still_frozen=bias_frozen,
                 same_seed_init_bitwise=init_same, off_arm_frozen_one_group=off_frozen, L=mm["L"]))


def _relearn_env(dev, seed=0):
    """A31/A32 共用构造: 固定 ns 的小场景批 (1,2,3,5) + 曝光子集手工设 (1,2,3) ⇒ 转移行 = n=5."""
    from . import relearn as RL
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 1)
    cfg = _small_cfg(d1_cm=1, exposure=1, e_exp=3, t_period=50, eta_tr=0.5, tr_warm=0.0,
                     chain_k=1, chain_from_pool=0)
    model = PredModel(d1_cm=True).to(dev)
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    w = TaskWeights().weights(dev)
    items = [D.sample_group(rng, 9, n=n) for n in (1, 2, 3, 5)]
    sb = build_batch(items, dev)
    pool = DataPool(64)
    with torch.no_grad():
        enc = model.encode(sb["x1"], is_scene=True)
        kk = ST.write_logits(model, enc["tokens"], cfg).argmax(-1).cpu()
    pool.admit(kk, sb["ns"].cpu(), sb["zseed"], 0)
    rl = RL.Relearn(cfg, dev)
    rl.rollover(9, 0)
    rl.exp = (1, 2, 3)                       # 手工设定, 只为断言构造确定性
    draw = R.draw_channel(sb["B"], cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = ST.chain_side(model, sb, w, cfg.mu, cfg, draw, tpl)
    return rl, model, opt, sb, out, pool


def a30(seed=0):
    """A30: 全关平价 (CPU 强制 — GPU 宽松模式同码两跑本就不逐位, 常数台账). exposure=0 ⇒
    chain_train_step(relearn=None) 同种子两跑 损失+全参数 逐位同, 且计量无新键."""
    dev = torch.device("cpu")
    outs = []
    for _ in range(2):
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed + 1)
        cfg = _small_cfg(d1_cm=1, chain_k=1, chain_from_pool=0)
        from .trainer import build_model
        model = build_model(cfg).to(dev)
        params = model.trainable_params()
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
        w = TaskWeights().weights(dev)
        sb, pb, mb, pool = _mini_batches(model, cfg, dev, rng)
        m = ST.chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, relearn=None)
        outs.append((m["L"], [p.detach().clone() for p in model.parameters()], m))
    same = outs[0][0] == outs[1][0] and all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
    no_new = all(k not in outs[0][2] for k in ("notes_x", "l_tr", "n_tr", "sr_loss", "eta_tr_now"))
    return ("PASS" if same and no_new else "FAIL", dict(bitwise_same=same, no_new_keys=no_new, L=outs[0][0]))


def a31(dev, seed=0):
    """A31: 梯度路由. (i) 只反传转移项 ⇒ b_c/α/Θ_E 梯度非零, W_D1 与新读者参数梯度为 None;
    (ii) 教学一步 ⇒ 主模型梯度不动 (None), 新读者参数确实变了."""
    rl, model, opt, sb, out, pool = _relearn_env(dev, seed)
    l_tr, tr_m = rl.transfer(out["writes"][-1]["x"], sb)
    opt.zero_grad(set_to_none=True)
    l_tr.backward()
    g_bc = float(model.D1.bias_c.grad.abs().sum()) if model.D1.bias_c.grad is not None else 0.0
    g_al = float(model.D1.alpha.grad.abs().sum()) if model.D1.alpha.grad is not None else 0.0
    g_E = sum(float(p.grad.abs().sum()) for p in model.E.parameters() if p.grad is not None)
    d1_none = model.D1.lin.weight.grad is None
    rd_none = all(p.grad is None for p in rl.reader.parameters())
    opt.zero_grad(set_to_none=True)
    r0 = [p.detach().clone() for p in rl.reader.parameters()]
    tm = rl.teach(sb, out["writes"][-1]["x"].detach(), pool)
    main_clean = all(p.grad is None for p in model.parameters())
    stepped = any(not torch.equal(a, b) for a, b in zip(r0, [p.detach() for p in rl.reader.parameters()]))
    ok = (tr_m["n_tr"] == 1 and g_bc > 0 and g_al > 0 and g_E > 0 and d1_none and rd_none
          and main_clean and stepped and tm["n_teach"] > 0)
    return ("PASS" if ok else "FAIL",
            dict(n_tr=tr_m["n_tr"], g_bc=g_bc, g_alpha=g_al, g_E=g_E, d1_grad_none=d1_none,
                 reader_grad_none=rd_none, main_clean_after_teach=main_clean, reader_stepped=stepped,
                 n_teach=tm["n_teach"]))


def a32(dev, seed=0):
    """A32: 曝光防火墙. 教学批 N ⊆ 𝒩_exp={1,2,3}; 转移批 N = {5}; 两批 ⊆ train_ns(9)."""
    rl, model, opt, sb, out, pool = _relearn_env(dev, seed)
    rl.transfer(out["writes"][-1]["x"], sb)
    rl.teach(sb, out["writes"][-1]["x"].detach(), pool)
    tset = set(rl.last_teach_ns.tolist())
    rset = set(rl.last_tr_ns.tolist())
    train9 = set(D.train_ns(9))
    ok = tset and tset <= {1, 2, 3} and rset == {5} and tset <= train9 and rset <= train9
    return ("PASS" if bool(ok) else "FAIL", dict(teach_ns=sorted(tset), tr_ns=sorted(rset)))


def a33(dev, seed=0):
    """A33: 换代作为. 期界重置 ⇒ 参数非拷贝 + 优化器状态清零 + 子集重抽确定性;
    五分带配额 (e_exp=20, k=128) = [2,4,4,6,4]."""
    from . import relearn as RL
    cfg = _small_cfg(exposure=1, e_exp=20, t_period=10)
    rl = RL.Relearn(cfg, dev)
    rl.rollover(GM.K_MAX, 0)
    quotas = [sum(1 for n in rl.exp if lo <= n <= hi) for lo, hi in RL.BANDS]
    p0 = [p.detach().clone() for p in rl.reader.parameters()]
    l = sum(p.float().sum() for p in rl.reader.parameters())
    l.backward()
    rl.opt.step()
    had_state = len(rl.opt.state) > 0
    rl.steps_in = 10
    row = rl.tick(GM.K_MAX, 10)                                   # 到期界 ⇒ rollover
    changed = any(not torch.equal(a, b) for a, b in zip(p0, [p.detach() for p in rl.reader.parameters()]))
    fresh_opt = len(rl.opt.state) == 0
    redraw = rl.exp == RL.draw_exposure(
        torch.Generator().manual_seed(RL._mix(int(cfg.seed) + 9001, 1)), GM.K_MAX, 20)
    ok = quotas == [2, 4, 4, 6, 4] and row is not None and changed and fresh_opt and had_state and redraw
    return ("PASS" if ok else "FAIL", dict(quotas=quotas, rolled=row is not None, params_changed=changed,
                                           opt_cleared=fresh_opt, redraw_deterministic=bool(redraw)))


def a34(dev, seed=0):
    """A34: 隔离. 新读者参数集与主模型参数集不交 (id 级); 新读者非空."""
    from . import relearn as RL
    torch.manual_seed(seed)
    model = PredModel(d1_cm=True).to(dev)
    rl = RL.Relearn(_small_cfg(exposure=1), dev)
    rl.rollover(9, 0)
    ids_m = {id(p) for p in model.parameters()}
    ids_r = {id(p) for p in rl.reader.parameters()}
    n_r = sum(p.numel() for p in rl.reader.parameters())
    ok = not (ids_m & ids_r) and n_r > 0
    return ("PASS" if ok else "FAIL", dict(disjoint=not (ids_m & ids_r), reader_params=n_r))


def a35(dev, seed=0):
    """A35 (乙案全关平价): gen_relearn=0 缺省 ⇒ chain_train_step 一步 L == A30 钉值 (CPU 逐位;
    genrl=None 分支零张量运算/零 RNG 消耗差异)."""
    dev = torch.device("cpu")
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(1)
    cfg = _small_cfg(d1_cm=1, chain_k=1, chain_from_pool=0)
    model = PredModel(d1_cm=True).to(dev)
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    w = TaskWeights().weights(dev)
    sb, pb, mb, _pool = _mini_batches(model, cfg, dev, rng)
    m = ST.chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng)
    ok = (m["L"] == A30_PIN) and ("n_im" not in m) and ("L_im" not in m) and ("n_sup_rows" not in m)
    return ("PASS" if ok else "FAIL"), dict(L=m["L"], pin=A30_PIN)


def a36(dev, seed=0):
    """A36 (乙案标签防火墙): 全批 N∉𝒩_exp ⇒ 六任务/t5/直读头零梯度, 掩码头非零; n_sup_rows=0;
    未曝行照常出纸 (写路径不断)."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(11)
    cfg = _small_cfg(d1_cm=1, chain_k=1, chain_from_pool=0, gen_relearn=1, e_exp=2, t_gen=50)
    model = PredModel(d1_cm=True).to(dev)
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    w = TaskWeights().weights(dev)
    sb, pb, mb, pool = _mini_batches(model, cfg, dev, rng, k=9)
    from . import relearn as RL
    grl = RL.GenRelearn(cfg, dev)
    grl.period = 0
    grl.k = 9
    avail = [n for n in range(1, 10) if n not in set(sb["ns"].tolist())][:2]
    grl.exp = tuple(avail)
    grl.bank = {}
    m = ST.chain_train_step(model, opt, params, sb, pb, mb, cfg, w, dev, rng, genrl=grl, pool=pool)

    def _zero(ps):
        return all((p.grad is None) or float(p.grad.abs().sum()) == 0.0 for p in ps)

    heads_zero = _zero([model.heads.t1.weight, model.heads.t2.weight, model.heads.t3.weight,
                        model.heads.t4.weight, model.heads.t5.weight, model.heads.t6.weight])
    read_zero = _zero(list(model.heads.read.parameters()))
    cell_nz = (model.heads.cell.weight.grad is not None
               and float(model.heads.cell.weight.grad.abs().sum()) > 0)
    ok = heads_zero and read_zero and cell_nz and m["n_sup_rows"] == 0 and m["n_im"] == 0
    return ("PASS" if ok else "FAIL"), dict(heads_zero=heads_zero, read_zero=read_zero, cell_nz=cell_nz,
                                            n_sup=m["n_sup_rows"], exp=list(grl.exp),
                                            ns=sb["ns"].tolist())


def a37(dev, seed=0):
    """A37 (换代作为): rollover ⇒ 非 CNN/非 W_D1 参数重初始化, E.cnn 逐位同 (keep=1), D1.lin 逐位同,
    子集重抽; 池不清空; keep_cnn=0 臂 E.cnn 亦重初始化."""
    torch.manual_seed(seed)
    cfg = _small_cfg(d1_cm=1, gen_relearn=1, e_exp=2, t_gen=50)
    model = PredModel(d1_cm=True).to(dev)
    from . import relearn as RL
    pool = DataPool(8)
    pool.admit(torch.zeros(3, GM.T, dtype=torch.long), torch.tensor([1, 2, 3]),
               torch.zeros(3, dtype=torch.long), 0)
    grl = RL.GenRelearn(cfg, dev)
    cnn0 = {kk: v.detach().clone() for kk, v in model.E.cnn.state_dict().items()}
    d10 = [p.detach().clone() for p in model.D1.lin.parameters()]
    pos0 = model.E.pos.detach().clone()
    row = grl.rollover(model, 9, 1)
    cnn_same = all(torch.equal(v, cnn0[kk]) for kk, v in model.E.cnn.state_dict().items())
    d1_same = all(torch.equal(p.detach(), r) for p, r in zip(model.D1.lin.parameters(), d10))
    pos_diff = not torch.equal(model.E.pos.detach(), pos0)
    cfg0 = _small_cfg(d1_cm=1, gen_relearn=1, e_exp=2, t_gen=50, gen_keep_cnn=0)
    m2 = PredModel(d1_cm=True).to(dev)
    cnn20 = {kk: v.detach().clone() for kk, v in m2.E.cnn.state_dict().items()}
    RL.GenRelearn(cfg0, dev).rollover(m2, 9, 1)
    cnn2_diff = any(not torch.equal(v, cnn20[kk]) for kk, v in m2.E.cnn.state_dict().items())
    ok = (cnn_same and d1_same and pos_diff and pool.size == 3
          and len(row["exp"]) == 2 and cnn2_diff and grl.bank is not None)
    return ("PASS" if ok else "FAIL"), dict(cnn_same=cnn_same, d1_same=d1_same, pos_diff=pos_diff,
                                            pool_kept=pool.size, exp=row["exp"], cnn2_diff=cnn2_diff)


def a38(dev, seed=0):
    """A38 (模仿路由): 池上一代曝光条目 × 银行新抽场景 → 写路径 CE: 梯度达 Θ_E 与 b_c 非零 (非零梯度
    断言), W_D1 零; 模仿/池读批 N ⊂ 𝒩_exp; 池读流梯度达任务头; 模仿目标为整数类别张量."""
    torch.manual_seed(seed)
    cfg = _small_cfg(d1_cm=1, gen_relearn=1, e_exp=2, t_gen=50, b_im=4, b_poolread=4)
    model = PredModel(d1_cm=True).to(dev)
    from . import relearn as RL
    grl = RL.GenRelearn(cfg, dev)
    grl.period = 0
    grl.k = 9
    grl.exp = (1, 2)
    grl.bank = RL.gen_scene_bank(grl.exp, RL.GenRelearn.BANK_PER_N, 123)
    pool = DataPool(8)
    pool.admit(torch.randint(0, 4, (4, GM.T)), torch.tensor([1, 2, 1, 2]),
               torch.zeros(4, dtype=torch.long), 0)
    l_im, l_pr, m = grl.streams(model, pool)
    ok_ns = (set(grl.last_im_ns.tolist()) <= {1, 2}) and (set(grl.last_pr_ns.tolist()) <= {1, 2})
    for p in model.parameters():
        p.grad = None
    l_im.backward()
    gE = sum(float(p.grad.abs().sum()) for p in model.E.parameters() if p.grad is not None)
    g_bc = float(model.D1.bias_c.grad.abs().sum()) if model.D1.bias_c.grad is not None else 0.0
    g_w = model.D1.lin.weight.grad
    w_zero = g_w is None or float(g_w.abs().sum()) == 0.0
    for p in model.parameters():
        p.grad = None
    l_pr.backward()
    g_t1 = model.heads.t1.weight.grad
    ok = (gE > 0 and g_bc > 0 and w_zero and ok_ns and m["n_im"] == 4 and m["n_pr"] == 4
          and g_t1 is not None and float(g_t1.abs().sum()) > 0)
    return ("PASS" if ok else "FAIL"), dict(gE=gE, g_bc=g_bc, w_zero=w_zero, ok_ns=ok_ns,
                                            n_im=m["n_im"], n_pr=m["n_pr"])


def a39(dev, seed=0):
    """A39 (乙案期首豁免 + 模仿墨权): in_warm 带界正确; im_wink>1 时 恒写空的模型 模仿损失严格大于
    逐格平均版 (墨格权重生效); 缺省口径 (im_wink=1/gen_warm=0) 的逐位平价由 A35 钉值覆盖."""
    torch.manual_seed(seed)
    from . import relearn as RL
    cfg = _small_cfg(d1_cm=1, gen_relearn=1, e_exp=2, t_gen=50, gen_warm=0.5, b_im=4, b_poolread=0)
    grl = RL.GenRelearn(cfg, dev)
    grl.period = 0
    grl.k = 9
    grl.exp = (1, 2)
    grl.steps_in = 10
    warm_in = bool(grl.in_warm)
    grl.steps_in = 30
    warm_out = not grl.in_warm
    grl.bank = RL.gen_scene_bank(grl.exp, RL.GenRelearn.BANK_PER_N, 123)
    pool = DataPool(8)
    kmix = torch.zeros(4, GM.T, dtype=torch.long)
    kmix[:, : GM.T // 2] = torch.randint(1, 4, (4, GM.T // 2))
    pool.admit(kmix, torch.tensor([1, 2, 1, 2]),
               torch.zeros(4, dtype=torch.long), 0)          # 半空半墨目标: 空写模型下 墨格 CE 大 / 空格 CE 小
    model = PredModel(d1_cm=True).to(dev)
    with torch.no_grad():
        model.D1.bias_c[0] = 5.0                             # 恒写空的写头
    grl.steps_in = 40
    l1, _, _ = grl.streams(model, pool)
    cfg.im_wink = 8.0
    grl2 = RL.GenRelearn(cfg, dev)
    grl2.period, grl2.k, grl2.exp, grl2.bank, grl2.steps_in = 0, 9, (1, 2), grl.bank, 40
    l8, _, _ = grl2.streams(model, pool)
    ok = warm_in and warm_out and l8 is not None and l1 is not None and float(l8) > float(l1) * 1.05
    return ("PASS" if ok else "FAIL"), dict(warm_in=warm_in, warm_out=warm_out,
                                            l_im_w1=float(l1) if l1 is not None else None,
                                            l_im_w8=float(l8) if l8 is not None else None)


def a40(dev, seed=0):
    """A40 (加法流, [U] 2026-08-27「输入两个scene，分别为N和m个物体，m = 1 - 3，输出笔记。除了输入外，流程和主流程内其他任务相同，不新造器官」):
    (a) 数据防火墙 + 内容对表: k∈{9,128} 各 100 项, n == n_a + m, m ∈ {1,2,3}, n_a / m / n / 候选 N 皆训练域, 标签 == targets(n_a+m, θ);
    (b) 写路径可达第二张场景: 批内滚动 x_m ⇒ 写 logits 改变 (pin 打在可达输入上);
    (c) 加法纸损失对 seg_cur / seg_org / 眼部首层权重 |g| 和 > 0 (非零梯度断言), 且模型无新参数 (参数键集 = 不开加法流同种子模型);
    (d) ab=None ⇒ chain_train_step 无加法项 (L_add None, ks_add None); add_on=0 的逐位平价由 A18 钉值覆盖."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 40)
    ok_a = True
    for k in (9, GM.K_MAX):
        tns = set(D.train_ns(k))
        for _ in range(100):
            it = D.sample_add_group(rng, k)
            if not (it["m"] in D.M_CHOICES and it["n"] == it["n_a"] + it["m"] and it["n_a"] in tns and it["m"] in tns
                    and it["n"] in tns and all(n in tns for n in it["cand_ns"]) and it["targets"] == D.targets(it["n"], it["theta"])):
                ok_a = False
    cfg = _small_cfg(d1_cm=1, add_on=1, b_add=4, add_workers=0, chain_k=1)
    model = PredModel(d1_cm=True).to(dev)
    keys_ref = set(PredModel(d1_cm=True).state_dict().keys())
    ab = build_batch([D.sample_add_group(rng, 9) for _ in range(4)], dev)
    w = TaskWeights().weights(dev)
    draw = R.draw_channel(4, cfg.s, rng, dev, cfg.occ_k)
    tpl = R.templates(draw.u, draw.s)
    out = ST.chain_side(model, ab, w, cfg.mu, cfg, draw, tpl)
    with torch.no_grad():
        e2r = model.E.enc_seq(model.E.tokenize(ab["x1"]), model.E.tokenize(ab["x_aux"].roll(1, 0)), seg=True)
        lg_r = ST.write_logits(model, e2r["tokens"], cfg)
    ok_b = not torch.equal(out["writes"][0]["logits"].detach(), lg_r)
    model.zero_grad(set_to_none=True)
    out["states"][1]["rows"].mean().backward()
    gs = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0
          for n, p in (("seg_cur", model.E.seg_cur), ("seg_org", model.E.seg_org), ("cnn0", model.E.cnn[0].weight))}
    ok_c = all(v > 0 for v in gs.values()) and set(model.state_dict().keys()) == keys_ref
    params = model.trainable_params()
    opt = torch.optim.AdamW(params, lr=1e-4)
    sb, pb, mb, _ = _mini_batches(model, cfg, dev, rng)
    mm = ST.chain_train_step(model, opt, params, sb, None, mb, cfg, w, dev, rng, ab=ab, rng_add=torch.Generator().manual_seed(1))
    mm0 = ST.chain_train_step(model, opt, params, sb, None, mb, cfg, w, dev, rng)
    ok_d = (mm["L_add"] is not None and mm["ks_add"] is not None and mm["ks_add"][0].shape == (4, GM.T)
            and mm0["L_add"] is None and mm0["ks_add"] is None and mm["accs_notes"].shape[0] == cfg.b_scene + 4)
    ok = ok_a and ok_b and ok_c and ok_d
    return ("PASS" if ok else "FAIL"), dict(data_firewall_and_labels=ok_a, write_sees_x_m=ok_b, grads=gs, no_new_params=ok_c,
                                            step_wiring=ok_d, L_add=mm["L_add"])
