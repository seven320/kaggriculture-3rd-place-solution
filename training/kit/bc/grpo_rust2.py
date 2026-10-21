"""exp223b: grpo_rust.py の高速版。Rust の batch モードで全局を1プロセスで進め、1手ごとに全局ぶんの観測をまとめて GPU で推論する。
   (1局ずつ CPU で推論すると 1反復 27分のうち27分が「打つ」だった。)
   定期的に、同じ物差し(P9 の相手50体 × 固定シード、貪欲)でも測る。
   環境変数: TAG INIT ITERS NGRP GROUP TEMP RSCALE LR BETA ENT MB EVAL_EVERY EVAL_SEEDS TPL"""
import os, sys, json, time, random, subprocess, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M
E = lambda k, d: type(d)(os.environ.get(k, d))
TAG, INIT = os.environ["TAG"], os.environ["INIT"]
ITERS, NGRP, GROUP = E("ITERS", 200), E("NGRP", 24), E("GROUP", 8)
TEMP, RSCALE, LR, BETA, ENT, MB, CLIP = E("TEMP", 0.7), E("RSCALE", 20000.0), E("LR", 2e-5), E("BETA", 0.1), E("ENT", 0.001), E("MB", 256), 0.2
EVAL_EVERY = E("EVAL_EVERY", 5); SUB = E("SUB", 1.0); EVAL_SEEDS = [int(x) for x in os.environ.get("EVAL_SEEDS", "730001 730002 730003 730004").split()]
ROOT = "/content/p9/p9_opponents/p9_opponents"; ARENA = "/content/lab/rust_arena/target/release/arena"
TPL = os.environ.get("TPL", "/content/lab/agents/bc_dmftS2_pace")
IDS = [l.strip() for l in open("/content/lab/rust_arena/ids.txt") if l.strip()]
WD = "/content/lab/grpo/%s" % TAG; os.makedirs(WD, exist_ok=True)
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")
DEV = P.DEV; NM, NS, M_STOP = 23, 11, 22
logf = open(WD + "/log.txt", "a")
def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
# 修理層(_pace)・特徴(featurize)・行動への変換(to_action)は提出物の main.py と同じものを使う
tp = os.path.join(TPL, "main.py"); sys.path.insert(0, TPL)
mod = types.ModuleType("tpl"); exec(compile(open(tp).read(), tp, "exec"), mod.__dict__)
featurize, to_action, pace = mod.featurize, mod.to_action, getattr(mod, "_pace", None)
tt = lambda a, dt=torch.float32: torch.as_tensor(a, device=DEV, dtype=dt)
@torch.no_grad()
def sample_batch(net, st, temp):
    """ppo_tf.sample(cap=True)に温度を足したもの。temp<=0 は貪欲。"""
    tiles, ou, pu, g, lm = st; B = len(g)
    pick = (lambda l: l.argmax(-1)) if temp <= 0 else (lambda l: torch.distributions.Categorical(logits=l.float() / temp).sample())
    with torch.autocast("cuda", dtype=torch.float16):
        h, pad = net.encode(tt(tiles, torch.float16), tt(ou), tt(pu), tt(g))
        lg, q = net.unit_logits(h, tt(ou), tt(lm, torch.uint8))
        live = tt(ou[..., 0] > 0, torch.bool)
        seeds = torch.round(tt(g[:, 27:32]) * 20).long(); used = torch.zeros_like(seeds); cols = []
        for u in range(lg.shape[1]):
            l = lg[:, u].float().clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
            c = pick(l); pl = (c >= 15) & (c < 20) & live[:, u]
            used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
        ul = torch.stack(cols, 1); ul = torch.where(live, ul, torch.full_like(ul, -1))
        uq = pick(net.unit_qty(q, ul)); uq = torch.where(ul >= 20, uq, torch.zeros_like(uq))
        prev = torch.full((B, 1), NM, device=DEV, dtype=torch.long); done = torch.zeros(B, dtype=torch.bool, device=DEV)
        ms = torch.full((B, NS), M_STOP, device=DEV, dtype=torch.long); mq = torch.zeros((B, NS), device=DEV, dtype=torch.long)
        for j in range(NS):
            mo, y = net.market(h, pad, prev); lj = mo[:, -1].float()
            if j == NS - 1: lj = torch.full_like(lj, torch.finfo(lj.dtype).min); lj[:, M_STOP] = 0
            cj = pick(lj); cj = torch.where(done, torch.full_like(cj, M_STOP), cj); ms[:, j] = cj
            qc = pick(net.market_qty(y[:, -1:], cj.view(-1, 1))[:, 0])
            mq[:, j] = torch.where((cj < 19) & ~done, qc, torch.zeros_like(qc))
            done = done | (cj == M_STOP); prev = torch.cat([prev, cj.clamp(max=NM).view(-1, 1)], 1)
            if bool(done.all()): break
    return ul.cpu().numpy(), uq.cpu().numpy(), ms.cpu().numpy(), mq.cpu().numpy()
def play_batch(net, games, temp, record):
    """games: [{"opp","seed","seat"}]。戻り値: 結果 [(自分, 相手)]、記録 {局: {KEYS: 配列}}"""
    pr = subprocess.Popen([ARENA, "batch", ROOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    pr.stdin.write(json.dumps({"games": games}) + "\n"); pr.stdin.flush()
    rec = {i: {k: [] for k in KEYS} for i in range(len(games))} if record else None
    while True:
        line = pr.stdout.readline()
        if not line: raise RuntimeError("arena batch が途中で終わった")
        msg = json.loads(line)
        if msg.get("done"): res = msg["results"]; break
        obs = msg["obs"]; idx = [i for i, o in enumerate(obs) if o is not None]
        F = [featurize(obs[i]) for i in idx]
        st = tuple(np.stack([f[k] for f in F]) for k in range(5))
        ul, uq, ms, mq = sample_batch(net, st, temp)
        acts = [None] * len(obs)
        for j, i in enumerate(idx):
            a = to_action(ul[j], uq[j], ms[j], mq[j], int((st[1][j][:, 0] > 0).sum()))
            if pace is not None: a = pace(a, obs[i])
            acts[i] = a
            if record:
                r = rec[i]
                for k, v in zip(KEYS, (st[0][j].astype(np.float16), st[1][j], st[2][j], st[3][j], st[4][j], ul[j], uq[j], ms[j], mq[j])): r[k].append(v)
        pr.stdin.write(json.dumps({"actions": acts}) + "\n"); pr.stdin.flush()
    pr.wait()
    if record: rec = {i: {k: np.stack(v) for k, v in r.items()} for i, r in rec.items()}
    return res, rec
net = M.load(INIT, DEV); ref = M.load(INIT, DEV)
for p_ in ref.parameters(): p_.requires_grad_(False)
net.eval(); ref.eval()
opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); scaler = torch.amp.GradScaler("cuda")
ck = WD + "/ckpt.pt"; start = 0
if os.path.exists(ck):
    c = torch.load(ck, map_location=DEV); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); start = c["it"] + 1; log("再開 it%d" % start)
EVG = [dict(opp=o, seed=s, seat=0) for o in IDS for s in EVAL_SEEDS]
def evaluate(tag):
    t = time.time(); res, _ = play_batch(net, EVG, 0.0, False)
    me = np.array([r[0] for r in res]); op = np.array([r[1] for r in res])
    log("%s [ものさし P9 の相手50体 × %dシード 貪欲 %d局] 勝率 %.3f 差 %+.0f 自分 中央値 %.0f 相手 中央値 %.0f(%.0fs)" % (
        tag, len(EVAL_SEEDS), len(res), (me > op).mean(), (me - op).mean(), np.median(me), np.median(op), time.time() - t))
    return me, op
log("設定: 相手 Rust P9 %d体 / 組 %d × %d回 / 温度 %.2f / 報酬 tanh(差/%.0f) / LR %g BETA %g ENT %g / 初期値 %s / まとめて推論(batch)/ 局面の使用割合 %.2f" % (len(IDS), NGRP, GROUP, TEMP, RSCALE, LR, BETA, ENT, INIT, SUB))
if start == 0: evaluate("it  -1")
rng = random.Random(4321 + start)
for it in range(start, ITERS):
    t0 = time.time()
    games = []
    for gi in range(NGRP):
        opp = rng.choice(IDS); seed = rng.randrange(10 ** 8, 10 ** 9)
        for k in range(GROUP): games.append(dict(opp=opp, seed=seed, seat=0, gid=gi))
    res, rec = play_batch(net, games, TEMP, True); tg = time.time() - t0
    mg = np.array([r[0] - r[1] for r in res]); rw = np.tanh(mg / RSCALE); gids = np.array([g["gid"] for g in games]); A_ = np.zeros(len(games))
    for g_ in set(gids.tolist()):
        ix = gids == g_; sd = rw[ix].std()
        A_[ix] = (rw[ix] - rw[ix].mean()) / (sd + 1e-6) if sd > 1e-6 else 0.0
    spread = float(np.mean([rw[gids == g_].std() for g_ in set(gids.tolist())]))
    sel = [i for i in range(len(games)) if A_[i] != 0]
    D = {k: np.concatenate([rec[i][k] for i in sel]) for k in KEYS}
    A = np.concatenate([np.full(len(rec[i]["g"]), A_[i], np.float32) for i in sel]); del rec
    if SUB < 1.0:     # 1反復で動かしすぎないよう、局面の一部だけ使う(r2 は 18万局面 × 1回で物差しが 0.995 → 0.900 に悪化した)
        keep = np.random.rand(len(A)) < SUB; D = {k: v[keep] for k, v in D.items()}; A = A[keep]
    D = {k: torch.as_tensor(v, device=DEV) for k, v in D.items()}; A = torch.as_tensor(A, device=DEV)
    for k in ("ul", "uq", "ms", "mq"): D[k] = D[k].long()
    D["lm"] = D["lm"].to(torch.uint8); D["tiles"] = D["tiles"].half()
    n = len(A); tu = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        PER0 = torch.cat([P.evaluate(net, [D[k][i:i + MB] for k in KEYS])[4].float() for i in range(0, n, MB)])
    stats = []; perm = torch.randperm(n, device=DEV)
    for i in range(0, n, MB):
        idx = perm[i:i + MB]; b = [D[k][idx] for k in KEYS]
        with torch.autocast("cuda", dtype=torch.float16):
            lp, ent, v, dist, per, pmask = P.evaluate(net, b)
        ent, per = ent.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
        rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
        pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            rd = P.evaluate(ref, b)[3]
        rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
        kl = P.kl_to_ref(dist, rd).mean(); loss = pl + BETA * kl - ENT * ent.mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
        stats.append((float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean())))
    del D, A, PER0; torch.cuda.empty_cache(); s_ = np.mean(stats, 0)
    log("it%3d 局 %d(勝率 %.3f 差 中央値 %+.0f 自分 中央値 %.0f)組内の報酬SD %.3f | 学習局面 %d KL(初期値) %.4f ent %.2f 切 %.3f | 打つ %.0fs 更新 %.0fs" % (
        it, len(games), (mg > 0).mean(), np.median(mg), np.median([r[0] for r in res]), spread, n, s_[0], s_[1], s_[2], tg, time.time() - tu))
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), it=it), ck)
    if (it + 1) % EVAL_EVERY == 0:
        torch.save(net.state_dict(), "%s/net_it%03d.pt" % (WD, it + 1)); evaluate("it%4d" % it)
log("== 終了 ==")
