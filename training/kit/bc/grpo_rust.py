"""exp223: GRPO を実戦に近い相手で。チームの Rust エンジン上で P9 の相手50体と打つ(公式環境と1コインまで一致を確認済み)。
   1反復: 組(相手・シード)× GROUP 回を CPU で並列に打ち(受け手 worker_rec.py が BC を温度 TEMP で確率的に打たせて局面と手を記録)、
   報酬 r = tanh(差/RSCALE)、組の平均との差を優位度にして(批評家なし)、PPO の切り取り + 初期値への KL で1回更新する。
   環境変数: TAG INIT ITERS NGRP GROUP TEMP RSCALE LR BETA ENT MB PAR"""
import os, sys, json, time, random, glob, subprocess, shutil
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M
E = lambda k, d: type(d)(os.environ.get(k, d))
TAG, INIT = os.environ["TAG"], os.environ["INIT"]
ITERS, NGRP, GROUP, PAR = E("ITERS", 60), E("NGRP", 24), E("GROUP", 8), E("PAR", 32)
TEMP, RSCALE, LR, BETA, ENT, MB, CLIP = E("TEMP", 0.7), E("RSCALE", 20000.0), E("LR", 2e-5), E("BETA", 0.1), E("ENT", 0.001), E("MB", 256), 0.2
ROOT = "/content/p9/p9_opponents/p9_opponents"; ARENA = "/content/lab/rust_arena/target/release/arena"; WREC = "/content/lab/rust_arena/worker_rec.py"
TPL = os.environ.get("TPL", "/content/lab/agents/bc_dmftS2_pace")          # main.py(修理層)と stock_target.json の雛形
IDS = [l.strip() for l in open("/content/lab/rust_arena/ids.txt") if l.strip()]
WD = "/content/lab/grpo/%s" % TAG; os.makedirs(WD, exist_ok=True)
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")
DEV = P.DEV
logf = open(WD + "/log.txt", "a")
def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
cfg = json.load(open(INIT + ".cfg.json")) if os.path.exists(INIT + ".cfg.json") else {}
net = M.load(INIT, DEV); ref = M.load(INIT, DEV)
for p_ in ref.parameters(): p_.requires_grad_(False)
opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); scaler = torch.amp.GradScaler("cuda")
ck = WD + "/ckpt.pt"; start = 0
if os.path.exists(ck):
    c = torch.load(ck, map_location=DEV); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); start = c["it"] + 1
    log("再開 it%d" % start)
log("設定: 相手 Rust P9 %d体 / 組 %d × %d回 / 温度 %.2f / 報酬 tanh(差/%.0f) / LR %g BETA %g ENT %g / 初期値 %s" % (len(IDS), NGRP, GROUP, TEMP, RSCALE, LR, BETA, ENT, INIT))
rng = random.Random(1234 + start)
for it in range(start, ITERS):
    t0 = time.time()
    # 1) 今の重みを受け手の置き場に書き出す
    AD = "%s/agent" % WD; os.makedirs(AD, exist_ok=True)
    for f in ("main.py", "model_tf.py", "obs_feat.py", "cfg.json", "stock_target.json"):
        if os.path.exists(TPL + "/" + f): shutil.copy(TPL + "/" + f, AD + "/" + f)
    torch.save(net.state_dict(), AD + "/policy.pt")
    GD = "%s/it%03d" % (WD, it); shutil.rmtree(GD, ignore_errors=True); os.makedirs(GD)
    tasks = []
    for gi in range(NGRP):
        opp = rng.choice(IDS); seed = rng.randrange(10 ** 8, 10 ** 9)
        for k in range(GROUP): tasks.append((gi, k, opp, seed))
    with open(GD + "/tasks.txt", "w") as f:
        for gi, k, opp, seed in tasks: f.write("%d %d %s %d\n" % (gi, k, opp, seed))
    # 2) 並列に打つ(1局 = arena 1本 + 受け手1本)
    cmd = ("cat %s/tasks.txt | xargs -P %d -L 1 bash -c 'REC_OUT=%s/g$0_k$1.npz SAMPLE_SEED=$((1000*$0+$1+%d)) TEMP=%s "
           "%s arena %s \"$2\" \"$3\" 0 %s/main.py python3 %s > %s/g$0_k$1.json 2>/dev/null || true'") % (
           GD, PAR, GD, it * 100000, TEMP, ARENA, ROOT, AD, WREC, GD)
    subprocess.run(cmd, shell=True, check=False)
    tg = time.time() - t0
    # 3) 報酬と組の優位度
    rows = []
    for gi, k, opp, seed in tasks:
        try:
            r = json.load(open("%s/g%d_k%d.json" % (GD, gi, k))); d = np.load("%s/g%d_k%d.npz" % (GD, gi, k))
            rows.append((gi, k, r["me"] - r["op"], r["me"], d))
        except Exception:
            pass
    if len(rows) < GROUP * 2: log("it%d 対局が集まらない(%d局)" % (it, len(rows))); continue
    mg = np.array([x[2] for x in rows]); rw = np.tanh(mg / RSCALE); gids = np.array([x[0] for x in rows]); A_ = np.zeros(len(rows))
    for g_ in set(gids.tolist()):
        ix = gids == g_; sd = rw[ix].std()
        A_[ix] = (rw[ix] - rw[ix].mean()) / (sd + 1e-6) if sd > 1e-6 else 0.0
    spread = float(np.mean([rw[gids == g_].std() for g_ in set(gids.tolist())]))
    D = {k: np.concatenate([x[4][k] for x in rows]) for k in KEYS}
    A = np.concatenate([np.full(len(x[4]["g"]), a, np.float32) for x, a in zip(rows, A_)])
    keep = A != 0
    D = {k: torch.as_tensor(v[keep], device=DEV) for k, v in D.items()}; A = torch.as_tensor(A[keep], device=DEV)
    for k in ("ul", "uq", "ms", "mq"): D[k] = D[k].long()
    D["lm"] = D["lm"].to(torch.uint8); D["tiles"] = D["tiles"].half()
    n = len(A); tu = time.time()
    # 4) 更新(1回、PPO の切り取り + 初期値への KL)
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
    s_ = np.mean(stats, 0)
    wins = float((mg > 0).mean())
    log("it%3d 局 %d(勝率 %.3f 差 中央値 %+.0f 自分 中央値 %.0f)組内の報酬SD %.3f | 学習局面 %d KL(初期値) %.4f ent %.2f 切 %.3f | 打つ %.0fs 更新 %.0fs" % (
        it, len(rows), wins, np.median(mg), np.median([x[3] for x in rows]), spread, n, s_[0], s_[1], s_[2], tg, time.time() - tu))
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), it=it), ck)
    if (it + 1) % E("SAVE_EVERY", 5) == 0: torch.save(net.state_dict(), "%s/net_it%03d.pt" % (WD, it + 1))
    shutil.rmtree(GD, ignore_errors=True)       # 1反復 ~45MB。重みだけ残す
log("== 終了 ==")
