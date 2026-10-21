"""exp176: 自己対戦＋リーグの PPO。両座席をエンジンの中で動かす(sp_lib)。
相手(1反復あたりの局数は環境変数で):
  self : 現行の方策どうし(両座席とも学習データになる)          NSELF
  v55  : v55 そのもの(公式環境と1コインまで一致を確認済み)      NV55
  d2   : 固定の d2(確率的)                                    ND2
  snap : 自分の過去の版(SNAP 反復ごとに保存、最大8体から選ぶ)   NSNAP
  tape : 学習用対局の相手の記録された手(元のシード)            NTAPE
更新は ppo_tf と同じ(決定ごとの比、別ネットの評価役、BC への KL)。
中間指標(EVAL 反復ごと): エンジン内で v55 と d2(貪欲)に対して 32シード x 両座席 = 64局(貪欲・種の上限)、加えて58局評価(凍った相手)。
使い方: python3 ppo_sp.py <tag> <初期の重み.pt>
"""
import json, os, sys, time, random, copy, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
from extract3 import acts_of
DEV = P.DEV; SC, WIN, GAMMA, LAM, CLIP = P.SCALE, P.WIN, P.GAMMA, P.LAM, P.CLIP
E = lambda k, d: type(d)(os.environ.get(k, d))
NSELF, NV55, ND2, NSNAP, NTAPE = E("NSELF", 12), E("NV55", 16), E("ND2", 8), E("NSNAP", 8), E("NTAPE", 12)
ITERS, VWARM, EVAL, SNAP = E("ITERS", 300), E("VWARM", 5), E("EVAL", 10), E("SNAP", 10)
LR, BETA, ENT, MB = E("LR", 1e-4), E("BETA", 0.1), E("ENT", 0.001), E("MB", 256)
MKTONLY = E("MKTONLY", 0)   # 1: ユニットは BC(ref)の貪欲に固定し、市場の決定だけを学習する(exp179)
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")
EVSEEDS = [2_000_000 + 7919 * k for k in range(32)]

def mixed_eval(net, opp, seeds=EVSEEDS, unit_net=None):
    """net(貪欲・種の上限)対 opp、各シードで両座席。unit_net があればユニットはそちら(市場だけ RL)。返り値: 勝率, 差, 自分の平均。"""
    mg = []; me = []
    me_agent = S.MixedOpp(unit_net, net) if unit_net is not None else S.PolicyOpp(net)
    for s in (0, 1):
        m = S.play(seeds, {s: me_agent, 1 - s: opp})
        mg.append(m[:, s] - m[:, 1 - s]); me.append(m[:, s])
    mg = np.concatenate(mg); me = np.concatenate(me)
    return float((mg > 0).mean() + 0.5 * (mg == 0).mean()), float(mg.mean()), float(me.mean())

def main():
    tag, init = sys.argv[1], sys.argv[2]
    logf = open("/content/local_logs/sp_%s.log" % tag, "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    net = M.load(init, DEV); ref = M.load(init, DEV); crit = M.load(init, DEV); d2 = M.load("/content/local_logs/tf_d2.pt", DEV)
    for p in ref.parameters(): p.requires_grad_(False)
    net.eval()                                   # ドロップアウトは切る(p1 の教訓)
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); opt_c = torch.optim.Adam(crit.parameters(), lr=1e-4, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda"); scaler_c = torch.amp.GradScaler("cuda")
    split = json.load(open(os.environ.get("SPLITFILE", "/content/local_logs/bc_d2.json"))); train_e, hold_e = split["train"], split["hold"]
    ck = "/content/local_logs/sp_%s.pt" % tag; start = 0; snaps = []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV); net.load_state_dict(c["net"]); crit.load_state_dict(c["crit"]); opt.load_state_dict(c["opt"]); opt_c.load_state_dict(c["opt_c"]); start = c["it"] + 1
        snaps = c.get("snaps", []); log("再開 it%d" % start)
    v55 = S.V55Opp()
    if start == 0 and not os.environ.get("SKIPEV0"):
        w, m, me = mixed_eval(net, S.V55Opp(), unit_net=ref if MKTONLY else None); log("it  -1 [中間指標 対v55 64局] 勝率 %.3f 差 %+.0f 自分 %.0f (初期)" % (w, m, me))
        w, m, me = mixed_eval(net, S.PolicyOpp(d2), unit_net=ref if MKTONLY else None); log("it  -1 [中間指標 対d2 64局] 勝率 %.3f 差 %+.0f 自分 %.0f (初期)" % (w, m, me))
    rng = random.Random(7)
    for it in range(start, ITERS):
        t0 = time.time()
        # --- 対局の組み立て: 局ごとに (種類, 学習側の座席, シード) ---
        games = []
        for kind, n in (("self", NSELF), ("v55", NV55), ("d2", ND2), ("snap", NSNAP if snaps else 0), ("tape", NTAPE)):
            for _ in range(n):
                if kind == "tape":
                    e = rng.choice(train_e); z = np.load("%s/%s.npz" % (P.CACHE, e)); mt = json.loads(str(z["meta"]))
                    games.append(dict(kind=kind, seat=mt["seat"], seed=mt["seed"], opp_acts=json.loads(str(z["opp_acts"]))))   # 相手の手は元の座席で再生
                else:
                    games.append(dict(kind=kind, seat=rng.randrange(2), seed=rng.randrange(10**9)))
        pool = S.SPPool([g["seed"] for g in games]); N = len(games)
        snap_net = None
        if snaps:
            snap_net = M.load("/content/local_logs/tf_d2.pt", DEV); snap_net.load_state_dict(torch.load(rng.choice(snaps), map_location=DEV)); snap_net.eval()
        # 学習データになる軌道: (局, 座席)。self は両座席
        trajs = [(i, g["seat"]) for i, g in enumerate(games)] + [(i, 1 - g["seat"]) for i, g in enumerate(games) if g["kind"] == "self"]
        buf = {k: [] for k in KEYS + ("v", "r")}
        prev = pool.money(); v55.pick = {}
        for t in range(719):
            st = {s: pool.state(s) for s in (0, 1)}
            U = {s: np.zeros((N, NU_, 3), np.int32) for s in (0, 1)}; O = {s: np.zeros((N, 10, 3), np.int32) for s in (0, 1)}
            # 学習側(と self の相手側)を現行の方策でまとめて標本
            tix = np.array([i for i, _ in trajs]); tseat = np.array([s for _, s in trajs])
            stL = tuple(np.stack([st[s][k][i] for i, s in zip(tix, tseat)]) for k in range(5))
            if MKTONLY: ul, uq, ms, mq, v = S.mixed_sample(ref, net, stL, False, crit=crit)
            else:
                outs = [P.sample(net, tuple(x[j:j + 64] for x in stL), crit=crit) for j in range(0, len(tix), 64)]
                ul, uq, ms, mq, v = [np.concatenate([o[k] for o in outs]) for k in range(5)]
            UL, OL = P.to_codes(ul, uq, ms, mq, stL[1])
            for j, (i, s) in enumerate(trajs): U[s][i], O[s][i] = UL[j], OL[j]
            for k, a in zip(KEYS, list(stL) + [ul, uq, ms, mq]): buf[k].append(a)
            buf["v"].append(v)
            # 固定の相手
            for kind, obj in (("v55", v55), ("d2", S.PolicyOpp(d2, greedy=False, cap=True)), ("snap", S.PolicyOpp(snap_net, greedy=False, cap=True) if snap_net else None), ("tape", None)):
                ids = [i for i, g in enumerate(games) if g["kind"] == kind]
                if not ids: continue
                for s in (0, 1):
                    sub = [i for i in ids if 1 - games[i]["seat"] == s]
                    if not sub: continue
                    sts = tuple(x[sub] for x in st[s])
                    if kind == "tape": u, o = S.TapeOpp([games[i]["opp_acts"] for i in sub]).act(sts, t, list(range(len(sub))), pool, s)
                    elif kind == "v55": u, o = v55.act(sts, t, sub, pool, s)      # step144 で毛糸店を見て継続を決める(sp_lib.V55Opp)
                    else: u, o = obj.act(sts, t, None, pool, s)
                    for j, i in enumerate(sub): U[s][i], O[s][i] = u[j], o[j]
            pool.step(U[0], O[0], U[1], O[1])
            cur = pool.money(); r = np.array([((cur[i, s] - cur[i, 1 - s]) - (prev[i, s] - prev[i, 1 - s])) / SC for i, s in trajs]); prev = cur
            if t == 718: r = r + WIN * np.sign(np.array([cur[i, s] - cur[i, 1 - s] for i, s in trajs]))
            buf["r"].append(r)
        tr = time.time() - t0
        fin = pool.money(); mg_by = {}
        for (i, s) in trajs[:N]: mg_by.setdefault(games[i]["kind"], []).append(fin[i, s] - fin[i, 1 - s])
        # --- GAE と更新(ppo_tf と同じ) ---
        V = np.stack(buf["v"]); Rw = np.stack(buf["r"]); T_ = len(Rw); Nn = V.shape[1]
        adv = np.zeros_like(Rw); last = np.zeros(Nn)
        for t in reversed(range(T_)):
            nv = V[t + 1] if t + 1 < T_ else np.zeros(Nn); delta = Rw[t] + GAMMA * nv - V[t]; last = delta + GAMMA * LAM * last; adv[t] = last
        ret = adv + V
        flat = lambda k, dt: torch.as_tensor(np.concatenate(buf[k]), device=DEV, dtype=dt)
        D = dict(tiles=flat("tiles", torch.float16), ou=flat("ou", torch.float32), pu=flat("pu", torch.float32), g=flat("g", torch.float32),
                 lm=flat("lm", torch.uint8), ul=flat("ul", torch.long), uq=flat("uq", torch.long), ms=flat("ms", torch.long), mq=flat("mq", torch.long))
        del buf
        A = torch.as_tensor(adv.reshape(-1), device=DEV, dtype=torch.float32); RT = torch.as_tensor(ret.reshape(-1), device=DEV, dtype=torch.float32)
        A = (A - A.mean()) / (A.std() + 1e-8); n = len(A)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            PER0 = torch.cat([P.evaluate(net, [D[k][i:i + MB] for k in KEYS])[4].float() for i in range(0, n, MB)])
        stats = []; vonly = it < VWARM
        for ep in range(2):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, MB):
                idx = perm[i:i + MB]; b = [D[k][idx] for k in KEYS]
                with torch.autocast("cuda", dtype=torch.float16):
                    hc, _ = crit.encode(b[0], b[1], b[2], b[3]); vc = crit.value(hc[:, -1]).squeeze(-1)
                vl = F.smooth_l1_loss(vc.float(), RT[idx]); opt_c.zero_grad(); scaler_c.scale(vl).backward(); scaler_c.unscale_(opt_c)
                torch.nn.utils.clip_grad_norm_(crit.parameters(), 0.5); scaler_c.step(opt_c); scaler_c.update()
                if vonly: continue
                with torch.autocast("cuda", dtype=torch.float16):
                    lp, ent, v, dist, per, pmask = P.evaluate(net, b)
                ent, per = ent.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
                if MKTONLY: pmask = torch.cat([torch.zeros_like(pmask[:, :20]), pmask[:, 20:]], 1)
                rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
                pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    rd = P.evaluate(ref, b)[3]
                rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
                if MKTONLY:
                    lsm, valid = dist[2], dist[3]; rsm = rd[2]; kl = ((lsm.exp() * (lsm - rsm)).sum(-1) * valid).sum(1).mean()
                else: kl = P.kl_to_ref(dist, rd).mean()
                loss = pl + BETA * kl - ENT * ent.mean()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
                stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean())))
        ev_ = 1 - np.var(ret.reshape(-1) - V.reshape(-1)) / (np.var(ret.reshape(-1)) + 1e-8); s_ = np.mean(stats, 0) if stats else [0, 0, 0, 0]
        log("it%4d " % it + " ".join("%s %+.0f(%.2f)" % (k, np.mean(v), np.mean(np.array(v) > 0)) for k, v in mg_by.items()) +
            " | KL %.3f ent %.2f 切 %.2f 価値EV %.2f %s| 収集 %.0fs 全体 %.0fs" % (s_[1], s_[2], s_[3], ev_, "価値のみ " if vonly else "", tr, time.time() - t0))
        del D
        if it % SNAP == SNAP - 1:
            p = "/content/local_logs/sp_%s_it%d.pt" % (tag, it); torch.save(net.state_dict(), p); snaps = (snaps + [p])[-8:]
        torch.save(dict(net=net.state_dict(), crit=crit.state_dict(), opt=opt.state_dict(), opt_c=opt_c.state_dict(), it=it, snaps=snaps), ck)
        try: torch.save(dict(net=net.state_dict(), crit=crit.state_dict(), it=it), "/content/drive/MyDrive/kaggriculture/runs/sp_%s_ck.pt" % tag)
        except Exception: pass
        if it % EVAL == EVAL - 1:
            w, m, me = mixed_eval(net, S.V55Opp(), unit_net=ref if MKTONLY else None); log("it%4d [中間指標 対v55 64局] 勝率 %.3f 差 %+.0f 自分 %.0f" % (it, w, m, me))
            w, m, me = mixed_eval(net, S.PolicyOpp(d2), unit_net=ref if MKTONLY else None); log("it%4d [中間指標 対d2 64局] 勝率 %.3f 差 %+.0f 自分 %.0f" % (it, w, m, me))
            if not MKTONLY: me, wr, mg = P.eval_policy(net, hold_e, True, 2, cap=True); log("it%4d [58局 凍った相手 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))

NU_ = S.NU

if __name__ == "__main__":
    main()
