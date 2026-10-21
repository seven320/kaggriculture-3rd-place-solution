"""exp198: Orbit Wars 1位(Tufa Labs「Scaling Reinforcement Learning to the Stars」)の型を移した自己対戦 PPO。

   1位から持ってきたもの:
     - 報酬は**勝ち +1 / 負け −1 だけ**(形を付けない。コインの差は使わない)
     - **「直前の最良版」への関門**: GATE 反復ごとに 現行 対 最良版 を直接対戦させ、GATE_WR 以上勝てば最良版を置き換える
     - **KL 罰則は「直前の最良版」に対して**(BC の初期値に縛り続けない)
     - 並列の局数を増やす(1位は数千。ここは NSELF 等で指定)
   1位が「やればよかった」と書いていたもの:
     - **過去の最良版を少し混ぜる**(リーグ。戦略の堂々巡りと自己過適合を防ぐ)
   こちらの事情で足したもの:
     - 初期値は BC(d2xl)。計算量が1位の約1/8なので、ゼロからではなく BC から始める
     - tape_ug(v57 の土台)を少しだけ相手に混ぜ、同時にものさしにも使う

   common/ は変えていない(sp_lib・ppo_tf・model_tf を import して使うだけ)。
   使い方: python3 ppo_orbit.py <tag> <初期の重み.pt>
   環境変数: NSELF NBEST NLEAGUE NTAPEUG ITERS GATE GATE_WR EVAL LR BETA ENT MB OPPDIR SPLITFILE
"""
import copy, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
if os.environ.get("FASTPOOL", "1") == "1":          # exp199: 全局を1つの VecMarket に入れる版(値は SPPool とビット一致、環境側は約5倍)
    import sp_fast; S.SPPool = sp_fast.BatchSPPool

DEV = P.DEV; GAMMA, LAM, CLIP = P.GAMMA, P.LAM, P.CLIP
E = lambda k, d: type(d)(os.environ.get(k, d))
NSELF, NBEST, NLEAGUE, NTAPEUG = E("NSELF", 48), E("NBEST", 24), E("NLEAGUE", 12), E("NTAPEUG", 12)
ITERS, GATE, GATE_WR, EVAL, VWARM = E("ITERS", 200), E("GATE", 5), E("GATE_WR", 0.70), E("EVAL", 5), E("VWARM", 3)
LR, BETA, ENT, MB = E("LR", 5e-5), E("BETA", 0.1), E("ENT", 0.001), E("MB", 256)
OPPDIR = os.environ["OPPDIR"]
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")
EVSEEDS = [2_000_000 + 7919 * k for k in range(64)]      # 対 tape_ug のものさし(128局。64局では揺れが大きく舵取りを誤った — weval で確認)
GATESEEDS = [5_000_000 + 7919 * k for k in range(32)]    # 関門用(ものさしとは別)


def h2h(a, b, seeds, greedy=True):
    """a 対 b を各シード両座席で。a から見た勝率と差。"""
    mg = []
    for s in (0, 1):
        m = S.play(seeds, {s: S.PolicyOpp(a, greedy=greedy, cap=True), 1 - s: S.PolicyOpp(b, greedy=greedy, cap=True)})
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg)
    return float((mg > 0).mean() + 0.5 * (mg == 0).mean()), float(mg.mean())


def vs_tape(net, seeds=EVSEEDS):
    mg = []
    for s in (0, 1):
        m = S.play(seeds, {s: S.PolicyOpp(net, greedy=True, cap=True), 1 - s: S.TapeRouterOpp(OPPDIR)})
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg)
    return float((mg > 0).mean()), float(mg.mean())


def main():
    tag, init = sys.argv[1], sys.argv[2]
    L = "/content/local_logs"; logf = open("%s/orbit_%s.log" % (L, tag), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    cfg = json.load(open(init + ".cfg.json")) if os.path.exists(init + ".cfg.json") else {}
    mk = lambda: M.Policy(**cfg).to(DEV)                    # 過去の版もすべて初期値と同じ構成で作る(ppo_sp の d2 構成の不具合を避ける)
    net = M.load(init, DEV); crit = M.load(init, DEV); best = M.load(init, DEV); ref = M.load(init, DEV)
    for p in list(best.parameters()) + list(ref.parameters()): p.requires_grad_(False)
    net.eval(); best.eval(); ref.eval()
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); opt_c = torch.optim.Adam(crit.parameters(), lr=1e-4, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda"); scaler_c = torch.amp.GradScaler("cuda")
    split = json.load(open(os.environ.get("SPLITFILE", L + "/bc_d2.json"))); hold_e = split["hold"]
    ck = "%s/orbit_%s.pt" % (L, tag); start = 0; gen = 0; league = []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(c["net"]); crit.load_state_dict(c["crit"]); opt.load_state_dict(c["opt"]); opt_c.load_state_dict(c["opt_c"])
        best.load_state_dict(c["best"]); ref.load_state_dict(c["best"]); start = c["it"] + 1; gen = c["gen"]; league = c["league"]
        log("再開 it%d(最良版は第%d世代、リーグ %d体)" % (start, gen, len(league)))
    else:
        w, m = vs_tape(net); log("it  -1 [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f (初期)" % (w, m))
    log("設定: NSELF %d NBEST %d NLEAGUE %d NTAPEUG %d / 関門 %d反復ごと・勝率 %.2f 以上で置き換え / LR %g BETA %g / 報酬は勝ち±1のみ" % (
        NSELF, NBEST, NLEAGUE, NTAPEUG, GATE, GATE_WR, LR, BETA))
    rng = random.Random(11)
    for it in range(start, ITERS):
        t0 = time.time()
        games = []
        for kind, n in (("self", NSELF), ("best", NBEST), ("league", NLEAGUE if league else 0), ("tape", NTAPEUG)):
            for _ in range(n): games.append(dict(kind=kind, seat=rng.randrange(2), seed=rng.randrange(10 ** 9)))
        pool = S.SPPool([g["seed"] for g in games]); N = len(games)
        lg_net = None
        if league:
            lg_net = mk(); lg_net.load_state_dict(torch.load(rng.choice(league), map_location=DEV)); lg_net.eval()
        tape = S.TapeRouterOpp(OPPDIR)
        trajs = [(i, g["seat"]) for i, g in enumerate(games)] + [(i, 1 - g["seat"]) for i, g in enumerate(games) if g["kind"] == "self"]
        buf = {k: [] for k in KEYS + ("v",)}
        for t in range(719):
            st = {s: pool.state(s) for s in (0, 1)}
            U = {s: np.zeros((N, S.NU, 3), np.int32) for s in (0, 1)}; O = {s: np.zeros((N, 10, 3), np.int32) for s in (0, 1)}
            tix = np.array([i for i, _ in trajs]); tseat = np.array([s for _, s in trajs])
            stL = tuple(np.stack([st[s][k][i] for i, s in zip(tix, tseat)]) for k in range(5))
            outs = [P.sample(net, tuple(x[j:j + 128] for x in stL), crit=crit) for j in range(0, len(tix), 128)]
            ul, uq, ms, mq, v = [np.concatenate([o[k] for o in outs]) for k in range(5)]
            UL, OL = P.to_codes(ul, uq, ms, mq, stL[1])
            for j, (i, s) in enumerate(trajs): U[s][i], O[s][i] = UL[j], OL[j]
            for k, a in zip(KEYS, list(stL) + [ul, uq, ms, mq]): buf[k].append(a)
            buf["v"].append(v)
            for kind, obj in (("best", S.PolicyOpp(best, greedy=False, cap=True)), ("league", S.PolicyOpp(lg_net, greedy=False, cap=True) if lg_net else None), ("tape", tape)):
                ids = [i for i, g in enumerate(games) if g["kind"] == kind]
                if not ids or obj is None: continue
                for s in (0, 1):
                    sub = [i for i in ids if 1 - games[i]["seat"] == s]
                    if not sub: continue
                    sts = tuple(x[sub] for x in st[s])
                    u, o = obj.act(sts, t, sub, pool, s) if kind == "tape" else obj.act(sts, t, None, pool, s)
                    for j, i in enumerate(sub): U[s][i], O[s][i] = u[j], o[j]
            pool.step(U[0], O[0], U[1], O[1])
        fin = pool.money(); tr = time.time() - t0
        # 報酬は最後の手の勝ち負けだけ(+1 / −1、引き分け 0)
        z = np.array([np.sign(fin[i, s] - fin[i, 1 - s]) for i, s in trajs])
        V = np.stack(buf["v"]); T_ = V.shape[0]; Nn = V.shape[1]
        Rw = np.zeros_like(V); Rw[-1] = z
        adv = np.zeros_like(Rw); last = np.zeros(Nn)
        for t in reversed(range(T_)):
            nv = V[t + 1] if t + 1 < T_ else np.zeros(Nn); delta = Rw[t] + GAMMA * nv - V[t]; last = delta + GAMMA * LAM * last; adv[t] = last
        ret = adv + V
        byk = {}
        for (i, s), zz in zip(trajs, z): byk.setdefault(games[i]["kind"], []).append(zz)
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
                rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
                pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    rd = P.evaluate(ref, b)[3]
                rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
                kl = P.kl_to_ref(dist, rd).mean()
                loss = pl + BETA * kl - ENT * ent.mean()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
                stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean())))
        del D
        ev_ = 1 - np.var(ret.reshape(-1) - V.reshape(-1)) / (np.var(ret.reshape(-1)) + 1e-8); s_ = np.mean(stats, 0) if stats else [0, 0, 0, 0]
        log("it%4d " % it + " ".join("%s %.2f(%d)" % (k, (np.array(v) > 0).mean(), len(v)) for k, v in byk.items()) +
            " | KL %.3f ent %.2f 切 %.2f 価値EV %.2f %s| 局面 %d 収集 %.0fs 全体 %.0fs" % (s_[1], s_[2], s_[3], ev_, "価値のみ " if vonly else "", n, tr, time.time() - t0))
        # 関門: 現行 対 最良版(貪欲・種の上限 = 提出と同じ打ち方、64局)
        if it % GATE == GATE - 1 and not vonly:
            w, m = h2h(net, best, GATESEEDS)
            if w >= GATE_WR:
                gen += 1; p = "%s/orbit_%s_gen%d.pt" % (L, tag, gen); torch.save(net.state_dict(), p)
                best.load_state_dict(net.state_dict()); ref.load_state_dict(net.state_dict()); league = (league + [p])[-8:]
                log("it%4d [関門] 現行 対 最良版 64局 勝率 %.3f 差 %+.0f → **置き換え(第%d世代)**" % (it, w, m, gen))
            else:
                log("it%4d [関門] 現行 対 最良版 64局 勝率 %.3f 差 %+.0f → 据え置き(第%d世代のまま)" % (it, w, m, gen))
        torch.save(dict(net=net.state_dict(), crit=crit.state_dict(), opt=opt.state_dict(), opt_c=opt_c.state_dict(), it=it,
                        best=best.state_dict(), gen=gen, league=league), ck)
        if it % EVAL == EVAL - 1:
            w, m = vs_tape(net); log("it%4d [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f(最良版は第%d世代)" % (it, w, m, gen))
            me, wr, mg = P.eval_policy(net, hold_e, True, 2, cap=True); log("it%4d [58局 凍った相手 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))


if __name__ == "__main__":
    main()
