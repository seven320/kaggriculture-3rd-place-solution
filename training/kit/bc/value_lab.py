"""exp226: 批評家の診断(PPO の外で)。BC 方策(固定)で局を集め、局ごとに学習/検証を分け、
いくつかの批評家の作り方で「未見の局の勝敗」をどれだけ当てるか(価値EV)を局面の時期ごとに測る。

局の集め方は ppo_orbit2 と同じ(sp_fast の BatchSPPool、方策は確率的に打つ)。相手は tape_ug(TapeRouter)と自己対戦。
状態は STRIDE 手ごとに間引いて保存(同じ局の局面は強く相関しているので、全手は要らない)。

作り方(VARIANTS):
  full_l1   : 今の PPO と同じ。批評家全体(BC の重みから)を smooth L1 で z(±1)に当てる、lr 1e-4
  head_l1   : 胴体を止めて価値ヘッドだけ
  full_ce   : 勝ち負けの2値分類(交差エントロピー)、V = 2p − 1
  full_mg   : 目標を tanh(最終の差額 / 20000) にする(EV は z に対して測る)
  full_l1_q : full_l1 を学習局 1/4 で(局数の効き目)
  full_l1_s : full_l1 を学習局 1/16 で(≒ PPO の数反復ぶん)
  orc_l1    : exp242(PTCG 2・3・6・9位の神託批評家): full_l1 + 相手の観測(相手の納屋・種・所持金を含む、学習のときだけ使える特権情報)。
              V = value(h_自分) + 追加の頭(h_自分, h_相手)、追加の頭の出口は0で初期化(最初は full_l1 と同じ)。ORACLE=1 で相手の観測も集める
  anti_l1   : exp242: 対称な反対称の批評家 V = (value(h_自分) − value(h_相手)) / 2(相手の観測を同じ網で読む)
比較の床: money = 「いまの所持金の差」だけのロジスティック(時期ごとに当てはめ)。
使い方: python3 value_lab.py <tag> <方策の重み> ;  環境変数 NGAMES STRIDE STEPS OPPDIR VARIANTS
"""
import json, os, sys, time, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S, sp_fast, fast_codes
S.SPPool = sp_fast.BatchSPPool; P.to_codes = fast_codes.to_codes_fast
DEV = "cuda"
E = lambda k, d: type(d)(os.environ.get(k, d))
NGAMES, STRIDE, STEPS, BATCH = E("NGAMES", 4000), E("STRIDE", 12), E("STEPS", 4000), E("BATCH", 64)
OPPDIR = os.environ["OPPDIR"]; L = "/content/local_logs"


def collect(net, n, seed0, log):
    """n 局ぶん(半分は対 tape、半分は自己対戦で両席を記録)の間引いた状態・時刻・最終の勝敗と差額。"""
    ORC = os.environ.get("ORACLE") == "1"; KS = ("tiles", "ou", "pu", "g", "t") + (("otiles", "oou", "opu", "og") if ORC else ())
    rng = np.random.RandomState(seed0); out = {k: [] for k in KS + ("gid",)}; Z = []; MG = []; gidn = 0
    per = 256
    for b0 in range(0, n, per):
        nb = min(per, n - b0); kinds = ["tape" if k % 2 == 0 else "self" for k in range(nb)]
        seats = rng.randint(0, 2, nb); seeds = rng.randint(0, 10 ** 9, nb)
        pool = S.SPPool(list(map(int, seeds))); tape = sp_fast.FastTapeRouterOpp(OPPDIR)
        allt = [(i, int(seats[i])) for i in range(nb)] + [(i, 1 - int(seats[i])) for i in range(nb) if kinds[i] == "self"]
        i0 = np.array([i for i, s in allt if s == 0], int); i1 = np.array([i for i, s in allt if s == 1], int)
        trajs = [(i, 0) for i in i0] + [(i, 1) for i in i1]; n0 = len(i0); rec = {k: [] for k in KS}
        t0 = time.time()
        for t in range(719):
            st = {s: pool.state(s) for s in (0, 1)}
            U = {s: np.zeros((nb, S.NU, 3), np.int32) for s in (0, 1)}; O = {s: np.zeros((nb, 10, 3), np.int32) for s in (0, 1)}
            stL = tuple(np.concatenate([st[0][k][i0], st[1][k][i1]]) for k in range(5))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outs = [P.sample(net, tuple(x[j:j + 256] for x in stL)) for j in range(0, len(trajs), 256)]
            ul, uq, ms, mq, _ = [np.concatenate([o[k] for o in outs]) for k in range(5)]
            UL, OL = P.to_codes(ul, uq, ms, mq, stL[1])
            U[0][i0], O[0][i0] = UL[:n0], OL[:n0]; U[1][i1], O[1][i1] = UL[n0:], OL[n0:]
            ids = [i for i in range(nb) if kinds[i] == "tape"]
            for s in (0, 1):
                sub = [i for i in ids if 1 - seats[i] == s]
                if sub:
                    sts = tuple(x[sub] for x in st[s]); u, o = tape.act(sts, t, sub, pool, s); U[s][sub], O[s][sub] = u, o
            if t % STRIDE == 0:
                for k, a in zip(("tiles", "ou", "pu", "g"), stL[:4]): rec[k].append(a.astype(np.float16) if k == "tiles" else a)
                rec["t"].append(np.full(len(trajs), t, np.int16))
                if ORC:   # 同じ局・同じ時刻の相手の席の観測(軌道 (i, s) の相手 = 席 1−s)
                    stO = tuple(np.concatenate([st[1][k][i0], st[0][k][i1]]) for k in range(4))
                    for k, a in zip(("otiles", "oou", "opu", "og"), stO): rec[k].append(a.astype(np.float16) if k == "otiles" else a)
            pool.step(U[0], O[0], U[1], O[1])
        fin = pool.money()
        z = np.array([np.sign(fin[i, s] - fin[i, 1 - s]) for i, s in trajs], np.float32)
        mg = np.array([fin[i, s] - fin[i, 1 - s] for i, s in trajs], np.float32)
        gid = np.array([gidn + i for i, _ in trajs]); gidn += nb
        K = len(rec["t"])
        for k in KS: out[k].append(np.concatenate(rec[k]))      # (K*ntraj, ...) 並びは 時刻→軌道
        out["gid"].append(np.tile(gid, K)); Z.append(np.tile(z, K)); MG.append(np.tile(mg, K))
        log("  集めた %d/%d 局(%.0f秒、軌道 %d、自分の勝率 対tape %.3f)" % (b0 + nb, n, time.time() - t0, len(trajs),
            float(np.mean([z[j] > 0 for j, (i, s) in enumerate(trajs) if kinds[i] == "tape"]))))
    D = {k: np.concatenate(v) for k, v in out.items()}; D["z"] = np.concatenate(Z); D["mg"] = np.concatenate(MG)
    return D


def ev(z, v):
    return float(1 - np.var(z - v) / (np.var(z) + 1e-8))


def report(name, z, v, t, log):
    b = [(0, 240), (240, 480), (480, 720)]
    parts = " ".join("t%d-%d %.3f" % (a, c, ev(z[(t >= a) & (t < c)], v[(t >= a) & (t < c)])) for a, c in b)
    log("  %-10s 価値EV 全体 %.3f | %s" % (name, ev(z, v), parts))
    return ev(z, v)


def money_floor(D, tr, te, log):
    """所持金の差(対数比)と時刻ビンごとのロジスティック。numpy のニュートン法で。"""
    g = D["g"]; a = np.log1p(np.maximum(g[:, 32], 0) * 1e4); b = np.log1p(np.maximum(g[:, 33], 0) * 1e4)
    x = a - b; t = D["t"].astype(int); v = np.zeros(len(x), np.float32)
    for lo in range(0, 720, 60):
        mtr = tr & (t >= lo) & (t < lo + 60); mte = te & (t >= lo) & (t < lo + 60)
        X = np.stack([np.ones(mtr.sum()), x[mtr]], 1); y = (D["z"][mtr] > 0).astype(float); w = np.zeros(2)
        for _ in range(25):
            p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-3 * np.eye(2); w -= np.linalg.solve(H, X.T @ (p - y))
        v[mte] = 2 / (1 + np.exp(-(w[0] + w[1] * x[mte]))) - 1
    return report("money", D["z"][te], v[te], t[te], log)


def train_critic(D, tr_idx, te_idx, init, kind, log, save=None):
    crit = M.load(init, DEV); crit.train(); orc = kind.startswith("orc"); anti = kind.startswith("anti"); extra = None
    if orc:
        d = crit.value.in_features if isinstance(crit.value, torch.nn.Linear) else [m for m in crit.value.modules() if isinstance(m, torch.nn.Linear)][0].in_features
        extra = torch.nn.Sequential(torch.nn.Linear(2 * d, d), torch.nn.GELU(), torch.nn.Linear(d, 1)).to(DEV)
        torch.nn.init.zeros_(extra[2].weight); torch.nn.init.zeros_(extra[2].bias)
    if kind.startswith("head"):
        for n_, p in crit.named_parameters(): p.requires_grad_(n_.startswith("value."))
    ps = [p for p in crit.parameters() if p.requires_grad] + (list(extra.parameters()) if extra is not None else [])
    opt = torch.optim.Adam(ps, lr=1e-4, eps=1e-5); scaler = torch.amp.GradScaler("cuda")
    tt = lambda k, ix, dt: torch.as_tensor(D[k][ix], device=DEV, dtype=dt)
    z = D["z"]; tgt = np.tanh(D["mg"] / 20000.0).astype(np.float32) if kind.endswith("mg") else z
    rng = np.random.RandomState(0); best = None

    def vout(b):
        h, _ = crit.encode(tt("tiles", b, torch.float16), tt("ou", b, torch.float32), tt("pu", b, torch.float32), tt("g", b, torch.float32))
        if not (orc or anti): return crit.value(h[:, -1]).squeeze(-1).float()
        ho, _ = crit.encode(tt("otiles", b, torch.float16), tt("oou", b, torch.float32), tt("opu", b, torch.float32), tt("og", b, torch.float32))
        if anti: return 0.5 * (crit.value(h[:, -1]) - crit.value(ho[:, -1])).squeeze(-1).float()
        return (crit.value(h[:, -1]) + extra(torch.cat([h[:, -1], ho[:, -1]], -1))).squeeze(-1).float()

    def predict(ix):
        crit.eval(); vs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for j in range(0, len(ix), 1024):
                b = ix[j:j + 1024]; o = vout(b)
                vs.append((torch.tanh(o) if kind.endswith("ce") else o).cpu().numpy())
        crit.train(); return np.concatenate(vs)
    bestv = None
    for step in range(STEPS + 1):
        if step % (STEPS // 4) == 0:
            v = predict(te_idx); e = report("%s@%d" % (kind, step), z[te_idx], np.clip(v, -1, 1), D["t"][te_idx], log)
            if best is None or e > best:
                best = e; bestv = np.clip(v, -1, 1)
                if save:   # 未見の局で最良の時点の批評家を残す(PPO の CRITINIT に使う)
                    torch.save(crit.state_dict(), save); json.dump(json.load(open(init + ".cfg.json")), open(save + ".cfg.json", "w")); log("  → 保存 %s(価値EV %.3f)" % (save, e))
        if step == STEPS: break
        b = tr_idx[rng.randint(0, len(tr_idx), 512)]
        with torch.autocast("cuda", dtype=torch.float16):
            o = vout(b)
        y = torch.as_tensor(tgt[b], device=DEV)
        # 2値分類: logit = 2·o(tanh(o) = 2σ(2o) − 1 なので V = tanh(o) と一致させる)
        loss = F.binary_cross_entropy_with_logits(2 * o, (y > 0).float()) if kind.endswith("ce") else F.smooth_l1_loss(o, y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(ps, 0.5); scaler.step(opt); scaler.update()
    train_critic.preds[kind] = bestv
    return best
train_critic.preds = {}


NB = 51; HL_LO, HL_HI = -60000.0, 60000.0   # exp250: HL-Gauss の区間(最終の差額)


def train_critic2(D, tr_idx, te_idx, init, kind, log, save=None):
    """exp250(PTCG 1・4・7・9位): 反対称の批評家(自分と相手の観測を同じ網で読む)の上で、価値ヘッドの形と目標を比べる。
    a2_l1  : 対照(anti_l1 と同じ)V = (v自 − v相)/2、smooth L1 で勝敗 z
    a2_ce  : 勝敗の2値分類。u = v自 − v相 を logit(×2)、V = tanh(u)(4位の分類の2値版)
    a2_c3  : 負/分/勝の3値分類(4位)。logits = l自 + 反転(l相)、V = P(勝) − P(負)
    a2_hl  : 最終の差額の分布(HL-Gauss 51ビン、7位)。logits = l自 + 反転(l相)、V = P(差>0) − P(差<0)
    a2_pool: 入力を global トークン ‖ 全トークン平均 にした頭(1位)、smooth L1
    a2_aux : a2_l1 + 補助頭で最終の差額 tanh(差/2万) を当てる(重み0.5、9位の補助タスク)"""
    crit = M.load(init, DEV); crit.train()
    d = [m for m in crit.value.modules() if isinstance(m, torch.nn.Linear)][0].in_features
    mkh = lambda i, o: torch.nn.Sequential(torch.nn.Linear(i, d), torch.nn.GELU(), torch.nn.Linear(d, o)).to(DEV)
    head = None
    if kind == "a2_c3": head = mkh(d, 3)
    if kind == "a2_hl": head = mkh(d, NB)
    if kind == "a2_pool": head = mkh(2 * d, 1)
    if kind == "a2_aux": head = mkh(d, 1)
    ps = list(crit.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.Adam(ps, lr=1e-4, eps=1e-5); scaler = torch.amp.GradScaler("cuda")
    tt = lambda k, ix, dt: torch.as_tensor(D[k][ix], device=DEV, dtype=dt)
    z = D["z"]; mgt = np.tanh(D["mg"] / 20000.0).astype(np.float32)
    centers = torch.linspace(HL_LO, HL_HI, NB, device=DEV); sgn = torch.sign(centers); width = (HL_HI - HL_LO) / (NB - 1)
    def hl_target(mg):                                       # 差額のまわりの正規分布をビンに積分(σ = 0.75 ビン)
        m = torch.as_tensor(mg, device=DEV).clamp(HL_LO, HL_HI).unsqueeze(1); sig = 0.75 * width
        edges = torch.cat([centers - width / 2, centers[-1:] + width / 2]).unsqueeze(0)
        cdf = 0.5 * (1 + torch.erf((edges - m) / (sig * 2 ** 0.5))); p = cdf[:, 1:] - cdf[:, :-1]
        return p / p.sum(1, keepdim=True).clamp(min=1e-8)
    def enc(b, o=""):
        h, _ = crit.encode(tt(o + "tiles", b, torch.float16), tt(o + "ou", b, torch.float32), tt(o + "pu", b, torch.float32), tt(o + "g", b, torch.float32))
        return h
    def outs(b):
        """(V, 学習に使う生の出力)"""
        h, ho = enc(b), enc(b, "o")
        if kind in ("a2_l1", "a2_ce", "a2_aux"):
            u = (crit.value(h[:, -1]) - crit.value(ho[:, -1])).squeeze(-1).float()
            if kind == "a2_ce": return torch.tanh(u), u
            if kind == "a2_aux": return 0.5 * u, (0.5 * u, 0.5 * (head(h[:, -1]) - head(ho[:, -1])).squeeze(-1).float())
            return 0.5 * u, 0.5 * u
        if kind == "a2_pool":
            f = lambda x: head(torch.cat([x[:, -1], x[:, :-1].mean(1)], -1)).squeeze(-1).float()
            v = 0.5 * (f(h) - f(ho)); return v, v
        if kind == "a2_c3":
            lg = (head(h[:, -1]) + head(ho[:, -1]).flip(-1)).float(); p = lg.softmax(-1)   # 並びは [負, 分, 勝]、相手から見ると反転
            return p[:, 2] - p[:, 0], lg
        if kind == "a2_hl":
            lg = (head(h[:, -1]) + head(ho[:, -1]).flip(-1)).float(); p = lg.softmax(-1)
            return (p * sgn).sum(-1), lg
    def predict(ix):
        crit.eval(); head is not None and head.eval(); vs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for j in range(0, len(ix), 1024): vs.append(outs(ix[j:j + 1024])[0].float().cpu().numpy())
        crit.train(); head is not None and head.train(); return np.concatenate(vs)
    rng = np.random.RandomState(0); best = None; bestv = None
    for step in range(STEPS + 1):
        if step % (STEPS // 4) == 0:
            v = predict(te_idx); e = report("%s@%d" % (kind, step), z[te_idx], np.clip(v, -1, 1), D["t"][te_idx], log)
            if best is None or e > best:
                best = e; bestv = np.clip(v, -1, 1)
                if save: torch.save(dict(crit=crit.state_dict(), head=head.state_dict() if head is not None else None, kind=kind), save); log("  → 保存 %s(価値EV %.3f)" % (save, e))
        if step == STEPS: break
        b = tr_idx[rng.randint(0, len(tr_idx), 512)]
        with torch.autocast("cuda", dtype=torch.float16):
            v, raw = outs(b)
        y = torch.as_tensor(z[b], device=DEV)
        if kind == "a2_ce": loss = F.binary_cross_entropy_with_logits(2 * raw, (y > 0).float())
        elif kind == "a2_c3": loss = F.cross_entropy(raw, (y.sign() + 1).long())
        elif kind == "a2_hl": loss = -(hl_target(D["mg"][b]) * raw.log_softmax(-1)).sum(-1).mean()
        elif kind == "a2_aux": loss = F.smooth_l1_loss(raw[0], y) + 0.5 * F.smooth_l1_loss(raw[1], torch.as_tensor(mgt[b], device=DEV))
        else: loss = F.smooth_l1_loss(raw, y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(ps, 0.5); scaler.step(opt); scaler.update()
    train_critic.preds[kind] = bestv
    return best


def main():
    tag, pol = sys.argv[1], sys.argv[2]
    logf = open("%s/vlab_%s.log" % (L, tag), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    cache = "%s/vlab_%s.npz" % (L, tag)
    if os.path.exists(cache):
        D = dict(np.load(cache)); log("保存済みの局面を読む: %s" % cache)
    else:
        net = M.load(pol, DEV); net.eval(); log("方策 %s で %d 局を集める(%d手ごとに間引く)" % (pol, NGAMES, STRIDE))
        D = collect(net, NGAMES, 1234, log); np.savez(cache, **D); del net; torch.cuda.empty_cache()
    games = np.unique(D["gid"]); rng = np.random.RandomState(7); rng.shuffle(games)
    te_g = set(games[: len(games) // 5].tolist())
    te = np.array([g in te_g for g in D["gid"]]); tr = ~te
    log("局面 %d(学習 %d / 検証 %d)、局 %d(検証 %d)、勝ちの割合 %.3f" % (len(te), tr.sum(), te.sum(), len(games), len(te_g), (D["z"] > 0).mean()))
    res = {"money": money_floor(D, tr, te, log)}
    tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
    trg = np.array(sorted(set(D["gid"][tr_idx].tolist())))
    for kind in os.environ.get("VARIANTS", "full_l1,head_l1,full_ce,full_mg,full_l1_q,full_l1_s").split(","):
        ix = tr_idx
        if kind.endswith("_q") or kind.endswith("_s"):
            keep = set(trg[: len(trg) // (4 if kind.endswith("_q") else 16)].tolist()); ix = np.array([i for i in tr_idx if D["gid"][i] in keep])
        k0 = kind[:-2] if kind.endswith("_q") or kind.endswith("_s") else kind
        log("== %s(学習の局面 %d、局 %d)==" % (kind, len(ix), len(set(D["gid"][ix].tolist()))))
        fn = train_critic2 if kind.startswith("a2_") else train_critic
        res[kind] = fn(D, ix, te_idx, pol, k0, log, save="%s/vlab_%s_%s.pt" % (L, tag, kind) if os.environ.get("SAVE") == "1" else None)
    json.dump(res, open("%s/vlab_%s.json" % (L, tag), "w")); log("最良の価値EV: %s" % json.dumps(res))
    # 対応比較: 検証の局ごとに二乗誤差の差(基準 full_l1 − 各変種)を平均し、局単位の標準誤差を付ける
    Pd = train_critic.preds; zt = D["z"][te_idx]; gt = D["gid"][te_idx]
    np.savez("%s/vlab_%s_preds.npz" % (L, tag), z=zt, gid=gt, t=D["t"][te_idx], **{k: v for k, v in Pd.items() if v is not None})
    BASE = os.environ.get("BASE", "full_l1")
    if BASE in Pd:
        ug = np.unique(gt)
        for k, v in Pd.items():
            if k == BASE or v is None: continue
            dse = (zt - Pd[BASE]) ** 2 - (zt - v) ** 2
            per = np.array([dse[gt == g].mean() for g in ug])
            log(("  対応比較 %s: 二乗誤差の改善(" + BASE + " − %s)局平均 %+.4f ± %.4f(%d局、良 %d / 悪 %d)" % (k, k, per.mean(), per.std() / np.sqrt(len(per)), len(per), (per > 0).sum(), (per < 0).sum())))


if __name__ == "__main__":
    main()
