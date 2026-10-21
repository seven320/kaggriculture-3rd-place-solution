"""exp197(B5): 家族全体を候補にした step144 のルーターを、我々の実戦局で作って測る。

   - 候補: ug 家族(ultimatum_game の11本)・gj 家族(GermanJurado1 の9本)。どちらも開幕144手が家族で共通。
   - 対局: 我々の実戦(v54/v55、元のシード・座席、相手は実際の手)。自己対戦の検証局は除く。
   - R[テープ, 対局] = kagsim.run_tapes の最終差。step144 の店は、家族の開幕を相手にぶつけて読む(router5_lib.shops144 と同じ)。
   - 比べる規則(5分割の交差検証・対応差):
       固定        : 家族の中で全体の勝率が最良の1本
       組の離散規則: router5_rule.py と同じ(組ごとに 8局以上・固定より z>2 なら差し替え)
       組のリッジ  : 組を作る2軒の 0/1 指標(8次元)から候補ごとの勝ちをリッジで当てる
   - 最後に全データで規則を作り、**main.py の FAM_RULE(組 → テープ番号)にそのまま入る辞書**で書き出す
     (リッジは28通りの組すべてについて最良の候補を並べれば辞書になる = 提出側のコードは変えなくてよい)。

   使い方: FAMDIR=/content/lab/fam GAMEDIR=/content/lab/ours OUT=/content/lab/fam_router.json NP=32 python3 fam_router.py
"""
import glob, hashlib, itertools, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.environ.get("KAGG_FAST", "/content/drive/MyDrive/kaggriculture/fast"))
import kagsim

FAMDIR = os.environ.get("FAMDIR", "/content/lab/fam"); GAMEDIR = os.environ.get("GAMEDIR", "/content/lab/ours")
OUT = os.environ.get("OUT", "/content/lab/fam_router.json"); NP = int(os.environ.get("NP", 32))
SH = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
K = 5


def load(d):
    out = []
    for f in sorted(glob.glob(d + "/*.npz")):
        z = np.load(f); m = json.loads(str(z["meta"]))
        out.append(dict(m, me=np.ascontiguousarray(z["me"]), op=np.ascontiguousarray(z["op"])))
    return out


def walk(buf, nsteps, npre):
    i = 0
    for _ in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1 + 3 * nu
        no = int(buf[i]); i += 1 + 3 * no
    return i


TAPES = load(FAMDIR)
GAMES = [g for g in load(GAMEDIR) if g["opp_name"] != g["me_name"]]
FAMS = {}
for i, t in enumerate(TAPES):
    FAMS.setdefault(t["me_name"], []).append(i)
print("候補: %s / 対局 %d 局" % ({k: len(v) for k, v in FAMS.items()}, len(GAMES)), flush=True)
for name, idx in FAMS.items():
    h = {hashlib.sha256(TAPES[i]["me"][:walk(TAPES[i]["me"], TAPES[i]["nme"], 144)].tobytes()).hexdigest()[:10] for i in idx}
    print("  %s: 開幕144手の種類 %d(1 なら家族として step144 で切り替えられる)" % (name, len(h)), flush=True)


def row(ti):
    t = TAPES[ti]; out = np.zeros(len(GAMES), np.float32)
    for k, g in enumerate(GAMES):
        r = np.asarray(kagsim.run_tapes(t["me"], t["nme"], g["op"], g["nop"], [g["seed"]], 720, 1))[g["seat"]]
        out[k] = r[0] - r[1]
    return ti, out


def shops144(args):
    ti, gi = args
    t, g = TAPES[ti], GAMES[gi]
    a, b = ((t["me"], t["nme"]), (g["op"], g["nop"])) if g["seat"] == 0 else ((g["op"], g["nop"]), (t["me"], t["nme"]))
    env = kagsim.VecMarket(a[0], a[1], b[0], b[1], [int(g["seed"])], 720, int(g["seat"])); env.keep_nonsell = False; env.reset()
    while env.step_index() < 144:
        env.step2(np.asarray(env.tape_orders())[None], np.asarray(env.opp_tape_orders()).reshape(1, 10, 3))
    sh = [int(x) for x in np.asarray(env.shops())[0] if x >= 0]
    return gi, sh


t0 = time.time()
R = np.zeros((len(TAPES), len(GAMES)), np.float32)
with Pool(NP) as p:
    for ti, r in p.imap_unordered(row, range(len(TAPES))):
        R[ti] = r
print("R %s (%.1f分)" % (R.shape, (time.time() - t0) / 60), flush=True)
W = (R > 0).astype(float)
G = len(GAMES)
rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(G), K)
result = {}

for name, idx in FAMS.items():
    idx = np.array(idx)
    with Pool(NP) as p:
        SHOPS = dict(p.map(shops144, [(int(idx[0]), gi) for gi in range(G)]))
    nsh = np.bincount([len(SHOPS[g]) for g in range(G)])
    key = lambda gi: "|".join(sorted(SH[x] for x in SHOPS[gi])[:2])       # TapeRouterOpp / main.py と同じ組の作り方
    X = np.zeros((G, 8))
    for gi in range(G):
        for nm in key(gi).split("|"):
            if nm: X[gi, SH.index(nm)] = 1
    keys = np.array([key(g) for g in range(G)])
    print("\n=== %s 家族(%d本)。step144 で開いている店の軒数の分布 %s、組の種類 %d ===" % (
        name, len(idx), dict(enumerate(nsh.tolist())), len(set(keys))), flush=True)

    def fixed(tr):
        return idx[int(W[np.ix_(idx, tr)].mean(1).argmax())]

    def discrete(tr):
        kf = fixed(tr); rule = {}
        for c in set(keys[tr]):
            m = tr[keys[tr] == c]
            if len(m) < 8: continue
            d = W[np.ix_(idx, m)] - W[kf, m][None, :]; mu = d.mean(1); se = d.std(1) / np.sqrt(len(m)) + 1e-9; b = int(mu.argmax())
            if mu[b] / se[b] > 2: rule[c] = int(idx[b])
        return kf, rule

    def ridge(tr, lam=float(os.environ.get("LAM", 5.0))):
        A = np.c_[X[tr], np.ones(len(tr))]
        P = np.linalg.solve(A.T @ A + lam * np.eye(9), A.T @ W[np.ix_(idx, tr)].T)
        return P

    wf = np.zeros(G); wd = np.zeros(G); wr = np.zeros(G); mf = np.zeros(G); md = np.zeros(G); mr = np.zeros(G)
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        kf = fixed(tr); kf2, rule = discrete(tr); P = ridge(tr)
        pd_ = np.array([rule.get(keys[g], kf2) for g in te])
        pr = idx[(np.c_[X[te], np.ones(len(te))] @ P).argmax(1)]
        wf[te] = W[kf, te]; wd[te] = W[pd_, te]; wr[te] = W[pr, te]
        mf[te] = R[kf, te]; md[te] = R[pd_, te]; mr[te] = R[pr, te]
    line = lambda lab, w, m, base: print("  %-22s 勝率 %.3f 差 %+7.0f%s" % (lab, w.mean(), m.mean(), "" if base is None else
        "  | 固定との対応差 %+.3f±%.3f (%.1fσ)" % ((w - base).mean(), (w - base).std(ddof=1) / G ** .5, abs((w - base).mean()) / ((w - base).std(ddof=1) / G ** .5 + 1e-12))), flush=True)
    line("固定(家族で最良の1本)", wf, mf, None)
    line("組の離散規則(いまの方法)", wd, md, wf)
    line("組のリッジ", wr, mr, wf)
    d = wr - wd
    print("  リッジ − 離散規則: %+.3f±%.3f (%.1fσ)" % (d.mean(), d.std(ddof=1) / G ** .5, abs(d.mean()) / (d.std(ddof=1) / G ** .5 + 1e-12)), flush=True)

    # 全データで規則を作って書き出す(28通りの組すべて。main.py の FAM_RULE に入る形)
    kf = fixed(np.arange(G)); _, rule_d = discrete(np.arange(G)); P = ridge(np.arange(G))
    pairs = ["|".join(sorted(p)) for p in itertools.combinations_with_replacement(SH, 2)]   # 同じ店が2回出る組もある(36通り)
    rule_r = {}
    for pk in pairs:
        x = np.zeros(8)
        for nm in pk.split("|"): x[SH.index(nm)] = 1
        rule_r[pk] = int(idx[int((np.r_[x, 1] @ P).argmax())])
    eid = lambda i: TAPES[i]["eid"]
    seen = sorted(set(keys)); miss = [k for k in seen if k not in rule_r]
    assert not miss, "辞書に無い組が実戦に出ている: %s" % miss
    result[name] = dict(fixed=eid(kf), tapes=[eid(i) for i in idx], pairs_seen=len(seen),
                        rule_discrete={k: eid(v) for k, v in rule_d.items()},
                        rule_ridge={k: eid(v) for k, v in rule_r.items()},
                        cv=dict(fixed=float(wf.mean()), discrete=float(wd.mean()), ridge=float(wr.mean()),
                                ridge_minus_fixed=float((wr - wf).mean()), ridge_minus_discrete=float(d.mean()), n=G))
    used = sorted(set(rule_r.values()))
    print("  全データでの固定 %s、リッジが使う候補 %d 本 %s" % (eid(kf), len(used), [eid(i) for i in used]), flush=True)

json.dump(result, open(OUT, "w"), indent=1, ensure_ascii=False)
print("\n保存 %s (%.1f分)" % (OUT, (time.time() - t0) / 60), flush=True)
