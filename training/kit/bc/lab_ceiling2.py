"""exp188(その2): 天井と、汎化する分を**対応のある誤差つき**で出す。
   使い方: SRC=<npzのdir> python3 lab_ceiling2.py <R.npz>
"""
import json, os, sys
import numpy as np

Z = np.load(sys.argv[1], allow_pickle=False)
R = Z["R"].astype(np.float64)
tape_eids = list(Z["tape_eids"]); game_eids = list(Z["game_eids"])
meta = json.loads(str(Z["meta"]))
W = (R > 0).astype(np.float64)
T, G = R.shape
src = os.environ.get("SRC", "/content/lab/sp")
TOPK = int(os.environ.get("TOPK", 30)); K = 5

base = lambda e: str(e).split("__")[0]
tpos = {}
for i, e in enumerate(tape_eids): tpos.setdefault(base(e), []).append(i)
diag = [(i, j) for j, e in enumerate(game_eids) for i in tpos.get(base(e), []) if tape_eids[i] == e]
d0 = np.array([R[i, j] for i, j in diag])
d1 = np.array([meta[j]["reward_me"] - meta[j]["reward_op"] for _, j in diag])
print("テープ %d x 対局 %d。恒等式(対角 %d 局): 一致 %d 局 / 最大ずれ %.0f" % (
    T, G, len(diag), int((np.abs(d0 - d1) < 1).sum()), np.abs(d0 - d1).max()))
mask = np.ones_like(W, bool)
for j, e in enumerate(game_eids):
    for i in tpos.get(base(e), []): mask[i, j] = False
Wm = np.where(mask, W, np.nan); Rm = np.where(mask, R, np.nan)

wr_t = np.nanmean(Wm, 1); mg_t = np.nanmean(Rm, 1)
cand = np.argsort(-np.nan_to_num(wr_t, nan=-1))[:TOPK]
kfix = int(np.nanargmax(wr_t))
print("固定で最良: %s 勝率 %.3f 差 %+.0f | 天井(局ごとオラクル): 勝率 %.3f 差 %+.0f" % (
    tape_eids[kfix], wr_t[kfix], mg_t[kfix], np.nanmax(Wm, 0).mean(), np.nanmax(Rm, 0).mean()))

UOPS = 18; MOPS = 7
def feats(buf, nsteps, npre):
    u = np.zeros(UOPS); m = np.zeros(MOPS); nun = 0.0; sellq = 0.0; buyq = 0.0; i = 0
    for s in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1; nun = max(nun, nu)
        for _ in range(nu):
            op = int(buf[i]); i += 3
            if 0 <= op < UOPS: u[op] += 1
        no = int(buf[i]); i += 1
        for _ in range(no):
            op, it, n = int(buf[i]), int(buf[i+1]), int(buf[i+2]); i += 3
            if 0 <= op < MOPS: m[op] += 1
            if op == 6: sellq += n
            elif op >= 3: buyq += n
    return np.concatenate([u, m, [nun, sellq/100.0, buyq/100.0]])

BUF = []
for e in game_eids:
    z = np.load("%s/%s.npz" % (src, e)); mm = json.loads(str(z["meta"]))
    BUF.append((np.ascontiguousarray(z["op"]), mm["nop"]))

def make_X(npre):
    X = np.array([feats(b, n, npre) for b, n in BUF])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(G), K)

def run(selector, name, base=None):
    won = np.zeros(G); marg = np.zeros(G)
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        pick = selector(tr, te)
        won[te] = np.nan_to_num(W[pick, te]); marg[te] = np.nan_to_num(R[pick, te])
    line = "  %-24s 勝率 %.3f  差 %+7.0f" % (name, won.mean(), marg.mean())
    if base is not None:
        dw = won - base[0]; dm = marg - base[1]
        sw = dw.std(ddof=1)/G**.5 + 1e-12
        line += " | 固定との対応差 勝率 %+.3f±%.3f (%.1fσ) 差 %+.0f±%.0f" % (
            dw.mean(), sw, abs(dw.mean())/sw, dm.mean(), dm.std(ddof=1)/G**.5)
    print(line, flush=True)
    return won, marg

def sel_fixed(tr, te):
    return np.full(len(te), cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, tr)], 1)))])

def sel_opp(tr, te):
    names = np.array([m["opp_name"] for m in meta])
    ba = cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, tr)], 1)))]
    out = np.empty(len(te), int)
    for i, g in enumerate(te):
        mtr = tr[names[tr] == names[g]]
        out[i] = cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, mtr)], 1)))] if len(mtr) >= 4 else ba
    return out

def sel_knn(X, kn):
    def f(tr, te):
        out = np.empty(len(te), int)
        for i, g in enumerate(te):
            nb = tr[np.argsort(((X[tr]-X[g])**2).sum(1))[:kn]]
            out[i] = cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, nb)], 1)))]
        return out
    return f

def sel_ridge(X, lam=10.0):
    def f(tr, te):
        A = np.c_[X[tr], np.ones(len(tr))]
        Y = np.nan_to_num(Wm[np.ix_(cand, tr)].T, nan=0.5)
        P = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ Y)
        S = np.c_[X[te], np.ones(len(te))] @ P
        return cand[S.argmax(1)]
    return f

print("\n交差検証(%d分割、候補は勝率上位 %d 本)" % (K, TOPK))
base = run(sel_fixed, "固定")
run(sel_opp, "相手の名前で選ぶ", base)
for npre in (24, 48, 144):
    X = make_X(npre)
    run(sel_knn(X, 25), "開幕%3d手 k-NN(25)" % npre, base)
    run(sel_ridge(X), "開幕%3d手 リッジ" % npre, base)
