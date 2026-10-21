"""exp188 の訂正: 開幕を共有するテープ(家族)の中でだけ、step N で分岐する価値を測る。
   exp180 の教訓「家族どうしは開幕が違うので step144 では切り替えられない」に従う。
   使い方: SRC=<npzのdir> python3 lab_fam.py <R.npz> [分岐する手数]
"""
import json, os, sys, hashlib, collections
import numpy as np

Z = np.load(sys.argv[1], allow_pickle=False)
NPRE = int(sys.argv[2]) if len(sys.argv) > 2 else 144
R = Z["R"].astype(np.float64); W = (R > 0).astype(np.float64)
tape_eids = list(Z["tape_eids"]); game_eids = list(Z["game_eids"]); meta = json.loads(str(Z["meta"]))
T, G = R.shape
src = os.environ.get("SRC", "/content/lab/sp")

def walk(buf, nsteps, npre):
    """先頭 npre 手ぶんの終端位置を返す(pack 形式を順に読む)。"""
    i = 0
    for s in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1 + 3 * nu
        no = int(buf[i]); i += 1 + 3 * no
    return i

UOPS = 18; MOPS = 7
def feats(buf, nsteps, npre):
    u = np.zeros(UOPS); m = np.zeros(MOPS); nun = 0.0; sq = 0.0; bq = 0.0; i = 0
    for s in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1; nun = max(nun, nu)
        for _ in range(nu):
            op = int(buf[i]); i += 3
            if 0 <= op < UOPS: u[op] += 1
        no = int(buf[i]); i += 1
        for _ in range(no):
            op, it, n = int(buf[i]), int(buf[i+1]), int(buf[i+2]); i += 3
            if 0 <= op < MOPS: m[op] += 1
            if op == 6: sq += n
            elif op >= 3: bq += n
    return np.concatenate([u, m, [nun, sq/100.0, bq/100.0]])

ME = {}; OPB = {}
for e in set(tape_eids) | set(game_eids):
    z = np.load("%s/%s.npz" % (src, e)); mm = json.loads(str(z["meta"]))
    ME[e] = (np.ascontiguousarray(z["me"]), mm["nme"])
    OPB[e] = (np.ascontiguousarray(z["op"]), mm["nop"])

fam = []
for e in tape_eids:
    b, n = ME[e]
    fam.append(hashlib.sha256(b[:walk(b, n, NPRE)].tobytes()).hexdigest()[:12])
fam = np.array(fam)
cnt = collections.Counter(fam)
print("テープ %d 本。開幕%d手で見た家族: %d 種。大きい順: %s" % (
    T, NPRE, len(cnt), [(k[:6], v) for k, v in cnt.most_common(8)]))

X = np.array([feats(*OPB[e], NPRE) for e in game_eids])
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
base = lambda e: str(e).split("__")[0]
tpos = {}
for i, e in enumerate(tape_eids): tpos.setdefault(base(e), []).append(i)
mask = np.ones_like(W, bool)
for j, e in enumerate(game_eids):
    for i in tpos.get(base(e), []): mask[i, j] = False
Wm = np.where(mask, W, np.nan)

rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(G), 5)

def cv(cand, sel):
    won = np.zeros(G)
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        won[te] = np.nan_to_num(W[sel(cand, tr, te), te])
    return won

def s_fix(cand, tr, te):
    return np.full(len(te), cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, tr)], 1)))])

def s_ridge(cand, tr, te, lam=10.0):
    A = np.c_[X[tr], np.ones(len(tr))]
    Y = np.nan_to_num(Wm[np.ix_(cand, tr)].T, nan=0.5)
    P = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ Y)
    return np.asarray(cand)[(np.c_[X[te], np.ones(len(te))] @ P).argmax(1)]

print("\n家族の中だけで step%d に分岐(5分割の交差検証、対応差)" % NPRE)
print("  %-14s %5s %7s %7s  %s" % ("家族", "本数", "固定", "分岐", "対応差(勝率)"))
tot_f = np.zeros(G); tot_r = np.zeros(G); used = 0
for f, nmem in cnt.most_common(12):
    idx = np.nonzero(fam == f)[0]
    if len(idx) < 3: continue
    wf = cv(idx, s_fix); wr = cv(idx, s_ridge)
    d = wr - wf; se = d.std(ddof=1) / G ** .5 + 1e-12
    print("  %-14s %5d %7.3f %7.3f  %+.3f±%.3f (%.1fσ)" % (f[:10], len(idx), wf.mean(), wr.mean(), d.mean(), se, abs(d.mean())/se))
    tot_f += wf; tot_r += wr; used += 1
if used:
    d = (tot_r - tot_f) / used; se = d.std(ddof=1) / G ** .5 + 1e-12
    print("  %-14s %5s %7.3f %7.3f  %+.3f±%.3f (%.1fσ)  ← 家族をまたいだ平均" % ("合計", used, tot_f.mean()/used, tot_r.mean()/used, d.mean(), se, abs(d.mean())/se))
