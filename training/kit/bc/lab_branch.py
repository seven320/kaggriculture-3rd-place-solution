"""exp188(その4・本番): **家族の中で step144 に分岐する価値**を、提出で観測できる情報だけで測る。
   比べるもの(すべて同じ対局・5分割の交差検証・対応差):
     1. 固定      : 家族の中で全体の勝率が最良の1本(いまの「家族の中の最良の固定テープ」)
     2. 店の抽選  : step144 の店の組ごとに最良の継続(exp180 のいまの規則)
     3. 相手の状態: step144 の相手の公開状態(所持金・雇用・区画・タイル・ユニット数)からリッジで選ぶ
     4. 店+状態   : 両方
   使い方: python3 lab_branch.py <R.npz> <obs144.npz>
"""
import json, os, sys
import numpy as np

Z = np.load(sys.argv[1], allow_pickle=False); O = np.load(sys.argv[2], allow_pickle=False)
R = Z["R"].astype(np.float64); W = (R > 0).astype(np.float64)
tape_eids = [str(x) for x in Z["tape_eids"]]; game_eids = [str(x) for x in Z["game_eids"]]
X = O["X"].astype(np.float64); fams = [str(x) for x in O["fams"]]; tape_fam = np.array([str(x) for x in O["tape_fam"]])
G = len(game_eids); K = 5
base_eid = lambda e: e.split("__")[0]
mask = np.ones_like(W, bool)
gb = {}
for j, e in enumerate(game_eids): gb.setdefault(base_eid(e), []).append(j)
for i, e in enumerate(tape_eids):
    for j in gb.get(base_eid(e), []): mask[i, j] = False      # 同じ対局から採ったテープは外す
Wm = np.where(mask, W, np.nan)

rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(G), K)
SHOP0 = 10   # X の中で店の 8次元が始まる位置(farm_pub 5 + ユニット 1 + タイル 4 = 10)


def cv(cand, pick_fn):
    won = np.zeros(G); marg = np.zeros(G)
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        p = pick_fn(cand, tr, te)
        won[te] = np.nan_to_num(W[p, te]); marg[te] = np.nan_to_num(R[p, te])
    return won, marg


def p_fixed(cand, tr, te):
    return np.full(len(te), cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, tr)], 1)))])


def p_shop(Xf):
    def f(cand, tr, te):
        key = lambda j: tuple(np.round(Xf[j, SHOP0:SHOP0 + 8]).astype(int))
        ba = cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, tr)], 1)))]
        out = np.empty(len(te), int)
        by = {}
        for j in tr: by.setdefault(key(j), []).append(j)
        for i, j in enumerate(te):
            m = by.get(key(j), [])
            out[i] = cand[int(np.nanargmax(np.nanmean(Wm[np.ix_(cand, m)], 1)))] if len(m) >= 8 else ba
        return out
    return f


def p_ridge(Xf, cols, lam=10.0):
    Xs = Xf[:, cols]; Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    def f(cand, tr, te):
        A = np.c_[Xs[tr], np.ones(len(tr))]
        Y = np.nan_to_num(Wm[np.ix_(cand, tr)].T, nan=0.5)
        P = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
        return np.asarray(cand)[(np.c_[Xs[te], np.ones(len(te))] @ P).argmax(1)]
    return f


print("家族 %d 種 / 対局 %d 局。候補は家族の中のテープだけ、分岐は step%d" % (len(fams), G, int(O["npre"])))
print("%-10s %5s %7s | %-22s %-22s %-22s" % ("家族", "本数", "固定", "店の抽選(いまの規則)", "相手の状態(リッジ)", "店+状態"))
tot = {}
for fi, f in enumerate(fams):
    cand = np.nonzero(tape_fam == f)[0]
    if len(cand) < 3: continue
    Xf = X[fi]
    wf, mf = cv(cand, p_fixed)
    outs = []
    for nm, fn in (("shop", p_shop(Xf)),
                   ("state", p_ridge(Xf, list(range(0, SHOP0)) + list(range(SHOP0 + 8, Xf.shape[1] - 3)))),
                   ("both", p_ridge(Xf, list(range(0, Xf.shape[1] - 3))))):
        w, m = cv(cand, fn); d = w - wf; se = d.std(ddof=1) / G ** .5 + 1e-12
        outs.append("%+.3f±%.3f(%.1fσ)" % (d.mean(), se, abs(d.mean()) / se))
        tot.setdefault(nm, []).append(d)
    tot.setdefault("fix", []).append(wf)
    print("%-10s %5d %7.3f | %-22s %-22s %-22s" % (f[:10], len(cand), wf.mean(), *outs))

print("\n家族をまたいだ平均(%d 家族)" % len(tot["fix"]))
fx = np.mean([w.mean() for w in tot["fix"]])
print("  固定: 勝率 %.3f" % fx)
for nm, lab in (("shop", "店の抽選(いまの規則)"), ("state", "相手の状態(リッジ)"), ("both", "店+状態")):
    d = np.stack(tot[nm]).mean(0); se = d.std(ddof=1) / G ** .5
    print("  %-22s 勝率 %.3f  固定との対応差 %+.3f±%.3f (%.1fσ)" % (lab, fx + d.mean(), d.mean(), se, abs(d.mean()) / (se + 1e-12)))
