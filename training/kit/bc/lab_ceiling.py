"""exp188: 「相手で分岐してテープを選ぶ」ことの天井と、実際に汎化する分だけを分ける。

入力: lab_matrix.py の R[テープ, 対局](差 = 自分 - 相手)と、対局ごとの meta。
出す数字:
  1. 固定で最良のテープの勝率(いまのやり方の上限)
  2. **天井**: 対局ごとに最良のテープを選べたときの勝率(未来を知っている選択。汎化しない)
  3. 相手の名前で選ぶ(交差検証。同じ相手の別の局に汎化するか)
  4. 相手の開幕の公開挙動で選ぶ(交差検証。**これが実際に使える分**)
  5. 分散の分解: テープの主効果 / 対局の主効果 / 交互作用(交互作用が小さければ分岐しても無駄)
使い方: python3 lab_ceiling.py <R.npz> [先頭N手=144]
"""
import json, os, sys, collections
import numpy as np

Z = np.load(sys.argv[1], allow_pickle=False)
NPRE = int(sys.argv[2]) if len(sys.argv) > 2 else 144
R = Z["R"].astype(np.float64)
tape_eids = list(Z["tape_eids"]); game_eids = list(Z["game_eids"])
meta = json.loads(str(Z["meta"]))
W = (R > 0).astype(np.float64)
T, G = R.shape
print("テープ %d 本 x 対局 %d 局" % (T, G))

# 恒等式: 対角(その局の本人のテープを、その局の相手・シードで再生)は実戦の結果と一致するはず
tpos = {e: i for i, e in enumerate(tape_eids)}
diag = [(tpos[e], j) for j, e in enumerate(game_eids) if e in tpos]
if diag:
    d0 = np.array([R[i, j] for i, j in diag])
    d1 = np.array([meta[j]["reward_me"] - meta[j]["reward_op"] for _, j in diag])
    print("0. 恒等式(対角 %d 局): 実戦と一致 %d 局 / 最大のずれ %.0f コイン" % (
        len(diag), int((np.abs(d0 - d1) < 1).sum()), np.abs(d0 - d1).max()))
    # 対角は「その相手・そのシードのために作られた手」なので、天井の計算からは外す
    for i, j in diag:
        R[i, j] = -np.inf; W[i, j] = np.nan

# ---------- 1. 固定で最良 ----------
wr_t = np.nanmean(W, 1); mg_t = np.where(np.isfinite(R), R, np.nan)
mg_t = np.nanmean(mg_t, 1)
kfix = int(wr_t.argmax())
print("\n1. 固定で最良のテープ: %s  勝率 %.3f±%.3f  差 %+.0f" % (
    tape_eids[kfix], wr_t[kfix], (wr_t[kfix] * (1 - wr_t[kfix]) / G) ** .5, mg_t[kfix]))
print("   (テープの勝率の分布: 中央値 %.3f 上位5%% %.3f 最良 %.3f)" % (
    np.median(wr_t), np.quantile(wr_t, .95), wr_t.max()))

# ---------- 2. 天井(対局ごとのオラクル) ----------
Wo = np.nanmax(W, 0); Ro = np.where(np.isfinite(R), R, -np.inf).max(0)
print("2. **天井**(対局ごとに最良を選べたとき。対角は除く): 勝率 %.3f  差 %+.0f  → 固定比 +%.3f / %+.0f" % (
    Wo.mean(), Ro.mean(), Wo.mean() - wr_t[kfix], Ro.mean() - mg_t[kfix]))

# ---------- 相手の開幕の挙動(公開情報) ----------
UOPS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "DROP", "PLACE", "PLANT", "WATER",
        "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"]
MOPS = ["NONE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]


def feats(buf, nsteps, npre):
    """pack 済みテープの先頭 npre 手から、相手の公開挙動の特徴を作る。"""
    u = np.zeros(len(UOPS)); m = np.zeros(len(MOPS)); nun = 0.0; sellq = 0.0; buyq = 0.0
    i = 0
    for s in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1
        nun = max(nun, nu)
        for _ in range(nu):
            op = int(buf[i]); i += 3
            if 0 <= op < len(UOPS): u[op] += 1
        no = int(buf[i]); i += 1
        for _ in range(no):
            op, it, n = int(buf[i]), int(buf[i + 1]), int(buf[i + 2]); i += 3
            if 0 <= op < len(MOPS): m[op] += 1
            if op == 6: sellq += n
            elif op >= 3: buyq += n
    return np.concatenate([u, m, [nun, sellq / 100.0, buyq / 100.0]])


src = os.environ.get("SRC", "/content/lab/sp")
X = []
for e in game_eids:
    z = np.load("%s/%s.npz" % (src, e))
    mm = json.loads(str(z["meta"]))
    X.append(feats(z["op"], mm["nop"], NPRE))
X = np.array(X)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)

# ---------- 5. 分散の分解 ----------
Rf = np.where(np.isfinite(R), R, np.nan)
gm = np.nanmean(Rf); a = np.nanmean(Rf, 1) - gm; b = np.nanmean(Rf, 0) - gm
inter = Rf - gm - a[:, None] - b[None, :]
tot = np.nanvar(Rf)
print("\n5. 差の分散の分解: テープの主効果 %.1f%% / 対局の主効果 %.1f%% / 交互作用 %.1f%%" % (
    100 * np.nanvar(a) / tot, 100 * np.nanvar(b) / tot, 100 * np.nanvar(inter) / tot))
print("   (交互作用の標準偏差 %.0f コイン — ここが小さいなら相手で分岐しても取れる余地は小さい)" % np.nanstd(inter))

# ---------- 3・4. 交差検証で「汎化する分」を測る ----------
rng = np.random.default_rng(0)
order = rng.permutation(G)
K = 5
folds = np.array_split(order, K)
TOPK = int(os.environ.get("TOPK", 30))
cand = np.argsort(-np.nan_to_num(wr_t, nan=-1))[:TOPK]          # 候補は全体の勝率上位 TOPK 本に絞る(過学習を抑える)


def cv(selector, name):
    won = np.zeros(G); marg = np.zeros(G)
    for k in range(K):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(K) if j != k])
        pick = selector(tr, te)
        won[te] = np.nan_to_num(W[pick, te]); marg[te] = np.where(np.isfinite(R[pick, te]), R[pick, te], 0)
    print("%s: 勝率 %.3f±%.3f  差 %+.0f  (固定比 %+.3f / %+.0f)" % (
        name, won.mean(), (won.mean() * (1 - won.mean()) / G) ** .5, marg.mean(),
        won.mean() - wr_t[kfix], marg.mean() - mg_t[kfix]))
    return won.mean()


def sel_fixed(tr, te):
    k = cand[int(np.nanmean(W[np.ix_(cand, tr)], 1).argmax())]
    return np.full(len(te), k)


def sel_by_opp(tr, te):
    names = np.array([m["opp_name"] for m in meta])
    best_all = cand[int(np.nanmean(W[np.ix_(cand, tr)], 1).argmax())]
    out = np.empty(len(te), int)
    for i, g in enumerate(te):
        mtr = tr[names[tr] == names[g]]
        out[i] = cand[int(np.nanmean(W[np.ix_(cand, mtr)], 1).argmax())] if len(mtr) >= 4 else best_all
    return out


def sel_by_feat(tr, te, kn=25):
    best_all = cand[int(np.nanmean(W[np.ix_(cand, tr)], 1).argmax())]
    out = np.empty(len(te), int)
    for i, g in enumerate(te):
        d = ((X[tr] - X[g]) ** 2).sum(1)
        nb = tr[np.argsort(d)[:kn]]
        wv = np.nanmean(W[np.ix_(cand, nb)], 1)
        out[i] = cand[int(np.nanargmax(wv))]
    return out


print("\n3・4. 交差検証(候補は勝率上位 %d 本、5分割)" % TOPK)
cv(sel_fixed, "  固定(交差検証)      ")
cv(sel_by_opp, "  相手の名前で選ぶ      ")
cv(lambda tr, te: sel_by_feat(tr, te), "  相手の開幕%d手で選ぶ " % NPRE)
