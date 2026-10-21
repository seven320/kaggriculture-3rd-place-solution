"""exp316b: 家畜店の数と勝ち負けの因果の向きを調べる。
店の軒数は時計どおり(144手で2軒、300手で4軒、500手で6軒、600手で8軒)だが、どの店が開くかは打ち方で変わる。
そこで「先に開いた店」と「あとで開く店」を分け、
  (1) 途中の所持金の差が、そのあと開く家畜店の数を予言するか  → 予言するなら逆向きの因果(負けている局で家畜店が開く)
  (2) 先に開いた家畜店の数が、そのあとの所持金の差を予言するか → こちらだけなら盤面の引きが原因
使い方: NSEED=192 python probe_shops2.py <自分> <相手>
"""
import os, sys, json, math
import numpy as np, torch
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
import model_tf as M
import sp_lib as S

DEV = "cuda"
N = int(os.environ.get("NSEED", "192"))
ME = sys.argv[1]; OP = sys.argv[2]
SNAP = [144, 300, 500, 718]
seeds = [int(os.environ.get("SEED0", "500009")) + 7919 * k for k in range(N)]
me = M.load(ME, DEV); me.eval(); op = M.load(OP, DEV); op.eval()
LIVE = {3, 6, 7}
lc = lambda r: sum(1 for v in r if int(v) in LIVE and int(v) >= 0)

pool = S.SPPool(seeds); idx = list(range(pool.n))
# 半分ずつ座席を入れ替える
seat_me = np.array([k % 2 for k in range(N)])
agm = S.PolicyOpp(me, greedy=True, cap=True); ago = S.PolicyOpp(op, greedy=True, cap=True)
shops = {}; money = {}
with torch.no_grad():
    for t in range(719):
        if t in SNAP:
            shops[t] = np.asarray(pool.shops()).copy(); money[t] = pool.money().copy()
        acts = {}
        for s in (0, 1):
            st = pool.state(s)
            Um, Om = agm.act(st, t, idx, pool, s); Uo, Oo = ago.act(st, t, idx, pool, s)
            use_me = (seat_me == s)
            bc = lambda a: use_me.reshape((-1,) + (1,) * (np.asarray(a).ndim - 1))
            U = np.where(bc(Um), Um, Uo); O = np.where(bc(Om), Om, Oo)
            acts[s] = (U, O)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
shops[719] = np.asarray(pool.shops()).copy(); money[719] = pool.money().copy()
mg = {t: np.array([money[t][i, seat_me[i]] - money[t][i, 1 - seat_me[i]] for i in range(N)]) for t in money}
L = {t: np.array([lc(r) for r in shops[t]]) for t in shops}
TF = max(L)

def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0: return float('nan'), float('nan')
    r = float(np.corrcoef(x, y)[0, 1]); n = len(x)
    z = r * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1e-9, 1 - r * r))
    return r, z

print("局数 %d  最終の家畜店 平均 %.2f  最終の差 %+.0f" % (N, L[TF].mean(), mg[TF].mean()), flush=True)
for t in [144, 300, 500]:
    print("t=%3d 家畜店 平均%.2f  そこまでの差 %+8.0f" % (t, L[t].mean(), mg[t].mean()), flush=True)
print(flush=True)
print("■ (1) 逆向きの検査: 途中の差 → そのあと増える家畜店の数", flush=True)
for a, b in [(144, 300), (300, 500), (500, TF)]:
    d = L[b] - L[a]; r, z = corr(mg[a], d)
    print("  t=%3d の差 → t=%3d までに増えた家畜店(平均%.2f軒)  r=%+.3f z=%+.2f" % (a, b, d.mean(), r, z), flush=True)
print(flush=True)
print("■ (2) 順向きの検査: 先に開いた家畜店 → そのあとの差の伸び", flush=True)
for a, b in [(144, 300), (300, 500), (500, TF)]:
    dm = mg[b] - mg[a]; r, z = corr(L[a], dm)
    print("  t=%3d の家畜店 → t=%3d までの差の伸び(平均%+.0f)  r=%+.3f z=%+.2f" % (a, b, dm.mean(), r, z), flush=True)
print(flush=True)
r, z = corr(L[TF], mg[TF]); print("参考 最終の家畜店 → 最終の差  r=%+.3f z=%+.2f" % (r, z), flush=True)
r, z = corr(L[144], mg[TF]); print("参考 144手の家畜店 → 最終の差  r=%+.3f z=%+.2f" % (r, z), flush=True)
json.dump({"seeds": seeds, "live": {str(k): L[k].tolist() for k in L}, "margin": {str(k): mg[k].tolist() for k in mg}},
          open("/content/lab/probe_shops2.json", "w"))
print("書き出した", flush=True)
