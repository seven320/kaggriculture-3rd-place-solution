"""exp316: 家畜を買う店(ICE_CREAM/SMOOTHIE/YARN)の構成が、局のどこで決まるかを調べる。
  - 盤面の引きが対局開始時に確定しているなら、対局する盤面を偏らせる課程(curriculum)が使える。
  - 打ち方(貪欲/標本)を変えても最後の店の組が同じなら、店は種だけで決まる。
使い方: NSEED=96 python probe_shops.py <重み>
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
import model_tf as M
import sp_lib as S

DEV = "cuda"
N = int(os.environ.get("NSEED", "96")); TMAX = int(os.environ.get("TMAX", "719"))
INIT = sys.argv[1] if len(sys.argv) > 1 else "/content/local_logs/orbit_g5g_gen45.pt"
SNAP = [0, 50, 100, 144, 200, 300, 400, 500, 600, 718]
seeds = [1000003 + 7919 * k for k in range(N)]
net = M.load(INIT, DEV); net.eval()
# 店の番号: obs_feat の SHOPS の並び(アルファベット順)
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
LIVE = {3, 6, 7}   # ICE_CREAM / SMOOTHIE / YARN

def live_count(row):
    return sum(1 for v in row if int(v) in LIVE and int(v) >= 0)

def run(greedy):
    pool = S.SPPool(seeds); idx = list(range(pool.n))
    ag = S.PolicyOpp(net, greedy=greedy, cap=True)
    out = {}
    with torch.no_grad():
        for t in range(TMAX):
            if t in SNAP: out[t] = np.asarray(pool.shops()).copy()
            acts = {}
            for s in (0, 1):
                st = pool.state(s); acts[s] = ag.act(st, t, idx, pool, s)
            pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    out[TMAX] = np.asarray(pool.shops()).copy()
    return out, pool.money()

A, mA = run(True)
print("A(貪欲) 終了。所持金の平均 %.0f" % mA.mean(), flush=True)
for t in sorted(A):
    a = A[t]
    cnt = [live_count(r) for r in a]
    nsh = [sum(1 for v in r if int(v) >= 0) for r in a]
    print("t=%3d 店の数 平均%.2f 家畜店 平均%.2f 内訳 %s" % (t, np.mean(nsh), np.mean(cnt),
          {k: cnt.count(k) for k in sorted(set(cnt))}), flush=True)
B, mB = run(False)
print("B(標本) 終了。所持金の平均 %.0f" % mB.mean(), flush=True)
kt = max(A)
sa = [sorted(int(v) for v in r if int(v) >= 0) for r in A[kt]]
sb = [sorted(int(v) for v in r if int(v) >= 0) for r in B[kt]]
same = sum(1 for x, y in zip(sa, sb) if x == y)
print("最後の店の組が A と B で一致した局 %d/%d" % (same, len(sa)), flush=True)
t0a = [sorted(int(v) for v in r if int(v) >= 0) for r in A[0]]
same0 = sum(1 for x, y in zip(t0a, sa) if x == y)
print("0手目の店の組が最後と同じ局 %d/%d" % (same0, len(sa)), flush=True)
for t in sorted(A):
    x = [sorted(int(v) for v in r if int(v) >= 0) for r in A[t]]
    print("t=%3d の組が最後と一致 %d/%d" % (t, sum(1 for p, q in zip(x, sa) if p == q), len(sa)), flush=True)
json.dump({"seeds": seeds, "final_live": [live_count(r) for r in A[kt]],
           "t0_live": [live_count(r) for r in A[0]]}, open("/content/lab/probe_shops.json", "w"))
print("書き出した /content/lab/probe_shops.json", flush=True)
