"""09-22: 方策の序盤の「買えるようになった手」と「買った手」の差を測る(貪欲の自己対戦、席0を見る)。
所持金は観測 g の 32 列目 × 50000(分析係の確認: t=0 で 3000、整数との誤差 3e-4 未満)。区画数は farm_pub(0)[:,2]×4。
  使い方: POL=<重み> POLV=<合法手の版> NSEED=32 python3 afford_scan.py 出力.json"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
import sp_fast; S.SPPool = sp_fast.BatchSPPool
import fast_codes; P.to_codes = fast_codes.to_codes_fast
out = sys.argv[1]; POL = os.environ["POL"]; V = int(os.environ.get("POLV", "0")); NSEED = int(os.environ.get("NSEED", "32"))
net = M.load(POL, P.DEV).eval(); pol = S.PolicyOpp(net, greedy=True, cap=True)
pool = S.SPPool([8900000 + 7919 * j for j in range(NSEED)]); n = pool.n
pool.legal_opt = np.full((n, 2), V, np.int32)
money = np.zeros((719, n)); quad = np.zeros((719, n))
with torch.no_grad():
    for t in range(719):
        acts = {}
        for s in (0, 1):
            st = pool.state(s)
            if s == 0:
                money[t] = np.asarray(st[3])[:, 32] * 50000
                e = pool.env; e.set_view(0); quad[t] = np.clip(np.rint(np.asarray(e.farm_pub(0))[:, 2] * 4), 0, 4)
            acts[s] = pol.act(st, t, list(range(n)), pool, s)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
first = lambda cond: np.where(cond.any(0), cond.argmax(0), 719)
q2 = first(quad >= 2); q3 = first(quad >= 3); q4 = first(quad >= 4)
a2 = first((quad == 1) & (money >= 1000)); a3 = first((quad == 2) & (money >= 2000)); a4 = first((quad == 3) & (money >= 4000))
R = dict(pol=POL, q2=q2.tolist(), q3=q3.tolist(), q4=q4.tolist(), a2=a2.tolist(), a3=a3.tolist(), a4=a4.tolist(),
         money_med=[float(np.median(money[t])) for t in range(0, 719, 6)])
json.dump(R, open(out, "w"))
for k, (a, q) in enumerate(((a2, q2), (a3, q3), (a4, q4)), 2):
    print("%d区画目: 買えるようになった手 中央値 %d(最小 %d) / 買った手 中央値 %d(最小 %d) / 差の中央値 %d" % (k, np.median(a), a.min(), np.median(q), q.min(), np.median(q - a)), flush=True)
print("所持金の中央値(手 60/96/120/130/150/180/220):", [int(np.median(money[t])) for t in (60, 96, 120, 130, 150, 180, 220)], flush=True)
print("== 済 ==", flush=True)
