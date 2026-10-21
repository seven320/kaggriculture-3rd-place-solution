"""exp290 の点検: 訓練用シミュレータ(BatchSPPool + LAND=1)で、第19世代の自己対戦 16局が本当に4区画目を買うか。"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import sp_fast, sp_lib as S, ppo_tf as P, model_tf as M
net = M.load("/content/local_logs/orbit_o37pace_gen19.pt", "cuda"); net.eval()
seeds = [777 + 7919 * k for k in range(16)]; pool = sp_fast.BatchSPPool(seeds)
print("LAND", sp_fast.LAND, "PACE", sp_fast.PACE, "land_ok", pool.land_ok.all(), "LAND_T", sp_fast._PACEMOD.LAND_T)
opp = [S.PolicyOpp(net, greedy=True, cap=True), S.PolicyOpp(net, greedy=True, cap=True)]
e = pool.env
for t in range(719):
    st = [pool.state(0), pool.state(1)]
    U, O = [], []
    for s in (0, 1):
        u, o = opp[s].act(st[s], t, None, pool, s); U.append(u); O.append(o)
    if t in (289, 300, 400, 500, 700):
        for s in (0, 1):
            e.set_view(s); fp = np.asarray(e.farm_pub(0))
            print("t=%d 席%d 区画 %s 所持金 %s" % (t, s, np.rint(fp[:, 2] * 4).astype(int).tolist(), (fp[:, 0] * 50000).astype(int).tolist()[:6]))
    pool.step(U[0], O[0], U[1], O[1])
