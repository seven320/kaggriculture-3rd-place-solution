"""exp319: 介入は方策に入ったのか、環境で覆っていただけか。
介入つきで学習した世代を **介入なし** で打たせて、狙った振る舞いが残っているかを測る。
残っていなければ「身につかない」= 環境側のガードレールは masking にすぎず、学習の中身を変えていない。
使い方: NSEED=64 python probe_stick.py <名前>=<重み> [<名前>=<重み> ...]
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf as M
import sp_lib as S            # 先に読む(fast の道を通す)
import kagpack

DEV = "cuda"; N = int(os.environ.get("NSEED", "64")); T1 = int(os.environ.get("T1", "240"))
seeds = [3000017 + 7919 * k for k in range(N)]
BUYSEED = kagpack.MOPI["BUY_SEED"]; BUYLAND = kagpack.MOPI["BUY_LAND"]
IT = {k: kagpack.ITI[k] for k in ("MELON", "WHEAT", "STRAWBERRY", "TOMATO", "CARROT")}
BUYANI = kagpack.MOPI.get("BUY_ANIMAL")

def run(name, path):
    net = M.load(path, DEV); net.eval()
    pool = S.SPPool(seeds); idx = list(range(pool.n))
    ag = S.PolicyOpp(net, greedy=True, cap=True)
    seed_cnt = {k: np.zeros(N) for k in IT}          # 序盤(T1まで)に出した種の注文の個数
    ani = np.zeros(N)                                 # 序盤に買った家畜
    q_step = {2: np.full(N, -1), 3: np.full(N, -1), 4: np.full(N, -1)}
    with torch.no_grad():
        for t in range(719):
            for s in (0, 1):
                st = pool.state(s); U, O = ag.act(st, t, idx, pool, s)
                if s == 0 and t < T1:
                    A = np.asarray(O)
                    for k, code in IT.items():
                        seed_cnt[k] += ((A[..., 0] == BUYSEED) & (A[..., 1] == code)).sum(1) * 1.0
                    if BUYANI is not None: ani += (A[..., 0] == BUYANI).sum(1) * 1.0
                if s == 0:
                    pool.envs[0].set_view(0)
                    nq = np.array([np.rint(np.asarray(e.farm_pub(0))[0, 2] * 4) for e in pool.envs])
                    for q in (2, 3, 4):
                        hit = (nq >= q) & (q_step[q] < 0); q_step[q][hit] = t
                if s == 0: a0 = (U, O)
                else: a1 = (U, O)
            pool.step(a0[0], a0[1], a1[0], a1[1])
    out = dict(name=name, money=float(pool.money()[:, 0].mean()))
    for k in IT: out["種_" + k] = float(seed_cnt[k].mean())
    out["家畜"] = float(ani.mean())
    for q in (2, 3, 4):
        v = q_step[q]; out["区画%d手番" % q] = float(v[v >= 0].mean()) if (v >= 0).any() else -1.0
        out["区画%d到達" % q] = float((v >= 0).mean())
    del net; torch.cuda.empty_cache()
    return out

rows = []
for a in sys.argv[1:]:
    nm, p = a.split("=", 1)
    r = run(nm, p); rows.append(r)
    print("%-12s 所持金 %7.0f | 序盤の種 メロン %.2f 小麦 %.2f イチゴ %.2f | 家畜 %.2f | 区画2 %.0f手(%.2f) 区画3 %.0f手(%.2f) 区画4 %.0f手(%.2f)"
          % (r["name"], r["money"], r["種_MELON"], r["種_WHEAT"], r["種_STRAWBERRY"], r["家畜"],
             r["区画2手番"], r["区画2到達"], r["区画3手番"], r["区画3到達"], r["区画4手番"], r["区画4到達"]), flush=True)
json.dump(rows, open("/content/lab/probe_stick.json", "w"), ensure_ascii=False)
print("書き出した", flush=True)
