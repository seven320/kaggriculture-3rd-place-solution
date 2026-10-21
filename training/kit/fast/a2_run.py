"""agent2 を kagsim で走らせる。空テープを積んで step_units で全上書き。"""
import sys, os, json
import numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack
from agent2 import Agent2, BS, OP
E = np.zeros((0,), dtype=np.int32)
MAXU = 40
BLANK = kagpack.pack([{"farmer": ["PASS"], "hands": [["PASS"]]*(MAXU-1), "market": []} for _ in range(720)])

def run(params, opp, seeds, trace=None):
    env = kagsim.VecMarket(BLANK[0], BLANK[1], opp[0], opp[1], list(seeds), 720, 0)
    env.keep_nonsell = False; env.reset()
    N = len(seeds); A = [Agent2(params) for _ in range(N)]
    for a in A: a._nk = 2
    while not env.done():
        ob = env.obs(); t = env.step_index(); day, hour = divmod(t, 24)
        uo = env.unit_obs(14); sh = env.seeds_held()
        U = np.zeros((N, 14, 3), np.int32); MK = np.zeros((N, 10, 3), np.int32)
        for i in range(N):
            b = uo[i, 0, 8:].reshape(100, 6)
            kind = np.rint(b[:, 0]*5).astype(int).reshape(BS, BS)
            what = np.rint(b[:, 1]*12).astype(int).reshape(BS, BS)
            watd = (b[:, 2] > .5).reshape(BS, BS); fedd = (b[:, 3] > .5).reshape(BS, BS)
            yld = np.rint(b[:, 4]*6).astype(int).reshape(BS, BS)
            nu = max(1, int((uo[i, :, 6] > 0).sum()))
            ch5 = uo[i, :nu, 8:].reshape(nu, 100, 6)[:, :, 5]
            fav = ((np.abs(ch5-0.5) < 0.1).any(0)).reshape(BS, BS)
            ux = np.rint(uo[i, :nu, 3]*9).astype(int); uy = np.rint(uo[i, :nu, 4]*9).astype(int)
            shed9 = np.rint(ob[i, 18:27]*100).astype(np.int64)
            shed = np.concatenate([shed9, np.zeros(3, np.int64)])
            A[i].mktinv = ob[i, 0:9]*10000.0
            A[i]._nk = min(2 + int((((kind == 3) | (kind == 4)) & (what >= 9)).sum())//6, max(1, nu-2)) if nu > 2 else 0
            invu = np.zeros((max(nu, 1), 12), np.int64)     # 手持ちは engine から取れないので粗く追跡
            if hasattr(A[i], "inv_track") and A[i].inv_track.shape[0] == max(nu, 1): invu = A[i].inv_track
            ops, mk = A[i].act(day, hour, kind, what, watd, fedd, yld, fav, ux, uy, shed,
                               float(ob[i, 36]*50000), sh[i], invu)
            # 手持ちの粗い更新
            for u in range(nu):
                o = int(ops[u, 0])
                if o == OP["PICKUP"]: invu[u, int(ops[u, 1])] += int(ops[u, 2])
                elif o == OP["FEED"]: invu[u, 0] = max(0, invu[u, 0]-1)
                elif o == OP["FERTILIZE"]: invu[u, 8] = max(0, invu[u, 8]-1)
                elif o == OP["PLACE"]: invu[u, int(ops[u, 1])] = max(0, invu[u, int(ops[u, 1])]-1)
                elif o == OP["COLLECT"]: invu[u, 8] += 1
            if hour == 23: invu[:] = 0
            A[i].inv_track = invu
            U[i, :nu] = ops[:nu]
            for j, o in enumerate(mk[:10]): MK[i, j] = o
            if trace is not None and i == 0 and hour == 12:
                trace.append(dict(day=day, money=int(ob[i, 36]*50000), nu=nu,
                                  plant=int((kind == 5).sum()), empty=int((kind == 0).sum()),
                                  lock=int((kind == 1).sum()), weed=int((kind == 2).sum()),
                                  anim=int((((kind == 3) | (kind == 4)) & (what >= 9)).sum()),
                                  past=int(((kind == 3) | (kind == 4)).sum()), shedW=int(shed9[0])))
        env.step_units(U, MK, E)
    m = env.money(); return m[:, 0]-m[:, 1], m
