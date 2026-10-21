"""Apply a SET of market deviations to the v17 route and measure the combination.

Single-deviation scans found three very large ones (FERT day2 hold +2,338, FERT
day3 hold +2,351, WHEAT day3 hold +2,169, each at win rate 0.992 against v17 on
disjoint seeds). They are all early-game holds, so they are unlikely to be
independent; this measures the set rather than assuming additivity.
"""
import os, sys, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim
from faithful import orders, TA, NA, SHED
from mkpolicy import M_SELL

NAMES = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERT"]
LIQ = 640

def apply_dev(mine, ob, day, devs, side=0):
    # obs() is the learner's view; the opponent's shed is the 27:36 block. Reading
    # the wrong one makes a self-mirror stop being a mirror -- caught by a null
    # check that came back -1 +/- 22 with a worst case of -6,421 instead of 0.
    shed = ob[:, SHED[side]] * 100.0
    N = mine.shape[0]
    for (prod, d, act) in devs:
        if d != day: continue
        m = (mine[:, :, 0] == M_SELL) & (mine[:, :, 1] == prod)
        if act == 1:
            mine[:, :, 0][m] = 0
        else:
            q = np.rint(shed[:, prod]).astype(np.int32)
            had = m.any(1)
            for i in range(N):
                if q[i] <= 0: continue
                if had[i]:
                    mine[i, int(np.argmax(m[i])), 2] = q[i]
                else:
                    free = np.where(mine[i, :, 0] == 0)[0]
                    if len(free): mine[i, free[0]] = (M_SELL, prod, q[i])

def play(seeds, seat, devs=()):
    N = len(seeds)
    env = kagsim.VecMarket(TA, NA, TA, NA, list(seeds), 720, seat)
    env.keep_nonsell = False; env.reset()
    while not env.done():
        ob = env.obs()
        mine = orders(ob, env, LIQ, 0)
        if devs: apply_dev(mine, ob, env.step_index() // 24, devs)
        env.step2(mine, orders(ob, env, LIQ, 1))
    m = env.money()
    return m[:, 0] - m[:, 1]

def score(seeds, devs=()):
    s = np.concatenate([play(seeds, se, devs) for se in (0, 1)])
    w = float(((s > 0) + 0.5*(s == 0)).mean())
    return w, s.mean(), s.std()/np.sqrt(len(s)), float(np.percentile(s, 5))

def label(devs):
    return " + ".join(f"{NAMES[p]}d{d}{'hold' if a==1 else 'dump'}" for p, d, a in devs) or "(none)"

if __name__ == "__main__":
    seeds = list(range(670000, 670000 + int(os.environ.get("NSEED", 128))))
    C = [(8,2,1), (8,3,1), (0,3,1)]
    print(f"{'set':<52}{'勝率':>7}{'コイン差':>10}{'±SE':>7}{'p5':>10}")
    for r in range(0, 4):
        for combo in itertools.combinations(C, r):
            w, d, se, p5 = score(seeds, combo)
            print(f"{label(combo):<52}{w:>7.3f}{d:>+10,.0f}{se:>7,.0f}{p5:>+10,.0f}", flush=True)
