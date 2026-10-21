"""Counterfactual scan of the NON-SELL market orders on the v17 route.

Same shape as fast/cf_v17.py: opponent is v17 itself, learner is v17 with exactly
one non-SELL change on exactly one day, same seeds, both seats played so the
mirror cancels to a margin of exactly 0. Win rate is the gate but saturates in a
self-mirror (84% exact ties), so the ranking response is the paired coin margin.

The 5th percentile of the paired delta is printed next to the mean because the
animal line is the trap: eight cows and five sheep eat bought wheat, an animal
that misses two days is gone for good, and a feed cut is free for a week before
it costs tens of thousands. A positive mean with a p5 of -20,000 is not a win.

Interventions per day d (only emitted where the tape actually has such an order):
  hire_drop   drop every HIRE that day
  hire_add    one extra HIRE, inserted straight after the day's last tape HIRE
  wheat_drop  drop BUY_PRODUCT WHEAT
  wheat_x2    double the BUY_PRODUCT WHEAT quantity
  wheat_half  halve it (a cash-freeing middle the brief did not ask for, but the
              route runs at $1-$5 on days 0-1 so partial cuts are the live axis)
  seed_drop   drop BUY_SEED
  fert_drop   drop BUY_PRODUCT FERTILIZER
  anim_drop   drop BUY_ANIMAL
  land_drop   drop BUY_LAND
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, '/content/drive/MyDrive/kaggriculture/fast')
import kagsim
from faithful import orders, TA, NA

LIQ  = 640
TAPE = json.load(open('/content/drive/MyDrive/kaggriculture/data/'
                      'boatlee_v29-r1-adaptive-market_n0_d1_0.tape.json'))
M_HIRE, M_LAND, M_SEED, M_PROD, M_ANIMAL, M_SELL = 1, 2, 3, 4, 5, 6
WHEAT, FERT = 0, 8

# ---------------------------------------------------------------- tape survey
def survey():
    """which days carry which non-SELL op, and where the day's last HIRE sits"""
    has = {k: set() for k in ("hire", "wheat", "seed", "fert", "anim", "land")}
    last_hire = {}
    for i, s in enumerate(TAPE):
        d = i // 24
        for o in (s.get("market") or []):
            if o[0] == "HIRE":            has["hire"].add(d); last_hire[d] = i
            elif o[0] == "BUY_LAND":      has["land"].add(d)
            elif o[0] == "BUY_SEED":      has["seed"].add(d)
            elif o[0] == "BUY_ANIMAL":    has["anim"].add(d)
            elif o[0] == "BUY_PRODUCT":
                if o[1] == "WHEAT":       has["wheat"].add(d)
                elif o[1] == "FERTILIZER":has["fert"].add(d)
    return has, last_hire

HAS, LAST_HIRE = survey()
SLOT_DROPS = 0            # rows where inserting a HIRE pushed a real order off

def intervene(mine, env, kind, day):
    """edit the learner's [N,10,3] order list in place for one intervention"""
    global SLOT_DROPS
    step = env.step_index()
    if kind.startswith("late_"):
        # a RULE, not a one-day poke: `day` is the first day it applies from.
        # Step 673 is the case that motivates it -- the tape spends 3 of the 10
        # market slots there on FERTILIZER/HIRE/WHEAT-SEED while the agent is
        # already liquidating, so faithful.orders has only 7 left for 9 products.
        if step // 24 < day:
            return mine
        op, it = mine[:, :, 0], mine[:, :, 1]
        if kind in ("late_seed", "late_buy"): op[op == M_SEED] = 0
        if kind in ("late_prod", "late_buy"): op[op == M_PROD] = 0
        if kind == "late_fert": op[(op == M_PROD) & (it == FERT)] = 0
        return mine
    if step // 24 != day:
        return mine
    op, it, qt = mine[:, :, 0], mine[:, :, 1], mine[:, :, 2]
    if kind == "hire_drop":
        op[op == M_HIRE] = 0
    elif kind == "hire_add":
        if step != LAST_HIRE.get(day, -1):
            return mine
        col = np.where(mine[0, :, 0] == M_HIRE)[0]
        if len(col) == 0:
            return mine
        j = int(col[-1]) + 1                       # right after the last tape HIRE
        SLOT_DROPS += int((mine[:, 9, 0] != 0).sum())
        mine[:, j+1:, :] = mine[:, j:9, :]         # shift right, slot 9 falls off
        mine[:, j, 0] = M_HIRE; mine[:, j, 1] = 0; mine[:, j, 2] = 1
    elif kind == "wheat_drop":
        op[(op == M_PROD) & (it == WHEAT)] = 0
    elif kind == "wheat_x2":
        m = (op == M_PROD) & (it == WHEAT); qt[m] = qt[m] * 2
    elif kind == "wheat_half":
        m = (op == M_PROD) & (it == WHEAT); qt[m] = np.maximum(1, qt[m] // 2)
    elif kind == "seed_drop":
        op[op == M_SEED] = 0
    elif kind == "fert_drop":
        op[(op == M_PROD) & (it == FERT)] = 0
    elif kind == "anim_drop":
        op[op == M_ANIMAL] = 0
    elif kind == "land_drop":
        op[op == M_LAND] = 0
    else:
        raise ValueError(kind)
    return mine

def play(seeds, seat, kind=None, day=None, combo=None):
    env = kagsim.VecMarket(TA, NA, TA, NA, list(seeds), 720, seat)
    env.keep_nonsell = False; env.reset()
    while not env.done():
        ob = env.obs()
        mine = orders(ob, env, LIQ, 0)
        if combo:
            for k, d in combo: mine = intervene(mine, env, k, d)
        elif kind is not None:
            mine = intervene(mine, env, kind, day)
        env.step2(mine, orders(ob, env, LIQ, 1))
    m = env.money()
    return m[:, 0], m[:, 1]

def evaluate(seeds, base, kind=None, day=None, combo=None):
    a = [play(seeds, s, kind, day, combo) for s in (0, 1)]
    mine = np.concatenate([x[0] for x in a]); opp = np.concatenate([x[1] for x in a])
    marg = mine - opp
    d = marg - base
    wr = float(((marg > 0).sum() + 0.5 * (marg == 0).sum()) / len(marg))
    return d.mean(), d.std(ddof=1)/np.sqrt(len(d)), np.percentile(d, 5), wr, float((marg == 0).mean())

def baseline(seeds):
    a = [play(seeds, s) for s in (0, 1)]
    return np.concatenate([x[0] - x[1] for x in a])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=256)
    ap.add_argument("--base",  type=int, default=500000)
    ap.add_argument("--out",   default="/content/drive/MyDrive/kaggriculture/runs/buy_cf.json")
    a = ap.parse_args()
    seeds = list(range(a.base, a.base + a.nseed))
    t0 = time.time()
    base = baseline(seeds)
    print(f"[null] mirror margin {base.mean():+.6f}  ties {np.mean(base==0):.3f}  "
          f"n={len(base)}  seeds {a.base}..{a.base+a.nseed-1}", flush=True)
    assert abs(base.mean()) < 1e-9, "mirror is not 0 -- harness is wrong, stop"

    jobs = []
    for d in range(30):
        if d in HAS["hire"]:  jobs += [("hire_drop", d), ("hire_add", d)]
        if d in HAS["wheat"]: jobs += [("wheat_drop", d), ("wheat_x2", d), ("wheat_half", d)]
        if d in HAS["seed"]:  jobs += [("seed_drop", d)]
        if d in HAS["fert"]:  jobs += [("fert_drop", d)]
        if d in HAS["anim"]:  jobs += [("anim_drop", d)]
        if d in HAS["land"]:  jobs += [("land_drop", d)]
    rows = []
    for n, (k, d) in enumerate(jobs):
        m, se, p5, wr, tie = evaluate(seeds, base, k, d)
        rows.append(dict(kind=k, day=d, mean=float(m), se=float(se), p5=float(p5),
                         wr=wr, tie=tie, z=float(m/max(se,1e-9))))
        if n % 20 == 0:
            print(f"  {n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: -r["mean"])
    hdr = f"{'intervention':<12}{'day':>4}{'Δmargin':>11}{'±SE':>8}{'z':>7}{'p5':>11}{'win':>7}{'tie':>7}"
    print("\n== 上位20 ==\n" + hdr)
    for r in rows[:20]:
        print(f"{r['kind']:<12}{r['day']:>4}{r['mean']:>+11,.0f}{r['se']:>8,.0f}"
              f"{r['z']:>+7.2f}{r['p5']:>+11,.0f}{r['wr']:>7.3f}{r['tie']:>7.3f}")
    print("\n== 下位10 ==\n" + hdr)
    for r in rows[-10:]:
        print(f"{r['kind']:<12}{r['day']:>4}{r['mean']:>+11,.0f}{r['se']:>8,.0f}"
              f"{r['z']:>+7.2f}{r['p5']:>+11,.0f}{r['wr']:>7.3f}{r['tie']:>7.3f}")
    json.dump(rows, open(a.out, "w"), indent=0)
    print(f"\n{len(jobs)} interventions, {time.time()-t0:.0f}s -> {a.out}")

if __name__ == "__main__":
    main()
