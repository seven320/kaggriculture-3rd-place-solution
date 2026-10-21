"""Sweep the terminal-liquidation step on EXACTLY the agent we would submit.

The earlier sweep re-sorted the tape's SELLs by value and hoisted them ahead of
its buys; the submitted agent does neither. Measuring one thing and shipping
another is how the real engine came back with v15 losing to v13 on the first seed
it was tried on. This builds the full order list the agent builds: the tape's own
orders in the tape's own order, with liquidation SELLs appended at the end.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim, kagpack
from mkpolicy import M_SELL, N_PROD

D = "/content/drive/MyDrive/kaggriculture/data/"
TA, NA = kagpack.load(D + "boatlee_v29-r1-adaptive-market_n0_d1_0.tape.json")
SHED = {0: slice(18,27), 1: slice(27,36)}
# _SELLABLE order of the submitted agent, as product ids
SELL_ORDER = [3,4,6,7,5,2,1,0,8]     # STRAWBERRY MELON MILK WOOL EGG TOMATO CARROT WHEAT FERT

def orders(ob, env, liq_step, side):
    N = ob.shape[0]
    shed = ob[:, SHED[side]] * 100.0
    tape = env.tape_orders()                      # [10,3] all ops, tape order
    out = np.zeros((N, 10, 3), dtype=np.int32)
    k = 0
    listed = set()
    for j in range(10):
        if tape[j,0] == 0: continue
        out[:, k, :] = tape[j]; k += 1
        if tape[j,0] == M_SELL: listed.add(int(tape[j,1]))
    if liq_step is not None and env.step_index() >= liq_step and k < 10:
        # The agent appends a liquidation SELL only when the shed actually holds
        # that item, so an empty product consumes no slot. Incrementing the slot
        # for every product instead filled all ten with zero-quantity orders and
        # silently dropped the real ones (found by diffing step 601 against the
        # engine: the agent emitted SELL WHEAT 47 / SELL FERTILIZER 4, we emitted
        # three padded HIRE slots). Packing is per env because sheds differ.
        q = np.rint(shed).astype(np.int32)            # obs is shed/100 in float32
        prio = np.full(N_PROD, 99, dtype=np.int32)
        for r, it in enumerate(SELL_ORDER):
            prio[it] = r
        elig = (q > 0)
        for it in listed:
            elig[:, it] = False
        key = np.where(elig, prio[None, :], 99)
        idx = np.argsort(key, axis=1, kind="stable")
        ks  = np.take_along_axis(key, idx, axis=1)
        qs  = np.take_along_axis(q,   idx, axis=1)
        for sl in range(min(N_PROD, 10 - k)):
            out[:, k + sl, 0] = np.where(ks[:, sl] < 99, M_SELL, 0)
            out[:, k + sl, 1] = idx[:, sl]
            out[:, k + sl, 2] = qs[:, sl]
    return out

def duel(la, lb, seeds):
    w = t = n = 0; marg = []
    for seat in (0,1):
        N = len(seeds)
        env = kagsim.VecMarket(TA, NA, TA, NA, list(seeds), 720, seat)
        env.keep_nonsell = False                  # we supply the complete order list
        env.reset()
        while not env.done():
            ob = env.obs()
            env.step2(orders(ob, env, la, 0), orders(ob, env, lb, 1))
        m = env.money()
        w += int((m[:,0] > m[:,1]).sum()); t += int((m[:,0] == m[:,1]).sum()); n += N
        marg.append(m[:,0]-m[:,1])
    marg = np.concatenate(marg)
    return (w+0.5*t)/n, n, marg.mean(), marg.std()/np.sqrt(len(marg)), t/n

if __name__ == "__main__":
    seeds = list(range(94000, 94000 + int(os.environ.get("NSEED", 300))))
    print("基準 = v13 (liq680) ── 提出物と同一構成")
    print(f"{'liq step':<10} {'勝率':>7} {'引分':>6} {'コイン差':>10} {'±SE':>7}")
    for L in (None, 640, 660, 690, 700, 705, 710, 714, 717):
        wr, n, d, se, tie = duel(L, 680, seeds)
        print(f"{str(L):<10} {wr:>7.3f} {tie:>6.3f} {d:>+10,.0f} {se:>7,.0f}", flush=True)
