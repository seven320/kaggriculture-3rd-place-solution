"""Test the new route with our market layer on top.

boat29 goes 1-1199 against the six routes that beat us on the ladder today. The
84-tape pool that ranked boat29 first was extracted from 2026-08-31 replays; these
came from 2026-09-03 games. The route we ship is a meta behind, and that -- not
market timing -- is what the 28 non-close losses were.

Our own additions (terminal liquidation, the SELL/non-SELL deviations) were tuned
on boat29 and may or may not transfer, so each is re-measured on the new chassis
rather than carried over.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim, kagpack

BASE = "/content/drive/MyDrive/kaggriculture"; D = BASE + "/data/"
M_SELL = 6
SELL_ORDER = [3, 4, 6, 7, 5, 2, 1, 0, 8]

def liq_orders(env, ob, side, liq_step):
    """that side's own tape orders + terminal liquidation from its own shed"""
    tape = env.tape_orders() if side == 0 else env.opp_tape_orders()
    shed = ob[:, (18 if side == 0 else 27):(27 if side == 0 else 36)] * 100.0
    N = ob.shape[0]
    out = np.zeros((N, 10, 3), dtype=np.int32); k = 0; listed = set()
    for j in range(10):
        if tape[j, 0] == 0: continue
        out[:, k, :] = tape[j]; k += 1
        if tape[j, 0] == M_SELL: listed.add(int(tape[j, 1]))
    if liq_step is not None and env.step_index() >= liq_step and k < 10:
        q = np.rint(shed).astype(np.int32)
        prio = np.full(9, 99, np.int32)
        for r, it in enumerate(SELL_ORDER): prio[it] = r
        elig = q > 0
        for it in listed: elig[:, it] = False
        key = np.where(elig, prio[None, :], 99)
        idx = np.argsort(key, 1, kind="stable")
        ks = np.take_along_axis(key, idx, 1); qs = np.take_along_axis(q, idx, 1)
        for sl in range(min(9, 10 - k)):
            out[:, k+sl, 0] = np.where(ks[:, sl] < 99, M_SELL, 0)
            out[:, k+sl, 1] = idx[:, sl]; out[:, k+sl, 2] = qs[:, sl]
    return out

def duel(ta, tb, seeds, liq_a, liq_b):
    w = t = n = 0; marg = []
    for seat in (0, 1):
        N = len(seeds)
        A, B_ = (ta, tb) if seat == 0 else (tb, ta)
        env = kagsim.VecMarket(A[0], A[1], B_[0], B_[1], list(seeds), 720, seat)
        env.keep_nonsell = False; env.reset()
        while not env.done():
            ob = env.obs()
            env.step2(liq_orders(env, ob, 0, liq_a), liq_orders(env, ob, 1, liq_b))
        m = env.money()
        w += int((m[:, 0] > m[:, 1]).sum()); t += int((m[:, 0] == m[:, 1]).sum()); n += N
        marg.append(m[:, 0] - m[:, 1])
    mg = np.concatenate(marg)
    return (w+0.5*t)/n, mg.mean()

if __name__ == "__main__":
    seeds = list(range(4200000, 4200000 + int(os.environ.get("NSEED", 150))))
    SKY = kagpack.load(D + "nemesis/sky_machine_105066395.tape.json")
    HIR = kagpack.load(D + "nemesis/hiros_105048345.tape.json")
    B29 = kagpack.load(D + "boatlee_v29-r1-adaptive-market_n0_d1_0.tape.json")
    print("ヌル: sky vs sky (清算なし)          ", duel(SKY, SKY, seeds, None, None))
    print("ヌル: sky vs sky (両方清算640)       ", duel(SKY, SKY, seeds, 640, 640))
    print()
    for L in (None, 560, 600, 640, 680, 700):
        w, m = duel(SKY, SKY, seeds, L, None)
        print(f"sky+liq{str(L):<5} vs sky素     勝率 {w:.3f}  コイン差 {m:>+9,.0f}", flush=True)
    print()
    for lab, T in (("hiros", HIR), ("boat29", B29)):
        w, m = duel(SKY, T, seeds, 640, 640)
        print(f"sky+liq640 vs {lab}+liq640   勝率 {w:.3f}  コイン差 {m:>+9,.0f}", flush=True)
