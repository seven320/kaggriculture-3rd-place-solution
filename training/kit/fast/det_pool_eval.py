"""Rank candidate routes against a pool of VERIFIED-DETERMINISTIC opponents.

Every field number we have trusted for two days came from data/nemesis: 54 tapes
selected by loss MARGIN, which is why its best member sat at rank 46 and why it
scores two verified-deterministic agents in the opposite order to the ladder
(LagrangianLocomotive 0.8659 at rank 14, l1af 0.8997 at rank 46).

data/pool_det holds 35 tapes from 21 teams whose two games agree 720/720 on the
farmer track under a fixed submissionId -- and the two games ran on different
seeds, so the agreement is open-loop, not luck. A frozen copy of such a team IS
that team, which is the one case where tape-vs-tape measures what the ladder
measures. Candidates play the pool with self excluded, both seats, four blocks.
"""
import sys, os, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/local")
import kagsim, kagpack
from newroute import liq_orders
from cf_combo import apply_dev

BASE = "/content/drive/MyDrive/kaggriculture"; D = BASE + "/data/"
FS = [tuple(x) for x in json.load(open(BASE + "/runs/l1af_field.json"))["sell"]]
E = np.zeros((0,), dtype=np.int32)

def vs(ME, liq, devs, opp, seeds):
    out = []
    for seat in (0, 1):
        tb, nb = opp
        env = (kagsim.VecMarket(ME[0], ME[1], tb, nb, list(seeds), 720, 0) if seat == 0
               else kagsim.VecMarket(tb, nb, ME[0], ME[1], list(seeds), 720, 1))
        env.keep_nonsell = False; env.reset()
        while not env.done():
            ob = env.obs(); day = env.step_index() // 24
            a = liq_orders(env, ob, 0, liq)
            if devs: apply_dev(a, ob, day, devs, 0)
            env.step2(a, E)
        m = env.money(); out.append(m[:, 0] - m[:, 1])
    return np.concatenate(out)

def wr(m): return float(((m > 0) + 0.5 * (m == 0)).mean())

def load_pool():
    meta = json.load(open(D + "pool_det/meta.json"))
    rows = meta if isinstance(meta, list) else meta.get("tapes", [])
    seen = set(); out = []
    for r in sorted(rows, key=lambda x: (x.get("rank") or 999)):
        f = D + "pool_det/" + r["file"]
        if not os.path.exists(f): continue
        h = open(f).read()[:3000]
        if h in seen: continue
        seen.add(h)
        try: out.append((r.get("team", r["file"])[:20], r.get("rank") or 999, kagpack.load(f)))
        except Exception: pass
    return out

def main():
    NS = int(os.environ.get("NSEED", 24))
    POOL = load_pool()
    print(f"決定的プール {len(POOL)}チーム  最上位 {POOL[0][0]}({POOL[0][1]}位)", flush=True)
    BLOCKS = [("A", 91100000), ("B", 92300000), ("C", 93500000), ("D", 94700000)]
    GRID = [560, 660, 700, None]
    cands = [("l1af(v34の経路)", 46, sorted(glob.glob(D + "nemesis/l1af*.tape.json"))[0])]
    for team, rk, _ in POOL[:8]:
        f = [D + "pool_det/" + r["file"] for r in json.load(open(D + "pool_det/meta.json"))
             if r.get("team", "").startswith(team[:12])]
        if f: cands.append((team, rk, f[0]))
    t0 = time.time(); res = []
    for label, rk, path in cands:
        try: ME = kagpack.load(path)
        except Exception: continue
        opp = [(t, p) for t, r, p in POOL if not t.startswith(label[:12])]
        top = None
        for liq in GRID:
            for dv, dn in ((FS, "介入あり"), ([], "介入なし")):
                w = float(np.mean([wr(vs(ME, liq, dv, p, list(range(BLOCKS[0][1], BLOCKS[0][1] + NS))))
                                   for _, p in opp]))
                if top is None or w > top[0]: top = (w, liq, dn, dv)
        blocks = []
        for b, s0 in BLOCKS:
            S = list(range(s0, s0 + NS))
            blocks.append(float(np.mean([wr(vs(ME, top[1], top[3], p, S)) for _, p in opp])))
        res.append((float(np.mean(blocks)), rk, label, top[1], top[2], blocks))
        print(f"  {rk:>4}位 {label:<24} {np.mean(blocks):.4f}  "
              f"(liq={top[1]}, {top[2]})  [{' '.join(f'{x:.3f}' for x in blocks)}]"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    res.sort(key=lambda x: -x[0])
    print("\n=== 決定的プールでの順位 ===")
    print(f"  {'勝率':>8}  {'ラダー':>6}  経路")
    for w, rk, label, liq, dn, _ in res:
        print(f"  {w:.4f}  {rk:>4}位  {label}  (liq={liq}, {dn})")
    lad = [r[1] for r in res]; pool = list(range(len(res)))
    if len(res) > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(lad, pool)
        print(f"\n  ラダー順位との順位相関: rho={rho:+.3f} (p={p:.3f})")
        print(f"  {'新プールはラダーを予測できている' if rho > 0.5 else 'まだ予測できていない -> プールも物差しにならない'}")
    json.dump([{"wr": w, "rank": rk, "route": lb, "liq": liq, "dev": dn} for w, rk, lb, liq, dn, _ in res],
              open(BASE + "/runs/det_pool_eval.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
