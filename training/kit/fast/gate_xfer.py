"""出荷済みの6ゲートを、別の車体(テープ)に載せ替えて効くか測る。

ゲートは v39 の経路の上で貪欲に積んだものなので、(品目,日,動作) の意味が
別のテープでも同じとは限らない。ラダーの公開スコアは v30/v31 の方が高いので、
「ラダーで一番良かった車体 + 一番良かった適応層」が出せるかを確かめる。
"""
import sys, os, glob, hashlib, re, json, time
import numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack
from newroute import liq_orders
from cf_combo import apply_dev
BASE = "/content/drive/MyDrive/kaggriculture"; D = BASE + "/data/"
E = np.zeros((0,), dtype=np.int32); M_SELL = 6
FEAT = list(range(0, 27)) + [36, 37, 38, 39]
ITEMS = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"]
NS = int(os.environ.get("NSEED", 96))
POL = [( tuple(r["k"]), np.array(r["th"]), np.array(r["mu"]), np.array(r["sd"]) )
       for r in json.load(open(BASE + "/runs/cfq_ship.json"))]
PL, seen = [], set()
for f in sorted(glob.glob(D + "pool_big/*.tape.json")):
    h = hashlib.md5(open(f, "rb").read()).hexdigest()
    if h in seen: continue
    seen.add(h)
    if "l1af" in re.sub(r"_\d+\.tape\.json$", "", os.path.basename(f)): continue
    PL.append(kagpack.load(f))
PL = PL[:12]

def load(v):
    ns = {"__name__": "m"}
    exec(compile(open(BASE + "/snap/%s.py" % v).read().split("def agent")[0], v, "exec"), ns)
    return kagpack.pack(ns["_TRACE"]), [tuple(x) for x in ns.get("_DEVS", [])], ns.get("_LIQ_STEP"), ns.get("_LIQ_CAP")

def rollout(me, devs, liq, cap, opp, seeds, pol=()):
    tb, nb = opp
    env = kagsim.VecMarket(me[0], me[1], tb, nb, list(seeds), 720, 0)
    env.keep_nonsell = False; env.reset()
    fired = {}
    while not env.done():
        ob = env.obs(); t = env.step_index(); day = t // 24
        out = liq_orders(env, ob, 0, liq)
        if devs: apply_dev(out, ob, day, devs, 0)
        for k, th, mu, sd in pol:
            if k[1] != day: continue
            if k not in fired:
                Z = np.hstack([(ob[:, FEAT] - mu) / sd, np.ones((ob.shape[0], 1))])
                fired[k] = (1.0 / (1.0 + np.exp(-np.clip(Z @ th, -30, 30)))) > 0.5
            m = fired[k]
            if m.any():
                tmp = out[m].copy(); apply_dev(tmp, ob[m], day, [k], 0); out[m] = tmp
        if cap and t >= liq:
            tape = env.tape_orders(); nt = int((tape[:, 0] != 0).sum())
            sel = out[:, :, 0] == M_SELL
            idx = np.arange(out.shape[1])[None, :] >= nt
            out[:, :, 2] = np.where(sel & idx, np.minimum(out[:, :, 2], cap), out[:, :, 2])
        env.step2(out, E)
    m = env.money(); return m[:, 0] - m[:, 1]

if __name__ == "__main__":
    S = list(range(96000000, 96000000 + NS))     # ゲート学習に使っていないシード帯
    TEO = list(range(8, 12))
    print("ゲート: %s" % [(ITEMS[k[0]], k[1], "保持" if k[2] == 1 else "全投") for k, *_ in POL])
    print("%-6s %10s %10s %10s %9s" % ("車体", "ゲート無", "ゲート有", "差", "勝ち局率"), flush=True)
    for v in ("v41", "v34", "v31", "v30"):
        me, devs, liq, cap = load(v)
        if v == "v41": cap = cap or 12
        b, g = [], []
        for oi in TEO:
            b.append(rollout(me, devs, liq, cap, PL[oi], S))
            g.append(rollout(me, devs, liq, cap, PL[oi], S, pol=POL))
        b = np.concatenate(b); g = np.concatenate(g); d = g - b
        print("%-6s %10s %10s %10s %9.3f" % (v, format(round(b.mean()), ","), format(round(g.mean()), ","),
              ("+" if d.mean() >= 0 else "-") + format(abs(round(d.mean())), ","), float((d > 0).mean())), flush=True)
