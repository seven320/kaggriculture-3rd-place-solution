"""現ラダー上位30から採った29本(data/league)をリーグ相手として使う。

kagsim は実エンジンと一致することを確認済み(同じ対戦で 54,140 対 54,059)。
旧プール(data/pool_big、09-05断面の12体)は弱すぎて、生成器を 73.9k と
過大評価していた(実際は 54k)。以後の判定はこのリーグで行う。
"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack
from newroute import liq_orders
B = "/content/drive/MyDrive/kaggriculture"
E = np.zeros((0,), dtype=np.int32)
META = {x["file"]: x for x in json.load(open(B + "/data/league/meta.json"))}
LEAGUE = []
for f in sorted(glob.glob(B + "/data/league/*.tape.json")):
    m = META.get(os.path.basename(f))
    if m: LEAGUE.append((m["team"], m["rank"], m["score"], m["kind"], kagpack.load(f)))
LEAGUE.sort(key=lambda r: r[1])

def duel_tape(me, opp, seeds, liq=620):
    env = kagsim.VecMarket(me[0], me[1], opp[0], opp[1], list(seeds), 720, 0)
    env.keep_nonsell = False; env.reset()
    while not env.done():
        ob = env.obs(); env.step2(liq_orders(env, ob, 0, liq), E)
    return env.money()

def duel_gen(params, opp, seeds):
    from dispatch import Dispatcher
    from gen_run import BLANK
    BS = 10
    env = kagsim.VecMarket(BLANK[0], BLANK[1], opp[0], opp[1], list(seeds), 720, 0)
    env.keep_nonsell = False; env.reset()
    DS = [Dispatcher(params) for _ in seeds]; N = len(seeds)
    while not env.done():
        ob = env.obs(); t = env.step_index(); day, hour = divmod(t, 24)
        uo = env.unit_obs(14); sh = env.seeds_held()
        U = np.zeros((N, 14, 3), np.int32); MK = np.zeros((N, 10, 3), np.int32)
        for i in range(N):
            b = uo[i, 0, 8:].reshape(100, 6)
            kind = np.rint(b[:, 0]*5).astype(int).reshape(BS, BS); what = np.rint(b[:, 1]*12).astype(int).reshape(BS, BS)
            watd = (b[:, 2] > .5).reshape(BS, BS); fedd = (b[:, 3] > .5).reshape(BS, BS)
            yld = np.rint(b[:, 4]*6).astype(int).reshape(BS, BS)
            nu = max(1, int((uo[i, :, 6] > 0).sum())); ch5 = uo[i, :nu, 8:].reshape(nu, 100, 6)[:, :, 5]
            fav = ((np.abs(ch5-0.5) < 0.1).any(0)).reshape(BS, BS)
            ux = np.rint(uo[i, :nu, 3]*9).astype(int); uy = np.rint(uo[i, :nu, 4]*9).astype(int)
            shed = np.rint(ob[i, 18:27]*100).astype(int); DS[i].mktinv = ob[i, 0:9]*10000
            ops, mk = DS[i].act(day, hour, kind, what, watd, fedd, yld, ux, uy, shed,
                                float(ob[i, 36]*50000), sh[i], None, fav)
            U[i, :nu] = ops[:nu]
            for j, o in enumerate(mk[:10]): MK[i, j] = o
        env.step_units(U, MK, E)
    return env.money()

def score(kind, arg, seeds, verbose=False):
    rows = []
    for team, rk, sc, knd, tp in LEAGUE:
        m = duel_tape(arg, tp, seeds) if kind == "tape" else duel_gen(arg, tp, seeds)
        d = m[:, 0] - m[:, 1]
        rows.append((team, rk, knd, float(d.mean()), float((d > 0).mean()), float(m[:, 0].mean())))
        if verbose: print("  %-20s %3d位 %-6s margin %9s 勝ち率 %.2f 自分 %9s" % (
            team[:20], rk, knd, format(round(d.mean()), ","), (d > 0).mean(), format(round(m[:, 0].mean()), ",")), flush=True)
    a = np.array([r[3] for r in rows]); w = np.array([r[4] for r in rows])
    return a.mean(), w.mean(), min(r[3] for r in rows), rows

if __name__ == "__main__":
    S = list(range(700000, 700000 + int(os.environ.get("NSEED", 8))))
    print("リーグ %d チーム(台本型 %d / ほぼ台本 %d / 適応型 %d)" % (
        len(LEAGUE), sum(1 for r in LEAGUE if r[3] == "台本型"),
        sum(1 for r in LEAGUE if r[3] == "ほぼ台本"), sum(1 for r in LEAGUE if r[3] == "適応型")), flush=True)
    for v in ("v50", "v49", "v47"):
        ns = {"__name__": "m"}
        exec(compile(open(B + "/snap/%s.py" % v).read().split("def agent")[0], v, "exec"), ns)
        a, w, worst, _ = score("tape", kagpack.pack(ns["_TRACE"]), S)
        print("%-6s 平均margin %9s  平均勝ち率 %.3f  最悪 %9s" % (v, format(round(a), ","), w, format(round(worst), ",")), flush=True)
    p = json.load(open(B + "/runs/gen_params.json"))
    if "mix_w" in p: p["plant_mix"] = [p.pop("mix_w"), p.pop("mix_c", 0.3), 0.0, p.pop("mix_s"), p.pop("mix_m")]
    a, w, worst, _ = score("gen", p, S)
    print("%-6s 平均margin %9s  平均勝ち率 %.3f  最悪 %9s" % ("生成器", format(round(a), ","), w, format(round(worst), ",")), flush=True)
