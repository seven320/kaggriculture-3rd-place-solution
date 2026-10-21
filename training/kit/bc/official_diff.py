"""exp310: 記録 / 公式エンジンの打ち直し / 我々の kagsim の3つを1手ずつ並べ、
最初にズレる手でその手番の農場の状態をすべて突き合わせる。
使い方(容器内): REPLAY=<path> TEAM=<team> [MAXSTEP=300] python3 official_diff.py"""
import json, os, sys, gzip, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack, extract3
from extract4 import state, ucand, q_class, codes, UQ_MAX, MQ_MAX, NU, NS, M_STOP, mcand, QNONE, BASE
from deploy import wide_blank

def rj(p): return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
d = rj(os.environ["REPLAY"]); team = os.environ["TEAM"]; MAX = int(os.environ.get("MAXSTEP", "300"))
tn = d["info"]["TeamNames"]; seat = tn.index(team); S = d["steps"]
rec = [S[i][seat]["observation"]["farms"][seat]["money"] for i in range(len(S))]

# --- 1) 公式エンジンで打ち直す。env.state の観測は「その手の開始時点」 ---
from kaggle_environments import make
acts = [[S[i][0].get("action"), S[i][1].get("action")] for i in range(len(S) - 1)]
cfg = dict(d.get("configuration") or {}); cfg["seed"] = int(d["info"]["seed"])
env = make("kaggriculture", configuration=cfg, debug=False)
env.reset(2)
off = [env.state[seat]["observation"]["farms"][seat]["money"]]
off_farm = [json.loads(json.dumps(env.state[seat]["observation"]["farms"][seat]))]
off_shed = [json.loads(json.dumps(env.state[seat]["observation"]["private"]["shed"]))]
for i, ac in enumerate(acts[:MAX]):
    try: env.step(ac)
    except Exception as e:
        print("公式エンジンで再生できない %d手目: %s" % (i, str(e)[:120])); break
    off.append(env.state[seat]["observation"]["farms"][seat]["money"])
    off_farm.append(json.loads(json.dumps(env.state[seat]["observation"]["farms"][seat])))
    off_shed.append(json.loads(json.dumps(env.state[seat]["observation"]["private"]["shed"])))
n = min(len(off), len(rec))
bad = [i for i in range(n) if abs(off[i] - rec[i]) > 0.5]
print("公式の打ち直し 対 記録: %d手中ズレ %d (最初の5つ %s)" % (n, len(bad), bad[:5]))

# --- 2) 我々の kagsim ---
me = extract3.acts_of(d, seat); opp = kagpack.pack(extract3.acts_of(d, 1 - seat)); w = wide_blank()
A, Bt = (w, opp) if seat == 0 else (opp, w)
e2 = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [int(d["info"]["seed"])], 720, seat); e2.keep_nonsell = False; e2.reset()
rej = []; ours = []
first = None
for i, a in enumerate(me[:MAX]):
    m = float(e2.money()[0][0]); ours.append(m)
    if first is None and i < len(off) and abs(m - off[i]) > 0.5: first = i
    tiles, ou, pu, g, lm = state(e2); live = [u for u in range(NU) if ou[u, 0] > 0]
    us = [a["farmer"]] + a["hands"]; ul = np.full(NU, -1, np.int8); uq = np.zeros(NU, np.int16)
    for u in live:
        if u < len(us):
            c = ucand(us[u]); q = int(us[u][2]) if len(us[u]) > 2 else QNONE
            if not lm[u, c]: rej.append((i, u, BASE[c] if c < len(BASE) else "c%d" % c)); c, q = 0, QNONE
            ul[u] = c; uq[u] = q_class(q, UQ_MAX) if c >= 20 else 0
        else: ul[u] = 0
    ms = np.full(NS, M_STOP, np.int8); mq = np.zeros(NS, np.int16)
    for j, o in enumerate(a["market"][:10]):
        c2 = mcand(o); ms[j] = c2
        if c2 < 19: mq[j] = q_class(int(o[2]), MQ_MAX)
    U, O = codes(ul, uq, live, ms, mq); e2.step_units(U[None], O[None])
print("我々の kagsim が公式とズレ始める手:", first)
if first is not None:
    lo = max(0, first - 3)
    print("手   記録      公式打直   我々      我々−公式   その手の注文")
    for i in range(lo, min(first + 3, len(ours))):
        print("%3d %9.0f %9.0f %9.0f %+9.0f   %s" % (i, rec[i], off[i], ours[i], ours[i] - off[i], str(me[i]["market"])[:64]))
    print("\nズレる直前(%d手目)の公式の納屋: %s" % (first - 1, {k: v for k, v in off_shed[first - 1].items() if v}))
    print("ズレた時(%d手目)の公式の納屋  : %s" % (first, {k: v for k, v in off_shed[first].items() if v}))
    f0, f1 = off_farm[first - 1], off_farm[first]
    print("公式の人数 %d→%d / 区画 %s" % (len(f0.get("hands", [])), len(f1.get("hands", [])), f1.get("unlocked_quadrants")))
    print("\nそれまでに我々が弾いた手 %d 件:" % len([r for r in rej if r[0] < first]))
    print("  内訳", collections.Counter(r[2] for r in rej if r[0] < first).most_common(8))
    print("  直近10", [r for r in rej if r[0] < first][-10:])
