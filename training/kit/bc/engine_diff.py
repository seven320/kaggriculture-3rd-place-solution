"""exp310: 我々の kagsim と公式エンジンの食い違いを1手ずつ突き合わせる。
`151/266/290` クラスタ(redblackbst など)の棋譜が再現できない原因を特定するため。
使い方(容器内): REPLAY=<path> TEAM="redblackbst" python3 engine_diff.py"""
import json, os, sys, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack, extract3
from extract4 import state, ucand, q_class, codes, UQ_MAX, MQ_MAX, NU, NS, M_STOP, mcand, QNONE, BASE
from deploy import wide_blank

def rj(p): return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

d = rj(os.environ["REPLAY"]); team = os.environ["TEAM"]
tn = d["info"]["TeamNames"]; seat = tn.index(team)
st = d["steps"]
# 公式が記録した、各手の開始時点の状態(観測)
off_money = [s[seat]["observation"]["farms"][seat]["money"] for s in st]
off_hands = [len(s[seat]["observation"]["farms"][seat]["hands"]) for s in st]
off_hires = [s[seat]["observation"]["farms"][seat].get("hires_today", 0) for s in st]
off_quad = [len(s[seat]["observation"]["farms"][seat]["unlocked_quadrants"]) for s in st]

me = extract3.acts_of(d, seat); opp = kagpack.pack(extract3.acts_of(d, 1 - seat)); w = wide_blank()
A, Bt = (w, opp) if seat == 0 else (opp, w)
env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [int(d["info"]["seed"])], 720, seat); env.keep_nonsell = (os.environ.get("KEEPNS") == "1"); env.reset()
print("手   所持金(我々/公式/差)      人数(我々/公式)  その手の注文")
first = None
for i, a in enumerate(me):
    mine = float(env.money()[0][0])
    tiles, ou, pu, g, lm = state(env)
    nh = int((ou[:, 0] > 0).sum()) - 1          # 農夫を除いた人数
    dm = mine - off_money[i]; dh = nh - off_hands[i]
    if first is None and (abs(dm) > 0.5 or dh != 0):
        first = i
    if first is not None and i <= first + 4:
        nhire = sum(1 for o in a["market"] if o and o[0] == "HIRE")
        print("%3d  %8.0f/%8.0f %+7.0f   人 %2d/%2d   HIRE×%d  %s" % (
            i, mine, off_money[i], dm, nh, off_hands[i], nhire, str(a["market"])[:70]))
    live = [u for u in range(NU) if ou[u, 0] > 0]
    us = [a["farmer"]] + a["hands"]
    ul = np.full(NU, -1, np.int8); uq = np.zeros(NU, np.int16)
    for u in live:
        if u < len(us):
            c = ucand(us[u]); q = int(us[u][2]) if len(us[u]) > 2 else QNONE
            if not lm[u, c]: c, q = 0, QNONE
            ul[u] = c; uq[u] = q_class(q, UQ_MAX) if c >= 20 else 0
        else: ul[u] = 0
    ms = np.full(NS, M_STOP, np.int8); mq = np.zeros(NS, np.int16)
    for j, o in enumerate(a["market"][:10]):
        c2 = mcand(o); ms[j] = c2
        if c2 < 19: mq[j] = q_class(int(o[2]), MQ_MAX)
    U, O = codes(ul, uq, live, ms, mq); env.step_units(U[None], O[None])
    if first is not None and i > first + 4: break
print("最初に食い違った手:", first)
