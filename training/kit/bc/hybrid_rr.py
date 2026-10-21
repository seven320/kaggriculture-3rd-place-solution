"""09-22: 「継ぎ足し」エージェントの対戦。席 A は序盤を OPEN の方策(と任意で介入E2 の規則)で打ち、自分の区画が
SWITCH_Q 個になった手から MAIN の方策に切り替える。席 B は固定の方策。両方とも貪欲・種の上限あり(提出と同じ打ち方)。
各方策は自分のマスクの版で打つ(BatchSPPool.legal_opt を局と席ごとに、切り替えた局は MAIN の版に)。
  使い方: CONFIGS="名前;OPEN;OPEN版;MAIN;MAIN版;E2(0/1);相手;相手版|..." NSEED=64 python3 hybrid_rr.py 出力.json
  OPEN を "-" にすると最初から MAIN(E2=1 なら MAIN の序盤に規則だけかける)。
  出力: {名前: {"money": [[A, B], ...], "q2": [...], "q3": [...]}}(A の2区画目・3区画目を取った手番)"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
import sp_fast; S.SPPool = sp_fast.BatchSPPool
import fast_codes; P.to_codes = fast_codes.to_codes_fast
import pace_fast as PF, kagpack

out = sys.argv[1]; NSEED = int(os.environ.get("NSEED", "64")); SEED0 = int(os.environ.get("SEED0", "8800000"))
SWITCH_Q = int(os.environ.get("SWITCH_Q", "2")); E2_T0, E2_T1 = 96, 140
BUYLAND = kagpack.MOPI["BUY_LAND"]; BUYSEED = kagpack.MOPI["BUY_SEED"]; STRAW = kagpack.ITI["STRAWBERRY"]; NONE = 255
_nets = {}
def net(p):
    if p not in _nets: _nets[p] = M.load(p, P.DEV).eval()
    return _nets[p]
def quads(pool, s):
    e = pool.env; e.set_view(s); q = np.clip(np.rint(np.asarray(e.farm_pub(0))[:, 2] * 4), 0, 4); e.set_view(0); return q
def e2_rule(O, rows):
    """エンジンに渡す市場の注文(局, 10, 3)を書き換える: イチゴの種の購入を外し、区画の購入を最後に"""
    for b in rows:
        keep = [tuple(r) for r in O[b] if r[0] != NONE and r[0] != 254 and r[0] != BUYLAND and not (r[0] == BUYSEED and r[1] == STRAW)]
        keep = keep[:9] + [(BUYLAND, 0, 1)]
        O[b] = [keep[k] if k < len(keep) else (NONE, 0, 0) for k in range(10)]
def land_last(O, rows):
    """区画の購入を市場の列の最後に置く(同じ手番の売上で買えるように)。他の注文はそのまま"""
    for b in rows:
        keep = [tuple(r) for r in O[b] if r[0] != NONE and r[0] != BUYLAND]
        keep = keep[:9] + [(BUYLAND, 0, 1)]
        O[b] = [keep[k] if k < len(keep) else (NONE, 0, 0) for k in range(10)]
def play(cfg, seeds, sA):
    name, op, opv, mn, mnv, e2, bb, bbv = cfg
    pool = S.SPPool(seeds); n = pool.n; sB = 1 - sA
    pool.legal_opt = np.full((n, 2), -1, np.int32); pool.legal_opt[:, sB] = int(bbv)
    sw = np.zeros(n, bool) if op != "-" else np.ones(n, bool)
    pA_open = S.PolicyOpp(net(op), greedy=True, cap=True) if op != "-" else None
    pA_main = S.PolicyOpp(net(mn), greedy=True, cap=True); pB = S.PolicyOpp(net(bb), greedy=True, cap=True); rB = os.environ.get("RULE_B", "")   # 相手にも規則(L4)をかけるとき
    q2 = np.full(n, 719); q3 = np.full(n, 719)
    for t in range(719):
        qa = quads(pool, sA)
        q2 = np.where((q2 == 719) & (qa >= 2), t, q2); q3 = np.where((q3 == 719) & (qa >= 3), t, q3)
        sw = sw | (qa >= SWITCH_Q)
        pool.legal_opt[:, sA] = np.where(sw, int(mnv), int(opv) if op != "-" else int(mnv))
        acts = {}
        stA = pool.state(sA); U = np.zeros((n, S.NU, 3), np.int32); O = np.zeros((n, 10, 3), np.int32)
        for flag, pol in ((False, pA_open), (True, pA_main)):
            idx = np.nonzero(sw == flag)[0]
            if pol is None or len(idx) == 0: continue
            u, o = pol.act(tuple(x[idx] for x in stA), t, list(idx), pool, sA); U[idx], O[idx] = u, o
        # 規則: 1 = 2区画目の介入E2(区画1つ・手96〜139、イチゴの種を外し区画の購入を最後に)
        #       L3 / L4 / L34 = 区画が2つ(L3)・3つ(L4)・両方(L34)のあいだ、毎手 区画の購入を市場の列の最後に置く(買えるようになったらすぐ買う)
        if e2 == "1" and E2_T0 <= t < E2_T1: e2_rule(O, np.nonzero(qa == 1)[0])
        if e2 in ("L3", "L4", "L34"):
            want = [2] if e2 == "L3" else ([3] if e2 == "L4" else [2, 3])
            land_last(O, np.nonzero(np.isin(qa, want))[0])
        if e2 == "L4t290" and t >= 290: land_last(O, np.nonzero(qa == 3)[0])   # 学習中の規則(apply_land、LAND_T=290)と同じ手番から
        acts[sA] = (U, O)
        stB = pool.state(sB); uB, oB = pB.act(stB, t, list(range(n)), pool, sB)
        if rB == "L4": land_last(oB, np.nonzero(quads(pool, sB) == 3)[0])
        acts[sB] = (uB, oB)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    m = pool.money()
    return [[float(m[i, sA]), float(m[i, sB])] for i in range(n)], q2.tolist(), q3.tolist()
R = json.load(open(out)) if os.path.exists(out) else {}
for k, c in enumerate([x.split(";") for x in os.environ["CONFIGS"].split("|") if x]):
    if c[0] in R: continue
    t0 = time.time(); seeds = [SEED0 + 7919 * j for j in range(NSEED)]; rows = []; q2 = []; q3 = []
    with torch.no_grad():
        for sA in (0, 1):
            r_, a_, b_ = play(c, seeds, sA); rows += r_; q2 += a_; q3 += b_
    v = np.array(rows); w = float((v[:, 0] > v[:, 1]).mean() + 0.5 * (v[:, 0] == v[:, 1]).mean())
    R[c[0]] = dict(money=rows, q2=q2, q3=q3, cfg=c); json.dump(R, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    print("%s  %-22s %d局 勝率 %.3f 差 %+.0f  所持金 A %.0f / B %.0f  A の2区画目 中央値 %d(128以下 %.2f) 3区画目 中央値 %d  %.0f秒" % (
        time.strftime("%H:%M"), c[0], len(v), w, (v[:, 0] - v[:, 1]).mean(), v[:, 0].mean(), v[:, 1].mean(), int(np.median(q2)), float((np.array(q2) <= 128).mean()), int(np.median(q3)), time.time() - t0), flush=True)
print("== 済 ==", flush=True)
