"""transformer 用の抽出。状態は実体(タイル・ユニット・店・品目)ごとの生の値、行動は合法手の候補番号。

教訓の反映:
  ・合法手はエンジン自身の apply_unit で判定(kagsim.legal_mask)。教師の手の99.95%が入り、残りは無効な手
  ・市場は候補23種の順序つき列＋終了。空の注文も枠を占める(process_market は枠の添字で両者を突き合わせる)
  ・抽出の軌道は**実戦と同じ復号経路**で進め、最終残高が公式と一致しない対局は捨てる(抽出と恒等式を一体化)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
from codec import QNONE, UQ_MAX, MQ_MAX, q_class, q_value
from extract3 import acts_of
from deploy import wide_blank
import kagsim, kagpack
NU = 20; NC = 44; NM = 23; NS = 11
IT = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"]
BASE = ["PASS","NORTH","SOUTH","EAST","WEST","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","DROP"]
M_EMPTY, M_STOP = 21, 22
ALL_Q = 9999
MASKSUB = os.environ.get("MASKSUB") == "1"   # 09-21: 1 で従来どおり「非合法な手は待機に置換」

def ucand(a):
    op = a[0] if a else "PASS"
    if op in BASE: return BASE.index(op)
    if op == "PLANT": return 15 + IT.index(a[1])
    if op == "PICKUP": return 20 + IT.index(a[1])
    if op == "PLACE": return 32 + IT.index(a[1])
    return 0

def mcand(o):
    if not o or o[0] == "PASS": return M_EMPTY
    op = o[0]
    if op == "HIRE": return 19
    if op == "BUY_LAND": return 20
    if len(o) < 3 or o[1] not in IT: return M_EMPTY          # 数量なしの品目注文は枠だけ占める無効な注文
    it = IT.index(o[1])
    if op == "SELL" and it < 9: return it
    if op == "BUY_SEED" and it < 5: return 9 + it
    if op == "BUY_PRODUCT" and it in (0, 8): return 14 if it == 0 else 15
    if op == "BUY_ANIMAL" and it >= 9: return 16 + (it - 9)
    return M_EMPTY

UOP = [kagpack.OPI[x] for x in BASE]
def unit_code(c, qc):
    q = q_value(int(qc), UQ_MAX); q = 1 if q == QNONE else q
    if c < 15: return (UOP[c], 0, 1)
    if c < 20: return (kagpack.OPI["PLANT"], c - 15, 1)
    if c < 32: return (kagpack.OPI["PICKUP"], c - 20, q)
    return (kagpack.OPI["PLACE"], c - 32, q)

def mkt_code(c, qc):
    if c == M_EMPTY: return (254, 0, 0)
    if c == 19: return (kagpack.MOPI["HIRE"], 0, 1)
    if c == 20: return (kagpack.MOPI["BUY_LAND"], 0, 1)
    q = q_value(int(qc), MQ_MAX); q = 1 if q == QNONE else q
    if c < 9:
        if q > MQ_MAX and q not in (999, 1000): q = ALL_Q          # 売りは保有量で止まる
        return (kagpack.MOPI["SELL"], c, q)
    q = min(q, MQ_MAX) if q not in (999, 1000) else q               # 買いは所持金が尽きるまで続くので「全部」にしない
    if c < 14: return (kagpack.MOPI["BUY_SEED"], c - 9, q)
    if c < 16: return (kagpack.MOPI["BUY_PRODUCT"], 0 if c == 14 else 8, q)
    return (kagpack.MOPI["BUY_ANIMAL"], 9 + (c - 16), q)

def codes(ul, uq, live, ms, mq):
    """候補番号 -> step_units の配列。生きていないユニットは 255(空テープの待機のまま)。"""
    U = np.full((NU, 3), 0, np.int32); U[:, 0] = 255
    for u in live: U[u] = unit_code(ul[u], uq[u])
    O = np.zeros((10, 3), np.int32); O[:, 0] = 255
    for j in range(10):
        c = int(ms[j])
        if c == M_STOP: break
        O[j] = mkt_code(c, mq[j])
    return U, O

def state(env):
    tiles = np.stack([np.concatenate([np.asarray(env.tiles_base(s))[0], np.asarray(env.tiles_dyn(s))[0]], 2).reshape(100, 13) for s in (0, 1)])
    up = np.asarray(env.unit_pos2(0))[0][:NU]; op = np.asarray(env.unit_pos2(1))[0][:NU]; inv = np.asarray(env.unit_inv(NU))[0]
    ou = np.concatenate([(up[:, :1] >= 0), up, inv], 1).astype(np.float32)
    pu = np.concatenate([(op[:, :1] >= 0), op], 1).astype(np.float32)
    g = np.concatenate([np.asarray(env.obs_legal())[0], np.asarray(env.farm_pub(0))[0], np.asarray(env.farm_pub(1))[0],
                        np.asarray(env.shops())[0].astype(np.float32), np.asarray(env.shed_all())[0]]).astype(np.float32)
    return tiles.astype(np.float16), ou, pu, g, np.asarray(env.legal_mask(NU))[0]

def extract_game(d, seat):
    me = acts_of(d, seat); opp = kagpack.pack(acts_of(d, 1 - seat)); w = wide_blank(NU - 1 + 1 if False else 720)
    w = wide_blank(); A, Bt = (w, opp) if seat == 0 else (opp, w)
    env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [int(d["info"]["seed"])], 720, seat); env.keep_nonsell = False; env.reset()
    T, OU, PU, G, L, UL, UQ, MS, MQ = [], [], [], [], [], [], [], [], []
    noops = 0
    for i, a in enumerate(me):
        tiles, ou, pu, g, lm = state(env)
        live = [u for u in range(NU) if ou[u, 0] > 0]
        us = [a["farmer"]] + a["hands"]
        ul = np.full(NU, -1, np.int8); uq = np.zeros(NU, np.int16)
        for u in live:
            if u < len(us):
                c = ucand(us[u]); q = int(us[u][2]) if len(us[u]) > 2 else QNONE
                # 09-21: ここで待機に置き換えていたのが、上位6チームの棋譜を再現できない唯一の原因だった。
                #   legal_mask は「手番開始時のスナップショット1つ」から全ユニット分を計算するので、
                #   同じ手番で前のユニットが DROP/PLACE/HARVEST した結果に依存する手(受け渡し)を非合法と誤判定する。
                #   公式エンジンには非合法という概念が無く、条件を満たさない手は no-op になるだけ。
                #   我々の実行(step_units)は逐次で公式と一致しているので、そのまま渡すのが正しい。
                #   MASKSUB=1 で従来の挙動に戻せる。(検証: 9チーム×6局が 0/6〜6/6 → 全て 6/6 誤差0)
                if not lm[u, c]:
                    noops += 1
                    if MASKSUB: c, q = 0, QNONE
                ul[u] = c; uq[u] = q_class(q, UQ_MAX) if c >= 20 else 0
            else: ul[u] = 0
        ms = np.full(NS, M_STOP, np.int8); mq = np.zeros(NS, np.int16)
        for j, o in enumerate(a["market"][:10]):
            c = mcand(o); ms[j] = c
            if c < 19: mq[j] = q_class(int(o[2]), MQ_MAX)
        T.append(tiles); OU.append(ou); PU.append(pu); G.append(g); L.append(lm); UL.append(ul); UQ.append(uq); MS.append(ms); MQ.append(mq)
        U, O = codes(ul, uq, live, ms, mq)
        env.step_units(U[None], O[None])
    m = env.money()[0]
    X = dict(tiles=np.asarray(T), ou=np.asarray(OU), pu=np.asarray(PU), g=np.asarray(G), legal=np.asarray(L),
             ul=np.asarray(UL), uq=np.asarray(UQ), ms=np.asarray(MS), mq=np.asarray(MQ))
    return X, (float(m[0]), float(m[1])), noops
