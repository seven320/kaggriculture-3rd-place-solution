"""テープ <-> BC の出力空間 の可逆符号器。

作る順序を間違えないこと: **先に恒等式(復元してエンジンに通して残高一致)を通し**、
それから学習する。過去のBCが $0 だったのは、この恒等式を通さずに
「ユニット動詞28種」だけを出力空間にしていたため。
実測(200テープ・136万手)で欠けていたもの:
  ・PLACE の商品9種(10,468手が捨てられていた)
  ・PICKUP/PLACE/PLANT の**数量**(41,588手、数量の有無自体も情報)
  ・**市場行動の全部**(SELL 76,992 手 = 収入の全経路)
  ・hands は 0〜16人(nu_max=14 を超える)
  ・market の空要素
"""
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
# 引数なしの動詞
BARE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE",
        "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
# 品目を取る動詞(数量は別ヘッド。数量なしは QNONE で表す)
ITEMED = ["PLANT", "PICKUP", "PLACE"]
ITEMS = sorted(set(CROPS + ANIMALS + PRODUCTS))
UVERB = BARE + ITEMED
U_IX = {v: i for i, v in enumerate(UVERB)}
I_IX = {v: i for i, v in enumerate(ITEMS)}
QNONE = -1                       # 「数量の指定が無かった」を表す。0 とは違う
# 数量の上限は**実測で決めた**(エンジンに通して残高が一致する最小値):
#   ユニット 16 / 市場 64。ただし 999/1000 は「全部」の番兵なので**潰さない**。
#   上限で潰すと 40本中1本が不一致になった(実験30)。
UQ_MAX = 20   # ユニット数量>16 は PLACE の 17/18 のみ(999/1000 は番兵)。20 で厳密
# 納屋の上限は100(ルール)。市場数量を64で切ると 92/96/100 が潰れて残高がずれる。
MQ_MAX = 128
SENTINEL = (999, 1000)


def q_class(q, qmax):
    """数量 -> クラス。QNONE / 0..qmax / 番兵。"""
    if q == QNONE: return 0
    if q in SENTINEL: return 1 + qmax + 1 + SENTINEL.index(q)
    return 1 + min(int(q), qmax)


def q_value(c, qmax):
    if c == 0: return QNONE
    if c >= 1 + qmax + 1: return SENTINEL[c - (1 + qmax + 1)]
    return c - 1


UQ_N = 1 + UQ_MAX + 1 + len(SENTINEL)     # ユニット数量のクラス数
MQ_N = 1 + MQ_MAX + 1 + len(SENTINEL)     # 市場数量のクラス数

MOPS = ["SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "HIRE", "BUY_LAND", "PASS", "EMPTY"]
M_IX = {v: i for i, v in enumerate(MOPS)}
M_ITEMED = {"SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}


def enc_unit(u):
    """1ユニットの行動 -> (動詞, 品目, 数量)。未知は例外にする(黙って捨てない)。"""
    if not isinstance(u, list) or not u:
        return (U_IX["PASS"], -1, QNONE)
    op = u[0]
    if op in U_IX and op in ITEMED:
        it = I_IX[u[1]]
        q = int(u[2]) if len(u) > 2 else QNONE
        return (U_IX[op], it, q)
    if op in U_IX:
        return (U_IX[op], -1, QNONE)
    raise KeyError("未知のユニット行動: %r" % (u,))


def dec_unit(t):
    v, it, q = t
    name = UVERB[v]
    if name in ITEMED:
        a = [name, ITEMS[it]]
        if q != QNONE:
            a.append(int(q))
        return a
    return [name]


def enc_mkt(o):
    if not isinstance(o, list) or not o:
        return (M_IX["EMPTY"], -1, QNONE)
    op = o[0]
    if op not in M_IX:
        raise KeyError("未知の市場行動: %r" % (o,))
    if op in M_ITEMED:
        return (M_IX[op], I_IX[o[1]], int(o[2]) if len(o) > 2 else QNONE)
    return (M_IX[op], -1, QNONE)


def dec_mkt(t):
    v, it, q = t
    name = MOPS[v]
    if name == "EMPTY":
        return []
    if name in M_ITEMED:
        a = [name, ITEMS[it]]
        if q != QNONE:
            a.append(int(q))
        return a
    return [name]


def enc_step(st):
    return {"farmer": enc_unit(st.get("farmer")),
            "hands": [enc_unit(h) for h in (st.get("hands") or [])],
            "market": [enc_mkt(o) for o in (st.get("market") or [])]}


def dec_step(e):
    return {"farmer": dec_unit(e["farmer"]),
            "hands": [dec_unit(h) for h in e["hands"]],
            "market": [dec_mkt(o) for o in e["market"]]}
