"""公式の観測(辞書)だけから、学習時と同じ特徴と合法手を作る(提出用。本番環境にはエンジンが無い)。

エンジン側(tiles_base/tiles_dyn/unit_pos2/unit_inv/obs_legal/farm_pub/shops/shed_all/legal_mask)と
全手で一致することを検証してから使う。エンジン内部の値のうち公式観測に無いものは、
逃走・収穫・掘る・雑草化のたびにタイルが初期値に作り直されるので常に初期値(sim.hpp の Tile s{}/e{}/w{})。
合法手は sim.hpp の apply_unit の前提条件をそのまま移したもの。
"""
import os
import numpy as np
IT = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"]
CROP = IT[:5]; FIRST = [2, 2, 8, 10, 10]
KIND = {"EMPTY":0, "LOCKED":1, "WEED":2, "COOP":3, "PASTURE":4, "PLANT":5}
SHOPS = ["BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"]
NU, NC = 20, 44
SHED_ADJ = {(4, 4), (5, 4), (4, 5), (5, 5)}


def _tile(t, day, step):
    """-> (kind, what, watered, fed, yld, fav, has, cared, dry, pend, planted, mls, fud)"""
    if t is None: return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, -1, -1)
    if t == "LOCKED": return (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, -1, -1)
    k = KIND[t["kind"]]
    if k == 2: return (2, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, -1, -1)
    if k == 5:
        return (5, CROP.index(t["crop"]), int(t["watered_today"]), 0, t["yield_units"], 0, 0, 0,
                t["consecutive_unwatered"], 0, t["planted_day"], t["max_lifespan_step"], t["fertilized_until_day"])
    a = t.get("animal")
    if a is None: return (k, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, -1, -1)
    return (k, IT.index(a), 0, int(t["fed_today"]), t["yield_units"], int(t["fertilizer_available"]), 1, int(t["cared_today"]),
            t["consecutive_unfed"], t["pending_care_bonus"], t["placed_day"], -1, -1)


def _tile_feats(farm, day, step):
    out = np.zeros((100, 13), np.float32); raw = []
    for y in range(10):
        for x in range(10):
            k, w, wat, fed, yl, fav, has, car, dry, pend, pl, mls, fud = r = _tile(farm["tiles"][y][x], day, step)
            raw.append(r)
            out[y*10+x] = [k/5, w/12, wat, fed, yl/6, fav, has, car, dry/4, pend/4,
                           ((day - pl)/30 if (k == 5 or has) else -1),
                           ((mls - step)/720 if mls >= 0 else -1),
                           ((fud - day)/30 if fud >= 0 else -1)]
    return out, raw


def featurize(obs):
    me = obs["player"]; op = 1 - me; day = obs["day"]; hour = obs["hour"]; step = day * 24 + hour
    F = obs["farms"]; P = obs["private"]
    t0, raw0 = _tile_feats(F[me], day, step); t1, _ = _tile_feats(F[op], day, step)
    tiles = np.stack([t0, t1]).astype(np.float16)
    def pos(f):
        ps = [tuple(f["farmer"])] + [tuple(h) for h in f["hands"]]
        a = np.full((NU, 2), -1, np.float32)
        for i, p in enumerate(ps[:NU]): a[i] = p
        return a, len(ps)
    up, nme = pos(F[me]); upo, nop = pos(F[op])
    inv = np.zeros((NU, 12), np.float32)
    for i, d in enumerate(P["inventories"][:NU]):
        for k, v in d.items(): inv[i, IT.index(k)] = v
    ou = np.concatenate([(up[:, :1] >= 0), up, inv], 1).astype(np.float32)
    pu = np.concatenate([(upo[:, :1] >= 0), upo], 1).astype(np.float32)
    mk = obs["market"]; shed = P["shed"]; seeds = P["seeds"]
    PR = IT[:9]
    ol = ([mk["inventory"][p] / 10000 for p in PR] + [mk["prices"][p] / 250 for p in PR] +
          [shed.get(p, 0) / 100 for p in PR] + [seeds.get(c, 0) / 20 for c in CROP] +
          [F[me]["money"] / 50000, F[op]["money"] / 50000, step / 720, hour / 24, day / 30])
    shed_total = sum(shed.values())
    fp0 = [F[me]["money"] / 50000, nme / 13, len(F[me]["unlocked_quadrants"]) / 4, F[me]["hires_today"] / 13, shed_total / 100]
    fp1 = [F[op]["money"] / 50000, nop / 13, len(F[op]["unlocked_quadrants"]) / 4, F[op]["hires_today"] / 13, -1]
    sh = [SHOPS.index(s) for s in obs["town"]["unlocked_shops"]][:16]; sh += [-1] * (16 - len(sh))
    sa = [shed.get(k, 0) for k in IT]
    g = np.asarray(ol + fp0 + fp1 + sh + sa, np.float32)
    legal = legal_mask(raw0, up, nme, inv, shed, shed_total, seeds, day)
    return tiles, ou, pu, g, legal


# 09-21(exp313): 合法判定の版。0=従来(手番開始のスナップショットだけで判定) 2=手番内に変わりうる種を常に許す。
#   公式エンジンは _apply_unit_action をユニットごとに**逐次**適用し、条件を満たさない手は no-op になるだけなので、
#   同じ手番で前のユニットが作った状態に依存する手(受け渡し)を先に非合法と決めつけるのは誤り。
#   kagg/fast/kagsim.cpp の LEGAL_OPT と同じ意味。**学習と提出で必ず同じ値にすること。**
LEGAL_OPT = int(os.environ.get("LEGAL_OPT", "0"))
_OPT_ALWAYS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(20, 44))   # 水やり〜全部置く、取り出す、置く


def legal_mask(raw, up, n, inv, shed, shed_total, seeds, day):
    """sim.hpp apply_unit の前提条件の移植。0待機 1-4移動 5水やり 6収穫 7施肥 8掘る 9小屋 10牧草地 11給餌
    12肥料採取 13世話 14全部置く 15-19植える 20-31取り出す 32-43置く"""
    L = np.zeros((NU, NC), np.uint8)
    if LEGAL_OPT >= 2:
        for u in range(min(n, NU)):
            x, y = int(up[u, 0]), int(up[u, 1]); L[u, 0] = 1
            L[u, 1] = y - 1 >= 0; L[u, 2] = y + 1 <= 9; L[u, 3] = x + 1 <= 9; L[u, 4] = x - 1 >= 0
            if LEGAL_OPT >= 3: L[u, 5:] = 1; continue
            for c in _OPT_ALWAYS: L[u, c] = 1
            for c in range(5):
                if seeds.get(CROP[c], 0) > 0: L[u, 15 + c] = 1
        return L
    for u in range(min(n, NU)):
        x, y = int(up[u, 0]), int(up[u, 1]); L[u, 0] = 1
        L[u, 1] = y - 1 >= 0; L[u, 2] = y + 1 <= 9; L[u, 3] = x + 1 <= 9; L[u, 4] = x - 1 >= 0
        k, w, wat, fed, yl, fav, has, car, dry, pend, pl, mls, fud = raw[y*10+x]
        adj = (x, y) in SHED_ADJ; iv = inv[u]
        if adj and iv.sum() > 0: L[u, 14] = 1
        for i in range(12):
            if adj and shed.get(IT[i], 0) > 0: L[u, 20+i] = 1
            if i >= 9:
                want = 3 if i == 9 else 4
                if k == want and not has:
                    L[u, 32+i] = int(iv[i] >= 1); continue          # 構造物の上では納屋へは置かない(エンジン通り)
            if adj and iv[i] > 0 and shed_total < 100: L[u, 32+i] = 1
        if k == 1: continue                                          # ロックされたマスでは以下は無効
        if k == 0:
            for c in range(5):
                if seeds.get(CROP[c], 0) > 0: L[u, 15+c] = 1
            L[u, 9] = 1; L[u, 10] = 1
        if k == 5 and not wat: L[u, 5] = 1
        if k not in (0, 2) and yl > 0 and ((k == 5 and day - pl >= FIRST[w]) or (k != 5 and has)): L[u, 6] = 1
        if k == 5 and iv[8] >= 1: L[u, 7] = 1
        if k != 0 and not has: L[u, 8] = 1
        if has and not fed and iv[0] >= 1: L[u, 11] = 1
        if has and fav: L[u, 12] = 1
        if has and not car: L[u, 13] = 1
    return L


# ---------------------------------------------------------------- 出力 -> 公式の行動の辞書
BASE = ["PASS","NORTH","SOUTH","EAST","WEST","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","DROP"]
M_EMPTY, M_STOP = 21, 22


def _q(qc, qmax):
    """数量クラス -> 数(codec.q_class の逆)。0=指定なし, 1..qmax+1 = 0..qmax, 以降は番兵 999/1000"""
    qc = int(qc)
    if qc == 0: return None
    if qc <= qmax + 1: return qc - 1
    return (999, 1000)[min(qc - (qmax + 2), 1)]


def to_action(ul, uq, ms, mq, n_units, UQ_MAX=20, MQ_MAX=128):
    """候補番号 -> {"farmer":[...], "hands":[[...],...], "market":[[...],...]}。
    市場の空の注文は [] で枠を占める(process_market は枠の添字で両者を突き合わせる)。"""
    acts = []
    for u in range(n_units):
        c = int(ul[u]) if ul[u] >= 0 else 0
        if c < 15: acts.append([BASE[c]])
        elif c < 20: acts.append(["PLANT", CROP[c - 15]])
        else:
            q = _q(uq[u], UQ_MAX); q = 1 if q is None else q
            acts.append(["PICKUP" if c < 32 else "PLACE", IT[(c - 20) % 12], int(q)])
    market = []
    for j in range(10):
        c = int(ms[j])
        if c == M_STOP: break
        if c == M_EMPTY: market.append([]); continue
        if c == 19: market.append(["HIRE"]); continue
        if c == 20: market.append(["BUY_LAND"]); continue
        q = _q(mq[j], MQ_MAX); q = 1 if q is None else q
        if c < 9:
            if q > MQ_MAX and q not in (999, 1000): q = 9999
            market.append(["SELL", IT[c], int(q)])
        else:
            q = min(q, MQ_MAX) if q not in (999, 1000) else q
            if c < 14: market.append(["BUY_SEED", CROP[c - 9], int(q)])
            elif c < 16: market.append(["BUY_PRODUCT", "WHEAT" if c == 14 else "FERTILIZER", int(q)])
            else: market.append(["BUY_ANIMAL", IT[9 + c - 16], int(q)])
    return {"farmer": acts[0] if acts else ["PASS"], "hands": acts[1:], "market": market}


# ---- 09-24 nmb(tools/mkagent_nmb.py が足した): 旧マスクと新マスクを両方作り、印を付けて返す ----
#   0 = 非合法 / 1 = 旧マスク(版0)でも合法 / 3 = 新マスク(版2)でだけ増えた手。sp_fast.BatchSPPool._lm(MASK_MIG=1 MIG_V=2 MIG_BASE=0)と同じ。
#   LEGAL_OPT は版を切り替えるためだけに使い、呼び出しの後で元に戻す。
NMB_MARK = True
_legal_mask_one = legal_mask


def legal_mask(raw, up, n, inv, shed, shed_total, seeds, day):
    global LEGAL_OPT
    o = LEGAL_OPT
    try:
        LEGAL_OPT = 0; L0 = _legal_mask_one(raw, up, n, inv, shed, shed_total, seeds, day)
        LEGAL_OPT = 2; L2 = _legal_mask_one(raw, up, n, inv, shed, shed_total, seeds, day)
    finally:
        LEGAL_OPT = o
    xp = (L2 != 0) & (L0 == 0)
    return ((L0 != 0) | xp).astype(np.uint8) | (xp.astype(np.uint8) * np.uint8(2))
