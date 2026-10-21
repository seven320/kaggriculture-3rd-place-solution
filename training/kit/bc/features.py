"""BC の特徴量とラベル。学習と推論で**同じ関数**を使う(ポケモンでの教訓:
測定と出荷で特徴がずれると -2.7pt 相当の交絡が入る)。

行動空間の分解:
  結合行動(farmer 1 + hands 最大14 + market 順序付き10)をそのまま1つのヘッドで
  予測するのは不可能。**ユニット単位の動詞**に分解する。市場注文は順序に意味が
  あり(売りが買いより前に来ないと資金が足りず黙って失敗する)、独立ヘッドでは
  その結合を保てないので、市場はルールベースのまま残す。
"""
import math

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
        "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
SHOPS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
         "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
         "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
         "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
         "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}

# 動詞(ラベル)
VERBS = (["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE",
          "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE",
          "DROP"]
         + [f"PLANT_{c}" for c in CROPS]
         # PLACE は動物だけでなく**商品も置ける**(=出荷/施肥の実体)。
         # 動物3種しか無かったため 120テープで 10,468 手が None になり学習から捨てられていた。
         # 実測の内訳: FERTILIZER 3043 / MILK 1834 / STRAWBERRY 1719 / WOOL 1464 /
         #             MELON 1196 / WHEAT 651 / CARROT 278 / EGG 247 / TOMATO 36
         + [f"PLACE_{a}" for a in ANIMALS]
         + [f"PLACE_{c}" for c in PRODUCTS]
         + [f"PICKUP_{i}" for i in ["WHEAT", "FERTILIZER"] + ANIMALS])
VERB_IX = {v: i for i, v in enumerate(VERBS)}
NV = len(VERBS)
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


def label_of(act):
    """エージェントが出した1ユニットの行動 -> ラベル。対象外は None。"""
    if not isinstance(act, list) or not act:
        return VERB_IX["PASS"]
    op = act[0]
    if op in ("PLANT",) and len(act) > 1:
        return VERB_IX.get(f"PLANT_{act[1]}")
    if op == "PLACE" and len(act) > 1:
        return VERB_IX.get(f"PLACE_{act[1]}")
    if op == "PICKUP" and len(act) > 1:
        return VERB_IX.get(f"PICKUP_{act[1]}")
    return VERB_IX.get(op)


def _tile_feat(t):
    """タイル1枚を16次元で表す。"""
    f = [0.0] * 16
    if t is None:
        f[0] = 1.0
        return f
    if t == "LOCKED":
        f[1] = 1.0
        return f
    if not isinstance(t, dict):
        f[1] = 1.0
        return f
    k = t.get("kind")
    if k == "WEED":
        f[2] = 1.0
    elif k == "PLANT":
        f[3] = 1.0
        c = t.get("crop")
        if c in CROPS:
            f[4 + CROPS.index(c)] = 1.0
        f[9] = 1.0 if t.get("watered_today") else 0.0
        f[10] = min(t.get("yield_units", 0), 8) / 8.0
        f[11] = min(t.get("consecutive_unwatered", 0), 3) / 3.0
    else:
        f[12] = 1.0                                   # 構造物
        a = t.get("animal")
        if a in ANIMALS:
            f[13] = 1.0
            f[14] = min(t.get("yield_units", 0), 6) / 6.0
            f[15] = 1.0 if t.get("fed_today") else 0.0
    return f


def global_feat(obs, p):
    farm = obs["farms"][p]
    priv = obs["private"]
    mk = obs["market"]
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    drain = {it: (0.0 if it == "FERTILIZER" else 1.0) for it in PRODUCTS}
    for sh in shops:
        items = SHOPS.get(sh, [])
        m = 2 if len(items) == 1 else 1
        for it in items:
            drain[it] = drain.get(it, 0) + 6 * m
    g = [day / 29.0, hour / 23.0,
         math.log1p(max(farm["money"], 0)) / 12.0,
         len(farm.get("hands", [])) / 16.0,
         (len(farm.get("unlocked_quadrants", ["NW"])) - 1) / 3.0,
         min(farm.get("hires_today", 0), 20) / 20.0]
    for it in PRODUCTS:
        g.append(min(mk["prices"].get(it, 0), 600) / 600.0)
        g.append(max(-1.0, min(1.0, (mk["inventory"].get(it, 10000) - 10000) / 800.0)))
        g.append(drain[it] / 32.0)
        g.append(min(priv["shed"].get(it, 0), 60) / 60.0)
    for c in CROPS:
        g.append(min(priv["seeds"].get(c, 0), 20) / 20.0)
    for a in ANIMALS:
        g.append(min(priv["shed"].get(a, 0), 6) / 6.0)
    # 盤面の集計
    cnt = {"empty": 0, "weed": 0, "plant": 0, "struct": 0, "animal": 0, "locked": 0}
    for row in farm["tiles"]:
        for t in row:
            if t is None: cnt["empty"] += 1
            elif t == "LOCKED": cnt["locked"] += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "WEED": cnt["weed"] += 1
                elif k == "PLANT": cnt["plant"] += 1
                else:
                    cnt["struct"] += 1
                    if t.get("animal"): cnt["animal"] += 1
    for k in ("empty", "weed", "plant", "struct", "animal", "locked"):
        g.append(cnt[k] / 100.0)
    return g


def unit_feat(obs, p, ui, pos, inv):
    farm = obs["farms"][p]
    tiles = farm["tiles"]
    x, y = int(pos[0]), int(pos[1])
    u = [x / 9.0, y / 9.0,
         min(abs(x - 4) + abs(y - 4), 12) / 12.0,
         1.0 if (x, y) in SHED_TILES else 0.0,
         1.0 if ui == 0 else 0.0,
         min(sum(v for v in inv.values() if isinstance(v, (int, float))), 40) / 40.0]
    for it in ["WHEAT", "FERTILIZER"] + ANIMALS:
        u.append(min(inv.get(it, 0), 12) / 12.0)
    u += _tile_feat(tiles[y][x] if 0 <= y < 10 and 0 <= x < 10 else None)
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        nx, ny = x + dx, y + dy
        u += _tile_feat(tiles[ny][nx] if 0 <= ny < 10 and 0 <= nx < 10 else "LOCKED")
    return u


NG = None
NU = None


def dims(obs, p):
    global NG, NU
    if NG is None:
        NG = len(global_feat(obs, p))
        NU = len(unit_feat(obs, p, 0, obs["farms"][p]["farmer"], {}))
    return NG, NU
