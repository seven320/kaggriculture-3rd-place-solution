"""W9 + WT の後付け後処理(単独版、Python 標準ライブラリのみ、2026-09-29)。

  W9  hour 23(日の最後の手): unit の行動を仮適用した後の「倉庫 − 既存 SELL + 全 unit の携行品」が倉庫容量(100)を超える分を、
      日末の自動搬入で捨てられる品目から(倉庫に無ければ安い品目から)追加 SELL にする。注文枠(10)を超える分は足さない。
  WT  最終手(step 718): 既存注文が全て正しい形の SELL で、仮適用後の倉庫の売り残し全品目が注文枠に収まる時だけ、売り残しを追加 SELL。

使い方: action = postprocess_w9t.apply(obs, configuration, action)
unit の行動は変えない。market 注文の後ろに SELL を足すだけ。例外時は元の action を返す。
挙動は waste_replace2.replace_waste(obs, configuration, action, [], flags=('W9', 'WT')) の行動出力と同じ
(同等性試験: test_postprocess_w9t.py、結果は validation.json)。

--------------------------------------------------------------------------------------------------------
出典(同梱コードの一部): 下の「公式ルールの抜き出し」区間の定数・関数(CROPS, ANIMALS, PRODUCTS, FARMER_MOVES, _shed_access_tiles, _is_shed_adjacent, _new_plant, _new_animal, _farmer_position, _set_farmer_position, _farmer_inventory, _inv_add, _inv_take, _apply_unit_action)は、
  kaggle-environments 1.32.7 の kaggle_environments/envs/kaggriculture/kaggriculture.py
  (sha256 bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e)
から、定義ごとに文字列を変更せずに抜き出したもの(ファイル内の順。他の定義は含まない)。
kaggle-environments は Kaggle による Apache License 2.0 のソフトウェア(https://github.com/Kaggle/kaggle-environments)。
  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
  the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
ライセンス全文は同じフォルダの LICENSE_kaggle_environments_Apache-2.0.txt。変更点: 抜き出し(一部の定義のみ)で、抜き出した定義の本文は無変更。
--------------------------------------------------------------------------------------------------------
"""
from collections import Counter
from copy import deepcopy

__all__ = ['apply']

# ---- 公式ルールの抜き出し ここから(kaggriculture.py sha256 bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e) ----

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}


ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}


PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}


def _shed_access_tiles(board_size):
    """Four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _is_shed_adjacent(pos, board_size):
    return tuple(pos) in {(x, y) for (x, y) in _shed_access_tiles(board_size)}


def _new_plant(crop, day, turns_per_day):
    cd = CROPS[crop]
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "watered_today": False,
        "consecutive_unwatered": 1,  # planting day counts as unwatered
        "yield_units": 0 if cd["ongoing"] else 1,
        "max_lifespan_step": (-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day),
        "fertilized_until_day": -1,
    }


def _new_animal(animal, day):
    a = ANIMALS[animal]
    return {
        "kind": a["structure"],
        "animal": animal,
        "placed_day": day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def _farmer_position(farm, idx):
    """idx 0 = main farmer, 1+ = hand index."""
    if idx == 0:
        return farm["farmer"]
    return farm["hands"][idx - 1] if idx - 1 < len(farm["hands"]) else None


def _set_farmer_position(farm, idx, pos):
    if idx == 0:
        farm["farmer"] = list(pos)
    else:
        farm["hands"][idx - 1] = list(pos)


def _farmer_inventory(private, idx):
    """Inventories list is [main_farmer, *hands]; grow it if idx is past the end."""
    while len(private["inventories"]) <= idx:
        private["inventories"].append({})
    return private["inventories"][idx]


def _inv_add(inv, item, n=1):
    inv[item] = inv.get(item, 0) + n


def _inv_take(inv, item, n=1):
    if inv.get(item, 0) < n:
        return False
    inv[item] -= n
    if inv[item] == 0:
        del inv[item]
    return True


def _apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
    """Process one farmer/hand's action. Invalid / illegal actions are silent no-ops."""
    if not isinstance(action, list) or not action:
        return
    op = action[0]
    pos = _farmer_position(farm, idx)
    if pos is None:
        return
    fx, fy = pos[0], pos[1]
    inv = _farmer_inventory(private, idx)

    if op in FARMER_MOVES:
        dx, dy = FARMER_MOVES[op]
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < board_size and 0 <= ny < board_size):
            return
        # Movement onto LOCKED tiles is allowed: a hand can spawn on a locked
        # shed-access tile, and blocking movement would strand it there forever.
        # Tile operations (PLANT, WATER, etc.) still no-op on LOCKED tiles.
        _set_farmer_position(farm, idx, (nx, ny))
        return

    if op == "PASS":
        return

    tile = farm["tiles"][fy][fx]

    # Shed operations resolve before the LOCKED guard. They use the tile only as
    # a standing position -- the shed itself is always owned -- and three of the
    # four shed-access tiles start LOCKED, so guarding them first would make the
    # shed unreachable from those tiles.
    if op == "DROP":
        if not _is_shed_adjacent((fx, fy), board_size):
            return
        shed = private["shed"]
        for item, n in list(inv.items()):
            if n <= 0:
                del inv[item]
                continue
            room = max(0, shed_capacity - sum(shed.values()))
            take = min(n, room)
            if take > 0:
                shed[item] = shed.get(item, 0) + take
            del inv[item]
        return

    if op == "PICKUP":
        if not _is_shed_adjacent((fx, fy), board_size):
            return
        if len(action) < 2:
            return
        item = action[1]
        n = int(action[2]) if len(action) >= 3 else 1
        if n <= 0:
            return
        # Seeds live in private["seeds"] and are consumed directly by PLANT;
        # they never pass through farmer inventory or the shed.
        available = private["shed"].get(item, 0)
        n = min(n, available)
        if n <= 0:
            return
        private["shed"][item] -= n
        _inv_add(inv, item, n)
        return

    if op == "PLACE":
        if len(action) < 2:
            return
        item = action[1]
        # Animal placement: standing on a matching unoccupied structure. A LOCKED
        # tile is the string "LOCKED", never a dict, so this branch cannot match
        # there and PLACE falls through to the shed path below.
        if (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            if _inv_take(inv, item, 1):
                farm["tiles"][fy][fx] = _new_animal(item, day)
            return
        # Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.
        if _is_shed_adjacent((fx, fy), board_size):
            n = int(action[2]) if len(action) >= 3 else 1
            if n <= 0:
                return
            n = min(n, inv.get(item, 0))
            if n <= 0:
                return
            current = sum(private["shed"].values())
            room = max(0, shed_capacity - current)
            n = min(n, room)
            if n <= 0:
                return
            inv[item] -= n
            if inv[item] == 0:
                del inv[item]
            private["shed"][item] = private["shed"].get(item, 0) + n
        return

    # Everything below mutates the tile the unit stands on, so it requires that
    # tile to be owned.
    if tile == "LOCKED":
        return

    if op == "PLANT":
        if len(action) < 2:
            return
        crop = action[1]
        if crop not in CROPS:
            return
        if tile is not None:
            return
        if private["seeds"].get(crop, 0) <= 0:
            return
        private["seeds"][crop] -= 1
        farm["tiles"][fy][fx] = _new_plant(crop, day, turns_per_day)
        return

    if op == "WATER":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if tile["watered_today"]:
            return
        tile["watered_today"] = True
        crop_data = CROPS[tile["crop"]]
        if not crop_data["ongoing"]:
            age_days = day - tile["planted_day"]
            window_start = (crop_data["max_yield_day"] + 1) // 2
            if window_start <= age_days <= crop_data["max_yield_day"]:
                bonus = 2 if tile["fertilized_until_day"] >= day else 1
                tile["yield_units"] = min(crop_data["max_yield"], tile["yield_units"] + bonus)
        return

    if op == "HARVEST":
        if not isinstance(tile, dict):
            return
        if tile.get("yield_units", 0) <= 0:
            return
        if tile.get("kind") == "PLANT":
            crop_data = CROPS[tile["crop"]]
            if day - tile["planted_day"] < crop_data["first_yield_day"]:
                # Ongoing crops only accumulate yield_units after first_yield_day,
                # so reaching here with yield_units > 0 indicates a bug.
                if crop_data["ongoing"]:
                    print(
                        f"WARNING: HARVEST on immature ongoing {tile['crop']} "
                        f"(planted day {tile['planted_day']}, current day {day}, "
                        f"first_yield_day {crop_data['first_yield_day']}, "
                        f"yield_units {tile['yield_units']}); should never happen"
                    )
                return
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, tile["crop"], units)
            if not crop_data["ongoing"]:
                farm["tiles"][fy][fx] = None
        elif "animal" in tile:
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, ANIMALS[tile["animal"]]["product"], units)
        return

    if op == "FERTILIZE":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if not _inv_take(inv, "FERTILIZER", 1):
            return
        # Active for `day`, `day+1`, `day+2` (3 days inclusive).
        tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)
        return

    if op == "DIG":
        if tile is None:
            return
        # Removes plants, weeds, empty coop/pasture. Does NOT remove a placed animal.
        if isinstance(tile, dict) and "animal" in tile:
            return
        farm["tiles"][fy][fx] = None
        return

    if op == "BUILD_COOP":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "COOP"}
        return

    if op == "BUILD_PASTURE":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "PASTURE"}
        return

    if op == "FEED":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["fed_today"]:
            return
        if not _inv_take(inv, "WHEAT", 1):
            return
        tile["fed_today"] = True
        return

    if op == "COLLECT_FERTILIZER":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if not tile["fertilizer_available"]:
            return
        tile["fertilizer_available"] = False
        _inv_add(inv, "FERTILIZER", 1)
        return

    if op == "CARE":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["cared_today"]:
            return
        tile["cared_today"] = True
        return

# ---- 公式ルールの抜き出し ここまで ----

# =====================================================================================================
# ここから下は当チームのコード(W9 + WT の後処理)。waste_replace2.replace_waste(obs, configuration, action, [],
# flags=('W9', 'WT')) と同じ出力を返すように、同関数の該当部分をそのまま移したもの。
# =====================================================================================================

FLAGS = ('W9', 'WT')


class _Ctx:
    def __init__(self, obs, cfg):
        get = lambda k, d: int(cfg.get(k, d) if hasattr(cfg, 'get') else getattr(cfg, k, d))
        self.board, self.tpd, self.cap, steps = get('boardSize', 10), get('turnsPerDay', 24), get('shedCapacity', 100), get('episodeSteps', 720)
        self.step = int(obs['step']); self.day, self.hour = self.step // self.tpd, self.step % self.tpd
        self.last_step = steps - 2                          # 最後に手を打つ step(718)
        self.last_day = self.last_step // self.tpd          # 29
        self.last_eod = (steps - 1) // self.tpd - 1         # 最後に日末処理が来る日(28)
        self.shed = [tuple(t) for t in _shed_access_tiles(self.board)]
        self.max_orders = get('maxMarketOrdersPerTurn', 10)

    def apply(self, farm, private, u, a):
        _apply_unit_action(farm, private, u, list(a), self.board, self.day, self.tpd, self.cap)


def _w9_orders(c, obs, private, market):
    """hour 23: unit 処理後の倉庫 − 既存 SELL + 携行品が容量を超える分を、日末に捨てられる品目(公式 _drop_inventories_to_shed の順)
    から、倉庫に無ければ安い品目から、追加 SELL にする。注文枠を超える分は足さない。"""
    if c.hour != c.tpd - 1:
        return []
    room_orders = c.max_orders - len(market)
    if room_orders <= 0:
        return []
    shed = {k: n for k, n in private['shed'].items() if n > 0}
    for o in market:
        if isinstance(o, list) and len(o) >= 3 and o[0] == 'SELL' and shed.get(o[1], 0) > 0:
            try:
                n = int(o[2])
            except (TypeError, ValueError):
                continue
            if n > 0:
                shed[o[1]] -= min(n, shed[o[1]])
    sim = dict(shed); lost = Counter(); order = []
    for inv in private['inventories']:
        for item, n in inv.items():
            if n <= 0:
                continue
            take = min(n, max(0, c.cap - sum(sim.values())))
            if take > 0:
                sim[item] = sim.get(item, 0) + take
            if n - take > 0:
                if item not in lost:
                    order.append(item)
                lost[item] += n - take
    need = sum(lost.values())
    if need <= 0:
        return []
    sells = {}
    for item in order:
        k = min(lost[item], shed.get(item, 0)) if item in PRODUCTS else 0
        if k > 0:
            sells[item] = sells.get(item, 0) + k; shed[item] -= k; need -= k
    prices = (obs.get('market') or {}).get('prices', {})
    for item in sorted((i for i in PRODUCTS if shed.get(i, 0) > 0), key=lambda i: (prices.get(i, 10**9), PRODUCTS.index(i))):
        if need <= 0:
            break
        k = min(need, shed[item])
        sells[item] = sells.get(item, 0) + k; shed[item] -= k; need -= k
    return [['SELL', item, n] for item, n in sells.items()][:room_orders]


def _terminal_sales(c, private, market):
    """最終手(step 718): 既存注文が全て正しい形の SELL で、unit 処理後の倉庫の売り残し全品目が注文枠に収まる時だけ、売り残しを足す。"""
    if len(market) > c.max_orders or any(not isinstance(o, list) or len(o) != 3 or o[0] != 'SELL' or o[1] not in PRODUCTS
                                         or type(o[2]) is not int or o[2] <= 0 for o in market):
        return []
    remaining = {p: int(private['shed'].get(p, 0)) for p in PRODUCTS}
    for _, item, n in market:
        remaining[item] = max(0, remaining[item] - n)
    extra = [['SELL', p, n] for p, n in remaining.items() if n > 0]
    if not extra or len(market) + len(extra) > c.max_orders:
        return []
    return extra


def _run(obs, configuration, action):
    c = _Ctx(obs, configuration or {})
    player = obs['player']
    farm = deepcopy(obs['farms'][player]); private = deepcopy(obs['private'])
    seeds0 = dict(obs['private'].get('seeds', {}))
    units = [list(action.get('farmer', ['PASS'])), *[list(a) for a in action.get('hands', [])]]
    demand = Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT')
    blocked = {k for k, n in demand.items() if n > seeds0.get(k, 0)}
    changed = False
    # unit 順に元の行動を観測のコピーへ公式ルールで仮適用する(unit の行動は変えない)
    for u in range(len(units)):
        a = units[u]
        pos = _farmer_position(farm, u)
        if pos is None:
            continue
        pos = list(pos); x, y = pos
        op = a[0] if a else None
        if op in FARMER_MOVES:
            c.apply(farm, private, u, a)
            continue
        if op == 'PLANT' and len(a) >= 2 and a[1] in blocked:
            continue                                        # 公式の atomic PLANT 検査で取り消される手(公式は PASS 扱い)
        _farmer_inventory(private, u)
        if op == 'PASS':
            continue
        c.apply(farm, private, u, a)
    market = list(action.get('market', []) or [])
    extra = _w9_orders(c, obs, private, market)
    if extra:
        market = market + extra; changed = True
    if c.step == c.last_step:
        extra = _terminal_sales(c, private, market)
        if extra:
            market = market + extra; changed = True
    if not changed:
        return action
    out = dict(action)
    out['farmer'] = units[0]
    out['hands'] = units[1:]
    out['market'] = market
    return out


def apply(obs, configuration, action):
    """W9 + WT を掛けた行動を返す。変更が無ければ渡された action そのもの(同じオブジェクト)を返す。

    obs: 公式の観測(dict、または dict のサブクラスである kaggle_environments の Struct)。step・player・farms・private・market を使う。
    configuration: 公式の configuration(dict / Struct / None。None なら既定値 boardSize 10・turnsPerDay 24・shedCapacity 100・
                   episodeSteps 720・maxMarketOrdersPerTurn 10)。
    action: 公式の行動 {'farmer': [...], 'hands': [[...], ...], 'market': [[...], ...]}。
    unit の行動は変えず、market の後ろに SELL 注文を足すだけ。obs・action は書き換えない。例外時は元の action を返す。
    """
    try:
        return _run(obs, configuration, action)
    except Exception:  # noqa: BLE001 - 後処理の失敗で手を落とさない
        return action
