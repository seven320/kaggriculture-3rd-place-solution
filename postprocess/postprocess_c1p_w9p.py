"""C1' + W9' = W3 + W5 + W6 + W9 + W9' + WT の後付け後処理(単独版、2026-09-30)。apply は Python 標準ライブラリのみで動く。

  W3  CARE で、今回の CARE が効く最初の生産日が試合内に残らないもの → 他の手
  W5  WATER で、(a) planted_day + first_yield_day > 29、(b) 29 日目の単発作物で収量窓の外か step + 2 + k > 718
      (k は最寄りの倉庫の接点への Manhattan 距離)、(c) 28 日目で増収も施肥による生産増もなく consecutive_unwatered == 0 → 他の手
  W6  最終日の継続収穫作物への WATER、最終日の FEED、期限は延びるが延びた日の中に収量機会のない FERTILIZE → 他の手
  W9  hour 23: 日末の自動搬入で倉庫(100)からあふれて捨てられる分を追加 SELL(注文枠 10 の空きまで)
  W9' hour 23: 注文に空の注文 [] がある時、[] を空き枠として扱う。[] を除いた注文列で W9 を計算し直し、W9 が足しきれなかった
      SELL を [] の枠へ前から入れる(他の注文の位置は変えない)。公式の _parse_order で [] は無操作だが、_process_market の
      q[:max_orders] で [] も 10 枠の 1 つに数えられるため。
  WT  最終手 step 718: 倉庫の売り残しを追加 SELL(既存注文が全て正しい形の SELL で、注文枠に収まる時だけ)
  「他の手」= その unit のモデルが可とした全ての手(移動を含む)から、PASS・状態が変わらない手・W3〜W7/W14 に当たる手を除いた、
            元の forward のスコアが最大の手(再 encode はしない)。候補が無ければ元の手のまま。

使い方(Kit 系 main.py。詳しくは README):
    import postprocess_c1p_w9p as _C1P
    _C1P_HOOK = _C1P.KitHook(_NET)                      # _NET を作った直後。_NET.unit_logits を包んでスコアを控える
    ...
    return _C1P_HOOK.apply(o, config, _liq(_act(o), o))   # agent の最後(網 → 提出物の後処理 → C1' + W9')
網を使わない形: action = apply(obs, configuration, action, unit_scores, finalize=None)
挙動は waste_replace2.replace_waste(obs, configuration, action, unit_scores, flags=('W3','W5','W6','W9','WT'), finalize=...)
の行動出力に、実験の run_m.MTop.m5(W9')を掛けたものと同じ(同等性試験: test_postprocess_c1p_w9p.py、結果は ../validation.json)。
例外・時間超過の時は元の action を返す。

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

__all__ = ['apply', 'scores_from_logits', 'KitHook', 'nn_action', 'NN_CLASS', 'FLAGS']

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
# ここから下は当チームのコード(C1' + W9' = W3 + W5 + W6 + W9 + W9' + WT の後処理)。
# waste_replace2.replace_waste(obs, configuration, action, unit_scores, flags=('W3', 'W5', 'W6', 'W9', 'WT'),
# finalize=..., budget_seconds=...) と同じ出力を返すように、同関数の該当部分をそのまま移し、
# W9'(実験の artifacts/sub121_market_pp_20260930/run_m.py の MTop.m5 と market_pp.fill_empty)を W9 の直後に足したもの。
# 移していないのは flag が OFF の番号の置き換え(W1・W2・W4・W7・W10・W10n・W14)と detect_only の経路だけ。
# ただし W1・W2・W4・W7・W14 の「判定」は残している(置き換え先の候補から外すのに使うため。waste_replace2 と同じ)。
# =====================================================================================================
import time

FLAGS = ('W3', 'W5', 'W6', 'W9', 'WT')                   # waste_replace2 の flag。W9' はこの後に常に掛ける(flag ではない)
DEFAULT_BUDGET_SECONDS = 1.0                            # 実験(run_c.py)と同じ。超えたら元の行動を返す


class _Timeout(Exception):
    pass


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


def _fertilize_useful(c, tile):
    day = c.day
    cd = CROPS[tile['crop']]
    old = tile.get('fertilized_until_day', -1)
    if day + 2 <= old:
        return False
    for d in range(max(day, old + 1), day + 3):             # 延長で新たに有効になる日だけ
        if cd['ongoing']:
            if d > c.last_eod:
                continue
            ds = d + 1 - tile['planted_day'] - cd['first_yield_day']
            if ds >= 0 and ds % cd['interval'] == 0 and ds // cd['interval'] + 1 <= cd['max_yield']:
                return True
        else:
            if d > c.last_day:
                continue
            age = d - tile['planted_day']
            if ((cd['max_yield_day'] + 1) // 2 <= age <= cd['max_yield_day'] and (d > day or not tile['watered_today'])
                    and tile.get('yield_units', 0) < cd['max_yield']):
                return True
    return False


def _care_useful(c, tile):
    a = ANIMALS[tile['animal']]
    first = tile['placed_day'] + a['first_yield_day']
    earliest = c.day + 2                                    # 今日の CARE は翌日の生産の後に積まれる
    nxt = first + max(0, (earliest - first + a['interval'] - 1) // a['interval']) * a['interval']
    return nxt <= c.last_eod + 1


def _water_value(c, tile):
    """(gain, bonus): 今日の WATER で単発作物の収量が増えるか / 継続作物の日末生産に施肥の上乗せが付くか。"""
    cd = CROPS[tile['crop']]
    P, D, y = tile['planted_day'], c.day, tile.get('yield_units', 0)
    ws = (cd['max_yield_day'] + 1) // 2
    gain = (not cd['ongoing']) and ws <= D - P <= cd['max_yield_day'] and y < cd['max_yield']
    dsf = D + 1 - P - cd['first_yield_day']
    bonus = (cd['ongoing'] and D <= c.last_eod and dsf >= 0 and dsf % cd['interval'] == 0
             and dsf // cd['interval'] + 1 <= cd['max_yield'] and tile.get('fertilized_until_day', -1) >= D and y < cd['max_yield'])
    return gain, bonus


def _water_kind(c, tile, pos):
    cd = CROPS[tile['crop']]
    D = c.day
    if D >= c.last_day and cd['ongoing']:
        return 'W6'
    if tile['planted_day'] + cd['first_yield_day'] > c.last_day:
        return 'W5'                                         # (a) 収穫できる日が来ない
    gain, bonus = _water_value(c, tile)
    if D == c.last_day:
        k = min(abs(pos[0] - s[0]) + abs(pos[1] - s[1]) for s in c.shed)
        return 'W5' if (not gain) or c.step + 2 + k > c.last_step else None          # (b)
    if D == c.last_day - 1:
        return 'W5' if not (gain or bonus or tile['consecutive_unwatered'] >= 1) else None   # (c)
    if tile['consecutive_unwatered'] == 0 and not (gain or bonus):
        return 'W14'                                        # 判定だけ(置き換えない。候補から外すのに使う)
    return None


def _classify(c, a, tile, pos, later_water):
    """状態が変わる手 a(仮適用前のマス tile)を W3〜W7・W14 で判定。当たらなければ None。"""
    op = a[0]
    if op == 'CARE':
        return 'W3' if isinstance(tile, dict) and 'animal' in tile and not _care_useful(c, tile) else None
    if op == 'PLANT':
        if tile is None and a[1] in CROPS:
            if c.day + CROPS[a[1]]['first_yield_day'] > c.last_day:
                return 'W4'
            if c.hour == c.tpd - 1 and not later_water:
                return 'W4'
        return None
    if op == 'WATER' and isinstance(tile, dict) and tile.get('kind') == 'PLANT':
        return _water_kind(c, tile, pos)
    if op == 'FEED':
        return 'W6' if c.day >= c.last_day else None
    if op == 'FERTILIZE' and isinstance(tile, dict) and tile.get('kind') == 'PLANT':
        if c.day + 2 <= tile.get('fertilized_until_day', -1):
            return 'W7'
        return None if _fertilize_useful(c, tile) else 'W6'
    return None


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


def _run(obs, configuration, action, unit_scores, finalize, deadline, events):
    c = _Ctx(obs, configuration or {})
    player = obs['player']
    farm = deepcopy(obs['farms'][player]); private = deepcopy(obs['private'])
    seeds0 = dict(obs['private'].get('seeds', {}))
    units = [list(action.get('farmer', ['PASS'])), *[list(a) for a in action.get('hands', [])]]
    demand = Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT')
    blocked = {k for k, n in demand.items() if n > seeds0.get(k, 0)}
    changed = False

    def record(u, pos, kind, a, new=None, **kw):
        events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=u, pos=pos, kind=kind, previous_action=a, action=new, **kw))

    def spare(u, crop):
        others = sum(1 for v, b in enumerate(units) if v != u and len(b) >= 2 and b[0] == 'PLANT' and b[1] == crop)
        return seeds0.get(crop, 0) - others > 0

    def later(u, pos, op):
        for v in range(u + 1, len(units)):
            p = _farmer_position(farm, v)
            if p is not None and list(p) == list(pos) and units[v] and units[v][0] == op:
                return True
        return False

    for u in range(len(units)):
        if deadline is not None and time.perf_counter() > deadline:
            raise _Timeout
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
            continue                                        # 公式が PASS 扱いにする。触ると他 unit の取消が変わる
        _farmer_inventory(private, u)
        before = (deepcopy(farm['tiles'][y][x]), dict(private['inventories'][u]), dict(private['shed']), dict(private['seeds']))
        state = lambda: (farm['tiles'][y][x], private['inventories'][u], private['shed'], private['seeds'])

        def restore():
            farm['tiles'][y][x] = deepcopy(before[0]); private['inventories'][u] = dict(before[1])
            private['shed'] = dict(before[2]); private['seeds'] = dict(before[3]); _set_farmer_position(farm, u, pos)

        tile0 = before[0]
        # 元の手の判定(状態が変わる手は仮適用済みのまま)
        if op == 'PASS':
            kind = 'W1'
        else:
            c.apply(farm, private, u, a)
            if state() == before and list(_farmer_position(farm, u)) == pos:
                kind = 'W2'
            else:
                kind = _classify(c, a, tile0, pos, later(u, pos, 'WATER'))
        applied = kind not in ('W1', 'W2')
        if kind is None or kind not in FLAGS:
            continue                                        # 元の手のまま(状態は仮適用済み、W1/W2 は不変)
        if applied:
            restore()                                       # 置き換えを考えるので元の手を取り消す
        scores = unit_scores[u] if u < len(unit_scores) else {}
        best = None
        for cand, s in list(scores.items()):               # W3/W5/W6: その unit のモデルが可とした全ての手(移動を含む)
            name = cand[0]
            if name == 'PASS':
                continue
            if name == 'PLANT' and (cand[1] in blocked or not spare(u, cand[1])):
                continue
            c.apply(farm, private, u, cand)
            effect = state() != before or list(_farmer_position(farm, u)) != pos
            restore()
            if not effect:
                continue
            if name not in FARMER_MOVES and _classify(c, list(cand), tile0, pos, later(u, pos, 'WATER')) is not None:
                continue
            if best is None or s > best[1]:
                best = (cand, s)
        if best is None:
            record(u, pos, kind, a)
            c.apply(farm, private, u, a)                    # 元の手のまま
            continue
        new = list(best[0])
        if new[0] in ('PICKUP', 'PLACE') and len(new) == 2:
            new = list(finalize(u, new)) if finalize is not None else new + [1]
            c.apply(farm, private, u, new)
            ok = state() != before
            restore()
            if not ok:                                      # 数量 0 などで効果がなくなる時は 1 個にする
                new = new[:2] + [1]
        record(u, pos, kind, a, new, score=float(best[1]))
        units[u] = new; changed = True
        c.apply(farm, private, u, new)
    market = list(action.get('market', []) or [])
    extra = _w9_orders(c, obs, private, market)
    if extra:
        events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=None, pos=None, kind='W9', previous_action=None,
                           action=extra, orders=extra))
        market = market + extra; changed = True
    w9p = False
    empty = [i for i, o in enumerate(market) if isinstance(o, list) and not o]
    if c.hour == c.tpd - 1 and empty:
        # W9': 空の注文 [](網が枠の位置合わせに出す)は公式の _parse_order で無操作だが、_process_market の q[:max_orders] で
        # 10 枠の 1 つに数えられる。W9 は「10 − 注文数([] を含む)」までしか SELL を足せないので、[] を除いた注文列で W9 を
        # 計算し直し、足りない SELL を [] の枠へ前から入れる(他の注文の位置は変えない)。
        compact = [o for o in market if not (isinstance(o, list) and not o)]
        extra = _w9_orders(c, obs, private, compact)[:len(empty)]
        if extra:
            slots = empty[:len(extra)]
            market = [list(o) if isinstance(o, list) else o for o in market]
            for i, o in zip(slots, extra):
                market[i] = list(o)
            events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=None, pos=None, kind="W9'", previous_action=None,
                               action=extra, orders=extra, slots=slots))
            w9p = True
    if c.step == c.last_step:
        extra = _terminal_sales(c, private, market)
        if extra:
            events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=None, pos=None, kind='WT', previous_action=None,
                               action=extra, orders=extra))
            market = market + extra; changed = True
    if not changed and not w9p:
        return action
    out = dict(action)
    if changed:                                             # W9' だけの時は unit の手を元の action のまま(実験の MTop.m5 と同じ)
        out['farmer'] = units[0]
        out['hands'] = units[1:]
    out['market'] = market
    return out


def apply(obs, configuration, action, unit_scores, finalize=None, budget_seconds=DEFAULT_BUDGET_SECONDS, events=None):
    """C1' + W9'(W3 + W5 + W6 + W9 + W9' + WT)を掛けた行動を返す。変更が無ければ渡された action そのもの(同じオブジェクト)を返す。

    obs: 公式の観測(dict、または dict のサブクラスである kaggle_environments の Struct)。
    configuration: 公式の configuration(dict / Struct / None。None なら既定値)。
    action: 公式の行動 {'farmer': [...], 'hands': [[...], ...], 'market': [[...], ...]}(網 → 提出物の後処理 の後の行動)。
    unit_scores: unit ごとの {行動の tuple: スコア} の列(unit 順 = farmer, hands...)。モデルが不可とした手は入れない。
                 PICKUP / PLACE は数量なしの ('PICKUP', 品目)。Kit 系の網なら scores_from_logits() か KitHook で作る。
                 空の列を渡すと W3/W5/W6 は置き換え先が無く元の手のままになり、W9/W9'/WT だけが効く。
    finalize: PICKUP / PLACE が選ばれた時に数量を付ける関数 finalize(u, ['PICKUP', 品目]) -> ['PICKUP', 品目, 数量]。
              None なら数量 1。
    budget_seconds: これを超えたら元の action を返す(None で無制限)。
    events: list を渡すと、当たった手と置き換え先を追記する(デバッグ用)。
    obs・action は書き換えない。例外・時間超過の時は元の action を返す。
    """
    deadline = None if budget_seconds is None else time.perf_counter() + budget_seconds
    ev = []
    try:
        out = _run(obs, configuration, action, unit_scores, finalize, deadline, ev)
    except Exception:  # noqa: BLE001 - 後処理の失敗(時間超過を含む)で手を落とさない
        return action
    if events is not None:
        events.extend(ev)
    return out


# ---- 当チームの Kit 系 transformer 方策(model_tf.Policy / model_ptr_infer.PolicyPtrInfer)用の補助 ----
# 網の unit_logits の出力は unit ごとに 44 クラス(下の nn_action の並び)。合法手でないクラスは -1e4(nmb の下駄の後も -1e3 未満)。
_BASE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
         "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
_CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
_ITEM_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]


def nn_action(c):
    """網の unit クラス番号(0..43)-> 行動の tuple(PICKUP / PLACE は数量なし)。"""
    if c < 15:
        return (_BASE[c],)
    if c < 20:
        return ('PLANT', _CROP_ORDER[c - 15])
    return ('PICKUP' if c < 32 else 'PLACE', _ITEM_ORDER[(c - 20) % 12])


NN_CLASS = {nn_action(c): c for c in range(44)}


def scores_from_logits(logits, n_units):
    """網の unit_logits の 1 局分(20 × 44、torch.Tensor / numpy / list)から unit_scores を作る。
    スコアは nmb の下駄の後の値をそのまま使い、-1e3 未満(モデルが不可とした手)は入れない。"""
    rows = logits.tolist() if hasattr(logits, 'tolist') else logits
    return [{nn_action(c): float(r[c]) for c in range(44) if r[c] >= -1e3} for r in rows[:n_units]]


def _q(qc, qmax):
    """数量クラス -> 数(obs_feat._q と同じ): 0 = 指定なし(None), 1..qmax+1 = 0..qmax, 以降は番兵 999/1000。"""
    qc = int(qc)
    if qc == 0:
        return None
    if qc <= qmax + 1:
        return qc - 1
    return (999, 1000)[min(qc - (qmax + 2), 1)]


class KitHook:
    """Kit 系 main.py の網(_NET)の unit_logits を包み、agent 1 回の forward の unit スコアと q を控える。
    apply(obs, configuration, action) で、控えたスコアを使って C1' + W9' を掛ける(控えは使った後に消す)。
    torch は提出物にあるものを使う(この関数群を使う時だけ import)。"""

    def __init__(self, net, budget_seconds=DEFAULT_BUDGET_SECONDS):
        self.net, self.budget_seconds = net, budget_seconds
        self.s = self.q = None; self.calls = 0
        orig = net.unit_logits

        def unit_logits(*args, **kwargs):
            s, q = orig(*args, **kwargs)
            self.s, self.q = s, q; self.calls += 1
            return s, q
        net.unit_logits = unit_logits

    def finalize(self, u, a):
        """PICKUP / PLACE の数量を網の unit_qty で付ける(obs_feat.to_action と同じ数量の復号、実験の run_c.py と同じ)。"""
        import torch
        with torch.inference_mode():
            cls = torch.zeros(1, self.q.shape[1], dtype=torch.long, device=self.q.device)
            cls[0, u] = NN_CLASS[tuple(a)]
            qc = int(self.net.unit_qty(self.q[0:1], cls).argmax(-1)[0, u])
        qty = _q(qc, 20)
        return [a[0], a[1], 1 if qty is None else int(qty)]

    def apply(self, obs, configuration, action, events=None):
        """直前の agent 呼び出しの forward(1 回だけのはず)のスコアで C1' + W9' を掛ける。
        forward が 1 回でなかった時(控えが無い・複数)は unit の置き換えをせず W9/W9'/WT だけを掛ける。"""
        s, q, calls = self.s, self.q, self.calls
        self.s = self.q = None; self.calls = 0
        try:
            n = 1 + len(action.get('hands', []) or [])
            ok = calls == 1 and s is not None and s.shape[0] == 1
            scores = scores_from_logits(s[0], n) if ok else []
        except Exception:  # noqa: BLE001
            scores, ok = [], False
        self.q = q if ok else None
        try:
            return apply(obs, configuration, action, scores, finalize=self.finalize if ok else None,
                         budget_seconds=self.budget_seconds, events=events)
        finally:
            self.q = None
