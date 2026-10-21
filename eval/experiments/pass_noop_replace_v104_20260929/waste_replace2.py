"""無駄な手の置き換え(番号ごとの ON/OFF を持つ、モデルに依存しない後処理、2026-09-29 版 2)。

waste_replace.py(W1〜W14 の単独腕で使った版)に W10n と WT を足したもの。既存の番号の挙動は同じ。
  W10n W10 の狭い版: pending_care_bonus > 0、今日の日末がその動物の生産日、fed_today 偽、unit が小麦を携行してそのマスにいる、
       同じ step の後続 unit がそのマスで FEED しない、最終日より前、元の手が HARVEST・FEED・移動でない → FEED
  WT   最終手(step 718)の売り残し: unit 処理後の倉庫に残る品目を追加 SELL(action_waste_audit_20260928_codex/terminal_sales.py の
       append_sales と同じ条件: 既存注文が全て SELL で、残り全品目が注文枠に収まる時だけ)


入力は (obs, configuration, モデルの行動, unit ごとの {行動: スコア}) だけ。unit 順に、先行 unit の(確定後の)行動を
観測のコピーへ公式ルール(kaggriculture.py の _apply_unit_action)で仮適用しながら進み、元の行動を次の番号で判定する。
移動は判定しない。市場の注文は変えない。例外・時間超過は元の行動を返す。

  W1 PASS
  W2 移動以外で、仮適用しても状態(マス・携行品・倉庫・種・位置)が変わらない手
  W3 CARE: 今回の CARE が効く最初の生産日が試合内に残らない(action_waste_followup §9)
  W4 PLANT: 植えた日 + first_yield_day > 29、または最終時刻(hour 23)で同じ step の後続 unit が同じマスへ WATER しない
     (waste_search_more_20260929/endgame.py の W4rule)
  W5 WATER: (a) planted_day + first_yield_day > 29、(b) 29日目の単発作物で収量窓外、または step+2+k > 718
     (k は最寄りの倉庫接点への Manhattan 距離)、(c) 28日目で増収も施肥の生産増もなく consecutive_unwatered == 0
     (endgame.py の W5rule。29日目の継続収穫作物は W6)
  W6 最終日の継続収穫作物への WATER、最終日の FEED、期限は延びるが延びた日の中に収量機会のない FERTILIZE(followup §10〜12)
  W7 期限が延びない FERTILIZE(重複施肥)

置き換え先:
  W1/W2: 同じマスで仮適用すると状態が変わる HARVEST / WATER / FERTILIZE / COLLECT_FERTILIZER / FEED / CARE / PLANT のうち、
         最終日の継続作物 WATER・最終日 FEED・収量機会のない FERTILIZE・効き目が試合後の CARE・hour 23 の PLANT を除いたもの
         (pass_noop_replace.py と同じ規則)。
  W3〜W7: その unit の、モデルが可とした全ての手(移動を含む)から、PASS・状態が変わらない手・W3〜W7 に当たる手を除いたもの。
PLANT はどちらも、同ターンの他 unit の PLANT 分を予約した残りの種がある時だけ。元の行動で種不足により全取消になる品目
(公式の atomic PLANT 検査)は触らない。選ぶのはスコア最大(同点は候補の並び順)。候補がなければ元の行動のまま。

unit_scores[u] は {行動の tuple: スコア}。PICKUP / PLACE は数量なし ('PICKUP', 品目) で渡し、選ばれた時だけ
finalize(u, action) で数量を付ける(無ければ 1)。
"""
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

ALL = ('W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W9', 'W10', 'W10n', 'W14', 'WT')
OTHER = ('W3', 'W4', 'W5', 'W6', 'W7', 'W14')          # 「他の手」へ置き換える番号
SAME_TILE = (('HARVEST',), ('WATER',), ('FERTILIZE',), ('COLLECT_FERTILIZER',), ('FEED',), ('CARE',),
             ('PLANT', 'WHEAT'), ('PLANT', 'CARROT'), ('PLANT', 'TOMATO'), ('PLANT', 'STRAWBERRY'), ('PLANT', 'MELON'))


def official_rules(path=None):
    """公式ファイルから単位ルールの定義だけを読む(環境は起動しない)。"""
    if path is None:
        import kaggle_environments.envs.kaggriculture.kaggriculture as mod
        path = mod.__file__
    tree = ast.parse(Path(path).read_text())
    constants = {'CROPS', 'ANIMALS', 'FARMER_MOVES', 'PRODUCTS'}
    funcs = {'_apply_unit_action', '_farmer_position', '_set_farmer_position', '_farmer_inventory',
             '_inv_add', '_inv_take', '_new_plant', '_new_animal', '_is_shed_adjacent', '_shed_access_tiles'}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in funcs)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    assert constants | funcs <= env.keys()
    return SimpleNamespace(**{k: v for k, v in env.items() if k != '__builtins__'})


class _Timeout(Exception):
    pass


class _Ctx:
    def __init__(self, K, obs, cfg):
        get = lambda k, d: int(cfg.get(k, d) if hasattr(cfg, 'get') else getattr(cfg, k, d))
        self.K = K
        self.board, self.tpd, self.cap, steps = get('boardSize', 10), get('turnsPerDay', 24), get('shedCapacity', 100), get('episodeSteps', 720)
        self.step = int(obs['step']); self.day, self.hour = self.step // self.tpd, self.step % self.tpd
        self.last_step = steps - 2                          # 最後に手を打つ step(718)
        self.last_day = self.last_step // self.tpd          # 29
        self.last_eod = (steps - 1) // self.tpd - 1         # 最後に日末処理が来る日(28)
        self.shed = [tuple(t) for t in K._shed_access_tiles(self.board)]
        self.max_orders = get('maxMarketOrdersPerTurn', 10)

    def apply(self, farm, private, u, a):
        self.K._apply_unit_action(farm, private, u, list(a), self.board, self.day, self.tpd, self.cap)


def _fertilize_useful(c, tile):
    K, day = c.K, c.day
    cd = K.CROPS[tile['crop']]
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
    a = c.K.ANIMALS[tile['animal']]
    first = tile['placed_day'] + a['first_yield_day']
    earliest = c.day + 2                                    # 今日の CARE は翌日の生産の後に積まれる
    nxt = first + max(0, (earliest - first + a['interval'] - 1) // a['interval']) * a['interval']
    return nxt <= c.last_eod + 1


def _water_value(c, tile):
    """(gain, bonus): 今日の WATER で単発作物の収量が増えるか / 継続作物の日末生産に施肥の上乗せが付くか(W5・W14 の式)。"""
    cd = c.K.CROPS[tile['crop']]
    P, D, y = tile['planted_day'], c.day, tile.get('yield_units', 0)
    ws = (cd['max_yield_day'] + 1) // 2
    gain = (not cd['ongoing']) and ws <= D - P <= cd['max_yield_day'] and y < cd['max_yield']
    dsf = D + 1 - P - cd['first_yield_day']
    bonus = (cd['ongoing'] and D <= c.last_eod and dsf >= 0 and dsf % cd['interval'] == 0
             and dsf // cd['interval'] + 1 <= cd['max_yield'] and tile.get('fertilized_until_day', -1) >= D and y < cd['max_yield'])
    return gain, bonus


def _water_kind(c, tile, pos):
    cd = c.K.CROPS[tile['crop']]
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
        return 'W14'                                        # 前日に給水済み。今日外しても日末に WEED にならない
    return None


def classify(c, a, tile, pos, later_water):
    """状態が変わる手 a(仮適用前のマス tile)を W3〜W7・W14 で判定。当たらなければ None。"""
    op = a[0]
    if op == 'CARE':
        return 'W3' if isinstance(tile, dict) and 'animal' in tile and not _care_useful(c, tile) else None
    if op == 'PLANT':
        if tile is None and a[1] in c.K.CROPS:
            if c.day + c.K.CROPS[a[1]]['first_yield_day'] > c.last_day:
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
    K = c.K
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
        k = min(lost[item], shed.get(item, 0)) if item in K.PRODUCTS else 0
        if k > 0:
            sells[item] = sells.get(item, 0) + k; shed[item] -= k; need -= k
    prices = (obs.get('market') or {}).get('prices', {})
    for item in sorted((i for i in K.PRODUCTS if shed.get(i, 0) > 0), key=lambda i: (prices.get(i, 10**9), K.PRODUCTS.index(i))):
        if need <= 0:
            break
        k = min(need, shed[item])
        sells[item] = sells.get(item, 0) + k; shed[item] -= k; need -= k
    return [['SELL', item, n] for item, n in sells.items()][:room_orders]


def _production_today(c, tile):
    a = c.K.ANIMALS[tile['animal']]
    ds = c.day + 1 - tile['placed_day'] - a['first_yield_day']
    return c.day <= c.last_eod and ds >= 0 and ds % a['interval'] == 0


def _terminal_sales(c, private, market):
    """terminal_sales.append_sales と同じ条件で、unit 処理後の倉庫の売り残しを足す注文を返す。"""
    if len(market) > c.max_orders or any(not isinstance(o, list) or len(o) != 3 or o[0] != 'SELL' or o[1] not in c.K.PRODUCTS
                                         or type(o[2]) is not int or o[2] <= 0 for o in market):
        return []
    remaining = {p: int(private['shed'].get(p, 0)) for p in c.K.PRODUCTS}
    for _, item, n in market:
        remaining[item] = max(0, remaining[item] - n)
    extra = [['SELL', p, n] for p, n in remaining.items() if n > 0]
    if not extra or len(market) + len(extra) > c.max_orders:
        return []
    return extra


def _run(K, obs, configuration, action, unit_scores, flags, detect_only, finalize, deadline, events):
    c = _Ctx(K, obs, configuration or {})
    player = obs['player']
    farm = deepcopy(obs['farms'][player]); private = deepcopy(obs['private'])
    seeds0 = dict(obs['private'].get('seeds', {}))
    units = [list(action.get('farmer', ['PASS'])), *[list(a) for a in action.get('hands', [])]]
    demand = Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT')
    blocked = {k for k, n in demand.items() if n > seeds0.get(k, 0)}
    changed = False
    act = lambda k: (k in flags) and not detect_only

    def record(u, pos, kind, a, new=None, **kw):
        events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=u, pos=pos, kind=kind, previous_action=a, action=new, **kw))

    def spare(u, crop):
        others = sum(1 for v, b in enumerate(units) if v != u and len(b) >= 2 and b[0] == 'PLANT' and b[1] == crop)
        return seeds0.get(crop, 0) - others > 0

    def later(u, pos, op):
        for v in range(u + 1, len(units)):
            p = K._farmer_position(farm, v)
            if p is not None and list(p) == list(pos) and units[v] and units[v][0] == op:
                return True
        return False

    for u in range(len(units)):
        if deadline is not None and time.perf_counter() > deadline:
            raise _Timeout
        a = units[u]
        pos = K._farmer_position(farm, u)
        if pos is None:
            continue
        pos = list(pos); x, y = pos
        op = a[0] if a else None
        if op in K.FARMER_MOVES:
            c.apply(farm, private, u, a)
            continue
        if op == 'PLANT' and len(a) >= 2 and a[1] in blocked:
            continue                                        # 公式が PASS 扱いにする。触ると他 unit の取消が変わる
        K._farmer_inventory(private, u)
        before = (deepcopy(farm['tiles'][y][x]), dict(private['inventories'][u]), dict(private['shed']), dict(private['seeds']))
        state = lambda: (farm['tiles'][y][x], private['inventories'][u], private['shed'], private['seeds'])

        def restore():
            farm['tiles'][y][x] = deepcopy(before[0]); private['inventories'][u] = dict(before[1])
            private['shed'] = dict(before[2]); private['seeds'] = dict(before[3]); K._set_farmer_position(farm, u, pos)

        tile0 = before[0]
        # W10+W11: 小麦を持ち、fed_today が偽の動物のマスで FEED 以外(移動は除く)を選んだ
        w10 = (op != 'FEED' and c.day < c.last_day and before[1].get('WHEAT', 0) > 0 and isinstance(tile0, dict)
               and 'animal' in tile0 and not tile0.get('fed_today') and not later(u, pos, 'FEED'))
        # 元の手の判定(状態が変わる手は仮適用済みのまま)
        if op == 'PASS':
            kind = 'W1'
        else:
            c.apply(farm, private, u, a)
            if state() == before and list(K._farmer_position(farm, u)) == pos:
                kind = 'W2'
            else:
                kind = classify(c, a, tile0, pos, later(u, pos, 'WATER'))
        applied = kind not in ('W1', 'W2')
        w10n = w10 and op != 'HARVEST' and tile0.get('pending_care_bonus', 0) > 0 and _production_today(c, tile0)
        if detect_only:
            if kind is not None:
                record(u, pos, kind, a)
            if w10:
                record(u, pos, 'W10', a, base_kind=kind)
            if w10n:
                record(u, pos, 'W10n', a, base_kind=kind)
            continue
        for fk, fire in (('W10', w10), ('W10n', w10n)):
            if fire and act(fk) and not (kind == 'W3' and act('W3')):
                break
        else:
            fk = None
        if fk is not None:
            if applied:
                restore()
            new = ['FEED']
            record(u, pos, fk, a, new, base_kind=kind)
            units[u] = new; changed = True
            c.apply(farm, private, u, new)
            continue
        if kind is None or not act(kind):
            continue                                        # 元の手のまま(状態は仮適用済み、W1/W2 は不変)
        if applied:
            restore()                                       # 置き換えを考えるので元の手を取り消す
        scores = unit_scores[u] if u < len(unit_scores) else {}
        same_tile = kind in ('W1', 'W2')
        best = None
        pool = [(k, scores[k]) for k in SAME_TILE if k in scores] if same_tile else list(scores.items())
        for cand, s in pool:
            name = cand[0]
            if name == 'PASS':
                continue
            if name == 'PLANT' and (cand[1] in blocked or not spare(u, cand[1]) or (same_tile and c.hour == c.tpd - 1)):
                continue
            if same_tile:
                if name == 'FEED' and c.day >= c.last_day:
                    continue
                if name == 'WATER' and c.day >= c.last_day and isinstance(tile0, dict) and tile0.get('kind') == 'PLANT' \
                        and K.CROPS[tile0['crop']]['ongoing']:
                    continue
            c.apply(farm, private, u, cand)
            effect = state() != before or list(K._farmer_position(farm, u)) != pos
            restore()
            if not effect:
                continue
            if same_tile:
                if name == 'FERTILIZE' and not _fertilize_useful(c, tile0):
                    continue
                if name == 'CARE' and not _care_useful(c, tile0):
                    continue
            elif name not in K.FARMER_MOVES and classify(c, list(cand), tile0, pos, later(u, pos, 'WATER')) is not None:
                continue
            if best is None or s > best[1]:
                best = (cand, s)
        if best is None:
            if not same_tile:                               # W1/W2 の候補なしは多いので記録しない
                record(u, pos, kind, a)
            c.apply(farm, private, u, a)                    # 元の手のまま(W1/W2 では状態は変わらない)
            continue
        new = list(best[0])
        if new[0] in ('PICKUP', 'PLACE') and len(new) == 2:
            new = list(finalize(u, new)) if finalize is not None else new + [1]
            c.apply(farm, private, u, new)
            ok = state() != before
            restore()
            if not ok:                                      # 数量 0 などで効果がなくなる時は 1 個にする
                new = new[:2] + [1]
        extra = {}
        if kind == 'W14':
            extra = dict(crop=tile0['crop'], planted_day=tile0['planted_day'])
        record(u, pos, kind, a, new, score=float(best[1]), **extra)
        units[u] = new; changed = True
        c.apply(farm, private, u, new)
    market = list(action.get('market', []) or [])
    if 'W9' in flags or detect_only:
        extra = _w9_orders(c, obs, private, market)
        if extra:
            events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=None, pos=None, kind='W9', previous_action=None,
                               action=None if detect_only else extra, orders=extra))
            if not detect_only:
                market = market + extra; changed = True
    if ('WT' in flags or detect_only) and c.step == c.last_step:
        extra = _terminal_sales(c, private, market)
        if extra:
            events.append(dict(step=c.step, day=c.day, hour=c.hour, unit=None, pos=None, kind='WT', previous_action=None,
                               action=None if detect_only else extra, orders=extra))
            if not detect_only:
                market = market + extra; changed = True
    if not changed:
        return action
    out = dict(action)
    out['farmer'] = units[0]
    out['hands'] = units[1:]
    out['market'] = market
    return out


def replace_waste(obs, configuration, action, unit_scores, *, flags=(), detect_only=False, finalize=None,
                  rules=None, budget_seconds=None):
    """(新しい行動, events, status) を返す。flags は ALL の部分集合。detect_only=True は判定だけ(行動は変えない)。
    events: 当たった手ごとに kind と置き換え先(action、候補なし・判定だけなら None)。例外・時間超過は元の行動と空の events。"""
    flags = frozenset(flags)
    assert flags <= set(ALL), flags
    if not flags and not detect_only:
        return action, [], 'off'
    deadline = None if budget_seconds is None else time.perf_counter() + budget_seconds
    events = []
    try:
        K = rules if rules is not None else _default_rules()
        return _run(K, obs, configuration, action, unit_scores, flags if not detect_only else frozenset(ALL),
                    detect_only, finalize, deadline, events), events, 'ok'
    except _Timeout:
        return action, [], 'timeout'
    except Exception as e:  # noqa: BLE001 - 後処理の失敗で手を落とさない
        return action, [], 'error:' + type(e).__name__


_RULES = None


def _default_rules():
    global _RULES
    if _RULES is None:
        _RULES = official_rules()
    return _RULES


# ---- 当チームの transformer 方策(unit_logits の出力)用の変換 ----
BASE = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE",
        "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]


def nn_action(c):
    if c < 15:
        return (BASE[c],)
    if c < 20:
        return ('PLANT', CROPS[c - 15])
    return ('PICKUP' if c < 32 else 'PLACE', ITEMS[(c - 20) % 12])


NN_CLASS = {nn_action(c): c for c in range(44)}


def nn_unit_scores(row):
    """row: 1 unit のスコア(合法手の印で -1e4 を入れた後、nmb の下駄も後)。-1e3 未満はモデルが不可とした手。"""
    return {nn_action(c): float(row[c]) for c in range(44) if row[c] >= -1e3}
