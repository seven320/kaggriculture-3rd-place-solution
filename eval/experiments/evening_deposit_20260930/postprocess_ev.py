"""EV(夕方の寄せ売り)の後付け処理(2026-10-01)。版 12 の portable(postprocess_c1p_w9p.py)の出力に掛ける。
  day <= 28(日末処理のある日)の hour 21〜23 に、売れる持ち物(SELL_SET)の価値が V 以上の unit を倉庫接点へ寄せて DROP させ、
  同じ手の市場で SELL を足す(unit の手 → 市場の順なので同じ手で約定)。
  出発は「間に合う範囲で一番遅く」: 倉庫接点(公式 _shed_access_tiles の 4 マス)までの Manhattan 距離 d、余裕 = (23 − hour) − d。
    - 余裕 == 0 かつ d <= K かつ 価値 >= V → 帰還を始める(以後その日の間、最短経路で接点へ移動 → 着いた手で DROP)。
    - d == 0 かつ hour >= 21 かつ 価値 >= V → その手で DROP(21/22 時に接点にいる unit は早めに売る)。
    - 倉庫の空きを超える時は PLACE 品目 空きまで(売れる品目で価値最大の 1 品目)。空き 0 なら何もしない。
  DROP/PLACE の後はネットの手に戻す(1 unit 1 日 1 回)。
  SELL: 寄せた品物のうち SELL_SET の品目を、ネットの同品目 SELL(先頭 10 件内)があれば数を足してまとめ、なければ [] の枠 → 末尾(10 件まで)。
  day 29 には掛けない。例外時は元の action を返す。
"""
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'postprocess'))
import postprocess_c1p_w9p as P  # noqa: E402

PRODUCTS = P.PRODUCTS
SELL_SET = [p for p in PRODUCTS if p not in ('WHEAT', 'FERTILIZER')]   # 翌日の FEED/FERTILIZE 用を売らない(保守側)
BASE_PRICE = {'WHEAT': 25, 'CARROT': 35, 'TOMATO': 60, 'STRAWBERRY': 120, 'MELON': 250, 'EGG': 50, 'MILK': 160, 'WOOL': 200,
              'FERTILIZER': 100}                                          # 公式 MARKET_PARAMS の base
LAST_HOUR = 23
KEEPWORK = os.environ.get('EV_KEEPWORK') == '1'                     # 変種: ネットの手が作業(移動・PASS 以外)の時は置き換えない(帰還は諦める)
IDLE = {'NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS'}
NOSELL = os.environ.get('EV_NOSELL') == '1'                           # 診断用: DROP だけで SELL を足さない


class GameState:
    def __init__(self):
        self.day = -1; self.committed = {}; self.done = set()

    def reset(self, day):
        self.day = day; self.committed = {}; self.done = set()


def dist(pos, tiles):
    return min(abs(pos[0] - x) + abs(pos[1] - y) for x, y in tiles)


def step_toward(pos, tiles):
    tx, ty = min(tiles, key=lambda t: (abs(pos[0] - t[0]) + abs(pos[1] - t[1]), t))
    dx, dy = tx - pos[0], ty - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return ['EAST'] if dx > 0 else ['WEST']
    return ['SOUTH'] if dy > 0 else ['NORTH']


def _norm(a):
    return list(a) if isinstance(a, list) and a else ['PASS']


def apply_ev(obs, cfg, action, st, K, V, events=None, one_unit=False, opts=None):
    try:
        if one_unit:
            return _ev1u(obs, cfg, action, st, K, V, events if events is not None else [], **(opts or {}))
        return _ev(obs, cfg, action, st, K, V, events if events is not None else [])
    except Exception as e:  # noqa: BLE001
        if events is not None:
            events.append(dict(kind='EV_error', step=int(obs.get('step', -1)), error=repr(e)))
        return action


def _ev(obs, cfg, action, st, K, V, events):
    c = P._Ctx(obs, cfg or {})
    if c.day != st.day:
        st.reset(c.day)
    if c.day > c.last_eod or c.hour < LAST_HOUR - K:
        return action
    player = obs['player']
    farm = obs['farms'][player]; private = obs['private']
    n_units = 1 + len(farm.get('hands', []) or [])
    invs = private.get('inventories', [])
    prices = (obs.get('market') or {}).get('prices', {})
    hands = [_norm(a) for a in (action.get('hands', []) or [])]
    units = [_norm(action.get('farmer', ['PASS']))] + hands
    while len(units) < n_units:
        units.append(['PASS'])
    orig = [list(a) for a in units]
    tiles = c.shed
    left = LAST_HOUR - c.hour
    # 帰還の開始判定(この手の観測で)
    for u in range(n_units):
        if u in st.done or u in st.committed:
            continue
        pos = P._farmer_position(farm, u)
        inv = invs[u] if u < len(invs) else {}
        sell = {k: n for k, n in inv.items() if k in SELL_SET and n > 0}
        if not sell:
            continue
        val = sum(prices.get(k, 0) * n for k, n in sell.items())
        if val < V:
            continue
        d = dist(pos, tiles)
        if d > K or d > left:
            continue
        if d == left or (d == 0 and c.hour >= 21):
            if orig[u][0] in ('DROP',):
                continue                                   # ネットが既に DROP する
            if KEEPWORK and orig[u][0] not in IDLE:
                continue
            st.committed[u] = dict(step=c.step, d=d, value=val, items=dict(sell))
            events.append(dict(kind='EV_commit', step=c.step, day=c.day, hour=c.hour, unit=u, d=d, value=val, items=dict(sell)))
    if not st.committed:
        return action
    # unit の手を順に置き換え、倉庫の空きは公式順(unit 0 → 1 → …)で進めて測る
    blocked = {k for k, n in Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT').items()
               if n > private.get('seeds', {}).get(k, 0)}
    farm_s = deepcopy(farm); priv_s = deepcopy(private)
    placed = Counter(); repl = []
    for u in range(n_units):
        a = units[u]
        if u in st.committed and KEEPWORK and orig[u][0] not in IDLE:
            events.append(dict(kind='EV_yield', step=c.step, day=c.day, hour=c.hour, unit=u, orig=orig[u]))
            del st.committed[u]; st.done.add(u)
        if u in st.committed:
            pos = P._farmer_position(farm, u)
            d = dist(pos, tiles)
            if d > 0:
                a = step_toward(pos, tiles)
                repl.append(dict(unit=u, orig=orig[u], new=a, phase='move'))
            else:
                inv = invs[u] if u < len(invs) else {}
                room = max(0, c.cap - sum(priv_s['shed'].values()))
                tot = sum(n for n in inv.values() if n > 0)
                sell = {k: n for k, n in inv.items() if k in SELL_SET and n > 0}
                info = dict(kind='EV_drop', step=c.step, day=c.day, hour=c.hour, unit=u, room=room, carried=tot,
                            commit=st.committed[u])
                if room <= 0 or not sell:
                    info['result'] = 'no_room' if room <= 0 else 'nothing'
                    a = units[u]
                elif tot <= room:
                    a = ['DROP']; placed.update(sell); info['placed'] = dict(sell)
                else:
                    item = max(sell, key=lambda k: (min(sell[k], room) * prices.get(k, 0), k))
                    n = min(sell[item], room); a = ['PLACE', item, n]; placed[item] += n; info['placed'] = {item: n}
                info['action'] = a
                events.append(info)
                if a is not units[u]:
                    repl.append(dict(unit=u, orig=orig[u], new=a, phase='drop'))
                del st.committed[u]; st.done.add(u)
        units[u] = a
        if len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked:
            continue
        c.apply(farm_s, priv_s, u, a)
    market = [list(o) if isinstance(o, list) else o for o in (action.get('market', []) or [])]
    added = []
    for item, k in sorted(placed.items(), key=lambda kv: -prices.get(kv[0], 0) * kv[1]) if not NOSELL else []:
        merged = False
        for o in market[:c.max_orders]:
            if isinstance(o, list) and len(o) >= 3 and o[0] == 'SELL' and o[1] == item:
                try:
                    o[2] = int(o[2]) + k; merged = True; added.append(('merge', item, k)); break
                except (TypeError, ValueError):
                    continue
        if merged:
            continue
        empty = [i for i, o in enumerate(market[:c.max_orders]) if isinstance(o, list) and not o]
        if empty:
            market[empty[0]] = ['SELL', item, k]; added.append(('slot', item, k))
        elif len(market) < c.max_orders:
            market.append(['SELL', item, k]); added.append(('append', item, k))
        else:
            added.append(('no_slot', item, k))
    if placed:
        events.append(dict(kind='EV_sell', step=c.step, day=c.day, hour=c.hour, added=added,
                           price=({i: prices.get(i) for i in placed}), base_ratio={i: prices.get(i, 0) / BASE_PRICE[i] for i in placed}))
    for r in repl:
        if r['orig'] != r['new']:
            events.append(dict(kind='EV_replace', step=c.step, day=c.day, hour=c.hour, **r))
    out = dict(action)
    out['farmer'] = units[0]
    out['hands'] = units[1:] if (len(units) > 1 or 'hands' in action) else []
    out['market'] = market
    return out


def project_shed(obs, cfg, action):
    """公式順(unit → 市場 SELL/BUY(資金は見ない)→ hour 23 なら日末搬入)で倉庫を進めた見込み。(shed, 日末に捨てる品目の Counter, DROP であふれた数)"""
    c = P._Ctx(obs, cfg or {})
    player = obs['player']
    farm = deepcopy(obs['farms'][player]); priv = deepcopy(obs['private'])
    units = [_norm(action.get('farmer', ['PASS']))] + [_norm(a) for a in (action.get('hands', []) or [])]
    blocked = {k for k, n in Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT').items()
               if n > obs['private'].get('seeds', {}).get(k, 0)}
    drop_lost = 0
    for u, a in enumerate(units):
        pos = P._farmer_position(farm, u)
        if pos is None:
            continue
        if a[0] == 'DROP' and P._is_shed_adjacent(pos, c.board):
            inv = P._farmer_inventory(priv, u)
            drop_lost += max(0, sum(n for n in inv.values() if n > 0) - max(0, c.cap - sum(priv['shed'].values())))
        if len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked:
            continue
        c.apply(farm, priv, u, a)
    shed = priv['shed']
    for o in list(action.get('market', []) or [])[:c.max_orders]:
        if not isinstance(o, list) or len(o) < 3:
            continue
        try:
            n = int(o[2])
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        if o[0] == 'SELL' and o[1] in PRODUCTS:
            shed[o[1]] = shed.get(o[1], 0) - min(n, shed.get(o[1], 0))
        elif (o[0] == 'BUY_PRODUCT' and o[1] in ('WHEAT', 'FERTILIZER')) or (o[0] == 'BUY_ANIMAL' and o[1] in P.ANIMALS):
            shed[o[1]] = shed.get(o[1], 0) + min(n, max(0, c.cap - sum(shed.values())))
    eod_lost = Counter()
    if c.hour == c.tpd - 1 and c.day <= c.last_eod:
        for inv in priv['inventories']:
            for item, n in inv.items():
                if n <= 0:
                    continue
                take = min(n, max(0, c.cap - sum(shed.values())))
                if take:
                    shed[item] = shed.get(item, 0) + take
                if n - take > 0:
                    eod_lost[item] += n - take
    return {k: n for k, n in shed.items() if n > 0}, eod_lost, drop_lost


# ---- EV21-1u(別パターン、2026-10-01 追加): 1 日 1 unit、近い unit だけ、対象品目を PLACE で入れて同じ手で売る(ネットの注文とまとめない)
TARGET_1U = ['STRAWBERRY', 'CARROT', 'FERTILIZER', 'WHEAT', 'MILK']        # cf_x21_summary.json の品目別寄与が大きいもの


def _reserve(c, farm):
    """その日のうちに使う見込み: WHEAT = 今日まだ FEED されていない動物の数、FERTILIZER = 施肥が効く作物の数(農場全体、保守側)。"""
    wheat = fert = 0
    for row in farm['tiles']:
        for t in row:
            if not isinstance(t, dict):
                continue
            if 'animal' in t and not t.get('fed_today'):
                wheat += 1
            elif t.get('kind') == 'PLANT':
                try:
                    fert += bool(P._fertilize_useful(c, t))
                except Exception:  # noqa: BLE001
                    pass
    return {'WHEAT': wheat, 'FERTILIZER': fert}


def _quadrant_of(x, y, board_size):
    """公式 kaggriculture.py の _quadrant_of と同じ判定。"""
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _ev1u(obs, cfg, action, st, K, V, events, days=None, target=None, min_n=1, no4q=False):
    """days = (最初の日, 最後の日) だけ掛ける(None は全日)。target = 対象品目(None は TARGET_1U)。min_n = 選んだ品目の最小個数。
    no4q = True: farm['unlocked_quadrants'] の 4 番目の区画にいる unit を候補から外す(区画は公式 _quadrant_of を位置にそのまま当てる。
    倉庫接点の 4 マスもそれぞれどれか 1 つの区画に属するので、4 番目の区画に属する接点マスの上の unit も外す。判定は帰還開始の判断時だけ)。"""
    c = P._Ctx(obs, cfg or {})
    if c.day != st.day:
        st.reset(c.day)
    if c.day > c.last_eod or c.hour < LAST_HOUR - K or (days is not None and not days[0] <= c.day <= days[1]):
        return action
    target = TARGET_1U if target is None else target
    player = obs['player']
    farm = obs['farms'][player]; private = obs['private']
    n_units = 1 + len(farm.get('hands', []) or [])
    invs = private.get('inventories', [])
    prices = (obs.get('market') or {}).get('prices', {})
    units = [_norm(action.get('farmer', ['PASS']))] + [_norm(a) for a in (action.get('hands', []) or [])]
    while len(units) < n_units:
        units.append(['PASS'])
    tiles = c.shed; left = LAST_HOUR - c.hour
    if not st.committed and not st.done:
        res = _reserve(c, farm); best = None
        uq = farm.get('unlocked_quadrants') or []
        q4 = uq[3] if no4q and len(uq) >= 4 else None
        for u in range(n_units):
            pos = P._farmer_position(farm, u)
            inv = invs[u] if u < len(invs) else {}
            tg = {k: n - min(n, res.get(k, 0)) for k, n in inv.items() if k in target and n > 0}
            tg = {k: n for k, n in tg.items() if n > 0}
            if not tg or units[u][0] == 'DROP':
                continue
            item = max(tg, key=lambda k: (prices.get(k, 0) * tg[k], k))
            val = prices.get(item, 0) * tg[item]
            if tg[item] < min_n:
                continue
            d = dist(pos, tiles)
            if val < V or d > K or d > left or not (d == left or (d == 0 and c.hour >= 21)):
                continue
            if q4 is not None and _quadrant_of(pos[0], pos[1], c.board) == q4:
                events.append(dict(kind='EV_skip4q', step=c.step, day=c.day, hour=c.hour, unit=u, d=d, value=val, q4=q4))
                continue
            if best is None or val > best[0]:
                best = (val, u, item, tg[item], d)
        if best is not None:
            val, u, item, n, d = best
            st.committed[u] = dict(step=c.step, d=d, value=val, item=item, n=n)
            events.append(dict(kind='EV_commit', step=c.step, day=c.day, hour=c.hour, unit=u, d=d, value=val, items={item: n}))
    if not st.committed:
        return action
    u = next(iter(st.committed)); cm = st.committed[u]
    pos = P._farmer_position(farm, u); d = dist(pos, tiles)
    orig = list(units[u]); market = [list(o) if isinstance(o, list) else o for o in (action.get('market', []) or [])]
    added = []
    if d > 0:
        units[u] = step_toward(pos, tiles); phase = 'move'
    else:
        farm_s = deepcopy(farm); priv_s = deepcopy(private)
        blocked = {k for k, n in Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT').items()
                   if n > private.get('seeds', {}).get(k, 0)}
        for v in range(u):
            a = units[v]
            if not (len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked):
                c.apply(farm_s, priv_s, v, a)
        room = max(0, c.cap - sum(priv_s['shed'].values()))
        inv = invs[u] if u < len(invs) else {}
        n = min(cm['n'], inv.get(cm['item'], 0), room)
        info = dict(kind='EV_drop', step=c.step, day=c.day, hour=c.hour, unit=u, room=room, carried=sum(x for x in inv.values() if x > 0), commit=cm)
        del st.committed[u]; st.done.add(u)
        if n <= 0:
            info['result'] = 'no_room' if room <= 0 else 'nothing'
            events.append(info)
            return action
        units[u] = ['PLACE', cm['item'], n]; phase = 'drop'; info['placed'] = {cm['item']: n}; info['action'] = units[u]
        events.append(info)
        o = ['SELL', cm['item'], n]
        empty = [i for i, x in enumerate(market[:c.max_orders]) if isinstance(x, list) and not x]
        if empty:
            market[empty[0]] = o; added.append(('slot', cm['item'], n))
        elif len(market) < c.max_orders:
            market.append(o); added.append(('append', cm['item'], n))
        else:
            added.append(('no_slot', cm['item'], n))
        events.append(dict(kind='EV_sell', step=c.step, day=c.day, hour=c.hour, added=added, price={cm['item']: prices.get(cm['item'])},
                           base_ratio={cm['item']: prices.get(cm['item'], 0) / BASE_PRICE[cm['item']]}))
    if orig != units[u]:
        events.append(dict(kind='EV_replace', step=c.step, day=c.day, hour=c.hour, unit=u, orig=orig, new=units[u], phase=phase))
    out = dict(action)
    out['farmer'] = units[0]
    out['hands'] = units[1:] if (len(units) > 1 or 'hands' in action) else []
    out['market'] = market
    return out
