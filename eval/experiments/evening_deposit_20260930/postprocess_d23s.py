"""D23S(2026-10-01): day <= 28 の 23 時に倉庫接点の上にいる全 unit を DROP させて同じ手で売る後付け処理。
postprocess_ev.py(走行中の実験が凍結しているため変更しない)の helper を import する別ファイル。"""
from collections import Counter
from copy import deepcopy

from postprocess_ev import P, BASE_PRICE, _norm

# ---- D23S(2026-10-01 追加): day <= 28 の 23 時に倉庫接点の上にいる全 unit を DROP(あふれる時は売る品目を PLACE 空きまで)させ、
# 倉庫に入った売る品目の数だけ同じ手に SELL を足す(ネットの注文はそのまま、[] の枠 → 末尾、10 件まで)。移動の寄せはしない。
D23S_SET = ['STRAWBERRY', 'MILK', 'MELON', 'CARROT', 'TOMATO', 'EGG']     # 小麦・肥料(使う)・羊毛(売ると損になりうる)は売らない
D23S_REPLACE = {'NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS', 'DROP', 'PICKUP'}


def apply_d23s(obs, cfg, action, events=None, min_ratio=None):
    """min_ratio: 売る品目のうち、23 時の価格 / 基準価格 が min_ratio 以上の品目だけ SELL を足す(None は全部、D23S_p1 は 1.0)。"""
    try:
        return _d23s(obs, cfg, action, events if events is not None else [], min_ratio)
    except Exception as e:  # noqa: BLE001
        if events is not None:
            events.append(dict(kind='EV_error', step=int(obs.get('step', -1)), error=repr(e)))
        return action


def _d23s(obs, cfg, action, events, min_ratio=None):
    c = P._Ctx(obs, cfg or {})
    if c.day > c.last_eod or c.hour != c.tpd - 1:
        return action
    player = obs['player']
    farm = obs['farms'][player]; private = obs['private']
    n_units = 1 + len(farm.get('hands', []) or [])
    invs = private.get('inventories', [])
    prices = (obs.get('market') or {}).get('prices', {})
    units = [_norm(action.get('farmer', ['PASS']))] + [_norm(a) for a in (action.get('hands', []) or [])]
    while len(units) < n_units:
        units.append(['PASS'])
    orig = [list(a) for a in units]
    on = [u for u in range(n_units) if P._is_shed_adjacent(P._farmer_position(farm, u), c.board)]
    if not on:
        return action
    blocked = {k for k, n in Counter(a[1] for a in units if len(a) >= 2 and a[0] == 'PLANT').items()
               if n > private.get('seeds', {}).get(k, 0)}
    farm_s = deepcopy(farm); priv_s = deepcopy(private); placed = Counter(); changed = False
    for u in range(n_units):
        a = units[u]
        if u in on:
            inv = invs[u] if u < len(invs) else {}
            tot = sum(n for n in inv.values() if n > 0)
            room = max(0, c.cap - sum(priv_s['shed'].values()))
            sell = {k: n for k, n in inv.items() if k in D23S_SET and n > 0}
            info = dict(kind='D23S_unit', step=c.step, day=c.day, hour=c.hour, unit=u, orig=orig[u], room=room, carried=tot,
                        inv={k: n for k, n in inv.items() if n > 0})
            if a[0] not in D23S_REPLACE:
                info['result'] = 'kept_other'
            elif tot <= 0:
                info['result'] = 'empty'
            elif tot <= room:
                a = ['DROP']; placed.update(sell); info['result'] = 'drop'; info['placed'] = dict(sell)
            elif sell and room > 0:
                item = max(sell, key=lambda k: (min(sell[k], room) * prices.get(k, 0), k))
                n = min(sell[item], room); a = ['PLACE', item, n]; placed[item] += n; info['result'] = 'place'; info['placed'] = {item: n}
            else:
                info['result'] = 'no_room_or_nothing_to_sell'
            info['action'] = a
            events.append(info)
            if a != orig[u]:
                changed = True
            units[u] = a
        if len(a) >= 2 and a[0] == 'PLANT' and a[1] in blocked:
            continue
        c.apply(farm_s, priv_s, u, a)
    market = [list(o) if isinstance(o, list) else o for o in (action.get('market', []) or [])]
    added = []
    for item, k in sorted(placed.items(), key=lambda kv: -prices.get(kv[0], 0) * kv[1]):
        if k <= 0:
            continue
        if min_ratio is not None and prices.get(item, 0) / BASE_PRICE[item] < min_ratio:
            added.append(('below_ratio', item, k)); continue
        empty = [i for i, x in enumerate(market[:c.max_orders]) if isinstance(x, list) and not x]
        if empty:
            market[empty[0]] = ['SELL', item, k]; added.append(('slot', item, k))
        elif len(market) < c.max_orders:
            market.append(['SELL', item, k]); added.append(('append', item, k))
        else:
            added.append(('no_slot', item, k))
    if added:
        events.append(dict(kind='EV_sell', step=c.step, day=c.day, hour=c.hour, added=added, price={i: prices.get(i) for i in placed},
                           base_ratio={i: prices.get(i, 0) / BASE_PRICE[i] for i in placed}))
    if not changed and not any(h in ('slot', 'append') for h, _, _ in added):
        return action
    out = dict(action)
    out['farmer'] = units[0]
    out['hands'] = units[1:] if (len(units) > 1 or 'hands' in action) else []
    out['market'] = market
    return out
