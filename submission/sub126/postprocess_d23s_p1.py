"""D23S_p1: 日末前(day 0〜28 の hour 23)に倉庫接点の上にいる unit を DROP させ、基準価格以上の品目を同じ手で売る後付け処理(2026-10-01、版 13)。
gen35 など Kit 系 main.py の土台(postprocess_c1p_w9p)の後・最終日の回収計画(collect_plan_d29)の前に掛ける。
  - day 0〜28 の hour 23 の手で、倉庫接点(公式 _shed_access_tiles の 4 マス)の上にいる unit のうち、元の手が
    NORTH/SOUTH/EAST/WEST/PASS/DROP/PICKUP のものを DROP に替える(持ち物が倉庫の空きを超える時は、売る品目のうち
    min(個数, 空き) × 価格 が最大の 1 品目を PLACE で空きまで。空き 0 や売る品目が無い時は元の手のまま)。
  - 倉庫に入る STRAWBERRY/MILK/MELON/CARROT/TOMATO/EGG のうち、その時点の価格 / 基準価格(公式 MARKET_PARAMS の base)>= 1.0 の
    品目だけ、入る数を同じ手の market に SELL で足す(空の注文 [] の枠 → 末尾、10 件まで。ネットの注文はそのまま)。
    WHEAT・FERTILIZER・WOOL は売らない(倉庫に入れるだけ)。day 29 には掛けない。
使い方: import postprocess_d23s_p1 as _D23S を main.py で 1 回、agent の最後で new = _D23S.apply(obs, configuration, action)。
  状態を持たない。例外の時は渡された行動をそのまま返す。
同じフォルダの postprocess_c1p_w9p.py(版 11/12 の土台、公式ルールの抜き出しを同梱)を import する。標準ライブラリのみ。
中身は評価に使った artifacts/evening_deposit_20260930/postprocess_d23s.py の apply_d23s・_d23s の写し(無変更)。
"""
from collections import Counter
from copy import deepcopy

import postprocess_c1p_w9p as P

BASE_PRICE = {'WHEAT': 25, 'CARROT': 35, 'TOMATO': 60, 'STRAWBERRY': 120, 'MELON': 250, 'EGG': 50, 'MILK': 160, 'WOOL': 200,
              'FERTILIZER': 100}                                          # 公式 MARKET_PARAMS の base
MIN_RATIO = 1.0                                                           # D23S_p1: 価格 / base >= 1.0 の品目だけ SELL を足す


def _norm(a):
    return list(a) if isinstance(a, list) and a else ['PASS']


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


def apply(obs, configuration, action, events=None):
    """提出物用の入口(D23S_p1 = min_ratio 1.0)。obs は公式観測(dict)、configuration は公式の configuration。"""
    return apply_d23s(obs, configuration, action, events=events, min_ratio=MIN_RATIO)
