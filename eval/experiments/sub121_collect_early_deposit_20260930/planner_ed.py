"""(sub121_collect_early_deposit_20260930: sub121_collect_ab_20260930/planner_cap.py の複製に「持ち物を抱える時間の罰」を足した版。
  品物の価値 = 値段 × 個数 × max(0, 1 − ALPHA × (倉庫に入れた step − 収穫した step))(開始時の持ち物は計画した step を収穫扱い)。ALPHA=0 で元と同じ価値。
  経路の途中に倉庫接点での DROP(疑似仕事 key=("D", 接点, k))を挟めるようにし、罰が減る分が寄り道の損を上回るときに挿入される。
  倉庫の上限の同時制約は途中の DROP を含む全預けに掛ける。先回り SELL の預け見込みも途中の DROP を含む。)
(元: sub121_travel_waste_20260930/planner.py の複製に倉庫の上限 100 の扱いと並列用の純関数 plan_step を足した版)
最終日(day 29、step 696〜718、日末処理なし)の回収だけを最大効率にする計画(複数 unit の orienteering)。
市場は無視(価格は固定値を渡す。売却による値下がりは考えない)。公式ルール(kaggriculture.py)に合わせた前提:
  - 移動は 1 手 1 マス(NORTH/SOUTH/EAST/WEST)、LOCKED にも入れる、unit 同士の衝突なし → 距離はマンハッタン距離
  - HARVEST 1 手でそのマスの yield_units を全部得る(畑は day - planted_day >= first_yield_day が要る)、
    動物マスは HARVEST(産物 yield_units 個)と COLLECT_FERTILIZER(fertilizer_available なら FERTILIZER 1)が別の 1 手
  - 非継続作物(WHEAT/CARROT/MELON)は WATER の手でその場で yield_units が +1(fertilized_until_day >= day なら +2、上限 max_yield)。
    条件: 今日まだ水やりしていない、(max_yield_day+1)//2 <= 経過日数 <= max_yield_day。→ 仕事は「HARVEST」と「WATER→HARVEST」の 2 種
    (FERTILIZE→WATER→HARVEST は計画では使わない。上限(緩和)には +2 として入れる)
  - 持ち物の上限なし(_inv_add に上限なし)。DROP は倉庫接点 (4,4),(5,4),(4,5),(5,5) で全持ち物を倉庫へ(倉庫 100 を超える分は捨てられる)
    → 実行時は倉庫の空きを見て、全部入るときだけ DROP、入らなければ価値の高い品物から PLACE <item> <n>(超過分は持ったまま)、空き 0 は待つ
  - step t の unit 行動の後に _decay_plants(step t) が走る: step >= max_lifespan_step かつ (step - mls) が偶数の手で yield_units -1
  - step 718 の後に日末処理は来ない → 持ち物は step <= last の DROP でしか倉庫に入らない
使い方(後付け処理): pp = CollectPlanner(start_step=708, prices=None); new_action = pp(obs, action)
    obs は公式観測(farms/private/step/player/market)、action は元の行動 dict。step >= start_step かつ day 29 のときだけ
    unit の手(farmer/hands)を計画の手に置き換える。market は元のまま。仕事のない unit は PASS。"""

LAST = 718
DAY29 = 696
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = {"WHEAT": (2, 4, 6, False), "CARROT": (2, 3, 4, False), "TOMATO": (8, 8, 4, True),
         "STRAWBERRY": (10, 10, 4, True), "MELON": (10, 12, 6, False)}  # first_yield_day, max_yield_day, max_yield, ongoing
PRODUCT_OF = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def dshed(p):
    return min(abs(p[0] - x) + abs(p[1] - y) for x, y in SHED)


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def ndecay(mls, a, b):
    """step a..b-1 のうち _decay_plants が yield を 1 減らす手の数。"""
    if mls < 0 or b <= a:
        return 0
    lo = max(a, mls)
    first = lo + ((lo - mls) % 2)
    return 0 if first > b - 1 else (b - 1 - first) // 2 + 1


class Job:
    """1 マスでの一続きの作業。key が同じ仕事(同じマスの HARVEST と WATER→HARVEST)は 1 つしか選べない。"""
    __slots__ = ('pos', 'ops', 'key', 'item', 'y0', 'mls', 'price', 'bonus', 'maxy', 'id')

    def __init__(self, pos, ops, key, item, y0, mls, price, bonus, maxy, id_):
        self.pos, self.ops, self.key, self.item, self.y0, self.mls = pos, ops, key, item, y0, mls
        self.price, self.bonus, self.maxy, self.id = price, bonus, maxy, id_

    def value_at(self, t, t0):
        """step t に最初の作業を始めたときに得る価値(観測は step t0)。"""
        y = self.y0 - ndecay(self.mls, t0, t)
        if y <= 0:
            return 0
        if self.bonus:
            y = min(self.maxy, y + self.bonus) - ndecay(self.mls, t, t + 1)
        return max(0, y) * self.price


def jobs_from_tiles(tiles, day, prices, fert_bonus=False):
    """tiles: {(x, y): tile}(公式の tile 表現)。fert_bonus=True は上限用(WATER の +1 を +2 と見なす)。"""
    out = []
    for (x, y), t in sorted(tiles.items()):
        if not isinstance(t, dict):
            continue
        if t.get('kind') == 'PLANT':
            fy, my, mx, ongoing = CROPS[t['crop']]; age = day - t['planted_day']
            if t.get('yield_units', 0) > 0 and age >= fy:
                pr = prices[t['crop']]
                out.append(Job((x, y), ('HARVEST',), (x, y, 'P'), t['crop'], t['yield_units'], t['max_lifespan_step'], pr, 0, mx, len(out)))
                if not ongoing and not t.get('watered_today') and (my + 1) // 2 <= age <= my:
                    b = 2 if (fert_bonus or t.get('fertilized_until_day', -1) >= day) else 1
                    out.append(Job((x, y), ('WATER', 'HARVEST'), (x, y, 'P'), t['crop'], t['yield_units'], t['max_lifespan_step'],
                                   pr, b, mx, len(out)))
        elif 'animal' in t:
            if t.get('yield_units', 0) > 0:
                it = PRODUCT_OF[t['animal']]
                out.append(Job((x, y), ('HARVEST',), (x, y, 'A'), it, t['yield_units'], -1, prices[it], 0, 99, len(out)))
            if t.get('fertilizer_available'):
                out.append(Job((x, y), ('COLLECT_FERTILIZER',), (x, y, 'F'), 'FERTILIZER', 1, -1, prices['FERTILIZER'], 0, 99, len(out)))
    return out


class Unit:
    __slots__ = ('idx', 'pos', 't', 'carry', 'n')

    def __init__(self, idx, pos, t, carry, n=0):
        self.idx, self.pos, self.t, self.carry, self.n = idx, tuple(pos), t, carry, n  # t = 次に手を打つ step、carry = 持ち物の価値、n = 持ち物の個数


# 倉庫の上限の同時制約: 同じ手に倉庫接点で預ける個数の合計 <= CTX['cap']。solve の間だけ有効(ends = unit → (預ける手, 個数))。
CTX = dict(cap=None, ends={}, load={})


ALPHA = 0.0
N_DROP_COPIES = 4  # 倉庫接点ごとの途中 DROP の疑似仕事の数


def is_drop(j):
    return j.ops == ('DROP',)


def route_deps(u, route, t0):
    """預けの列 [(預ける手, 個数)](途中の DROP と最後の預け。個数 0 は含めない)。"""
    t, p, n = u.t, u.pos, u.n
    out = []
    for j in route:
        t += man(p, j.pos)
        if is_drop(j):
            if n > 0:
                out.append((t, n))
            n = 0
        else:
            n += j.value_at(t, t0) // max(1, j.price)
        t += len(j.ops); p = j.pos
    if n > 0:
        out.append((t + dshed(p), n))
    return out


def route_end(u, route, t0):
    """最後の預け (手, 個数) または None(互換用)。"""
    d = route_deps(u, route, t0)
    return d[-1] if d else None


def _ctx_fits(u, route, t0):
    old = dict()
    for tt, nn in CTX['ends'].get(u.idx, []):
        old[tt] = old.get(tt, 0) + nn
    new = dict()
    for tt, nn in route_deps(u, route, t0):
        new[tt] = new.get(tt, 0) + nn
    return all(CTX['load'].get(tt, 0) - old.get(tt, 0) + nn <= CTX['cap'] for tt, nn in new.items())


def _ctx_set(u, route, t0):
    for tt, nn in CTX['ends'].pop(u.idx, []):
        CTX['load'][tt] -= nn
    d = route_deps(u, route, t0)
    if d:
        CTX['ends'][u.idx] = d
        for tt, nn in d:
            CTX['load'][tt] = CTX['load'].get(tt, 0) + nn


def _w(td, th):
    return max(0.0, 1.0 - ALPHA * (td - th))


def route_value(u, route, t0, last=LAST):
    """(価値, 実行可能か)。route は Job(途中 DROP を含む)の列 → 最後に最寄りの倉庫接点へ行き DROP(step <= last)。
    価値 = Σ 品物の値段×個数×max(0, 1 − ALPHA×(預けた手 − 収穫した手))。開始時の持ち物の収穫手は u.t。"""
    if not route:  # 何もしない(持ち物は戻れれば DROP)
        td = u.t + dshed(u.pos)
        return (u.carry * _w(td, u.t) if td <= last else 0), True
    t, p, v = u.t, u.pos, 0.0
    pend = [(u.carry, u.t)] if u.carry else []
    for j in route:
        t += man(p, j.pos)
        if is_drop(j):
            v += sum(x * _w(t, th) for x, th in pend); pend = []
        else:
            x = j.value_at(t, t0)
            if x:
                pend.append((x, t + len(j.ops) - 1))
        t += len(j.ops); p = j.pos
    td = t + dshed(p)
    v += sum(x * _w(td, th) for x, th in pend)
    ok = td <= last
    if ok and CTX['cap'] is not None:
        ok = _ctx_fits(u, route, t0)
    return v, ok


def _time(u, route):
    t, p = u.t, u.pos
    for j in route:
        t += man(p, j.pos) + len(j.ops); p = j.pos
    return t + dshed(p)


def solve(units, jobs, t0, last=LAST, init=None, max_rounds=6, cap=None):
    """貪欲挿入(価値差/追加手数 と 価値差 の 2 通り)+ 局所改善(抜いて入れ直し・同じマスの別作業・未割当との交換)。
    init(unit idx → Job 列)があればそれを起点にし、改善しか受け入れない(再計画で価値が下がらない)。"""
    starts = []
    if init is not None:
        starts.append(('init', init))
    starts += [('ratio', None), ('delta', None)]
    best_routes, best_total = None, -1
    try:
        for mode, r0 in starts:
            CTX['cap'] = cap; CTX['ends'] = {}; CTX['load'] = {}
            routes = {u.idx: [] for u in units}
            for u in units:
                _ctx_set(u, [], t0)
            if r0 is not None:  # 前回の計画の残り: 実行可能(同時制約込み)な範囲で先頭から戻す
                for u in units:
                    r = list(r0.get(u.idx, []))
                    while r and not route_value(u, r, t0, last)[1]:
                        r.pop()
                    routes[u.idx] = r; _ctx_set(u, r, t0)
            if mode != 'init':
                _greedy(units, jobs, routes, t0, last, mode)
            _improve(units, jobs, routes, t0, last, max_rounds)
            tot = sum(route_value(u, routes[u.idx], t0, last)[0] for u in units)
            if tot > best_total:
                best_total, best_routes = tot, routes
    finally:
        CTX['cap'] = None; CTX['ends'] = {}; CTX['load'] = {}
    return best_routes, best_total


_U = {}


def _cur(units, routes, t0, last):
    return {u.idx: route_value(u, routes[u.idx], t0, last)[0] for u in units}


def _greedy(units, jobs, routes, t0, last, mode):
    _U.clear(); _U.update({u.idx: u for u in units})
    used = {j.key for r in routes.values() for j in r}
    cur = _cur(units, routes, t0, last)
    while True:
        best = None
        for j in jobs:
            if j.key in used:
                continue
            for u in units:
                r = routes[u.idx]; base_t = _time(u, r)
                if u.t + man(u.pos, j.pos) + len(j.ops) + dshed(j.pos) > last:
                    continue
                for k in range(len(r) + 1):
                    nr = r[:k] + [j] + r[k:]
                    v, ok = route_value(u, nr, t0, last)
                    if not ok:
                        continue
                    dv = v - cur[u.idx]
                    if dv <= 0:
                        continue
                    sc = dv / max(1, _time(u, nr) - base_t) if mode == 'ratio' else dv
                    if best is None or sc > best[0]:
                        best = (sc, u.idx, k, j, v)
        if best is None:
            return
        _, ui, k, j, v = best
        routes[ui].insert(k, j); used.add(j.key); cur[ui] = v; _ctx_set(_U[ui], routes[ui], t0)


def _best_insert(units, routes, cur, j, t0, last):
    best = None
    for u in units:
        if u.t + man(u.pos, j.pos) + len(j.ops) + dshed(j.pos) > last:
            continue
        r = routes[u.idx]
        for k in range(len(r) + 1):
            nr = r[:k] + [j] + r[k:]
            v, ok = route_value(u, nr, t0, last)
            if ok and (best is None or v - cur[u.idx] > best[0]):
                best = (v - cur[u.idx], u.idx, k, v)
    return best


def _improve(units, jobs, routes, t0, last, max_rounds):
    byidx = {u.idx: u for u in units}
    variants = {}
    for j in jobs:
        variants.setdefault(j.key, []).append(j)
    for _ in range(max_rounds):
        changed = False
        cur = _cur(units, routes, t0, last)
        # 1) 抜いて(同じマスの別作業も候補に)最良位置へ入れ直し
        for ui in list(routes):
            k = 0
            while k < len(routes[ui]):
                j = routes[ui][k]; u = byidx[ui]
                rem = routes[ui][:k] + routes[ui][k + 1:]
                vrem, ok = route_value(u, rem, t0, last)
                if not ok:
                    k += 1; continue
                saved = routes[ui]; routes[ui] = rem; c2 = dict(cur); c2[ui] = vrem; _ctx_set(u, rem, t0)
                best = None
                for jv in variants.get(j.key, [j]):
                    b = _best_insert(units, routes, c2, jv, t0, last)
                    if b is not None and (best is None or b[0] > best[0][0]):
                        best = (b, jv)
                if best is not None and (vrem - cur[ui]) + best[0][0] > 1e-9:
                    b, jv = best
                    routes[b[1]].insert(b[2], jv); c2[b[1]] = b[3]; cur = c2; changed = True
                    _ctx_set(byidx[b[1]], routes[b[1]], t0)
                else:
                    routes[ui] = saved; k += 1; _ctx_set(u, saved, t0)
        # 2) 未割当の仕事: 空きに挿入、なければ同じ unit の 1 件と交換
        used = {j.key for r in routes.values() for j in r}
        for j in jobs:
            if j.key in used:
                continue
            b = _best_insert(units, routes, cur, j, t0, last)
            if b is not None and b[0] > 1e-9:
                routes[b[1]].insert(b[2], j); cur[b[1]] = b[3]; used.add(j.key); changed = True
                _ctx_set(byidx[b[1]], routes[b[1]], t0)
                continue
            best = None
            for ui, r in routes.items():
                u = byidx[ui]
                if u.t + man(u.pos, j.pos) + len(j.ops) + dshed(j.pos) > last:
                    continue
                for k in range(len(r)):
                    base = r[:k] + r[k + 1:]
                    for kk in range(len(base) + 1):
                        nr = base[:kk] + [j] + base[kk:]
                        v, ok = route_value(u, nr, t0, last)
                        if ok and v - cur[ui] > 1e-9 and (best is None or v - cur[ui] > best[0]):
                            best = (v - cur[ui], ui, nr, r[k])
            if best is not None:
                _, ui, nr, out = best
                routes[ui] = nr; cur[ui] = route_value(byidx[ui], nr, t0, last)[0]
                used.add(j.key); used.discard(out.key); changed = True; _ctx_set(byidx[ui], nr, t0)
        if not changed:
            return


def unit_bound(u, jobs, t0, last=LAST):
    """1 unit の緩和上限(他 unit との取り合い無視): 入るマス数(移動 >= 訪問マス数)+ 作業手数 <= last - u.t の群ナップサック、
    各仕事は最速到着時刻の価値(時間とともに減るだけなので最大)、マスごとに作業の組み合わせを 1 つ選ぶ。持ち物は戻れれば加算。"""
    cap = last - u.t
    if cap < 0:
        return 0
    groups = {}
    for j in jobs:
        d = man(u.pos, j.pos)
        if u.t + d + len(j.ops) + dshed(j.pos) > last:
            continue
        groups.setdefault(j.pos, {}).setdefault(j.key[2], []).append((len(j.ops), j.value_at(u.t + d, t0)))
    dp = [0] * (cap + 1)
    for pos, kinds in groups.items():
        enter = 0 if pos == u.pos else 1
        opts = []
        if 'P' in kinds:
            opts += [(enter + c, v) for c, v in kinds['P']]
        else:
            a = max(kinds.get('A', []), default=None); f = max(kinds.get('F', []), default=None)
            for x in (a, f):
                if x:
                    opts.append((enter + x[0], x[1]))
            if a and f:
                opts.append((enter + a[0] + f[0], a[1] + f[1]))
        nd = dp[:]
        for c, v in opts:
            for b in range(c, cap + 1):
                if dp[b - c] + v > nd[b]:
                    nd[b] = dp[b - c] + v
        dp = nd
    carry = u.carry if u.t + dshed(u.pos) <= last else 0
    return carry + dp[cap]


def joint_bound(units, jobs, t0, last=LAST):
    """全 unit の緩和上限 = 持ち物(戻れる分)+ min(各 unit の緩和上限の仕事分の和, マスごとに最良の作業の最速価値の和)。
    jobs は jobs_from_tiles(..., fert_bonus=True) を渡す(FERTILIZE→WATER の +2 を手数・肥料の入手なしで許す = 緩和)。"""
    carry = sum(u.carry for u in units if u.t + dshed(u.pos) <= last)
    s = sum(unit_bound(u, jobs, t0, last) for u in units) - carry
    best = {}
    for j in jobs:
        vs = [j.value_at(u.t + man(u.pos, j.pos), t0) for u in units
              if u.t + man(u.pos, j.pos) + len(j.ops) + dshed(j.pos) <= last]
        if vs:
            best[j.key] = max(best.get(j.key, 0), max(vs))
    return carry + min(s, sum(best.values()))


def first_action(u, route):
    """route の最初の手(公式の unit 行動)。"""
    if route:
        tgt = route[0].pos
        if u.pos == tgt:
            return [route[0].ops[0]]
    else:
        if u.carry <= 0:
            return ['PASS']
        if dshed(u.pos) == 0:
            return ['DROP']
        tgt = min(SHED, key=lambda s: (man(u.pos, s), s))
    (x, y), (tx, ty) = u.pos, tgt
    if tx != x:
        return ['EAST' if tx > x else 'WEST']
    return ['SOUTH' if ty > y else 'NORTH']


def carry_value(inv, prices):
    return sum(n * prices[k] for k, n in (inv or {}).items() if k in PRODUCTS and n > 0)


def units_from_obs(farm, private, t, prices):
    pos = [farm['farmer']] + list(farm['hands'])
    invs = private['inventories']
    return [Unit(i, p, t, carry_value(invs[i] if i < len(invs) else {}, prices),
                 sum(n for n in (invs[i] if i < len(invs) else {}).values() if n > 0)) for i, p in enumerate(pos)]


def tiles_of(farm):
    return {(x, y): t for y, row in enumerate(farm['tiles']) for x, t in enumerate(row)}


SHED_CAP = 100


def shed_deposit(inv, room, prices):
    """倉庫接点での預け方(倉庫の上限 100 を超えて捨てない)。戻り値 (行動, 入る個数)。
    全部入るなら DROP。入らなければ価値の高い品物から PLACE <item> <n>(公式 393〜409 行: 空きを超える分は持ったまま残る)。空きが 0 なら待つ(PASS)。"""
    items = {k: n for k, n in (inv or {}).items() if n > 0}
    tot = sum(items.values())
    if tot <= room:
        return ['DROP'], tot
    if room <= 0:
        return ['PASS'], 0
    k = max(items, key=lambda k: (prices.get(k, 0), k))
    n = min(items[k], room)
    return ['PLACE', k, n], n


def sell_ahead(level, dep_now, dep_future, shed_after, prices, t, last=LAST, cap=SHED_CAP):
    """倉庫あふれの先回り売り(最終日、2 手前から)。level = この手の頭の倉庫の個数、dep_now = この手に預ける個数、
    dep_future = {手: 預ける見込みの個数}、shed_after = この手の預け後の倉庫の中身。元の注文の売りは当てにしない(保守的)。
    x_t >= max(t+1 のあふれ, ceil(t+2 のあふれ / 2))。品目は価格の低い順(同価格は品目名順)。t = last では売っても救えないので売らない。"""
    if t >= last:
        return []
    o1 = level + dep_now + dep_future.get(t + 1, 0) - cap
    o2 = level + dep_now + dep_future.get(t + 1, 0) + dep_future.get(t + 2, 0) - cap
    x = max(0, o1, -(-o2 // 2))
    out = []
    for k in sorted((k for k, n in shed_after.items() if n > 0 and k in PRODUCTS), key=lambda k: (prices.get(k, 0), k)):
        if x <= 0:
            break
        n = min(x, shed_after[k]); out.append(['SELL', k, n]); x -= n
    return out


def drop_jobs(n0):
    """途中 DROP の疑似仕事(値は 0、倉庫接点で 1 手)。"""
    return [Job(s, ('DROP',), ('D', s, k), None, 0, -1, 0, 0, 0, n0 + i * N_DROP_COPIES + k)
            for i, s in enumerate(SHED) for k in range(N_DROP_COPIES)]


def plan_step(state, prev, prices, last=LAST, shed_cap=SHED_CAP, alpha=None):
    """1 手ぶんの計画(純関数、並列用)。state = dict(step, farmer, hands, tiles(10x10), inventories, shed)。
    prev = 前回の計画の残り {unit idx: [(key, ops), ...]} または None。戻り値 (unit 行動の列, 先回り売りの注文, 次の prev, 情報)。
    計画には「同じ手に預ける個数の合計 <= 100 − 今の倉庫の個数」の同時制約を入れる。"""
    global ALPHA
    if alpha is not None:
        ALPHA = float(alpha)
    t = int(state['step'])
    farm = dict(farmer=state['farmer'], hands=state['hands'], tiles=state['tiles'])
    priv = dict(inventories=state['inventories'])
    units = units_from_obs(farm, priv, t, prices)
    jobs = jobs_from_tiles(tiles_of(farm), t // 24, prices)
    if ALPHA > 0:
        jobs += drop_jobs(len(jobs))
    level = sum(v for v in state['shed'].values())
    init = None
    if prev is not None:
        by = {}
        for j in jobs:
            by.setdefault(j.key, {})[j.ops] = j
        init = {}
        for ui, r in prev.items():
            nr = []
            for key, ops in r:
                v = by.get(key)
                if v:
                    nr.append(v.get(ops) or v.get(ops[-1:]) or next(iter(v.values())))
            init[ui] = nr
    routes, total = solve(units, jobs, t, last, init=init, cap=max(0, shed_cap - level))
    room = shed_cap - level
    acts, nxt, info = [], {}, dict(deposit=0, place=0, wait_full=0, pred=total)
    invs = state['inventories']; shed_after = dict(state['shed']); dep_future = {}
    for u in units:
        r = routes.get(u.idx, [])
        a = first_action(u, r)
        inv = invs[u.idx] if u.idx < len(invs) else {}
        if a[0] == 'DROP' or (not r and dshed(u.pos) == 0 and any(n > 0 for n in inv.values())):
            a, n = shed_deposit(inv, room, prices)
            room -= n; info['deposit'] += n
            info['place'] += a[0] == 'PLACE'; info['wait_full'] += a[0] == 'PASS'
            if a[0] == 'DROP':
                for k, m in inv.items():
                    shed_after[k] = shed_after.get(k, 0) + max(0, m)
            elif a[0] == 'PLACE':
                shed_after[a[1]] = shed_after.get(a[1], 0) + a[2]
            rest = sum(m for m in inv.values() if m > 0) - n
            if rest > 0:
                dep_future[t + 1] = dep_future.get(t + 1, 0) + rest
        else:
            for e in route_deps(u, r, t):
                if e[0] > t:
                    dep_future[e[0]] = dep_future.get(e[0], 0) + e[1]
        acts.append(a)
        done = r and u.pos == r[0].pos and len(r[0].ops) == 1
        nxt[u.idx] = [(j.key, j.ops) for j in (r[1:] if done else r)]
    sells = sell_ahead(level, info['deposit'], dep_future, shed_after, prices, t, last, shed_cap)
    info['sell_ahead'] = sum(o[2] for o in sells); info['dep_future'] = {str(k): v for k, v in sorted(dep_future.items())}
    return acts, sells, nxt, info


def merge_market(mk, sells):
    """先回り売りを market に入れる: [] の枠 → 末尾の空き(10 件まで)→ 元の注文を後ろから置き換え。戻り値 (新しい注文, 消した元の注文)。"""
    mk = [list(o) if isinstance(o, list) else o for o in (mk or [])]
    removed = []
    for o in sells:
        for i, x in enumerate(mk):
            if isinstance(x, list) and not x:
                mk[i] = o; break
        else:
            if len(mk) < 10:
                mk.append(o)
            else:
                for i in range(len(mk) - 1, -1, -1):
                    if mk[i] not in sells:
                        removed.append(mk[i]); mk[i] = o; break
    return mk, removed


class CollectPlanner:
    """後付け処理(1 試合 1 インスタンス)。day 29 の start_step 以降、unit の手を計画に置き換える(market は元のまま)。
    倉庫の上限: 計画に同じ手の預け個数の同時制約、預けは shed_deposit(捨てない)、あふれ見込みは 2 手前から sell_ahead で先に売る
    (market へは merge_market)。価格は最初の呼び出しの観測価格で固定(売却による値下がりは無視)。"""

    def __init__(self, start_step=DAY29, prices=None, last=LAST):
        self.start, self.prices, self.last = start_step, prices, last
        self.prev = None

    def __call__(self, obs, action):
        t = int(obs['step'])
        if t < max(self.start, DAY29) or t > LAST:
            return action
        if self.prices is None:
            self.prices = dict(obs['market']['prices'])
        p = obs['player']; farm = obs['farms'][p]; priv = obs['private']
        state = dict(step=t, farmer=farm['farmer'], hands=farm['hands'], tiles=farm['tiles'],
                     inventories=priv['inventories'], shed=priv['shed'])
        acts, sells, self.prev, _ = plan_step(state, self.prev, self.prices, self.last)
        out = dict(action) if isinstance(action, dict) else {}
        out['farmer'] = acts[0]; out['hands'] = acts[1:]
        if sells:
            out['market'], _ = merge_market(out.get('market'), sells)
        return out
