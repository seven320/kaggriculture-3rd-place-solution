"""市場・雇用・土地の注文だけを変える後付け処理(unit の手は変えない)。公式観測(dict)と行動を受け取り、(新しい行動, events) を返す。
flags:
  M1  段階売り(sub123s の _pace と同じ式): 710<=step<=717 で market を
      [SELL 品目 ceil(納屋在庫/残り手数)](価値順、在庫>0) + 元の注文のうち SELL・BUY_* 以外(HIRE・BUY_LAND)に置き換える。
      残り手数 = 718-step+1。step 718 は提出物の _liq(SELL 1000 × 9)のまま。
  M1x 段階売りの「網より少なく売らない」版: 710<=step<=717 で品目ごとに q=ceil(納屋在庫/残り手数)。網がその品目を SELL していれば
      数量を max(網, q) に上げ、していなければ SELL q を [] の枠(前から)→ 末尾(10 枠まで)に足す。網の他の注文(BUY・HIRE 等)と位置は変えない。
  M2  元が取れない購入の取り消し(公式ルールの日程で判定):
      BUY_SEED c: d0=(step+1)//24 として d0+first_yield_day > 29(最終日までに 1 度も収穫できない)
      BUY_ANIMAL a: d0 に置いて第 28 日の日替わりまでの生産回数 P、2P×その品の現在価格 + 肥料 (29-d0)×肥料の現在価格 < 購入額
      BUY_PRODUCT: step>=696(第 29 日。日替わりが来ないので餌・肥料が生産に効かない)
  M3  hour 23 または step 718 の HIRE の取り消し(雇った hand は 1 手も動けない)
  M5  hour 23 の W9 の空き枠詰め: 網は market に [](空の注文、枠の位置合わせ用)を出す。W9(waste_replace2._w9_orders)は
      「10 − 注文数(空も数える)」までしか SELL を足せないので、[] で 10 枠が埋まると日末に倉庫から溢れる携行品を売って空けられず、
      公式 _drop_inventories_to_shed で捨てられる。[] を除いた注文列で W9 を計算し直し、足りない分の SELL を [] の枠に入れる
      (他の注文の位置は変えない)。計算は run_m.MTop(waste_replace2 を使う)、置き方は fill_empty。
公式根拠: kaggriculture.py の _process_market(unit 行動の後に市場)、_end_of_day(hands=[]、第 29 日の日替わりは step 719 で来ない)、
_daily_refresh_plants/_animals(生産は日替わり)、HARVEST の first_yield_day 条件。"""
import math

VALUE_ORDER = ['STRAWBERRY', 'MELON', 'MILK', 'WOOL', 'TOMATO', 'FERTILIZER', 'EGG', 'CARROT', 'WHEAT']   # sub123s の _VALUE_ORDER
CROP_FYD = {'WHEAT': 2, 'CARROT': 2, 'TOMATO': 8, 'STRAWBERRY': 10, 'MELON': 10}
ANIMALS = {'GOOSE': (300, 4, 1, 'EGG'), 'COW': (400, 8, 2, 'MILK'), 'SHEEP': (500, 6, 3, 'WOOL')}   # cost, first_yield_day, interval, product
LIQ_FROM, LAST = 710, 718


def productions(d0, animal):
    _, fyd, iv, _ = ANIMALS[animal]; first = d0 + fyd - 1
    return 0 if first > 28 else (28 - first) // iv + 1


def dead_order(o, step, price_of):
    op = o[0] if isinstance(o, list) and o else None
    d0 = (step + 1) // 24
    if op == 'BUY_SEED' and len(o) > 1 and o[1] in CROP_FYD:
        return d0 + CROP_FYD[o[1]] > 29
    if op == 'BUY_ANIMAL' and len(o) > 1 and o[1] in ANIMALS:
        cost, _, _, prod = ANIMALS[o[1]]
        return 2 * productions(d0, o[1]) * price_of(prod) + max(0, 29 - d0) * price_of('FERTILIZER') < cost
    if op == 'BUY_PRODUCT':
        return step >= 696
    return False


def fill_empty(market, extra):
    """extra の SELL を market の [] の枠へ前から順に入れる(他の注文の位置は変えない)。"""
    out = [list(o) if isinstance(o, list) else o for o in market]; k = 0
    for i, o in enumerate(out):
        if k < len(extra) and isinstance(o, list) and not o:
            out[i] = list(extra[k]); k += 1
    assert k == len(extra), (market, extra)
    return out


def needs_obs(action, step, flags):
    """公式観測(pool.debug)が要る手か。要らない手は行動を変えない。"""
    m = action.get('market') or []
    if 'M5' in flags and step % 24 == 23 and any(isinstance(o, list) and not o for o in m):
        return True
    if ('M1' in flags or 'M1x' in flags) and LIQ_FROM <= step < LAST:
        return True
    if 'M2' in flags and any(isinstance(o, list) and o and o[0] in ('BUY_SEED', 'BUY_ANIMAL', 'BUY_PRODUCT') for o in m):
        return True
    if 'M3' in flags and (step % 24 == 23 or step == LAST) and any(isinstance(o, list) and o and o[0] == 'HIRE' for o in m):
        return True
    return False


def apply(obs, action, flags, market_price):
    step = int(obs['step']); events = []
    old = [list(o) for o in (action.get('market') or []) if isinstance(o, list)]
    mk = list(old)
    if 'M2' in flags:
        inv = obs['market']['inventory']
        keep = []
        for o in mk:
            if dead_order(o, step, lambda it: market_price(it, inv[it])):
                events.append(dict(kind='M2', step=step, order=o))
            else:
                keep.append(o)
        mk = keep
    if 'M3' in flags and (step % 24 == 23 or step == LAST):
        keep = []
        for o in mk:
            if o and o[0] == 'HIRE':
                events.append(dict(kind='M3', step=step, order=o, hires_today=int(obs['farms'][obs['player']].get('hires_today', 0))))
            else:
                keep.append(o)
        mk = keep
    if 'M1' in flags and LIQ_FROM <= step < LAST:
        shed = obs['private']['shed']; left = LAST - step + 1
        sells = [['SELL', it, -(-int(shed.get(it, 0)) // left)] for it in VALUE_ORDER if int(shed.get(it, 0)) > 0]
        rest = [o for o in mk if o and o[0] != 'SELL' and not str(o[0]).startswith('BUY')]
        new = (sells + rest)[:10]
        if new != mk:
            events.append(dict(kind='M1', step=step, before=mk, after=new,
                               dropped_buy=[o for o in mk if o and str(o[0]).startswith('BUY')]))
        mk = new
    if 'M1x' in flags and LIQ_FROM <= step < LAST:
        shed = obs['private']['shed']; left = LAST - step + 1
        new = [list(o) for o in mk]; add = []
        for it in VALUE_ORDER:
            n = int(shed.get(it, 0))
            if n <= 0:
                continue
            q = -(-n // left)
            idx = [i for i, o in enumerate(new) if len(o) >= 3 and o[0] == 'SELL' and o[1] == it]
            if idx:
                i = idx[0]
                if int(new[i][2]) < q:
                    new[i][2] = q
            else:
                add.append(['SELL', it, q])
        for o in add:
            e = [i for i, x in enumerate(new) if not x]
            if e:
                new[e[0]] = o
            elif len(new) < 10:
                new.append(o)
        if new != mk:
            events.append(dict(kind='M1x', step=step, before=mk, after=new))
        mk = new
    if mk == old:
        return action, []
    out = dict(action); out['market'] = mk
    return out, events
