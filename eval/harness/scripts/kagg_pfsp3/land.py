"""Land4 intervention and receipt audit. Rust remains the only environment.

Default constants below are pinned with engine/constants.rs. The audit models
inventory-affecting operations only, then checks the actual Rust observations.
It never supplies a next environment state or guesses a purchase from demand.
"""
from collections import Counter
from copy import deepcopy
import math
import statistics
import numpy as np
import torch

PACKAGES = {
    'STRAWBERRY15': [('BUY_SEED', 'STRAWBERRY', 15)],
    'SHEEP6': [('BUY_ANIMAL', 'SHEEP', 6), ('BUY_PRODUCT', 'WHEAT', 6)],
    'TOMATO10': [('BUY_SEED', 'TOMATO', 10)],
}
CROPS = dict(zip(('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON'),
                 ((10, 2, 4, 6, False), (20, 2, 3, 4, False),
                  (50, 8, 8, 4, True), (100, 10, 10, 4, True), (80, 10, 12, 6, False))))
ANIMALS = {'GOOSE': (300, 'COOP', 'EGG'), 'COW': (400, 'PASTURE', 'MILK'),
           'SHEEP': (500, 'PASTURE', 'WOOL')}
PRICE_PARAMS = dict(zip(('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON',
                        'EGG', 'MILK', 'WOOL', 'FERTILIZER'),
    ((25,400,'sqrt',.8,'log',.2), (35,450,'hinge',1.,'sqrt',.7),
     (60,200,'hinge',.4,'sqrt',.6), (120,100,'sqrt',.7,'linear',1.6),
     (250,300,'log',.2,'sq',3.6), (50,332,'hinge',.4,'log',.2),
     (160,122,'sqrt',.6,'linear',1.6), (200,105,'log',.2,'sq',3.2),
     (100,200,'linear',.4,'linear',.4))))
INVENTORY_ITEMS = tuple(PRICE_PARAMS) + tuple(ANIMALS)  # Native ou[..., 3:15].


class ReceiptError(RuntimeError):
    pass


def ratio(a, b):
    return a / b if b else None


def audit_money(value):
    """Match the bridge's number(): integral and strictly below 9e15."""
    return int(value) if abs(value) < 9e15 and value % 1 == 0 else value


def schedule_settings(land_schedule=None):
    """Resolve study.json settings without mutating them; old studies use v1."""
    cfg=dict(variant='v1',bonus_peak=.1,bonus_zero_iteration=40,p_mode='r_update',
             p_initial=.5,p_zero_from=81,supply_until=30,
             aux_plant_se_peak=0.,aux_plant_se_zero_iteration=40,
             aux_hire_buyseed_se_peak=0.,aux_hire_buyseed_se_zero_iteration=40)
    if land_schedule is not None: cfg.update(land_schedule)
    if cfg['variant'] not in ('v1','v2','v3','v3c') or cfg['p_mode'] not in ('r_update','fixed'):
        raise ValueError('invalid land schedule variant/p_mode')
    if not math.isfinite(cfg['bonus_peak']) or cfg['bonus_peak'] < 0:
        raise ValueError('invalid land bonus_peak')
    if not math.isfinite(cfg['p_initial']) or not 0 <= cfg['p_initial'] <= .5:
        raise ValueError('invalid land p_initial')
    for key,minimum in (('bonus_zero_iteration',2),('p_zero_from',1),('supply_until',0)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f'invalid land {key}')
    peak=cfg.get('aux_buy_land_peak',0.)
    zero=cfg.get('aux_buy_land_zero_iteration',60)
    if not math.isfinite(peak) or peak < 0: raise ValueError('invalid aux_buy_land_peak')
    if type(zero) is not int or zero < 2: raise ValueError('invalid aux_buy_land_zero_iteration')
    for name in ('aux_plant_se','aux_hire_buyseed_se'):
        peak=cfg[name+'_peak']; zero=cfg[name+'_zero_iteration']
        if not math.isfinite(peak) or peak < 0: raise ValueError(f'invalid {name}_peak')
        if type(zero) is not int or zero < 2: raise ValueError(f'invalid {name}_zero_iteration')
    peak=cfg.get('se_harvest_peak',0.)
    zero=cfg.get('se_harvest_zero_iteration',40); start=cfg.get('se_harvest_start_iteration',1)
    if not math.isfinite(peak) or peak < 0: raise ValueError('invalid se_harvest_peak')
    if type(zero) is not int or type(start) is not int or start < 1 or zero <= start:
        raise ValueError('invalid se_harvest_zero_iteration/se_harvest_start_iteration')
    return cfg


def se_harvest_coefficient(n, land_schedule=None):
    """Reward added to the learner's step reward for each harvest on the fourth (south-east) quadrant
    (user 2026-09-19: buying is settled, using it is not). se_harvest_peak at se_harvest_start_iteration,
    linear to 0 at se_harvest_zero_iteration, held at the peak before the start iteration."""
    cfg=schedule_settings(land_schedule)
    if (type(n) is not int or n < 1): raise ValueError('invalid iteration')
    zero=cfg.get('se_harvest_zero_iteration',40); start=cfg.get('se_harvest_start_iteration',1)
    return cfg.get('se_harvest_peak',0.) * min(1.,max(0.,(zero-n)/(zero-start)))


def aux_buy_land_coefficient(n, land_schedule=None):
    cfg=schedule_settings(land_schedule)
    if (type(n) is not int or n < 1): raise ValueError('invalid iteration')
    zero=cfg.get('aux_buy_land_zero_iteration',60)
    return cfg.get('aux_buy_land_peak',0.) * max(0.,(zero-n)/(zero-1))


def aux_plant_se_coefficient(n, land_schedule=None):
    cfg=schedule_settings(land_schedule)
    if (type(n) is not int or n < 1): raise ValueError('invalid iteration')
    zero=cfg['aux_plant_se_zero_iteration']
    return cfg['aux_plant_se_peak'] * max(0.,(zero-n)/(zero-1))


def aux_hire_buyseed_se_coefficient(n, land_schedule=None):
    cfg=schedule_settings(land_schedule)
    if (type(n) is not int or n < 1): raise ValueError('invalid iteration')
    zero=cfg['aux_hire_buyseed_se_zero_iteration']
    return cfg['aux_hire_buyseed_se_peak'] * max(0.,(zero-n)/(zero-1))


SE_INDEX = [y*10+x for y in range(5,10) for x in range(5,10)]


def plant_se_mask(g, ou, unit_log_probs, actor_rows):
    """Eligible (row, unit) pairs, using the current seed-capped distribution."""
    land4=torch.round(g[:,39]*4)==4
    return (actor_rows.bool()[:,None] & land4[:,None] & (ou[:,:,0]>0)
        & (ou[:,:,1]>=5) & (ou[:,:,1]<=9) & (ou[:,:,2]>=5) & (ou[:,:,2]<=9)
        & (unit_log_probs[:,:,15:20]>-1e3).any(-1))


def hire_buyseed_se_mask(g, tiles, actor_rows):
    """Eligible rows with an empty tile in the unlocked south-east quadrant."""
    land4=torch.round(g[:,39]*4)==4
    se_empty_any=(tiles[:,0,SE_INDEX,0]==0).any(-1)
    return actor_rows.bool() & land4 & se_empty_any


def schedule(n, p_next=.5, *, smoke=False, land_schedule=None):
    cfg=schedule_settings(land_schedule)
    if (type(n) is not int or n < 1) or not math.isfinite(p_next) or not 0 <= p_next <= .5:
        raise ValueError('invalid iteration/p')
    return dict(p=0. if n >= cfg['p_zero_from'] else cfg['p_initial'] if n <= 10 or cfg['p_mode']=='fixed' else float(p_next),
                b=cfg['bonus_peak'] * max(0., (cfg['bonus_zero_iteration']-n)/(cfg['bonus_zero_iteration']-1)),
                supply=n < cfg['supply_until'],warmup=n <= 2 and not smoke)


def update_p(n, p, counts, last_update, *, land_schedule=None):
    """Set the next iteration's p; block monitoring is independent of p_mode."""
    cfg=schedule_settings(land_schedule)
    if n >= cfg['p_zero_from']-1:
        return 0., n if n % 10 == 0 else last_update
    if cfg['p_mode']=='fixed':
        return cfg['p_initial'], n if n % 10 == 0 and last_update < n else last_update
    if n % 10 == 0 and last_update < n:
        r = ratio(counts['natural'], counts['unassigned'])
        return (p if r is None else float(np.clip(cfg['p_initial']-r, 0, cfg['p_initial']))), n
    return p, last_update


def assignment(seed, n, learner_player, p, learning_seed=29):
    if not math.isfinite(p) or not 0 <= p <= .5:
        raise ValueError('invalid p')
    def stream(i):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [20260918, learning_seed, n, seed, i])))
    return dict(target_player=int(stream(1).integers(2)) if learner_player is None else learner_player,
                assigned=bool(stream(2).random() < p), D=int(stream(3).integers(8,19)),
                package=list(PACKAGES)[int(stream(4).integers(3))])


def land_count(obs):
    return len(obs['farms'][obs['player']]['unlocked_quadrants'])


def buy_land_eligible(observations, *, lands=None, money=None):
    """Same pre-step quantities and sources as LandController.before_step.

    Keep the state flag separate from learner/forced, which are update masks.
    Native money is passed in float64 without a feature-based estimate.
    """
    counts=tuple(map(land_count,observations)) if lands is None else lands
    return [bool(int(counts[p]) == 3
        and float(observations[p]['farms'][p]['money'] if money is None else money[p]) >= 4000
        and 8 <= observations[p]['day'] <= 18) for p in (0,1)]


def inventory(obs, kind, item):
    pr = obs['private']
    if kind == 'BUY_SEED':
        return pr['seeds'].get(item, 0)
    return pr['shed'].get(item, 0) + sum(v.get(item,0) for v in pr['inventories'])


def market_price(item, count):
    base, scale, below, bt, above, at = PRICE_PARAMS[item]
    def shape(s, x):
        return {'linear': lambda: x, 'sqrt': lambda: math.sqrt(x), 'sq': lambda: x*x,
                'log': lambda: math.log(1+x), 'hinge': lambda: x/scale+8*max(0,x/scale-1)**2}[s]()
    low = count < 10000
    f, target, sign = (below, bt, 1) if low else (above, at, -1)
    return max(1, round(base + sign*base*target*shape(f,abs(count-10000))/shape(f,scale)))


def positive(d):
    return {k: int(v) for k,v in d.items() if v}


def ordered_inventory(values, order=None):
    vals = positive(values)
    if order is None:
        if len(vals) > 1:
            raise ReceiptError('inventory insertion order unavailable')
        order = list(vals)
    if set(order) != set(vals) or len(order) != len(vals):
        raise ReceiptError('inventory order does not match contents')
    return {k: vals[k] for k in order}


def _take(inv, item, n):
    if inv.get(item,0) < n:
        return False
    inv[item] = inv.get(item,0)-n
    if inv[item] == 0:
        inv.pop(item, None)
    return True


def _add(inv, item, n):
    if n:
        inv[item] = inv.get(item,0)+n


def _drop(inv, shed, capacity):
    for item, n in inv.items():
        _add(shed,item,min(n,max(0,capacity-sum(shed.values()))))
    inv.clear()


def _unit_phase(obs, action, orders=None, capacity=100):
    """Audit sequential unit effects before market; no random/environment step."""
    pr = obs['private'].copy(); farm = obs['farms'][obs['player']].copy()
    pr['seeds'] = pr['seeds'].copy(); pr['shed'] = pr['shed'].copy()
    if 'unlocked_quadrants' in farm: farm['unlocked_quadrants'] = farm['unlocked_quadrants'].copy()
    farm['tiles'] = farm['tiles'].copy()
    copied_rows = set(); copied_tiles = set()
    invs = [ordered_inventory(v, None if orders is None else orders[i])
            for i,v in enumerate(pr['inventories'])]
    units = [action.get('farmer',['PASS']), *action.get('hands',[])]
    positions = [farm['farmer'], *farm['hands']]
    while len(invs) < len(positions):
        invs.append({})
    demand = Counter(a[1] for a in units if a and a[0] == 'PLANT')
    blocked = {c for c,n in demand.items() if n > pr['seeds'].get(c,0)}
    harvested = []
    for i, (pos,a) in enumerate(zip(positions,units)):
        if not a:
            continue
        op = a[0]; x,y = pos
        # 同じ区画の複数 unit は、先行 unit が更新した複製を読む。
        if op not in ('PASS','NORTH','SOUTH','EAST','WEST','DROP','PICKUP'):
            if y not in copied_rows:
                farm['tiles'][y] = farm['tiles'][y].copy(); copied_rows.add(y)
            if (x,y) not in copied_tiles:
                tile = farm['tiles'][y][x]
                if isinstance(tile,dict): farm['tiles'][y][x] = tile.copy()
                copied_tiles.add((x,y))
        tile = farm['tiles'][y][x]; inv = invs[i]
        shed = pr['shed']; adjacent = x in (4,5) and y in (4,5)
        if op in ('PASS','NORTH','SOUTH','EAST','WEST'):
            continue  # a unit cannot move and use an item in the same step
        if op == 'DROP':
            if adjacent: _drop(inv,shed,capacity)
            continue
        if op == 'PICKUP':
            if adjacent:
                n = min(int(a[2]),shed.get(a[1],0)); _take(shed,a[1],n); _add(inv,a[1],n)
            continue
        if op == 'PLACE':
            item = a[1]
            if item in ANIMALS and isinstance(tile,dict) and tile.get('kind') == ANIMALS[item][1] and not tile.get('animal'):
                if _take(inv,item,1):
                    tile.update(animal=item,placed_day=obs['day'],yield_units=0,fed_today=False,
                                cared_today=False,fertilizer_available=False)
            elif adjacent:
                n = min(int(a[2]),inv.get(item,0),max(0,capacity-sum(shed.values())))
                _take(inv,item,n); _add(shed,item,n)
            continue
        if tile == 'LOCKED':
            continue
        if op == 'PLANT':
            item = a[1]
            if tile is None and item not in blocked and pr['seeds'].get(item,0):
                pr['seeds'][item] -= 1
                farm['tiles'][y][x] = dict(crop=item,planted_day=obs['day'],yield_units=0,
                                         watered_today=False,fertilized_until_day=-1)
        elif op == 'HARVEST' and isinstance(tile,dict):
            item = tile.get('crop'); n = tile.get('yield_units',0)
            if item and n and obs['day']-tile['planted_day'] >= CROPS[item][1]:
                _add(inv,item,n); tile['yield_units'] = 0
                harvested.append((x,y,item,n))
                if not CROPS[item][4]: farm['tiles'][y][x] = None
            elif tile.get('animal') and n:
                item = ANIMALS[tile['animal']][2]; _add(inv,item,n); tile['yield_units'] = 0
                harvested.append((x,y,item,n))
        elif op == 'WATER' and isinstance(tile,dict) and tile.get('crop') and not tile.get('watered_today'):
            tile['watered_today'] = True; c = CROPS[tile['crop']]; age = obs['day']-tile['planted_day']
            if not c[4] and math.ceil(c[2]/2) <= age <= c[2]:
                tile['yield_units'] = min(c[3],tile.get('yield_units',0)+(2 if tile.get('fertilized_until_day',-1) >= obs['day'] else 1))
        elif op == 'FEED' and isinstance(tile,dict) and tile.get('animal') and not tile.get('fed_today'):
            if _take(inv,'WHEAT',1): tile['fed_today'] = True
        elif op == 'FERTILIZE' and isinstance(tile,dict) and tile.get('crop'):
            if _take(inv,'FERTILIZER',1): tile['fertilized_until_day'] = max(tile.get('fertilized_until_day',-1),obs['day']+2)
        elif op == 'COLLECT_FERTILIZER' and isinstance(tile,dict) and tile.get('animal') and tile.get('fertilizer_available'):
            _add(inv,'FERTILIZER',1); tile['fertilizer_available'] = False
        elif op == 'DIG' and tile is not None and not (isinstance(tile,dict) and tile.get('animal')):
            farm['tiles'][y][x] = None
        elif op in ('BUILD_COOP','BUILD_PASTURE') and tile is None:
            farm['tiles'][y][x] = {'kind':op[6:]}
    pr['inventories'] = invs
    return pr, farm, harvested


def material_receipts(pre, post, executed_action, *, inventory_orders=None, capacity=100):
    """Return per-player BUY_* success counts, with both markets/slots audited.

    pre/post each contain two full observations. Without known insertion order,
    ambiguous multi-item carried inventory is a collection error, even when a
    particular outcome happens to look plausible. This deliberately fails closed.
    """
    if len(pre) != 2 or len(post) != 2 or len(executed_action) != 2:
        raise ReceiptError('both players and their executed actions required')
    pr, farms = [], []
    for p in (0,1):
        a,f,_ = _unit_phase(pre[p],executed_action[p],None if inventory_orders is None else inventory_orders[p],capacity)
        pr.append(a); farms.append(f)
    market = dict(pre[0]['market']['inventory'])
    for item, price in pre[0]['market']['prices'].items():
        if market_price(item,market[item]) != price:
            raise ReceiptError('unsupported market parameters')
    receipts = [Counter(),Counter()]
    orders = [a.get('market',[])[:10] for a in executed_action]
    for slot in range(max(map(len,orders), default=0)):
        pending = [None,None]
        for p in (0,1):
            a = orders[p][slot] if slot < len(orders[p]) else []
            if not a: continue  # preserve empty slots
            if a[0] == 'HIRE':
                aa,bb = 1,1
                for _ in range(farms[p]['hires_today']): aa,bb = bb,aa+bb
                if farms[p]['money'] >= aa:
                    farms[p]['money'] -= aa; farms[p]['hires_today'] += 1
                    pr[p]['inventories'].append({})
            elif a[0] == 'BUY_LAND':
                n = len(farms[p]['unlocked_quadrants'])
                if n < 4 and farms[p]['money'] >= (1000,2000,4000)[n-1]:
                    farms[p]['money'] -= (1000,2000,4000)[n-1]
                    farms[p]['unlocked_quadrants'].append(('NW','NE','SW','SE')[n])
            elif len(a) == 3 and a[2] > 0:
                if a[0] not in ('SELL','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL'):
                    raise ReceiptError('unknown market command')
                pending[p] = list(a)
        for _ in range(99999):
            quotes = [None,None]
            for p,a in enumerate(pending):
                if a is None: continue
                op,item,_ = a
                if op == 'SELL' and item in PRICE_PARAMS: quotes[p] = market_price(item,market[item])
                elif op == 'BUY_PRODUCT' and item in ('WHEAT','FERTILIZER'): quotes[p] = market_price(item,market[item]-1)
                elif op == 'BUY_SEED' and item in CROPS: quotes[p] = CROPS[item][0]
                elif op == 'BUY_ANIMAL' and item in ANIMALS: quotes[p] = ANIMALS[item][0]
                else: pending[p] = None
            if all(q is None for q in quotes): break
            for p, price in enumerate(quotes):
                if price is None: continue
                op,item,n = pending[p]; shed = pr[p]['shed']; success = False
                if op == 'SELL':
                    success = _take(shed,item,1)
                    if success:
                        farms[p]['money'] += price
                        if price > 1: market[item] += 1
                elif farms[p]['money'] >= price and (op == 'BUY_SEED' or sum(shed.values()) < capacity):
                    success = True; farms[p]['money'] -= price
                    _add(pr[p]['seeds'] if op == 'BUY_SEED' else shed,item,1)
                    receipts[p][(op,item)] += 1
                    if op == 'BUY_PRODUCT': market[item] -= 1
                pending[p] = [op,item,n-1] if success and n > 1 else None
    if (pre[0]['step']+1) % 24 == 0:
        for p in (0,1):
            for inv in pr[p]['inventories']: _drop(inv,pr[p]['shed'],capacity)
            pr[p]['inventories'] = [{}]
    for p in (0,1):
        actual = post[p]['private']
        for k in ('shed','seeds'):
            if positive(pr[p][k]) != positive(actual[k]):
                raise ReceiptError(f'player {p} {k} audit mismatch: {positive(pr[p][k])} != {positive(actual[k])}')
        if [positive(v) for v in pr[p]['inventories']] != [positive(v) for v in actual['inventories']]:
            raise ReceiptError(f'player {p} carried inventory audit mismatch')
        if not math.isclose(farms[p]['money'],post[p]['farms'][p]['money'],abs_tol=1e-7,rel_tol=0):
            raise ReceiptError(f'player {p} money audit mismatch')
        requested = Counter()
        for a in orders[p]:
            if a and a[0].startswith('BUY_') and len(a) == 3: requested[a[0],a[1]] += a[2]
        if any(n < 0 or n > requested[k] for k,n in receipts[p].items()):
            raise ReceiptError('receipt outside requested range')
    return receipts


def se_metrics(obs):
    plants, animals = Counter(),Counter(); empty = 0
    tiles = obs['farms'][obs['player']]['tiles']
    for y in range(5,10):
        for x in range(5,10):
            t = tiles[y][x]
            if t is None: empty += 1
            elif isinstance(t,dict):
                if t.get('crop'): plants[t['crop']] += 1
                if t.get('animal'): animals[t['animal']] += 1
    return dict(se_plants=sum(plants.values()),se_animals=sum(animals.values()),se_empty=empty,
                se_crops=dict(plants),se_animal_types=dict(animals))


class LandController:
    def __init__(self, seed, iteration, learner_player=None, *, p=.5, b=None,
                 supply=None, enabled=True, assigned=None, target_player=None, D=None, package=None,
                 land_schedule=None, verify_inventory_orders=False):
        self.verify_inventory_orders = verify_inventory_orders
        self.seed, self.iteration, self.learner_player = seed,iteration,learner_player
        cfg = schedule(iteration,p,land_schedule=land_schedule); self.b = cfg['b'] if b is None else b
        if not math.isfinite(self.b) or self.b < 0: raise ValueError('invalid bonus')
        self.se_reward = se_harvest_coefficient(iteration,land_schedule)
        self.enabled = enabled; self.supply_enabled = cfg['supply'] if supply is None else supply
        self.choice = assignment(seed,iteration,learner_player,cfg['p'])
        for k,v in dict(assigned=assigned,target_player=target_player,D=D,package=package).items():
            if v is not None: self.choice[k] = v
        if not enabled: self.choice['assigned'] = False; self.b = 0.; self.supply_enabled = False; self.se_reward = 0.
        self.learners = (0,1) if learner_player is None else (learner_player,)
        self.rows = [dict(land4_success=False,land4_forced=False,purchase_step=None,purchase_day=None,
                    bonus_total=0.,se_harvests=0,se_harvest_bonus=0.,se_plant_actions=0,
                    supply_required={},supply_purchased={},supply_status='ineligible',
                    first_se_harvest_step=None,**{f'{k}_{dt}':None for dt in (24,72)
                    for k in ('se_plants','se_animals','se_empty','se_crops','se_animal_types')}) for _ in (0,1)]
        self.orders = None; self.land_attempts = 0; self.forced_steps = 0
        self.insufficient_funds = False; self.shed_full = False; self.waits = Counter(); self.audit = []
        self._inventory_presence = None

    def _sync_orders(self, observations):
        if self.orders is None:
            self.orders = [[list(ordered_inventory(v)) for v in o['private']['inventories']] for o in observations]
        else:
            for p,o in enumerate(observations):
                old = self.orders[p]; updated = []
                for i,inv in enumerate(o['private']['inventories']):
                    keys = [k for k in (old[i] if i < len(old) else []) if inv.get(k,0)]
                    added = [k for k,v in inv.items() if v and k not in keys]
                    if len(added) > 1: raise ReceiptError('multiple unordered acquisitions')
                    updated.append(keys+added)
                self.orders[p] = updated

    def _sync_native_orders(self, units):
        """Same insertion history as _sync_orders, using exact nonzero flags.

        Start at episode initialization, before a possible forced purchase: a
        debug JSON object alone cannot recover an existing multi-item order.
        Only presence is read from float32 ou, never quantities for receipts.
        """
        present = units[...,3:] > 0
        if self._inventory_presence is None:
            self.orders = [[[] for _ in player] for player in units]
            changed = present.any(axis=-1)
        else:
            changed = (present != self._inventory_presence).any(axis=-1)
        for p,i in zip(*np.nonzero(changed)):
            keys = [k for k in self.orders[p][i] if present[p,i,INVENTORY_ITEMS.index(k)]]
            added = [INVENTORY_ITEMS[j] for j in np.flatnonzero(present[p,i])
                     if INVENTORY_ITEMS[j] not in keys]
            if len(added) > 1: raise ReceiptError('multiple unordered acquisitions')
            self.orders[p][i] = keys+added
        self._inventory_presence = present

    def needs_pre_observation(self, step):
        row = self.rows[self.choice['target_player']]
        return row['supply_status'] == 'pending' and 1 <= step-row['purchase_step'] <= 24

    def needs_harvest_observation(self, units):
        self._harvest_needed = [False,False]
        for p,row in enumerate(self.rows):
            # With the SE harvest reward on, every harvest is counted, not only the first.
            if (row['first_se_harvest_step'] is not None and not (self.enabled and p in self.learners)) or (
                    row['purchase_step'] is None and not (self.bought[p] and self.pre_lands[p] < 4)):
                continue
            a = self.executed[p]
            self._harvest_needed[p] = any(
                u[0] and u[1] >= 5 and u[2] >= 5 and command and command[0] == 'HARVEST'
                for u,command in zip(units[p],(a.get('farmer',['PASS']),*a.get('hands',[]))))
        return any(self._harvest_needed)

    def needs_post_observation(self, lands, step):
        if self.needs_pre_observation(self.pre[0]['step']): return True
        for p,row in enumerate(self.rows):
            if (self.forced[p] and self.reason[p] == 'land' and self.supply_enabled
                    and self.bought[p] and self.pre_lands[p] < 4 and lands[p] == 4
                    and not row['land4_success']):
                return True  # Fix required quantities from exact post-purchase inventory.
            if row['purchase_step'] is not None and step-(row['purchase_step']+1) in (24,72):
                return True
        return False

    def before_step(self, observations, actions, *, lands=None, money=None, units=None):
        self._native_step = units is not None
        if not self._native_step: self._sync_orders(observations)
        self.pre = observations
        # NativePool.step only reads actions. Copy on intervention, and snapshot
        # the two audit actions below; untouched sampled actions need no deepcopy.
        self.sampled = actions if self._native_step else deepcopy(actions)
        executed = list(actions) if self._native_step else deepcopy(actions)
        self.forced = [False,False]; self.reason = [None,None]; self.requested = [{},{}]
        p = self.choice['target_player']; row = self.rows[p]; obs = observations[p]
        step,day = obs['step'],obs['day']
        self.pre_lands = tuple(map(land_count,observations)) if lands is None else tuple(lands)
        land = int(self.pre_lands[p])
        if self._native_step:
            if self.enabled and self.choice['assigned'] and self.supply_enabled and (
                    (not row['land4_success'] and step <= 455) or row['supply_status'] == 'pending'):
                self._sync_native_orders(units)
            else: self.orders = self._inventory_presence = None
        cash = float(obs['farms'][p]['money'] if money is None else money[p])
        if not math.isfinite(cash): raise ValueError('nonfinite money')
        if self.enabled and self.choice['assigned'] and not row['land4_success'] and self.choice['D'] <= day <= 18 and step <= 455 and land == 3 and cash >= 4000:
            executed[p] = dict(executed[p],market=[['BUY_LAND']]); self.forced[p] = True; self.reason[p] = 'land'
            self.land_attempts += 1
        elif row['land4_forced'] and self.supply_enabled and row['supply_status'] == 'pending':
            dt = step-row['purchase_step']
            if 1 <= dt <= 24:
                room = max(0,100-sum(obs['private']['shed'].values())); budget = cash; added = []
                targets = {(op,item) for op,item,_ in PACKAGES[self.choice['package']]}
                for op,item,_ in PACKAGES[self.choice['package']]:
                    left = max(0,row['supply_required'][item]-row['supply_purchased'][item])
                    if not left: continue
                    price = CROPS[item][0] if op == 'BUY_SEED' else ANIMALS[item][0] if op == 'BUY_ANIMAL' else market_price(item,obs['market']['inventory'][item]-1)
                    n = min(left,int(budget//price)) if op != 'BUY_PRODUCT' else left if budget >= price else 0
                    if budget < price: self.insufficient_funds = True; self.waits['insufficient_funds'] += 1
                    if op != 'BUY_SEED':
                        if not room: self.shed_full = True; self.waits['shed_full'] += 1
                        n = min(n,room)
                    if n:
                        added.append([op,item,n]); self.requested[p][item] = n
                        budget = max(0,budget-price*n)
                        if op != 'BUY_SEED': room -= n
                original = executed[p].get('market',[])
                kept = [a for a in original if not (len(a) >= 2 and (a[0],a[1]) in targets)]
                if added or kept != original:
                    executed[p] = dict(executed[p],market=(added+kept)[:10])
                    self.forced[p] = True; self.reason[p] = 'supply'
                else: self.waits['no_market_change'] += 1
            else:
                row['supply_status'] = 'expired'
        self.executed = executed
        self.bought = [any(a and a[0] == 'BUY_LAND' for a in action.get('market',[])) for action in executed]
        self.forced_steps += sum(self.forced)
        return executed, self.forced.copy()

    def after_step(self, observations, *, lands=None):
        bonus = [0.,0.]; receipts = [Counter(),Counter()]
        p = self.choice['target_player']; supply = self.rows[p]
        if self.needs_pre_observation(self.pre[p]['step']):
            orders = self.orders
            if self._native_step:
                # Units beyond native's 20 action rows cannot act and therefore
                # carry nothing. Validate that fact instead of inventing order.
                orders = [order + [list(ordered_inventory(inv)) for inv in
                          self.pre[q]['private']['inventories'][len(order):]]
                          for q,order in enumerate(orders)]
                if self.verify_inventory_orders:
                    # Keep _sync_native_orders as the independent history. The
                    # buffer dictionaries carry Rust's order, not JSON key order.
                    actual = [[list(inv) for inv in obs['private']['inventories']] for obs in self.pre]
                    for q in (0,1):
                        invs = self.pre[q]['private']['inventories']
                        # No action row exists beyond 20: require empty contents
                        # explicitly, and compare every exported unit, not 20.
                        if any(invs[len(self.orders[q]):]):
                            raise ReceiptError('inventory order history unavailable beyond native action rows')
                        if orders[q][:len(invs)] != actual[q]:
                            raise ReceiptError(f'player {q} native inventory order mismatch')
                    orders = actual
            receipts = material_receipts(self.pre,observations,self.executed,inventory_orders=orders)
            for op,item,_ in PACKAGES[self.choice['package']]:
                supply['supply_purchased'][item] += receipts[p][op,item]
            if all(supply['supply_purchased'][i] >= n for i,n in supply['supply_required'].items()):
                supply['supply_status'] = 'completed'
            elif self.pre[p]['step']-supply['purchase_step'] == 24:
                supply['supply_status'] = 'expired'
        for q, obs in enumerate(observations):
            row = self.rows[q]; pre = self.pre[q]; s = pre['step']
            pre_land = self.pre_lands[q]; post_land = land_count(obs) if lands is None else int(lands[q])
            success = self.bought[q] and pre_land < 4 and post_land == 4 and not row['land4_success']
            if success:
                row.update(land4_success=True,land4_forced=self.forced[q],purchase_step=s,purchase_day=pre['day'])
                if q in self.learners and not self.forced[q]: bonus[q] = self.b; row['bonus_total'] += self.b
                if self.forced[q] and self.reason[q] == 'land' and self.supply_enabled:
                    row['supply_required'] = {item:max(0,n-inventory(obs,op,item)) for op,item,n in PACKAGES[self.choice['package']]}
                    row['supply_purchased'] = {i:0 for i in row['supply_required']}
                    row['supply_status'] = 'pending' if any(row['supply_required'].values()) else 'completed'
            if row['purchase_step'] is not None:
                dt = obs['step']-(row['purchase_step']+1)
                if dt in (24,72): row.update({f'{k}_{dt}':v for k,v in se_metrics(obs).items()})
                count_every = self.enabled and q in self.learners  # count every SE harvest while training (the reward itself is added only while se_reward > 0)
                if (row['first_se_harvest_step'] is None or count_every) and (not self._native_step or self._harvest_needed[q]):
                    # Cargo insertion order affects receipts, not the existence
                    # of a harvest. Keep the sequential tile/unit simulation.
                    orders = ([list(inv) for inv in pre['private']['inventories']]
                              if self._native_step else self.orders[q])
                    _,_,harvest = _unit_phase(pre,self.executed[q],orders)
                    n = sum(1 for x,y,_,_ in harvest if x >= 5 and y >= 5)
                    if n and row['first_se_harvest_step'] is None: row['first_se_harvest_step'] = s
                    if n and count_every:
                        row['se_harvests'] += n; row['se_harvest_bonus'] += n*self.se_reward
                        bonus[q] += n*self.se_reward; row['bonus_total'] += n*self.se_reward
            if self.forced[q]:
                self.audit.append(dict(seed=self.seed,player=q,step=s,forced_reason=self.reason[q],
                    pre_land=int(pre_land),post_land=post_land,money_before=audit_money(pre['farms'][q]['money']),
                    money_after=audit_money(obs['farms'][q]['money']),sampled_action=deepcopy(self.sampled[q]),executed_action=deepcopy(self.executed[q]),
                    requested_qty=self.requested[q],received_qty={item:n for (_,item),n in receipts[q].items()},
                    failure_reason=('land_not_acquired' if not success else None) if self.reason[q] == 'land' else
                    ('partial_or_failed' if any(receipts[q][op,item] < self.requested[q].get(item,0) for op,item,_ in PACKAGES[self.choice['package']]) else None)))
        if not self._native_step: self._sync_orders(observations)
        return bonus

    def finish(self, observations, opponent=None):
        money = [observations[0]['farms'][p]['money'] for p in (0,1)]
        z = int(money[0] > money[1])-int(money[0] < money[1]); outcomes = []
        for p in self.learners:
            row = deepcopy(self.rows[p])
            if row['supply_status'] == 'pending': row['supply_status'] = 'episode_end'
            outcomes.append(dict(row,seed=self.seed,opponent=opponent,learner_player=p,**self.choice,
                land_attempts=self.land_attempts if p==self.choice['target_player'] else 0,
                forced_steps=self.forced_steps if p==self.choice['target_player'] else 0,
                wait_reasons=dict(self.waits) if p==self.choice['target_player'] else {},
                insufficient_funds=self.insufficient_funds if p==self.choice['target_player'] else False,
                shed_full=self.shed_full if p==self.choice['target_player'] else False,
                final_land_count=land_count(observations[p]),result='win' if money[p] > money[1-p] else 'loss' if money[p] < money[1-p] else 'draw',
                own_money=money[p],opponent_money=money[1-p],margin=money[p]-money[1-p]))
        return dict(seed=self.seed,opponent=opponent,learner_player=self.learner_player,**self.choice,
                    money=money,reward_player0=z,reward_kind='wdl',learner_outcomes=outcomes,
                    land_attempts=self.land_attempts,forced_steps=self.forced_steps,
                    insufficient_funds=self.insufficient_funds,shed_full=self.shed_full,wait_reasons=dict(self.waits))


def summarize_games(games):
    if len({g['seed'] for g in games}) != len(games): raise ValueError('duplicate game seeds')
    targets = [next(r for r in g['learner_outcomes'] if r['learner_player'] == g['target_player']) for g in games]
    rows = [r for g in games for r in g['learner_outcomes']]
    assigned = sum(g['assigned'] for g in games); unassigned = len(games)-assigned
    natural = sum(not g['assigned'] and r['land4_success'] and not r['land4_forced'] for g,r in zip(games,targets))
    eligible = [g for g,r in zip(games,targets) if r['supply_status'] != 'ineligible']
    completed = sum(r['supply_status'] == 'completed' for r in targets)
    attempts = sum(g['land_attempts'] for g in games); success = sum(r['land4_forced'] for r in targets)
    forced = sum(g['forced_steps'] > 0 for g in games)
    days = [r['purchase_day'] for r in targets if r['land4_success']]
    funds = sum(g['insufficient_funds'] for g in eligible)
    per_group = {}
    for key in ('purchase_day','package'):
        per_group[key] = {}
        for g,r in zip(games,targets):
            name = str(r[key]); cell = per_group[key].setdefault(name,dict(games=0,purchases=0,eligible=0,completed=0,insufficient_funds=0,shed_full=0))
            cell['games'] += 1; cell['purchases'] += int(r['land4_success']); cell['completed'] += int(r['supply_status'] == 'completed')
            cell['eligible'] += int(r['supply_status'] != 'ineligible')
            cell['insufficient_funds'] += int(g['insufficient_funds']); cell['shed_full'] += int(g['shed_full'])
    return dict(games=len(games),assigned_games=assigned,assigned_rate=ratio(assigned,len(games)),
        forced_games=forced,forced_rate=ratio(forced,len(games)),land_attempts=attempts,forced_success=success,
        forced_success_rate=ratio(success,attempts),unassigned_games=unassigned,natural_land4_games_unassigned=natural,r=ratio(natural,unassigned),
        bonus_sum=sum(r['bonus_total'] for r in rows),all_learner_natural_land4_games=sum(r['land4_success'] and not r['land4_forced'] for r in rows),
        all_learner_games=len(rows),purchase_day_median=statistics.median(days) if days else None,
        se_plant_actions_mean=ratio(sum(r.get('se_plant_actions',0) for r in rows),len(rows)),
        se_plant_actions_games=sum(r.get('se_plant_actions',0)>0 for r in rows),
        supply_eligible_games=len(eligible),supply_completed_games=completed,supply_completion_rate=ratio(completed,len(eligible)),
        insufficient_funds_games=funds,insufficient_funds_rate=ratio(funds,len(eligible)),shed_full_games=sum(g['shed_full'] for g in eligible),
        pfsp_recorded_games=sum(bool(g['opponent']) and not g['assigned'] for g in games),
        pfsp_excluded_assigned_games=sum(bool(g['opponent']) and g['assigned'] for g in games),by_group=per_group)
