"""Exact public/private fields needed by v99's tomato rule on native rollouts."""
from weakref import WeakKeyDictionary

SHOPS=('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP',
       'PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')
_CACHE=WeakKeyDictionary()

def complete(obs):
    if '_native_pool' not in obs: return obs
    pool=obs['_native_pool'];index=obs['_native_index']
    # CpuSubsetSampler may point to the concatenated MultiPool facade.
    if hasattr(pool,'pools'):
        for child in pool.pools:
            if index<child.batch:
                pool=child;break
            index-=child.batch
        else: raise IndexError('native observation row outside MultiPool')
    clock=int(pool.meta[index,0])
    cached=_CACHE.get(pool)
    if cached is None or cached[0]!=clock:
        pairs=pool.land_observations(list(range(pool.count)))
        for game,pair in enumerate(pairs):
            for player,full in enumerate(pair):
                assert full['step']==clock
                shops=pool.arrays['g'][2*game+player,47:63]
                full['town']={'unlocked_shops':[SHOPS[int(n)] for n in shops if n>=0]}
        cached=(clock,pairs);_CACHE[pool]=cached
    return cached[1][index//2][index%2]

def wrap_tomato_repair(module):
    """Keep original rule code, but supply fields omitted by compact observations."""
    import atexit
    if hasattr(module,'_dump_fire'): atexit.unregister(module._dump_fire)
    original=module._pace
    def repair(action,obs):
        # Outside these times the original rule only uses shed, already present.
        step=int(obs['step']) if 'step' in obs else int(obs['day'])*24+int(obs['hour'])
        if module.BUY_D0<=int(obs['day'])<=module.PLANT_D1 or step>=718:
            obs=complete(obs)
        return original(action,obs)
    repair.tomato_module=module
    return repair
