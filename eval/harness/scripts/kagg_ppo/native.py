"""Validated-size NumPy buffers around the isolated Rust batch FFI."""
import ctypes as C
import json
import operator
from copy import deepcopy
from scripts.kagg_bc_kit.contracts import apply_masks
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

ITEMS=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP']
BASE=['PASS','NORTH','SOUTH','EAST','WEST','WATER','HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','FEED','COLLECT_FERTILIZER','CARE','DROP']
import os
# Built by `cargo build --release` in engine/rust/ (crate kagg-ppo-native); override with KAGG_NATIVE_LIB.
LIBRARY=Path(os.environ.get('KAGG_NATIVE_LIB',ROOT.parents[1]/'engine/rust/target/release/libkagg_ppo_native.so'))

LAND_SCHEMA = 1
LAND_UNITS = 241
LAND_OBSERVATION_FORMATS = ('json', 'buffer')
# Second axis 2 is player 0's farm 0 and player 1's farm 1, once each.
# It does not contain both farms per observation. Opponent tiles are omitted.
LAND_BUFFERS = (
    ('clock', (3,), np.int64),
    ('farm_meta', (2, 4), np.int64),
    ('money', (2,), np.float64),
    ('tile', (2, 100, 8), np.int64),
    ('position', (2, LAND_UNITS, 2), np.int64),
    ('inventory_qty', (2, LAND_UNITS, 12), np.int64),
    ('inventory_order', (2, LAND_UNITS, 12), np.int8),
    ('inventory_order_len', (2, LAND_UNITS), np.uint8),
    ('shed', (2, 12), np.int64),
    ('seeds', (2, 5), np.int64),
    ('market_inventory', (9,), np.int64),
    ('market_prices', (9,), np.float64),
)
LAND_EMPTY, LAND_LOCKED, LAND_WEED, LAND_PLANT, LAND_COOP, LAND_PASTURE = range(6)


def validate_land_observation_format(value):
    if value not in LAND_OBSERVATION_FORMATS:
        raise ValueError(f'unknown land_observation_format: {value!r}')
    return value


def _land_array(array, dtype, shape):
    if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
            or array.shape != shape
            or not all((array.flags.c_contiguous, array.flags.aligned, array.flags.writeable))):
        raise ValueError(f'invalid land buffer: expected {np.dtype(dtype)} {shape}')
    return array


def _land_number(value):
    # Same integer test and strict boundary as bridge::number().
    return int(value) if abs(value) < 9e15 and value % 1 == 0 else value


def _land_tiles(rows):
    values = rows.tolist()
    for row in values:
        if not LAND_EMPTY <= row[0] <= LAND_PASTURE:
            raise ValueError(f'invalid land tile kind: {row[0]}')
    tiles = []
    for kind, item, planted, yield_units, watered, fertilized, fed, fertilizer in values:
        if kind == LAND_EMPTY: value = None
        elif kind == LAND_LOCKED: value = 'LOCKED'
        elif kind == LAND_WEED: value = {'kind': 'WEED'}
        elif kind == LAND_PLANT:
            if not 0 <= item < 5: raise ValueError('invalid land crop ID')
            value = dict(kind='PLANT', crop=ITEMS[item], planted_day=planted,
                         watered_today=bool(watered), yield_units=yield_units,
                         fertilized_until_day=fertilized)
        else:  # The range check and selection leave only COOP and PASTURE.
            value = {'kind': 'COOP' if kind == LAND_COOP else 'PASTURE'}
            if item != -1:
                if not 9 <= item < 12: raise ValueError('invalid land animal ID')
                value.update(animal=ITEMS[item], yield_units=yield_units,
                             fed_today=bool(fed), fertilizer_available=bool(fertilizer))
        tiles.append(value)
    return [tiles[i:i+10] for i in range(0,100,10)]


def encode_actions(actions):
    array=np.zeros((len(actions),92),np.int64)
    for b,action in enumerate(actions):
        units=[action.get('farmer',['PASS']),*action.get('hands',[])]
        orders=action.get('market',[])
        if not 1<=len(units)<=20 or len(orders)>10:raise ValueError('unsupported action count')
        array[b,:2]=[len(units)-1,len(orders)]
        for u,unit in enumerate(units):
            op=unit[0]
            if op in BASE:code=BASE.index(op);quantity=0
            elif op=='PLANT':code=15+ITEMS[:5].index(unit[1]);quantity=0
            elif op in ('PICKUP','PLACE'):
                code=(20 if op=='PICKUP' else 32)+ITEMS.index(unit[1]);quantity=int(unit[2])
            else:raise ValueError(unit)
            if not 0<=quantity<2**32:raise ValueError(unit)
            array[b,2+3*u]=code;array[b,4+3*u]=quantity
        for i,order in enumerate(orders):
            quantity=0
            if not order:code=21
            elif order[0]=='HIRE':code=19
            elif order[0]=='BUY_LAND':code=20
            else:
                op,item,quantity=order;quantity=int(quantity)
                if op=='SELL':code=ITEMS[:9].index(item)
                elif op=='BUY_SEED':code=9+ITEMS[:5].index(item)
                elif op=='BUY_PRODUCT':code=14+['WHEAT','FERTILIZER'].index(item)
                elif op=='BUY_ANIMAL':code=16+ITEMS[9:].index(item)
                else:raise ValueError(order)
            if not 0<=quantity<2**32:raise ValueError(order)
            array[b,62+3*i]=code;array[b,64+3*i]=quantity
    return array


class NativePool:
    def __init__(self,count=16,episode_steps=720,seed_start=10029,*,land_observation_format='json',mask_versions=None):
        self.land_observation_format=validate_land_observation_format(land_observation_format)
        self.lib=C.CDLL(str(LIBRARY));self.batch=count*2;self.count=count
        self.mask_versions=np.zeros(self.batch,np.uint8) if mask_versions is None else np.asarray(mask_versions).copy()
        if self.mask_versions.shape != (self.batch,) or not np.isin(self.mask_versions,(0,2)).all():
            raise ValueError("invalid player mask versions")
        self.lib.mp_new.argtypes=[C.c_int64,C.c_size_t,C.c_uint32];self.lib.mp_new.restype=C.c_void_p
        self.lib.mp_free.argtypes=[C.c_void_p];self.lib.mp_free.restype=None
        self.lib.mp_step.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];self.lib.mp_step.restype=C.c_int32
        self.lib.mp_features.argtypes=[C.c_void_p,C.c_size_t]+[C.c_void_p]*7;self.lib.mp_features.restype=C.c_int32
        self.lib.mp_debug.argtypes=[C.c_void_p];self.lib.mp_debug.restype=C.c_char_p
        self.handle=self.lib.mp_new(seed_start,count,episode_steps)
        if not self.handle:raise ValueError('invalid native pool configuration')
        self.tiles=np.empty((self.batch,2,100,13),np.float32)
        self.arrays={'ou':np.empty((self.batch,20,15),np.float32),
                     'pu':np.empty((self.batch,20,3),np.float32),
                     'g':np.empty((self.batch,75),np.float32),
                     'legal':np.empty((self.batch,20,44),np.uint8)}
        self.meta=np.empty((self.batch,18),np.float64);self.done=np.empty(self.batch,np.uint8)
        try:
            self.refresh()
            if self.land_observation_format == 'buffer':
                self._init_land_buffers()
                self.land_observations([0])  # Fail on missing symbol/schema before collection starts.
        except BaseException:
            self.close();raise

    def refresh(self):
        if not self.handle:raise RuntimeError('pool closed')
        arrays=[self.tiles,*self.arrays.values(),self.meta,self.done]
        code=self.lib.mp_features(self.handle,self.batch,*[a.ctypes.data for a in arrays])
        if code:raise RuntimeError(f'native features failed: {code}')
        self.features={'tiles':self.tiles.astype(np.float16),**self.arrays}
        apply_masks(self.features,self.mask_versions)
        self.observations=[]
        # Only fields consumed by the unchanged repair layer and terminal reward.
        for b,row in enumerate(self.meta):
            self.observations.append({'step':int(row[0]),'day':int(row[1]),'hour':int(row[2]),'player':int(row[3]),
                'farms':[{'money':float(row[4])},{'money':float(row[5])}],
                'private':{'shed':{name:int(n) for name,n in zip(ITEMS,row[6:]) if n}},
                '_native_pool':self,'_native_index':b})

    def step(self,actions):
        if len(actions)!=self.batch:raise ValueError('one action per player required')
        data=encode_actions(actions)
        code=self.lib.mp_step(self.handle,data.ctypes.data,data.size)
        if code:raise RuntimeError(f'native step failed: {code}')
        self.refresh()

    def debug(self):
        if not self.handle:raise RuntimeError('pool closed')
        return json.loads(self.lib.mp_debug(self.handle))

    def _init_land_buffers(self):
        try: ffi = self.lib.mp_land_observations
        except AttributeError as exc:
            raise RuntimeError('buffer requested but mp_land_observations symbol is missing') from exc
        flags = ('C_CONTIGUOUS', 'ALIGNED', 'WRITEABLE')
        args = [C.c_void_p, C.c_int64, np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=flags), C.c_size_t]
        for _, tail, dtype in LAND_BUFFERS:
            args.extend((np.ctypeslib.ndpointer(dtype=dtype, ndim=1+len(tail), flags=flags), C.c_size_t))
        ffi.argtypes = args; ffi.restype = C.c_int32
        self._land_ids = np.empty(self.count, dtype=np.int64)
        self._land_buffers = {name: np.empty((self.count, *tail), dtype=dtype)
                              for name, tail, dtype in LAND_BUFFERS}
        self._land_buffers['inventory_order'].fill(-1)

    def land_observations(self, ids):
        """Detached land dictionaries, requested games in order, [player0, player1].

        The buffer's farm axis is ownership: farm 0 for player 0 and farm 1 for
        player 1. Each observation has its own farm's tiles/units and only the
        opponent's money. Do not call concurrently with step/debug/close.
        Buffers are allocated only by a buffer-enabled constructor and reused.
        """
        if not self.handle: raise RuntimeError('pool closed')
        if self.land_observation_format != 'buffer': raise ValueError('land buffer was not enabled')
        requested = []
        for value in ids:
            if isinstance(value, (bool, np.bool_)): raise ValueError('invalid land game ID')
            try: value = operator.index(value)
            except TypeError as exc: raise ValueError('land game ID must be an integer') from exc
            if not 0 <= value < self.count: raise ValueError(f'land game ID out of range: {value}')
            requested.append(value)
        unique = list(dict.fromkeys(requested))
        m = len(unique)
        if not m: return []
        _land_array(self._land_ids, np.int64, (self.count,))
        ids_buffer = _land_array(self._land_ids[:m], np.int64, (m,))
        ids_buffer[:] = unique
        arrays = {}; args = [self.handle, LAND_SCHEMA, ids_buffer, m]
        for name, tail, dtype in LAND_BUFFERS:
            array = _land_array(self._land_buffers[name], dtype, (self.count, *tail))
            arrays[name] = _land_array(array[:m], dtype, (m, *tail))
            args.extend((arrays[name], arrays[name].size))
        code = self.lib.mp_land_observations(*args)
        if code: raise RuntimeError(f'native land observations failed (schema {LAND_SCHEMA}): {code}')
        # Convert fixed fields in batches; never index NumPy scalars in tile or
        # item loops. Unit arrays are sliced to actual counts before tolist().
        fixed = {name: arrays[name].tolist() for name in
                 ('clock', 'farm_meta', 'money', 'shed', 'seeds', 'market_inventory', 'market_prices')}
        expected_clocks = self.meta[2*np.asarray(unique), :3].tolist()
        games = {}
        for g, game_id in enumerate(unique):
            step, day, hour = fixed['clock'][g]
            if fixed['clock'][g] != expected_clocks[g]:
                raise ValueError(f'land observation clock mismatch for game {game_id}')
            pair = []; money = list(map(_land_number, fixed['money'][g]))
            for p in (0, 1):
                mask, hires, npos, ninv = fixed['farm_meta'][g][p]
                if not 1 <= npos <= LAND_UNITS or not 0 <= ninv <= LAND_UNITS:
                    raise ValueError('land unit count exceeds buffer capacity')
                positions = arrays['position'][g, p, :npos].tolist()
                quantities = arrays['inventory_qty'][g, p, :ninv].tolist()
                orders = arrays['inventory_order'][g, p, :ninv].tolist()
                lengths = arrays['inventory_order_len'][g, p, :ninv].tolist()
                inventories = []
                for qty, order, length in zip(quantities, orders, lengths):
                    order = order[:length]
                    if (length > 12 or len(set(order)) != length or any(not 0 <= i < 12 for i in order)
                            or set(order) != {i for i, q in enumerate(qty) if q > 0}):
                        raise ValueError('invalid land inventory order')
                    inventories.append({ITEMS[i]: qty[i] for i in order})
                farms = [{'money': money[0]}, {'money': money[1]}]
                farms[p].update(tiles=_land_tiles(arrays['tile'][g, p]), farmer=positions[0], hands=positions[1:],
                                unlocked_quadrants=[name for bit, name in enumerate(('NW', 'NE', 'SW', 'SE')) if mask & (1 << bit)],
                                hires_today=hires)
                pair.append(dict(step=step, day=day, hour=hour, player=p, farms=farms,
                    market=dict(inventory=dict(zip(ITEMS[:9], fixed['market_inventory'][g])),
                                prices=dict(zip(ITEMS[:9], map(_land_number, fixed['market_prices'][g])))),
                    private=dict(shed=dict(zip(ITEMS, fixed['shed'][g][p])),
                                 seeds=dict(zip(ITEMS[:5], fixed['seeds'][g][p])), inventories=inventories)))
            games[game_id] = pair
        # Duplicate IDs use one FFI row, but never share mutable dictionaries.
        seen = set(); result = []
        for game_id in requested:
            result.append(deepcopy(games[game_id]) if game_id in seen else games[game_id])
            seen.add(game_id)
        return result

    def close(self):
        if self.handle:self.lib.mp_free(self.handle);self.handle=None
        self.observations=[]

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
