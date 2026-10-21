"""MMPQ-BCw retained Best Gen21 submission agent.

The policy uses the same relative new-mask penalty as PPO inference. On the final
playable step, market orders are replaced with SELL 1000 for all nine products
so no usable inventory is intentionally carried past the end of the game.
"""
import os
import sys

import numpy as np
import torch

_C = ([os.path.dirname(os.path.abspath(__file__))] if "__file__" in globals() else []) + [
    os.path.dirname(os.path.abspath(sys._getframe(0).f_code.co_filename)),
    "/kaggle_simulations/agent",
    os.getcwd(),
]
_D = next((directory for directory in _C if os.path.exists(os.path.join(directory, "policy.pt"))), _C[-1])
sys.path.insert(0, _D)

import json
import model_tf as M
from obs_feat import featurize, to_action

with open(os.path.join(_D, "cfg.json"), encoding="utf-8") as config_file:
    _CFG = json.load(config_file)
_PTR = _CFG.pop("ptr", False)
if _PTR:
    import model_ptr_infer as _MP
    _POLICY = _MP.PolicyPtrInfer
else:
    _POLICY = M.Policy

_NET = _POLICY(**_CFG)
_NET.load_state_dict(torch.load(os.path.join(_D, "policy.pt"), map_location="cpu"))
_NET.eval()
# ---- 09-30 土台(W3+W5+W6+W9'+WT)の後付け後処理: _NET の unit_logits を包んで agent 1 回の forward のスコアを控える ----
import postprocess_c1p_w9p as _C1P
_C1P_HOOK = _C1P.KitHook(_NET)
# ---- 09-30 最終日の回収計画(探索版、開始 step 708、α = 0.01)。1 試合 1 インスタンス ----
import collect_plan_d29 as _CP
_CP_PLAN = _CP.CollectPlanner(start_step=708, alpha=0.01)
# ---- 10-01 D23S_p1(day 0〜28 の hour 23、倉庫接点の unit を DROP、価格 / 基準価格 >= 1.0 の品目を同じ手で SELL)。状態なし ----
import postprocess_d23s_p1 as _D23S
import melon16 as _MELON16
import melon_replant as _MREP
torch.set_num_threads(1)

NM, NS, M_STOP = 23, 11, 22
_PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
_OUTPUTS = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
_VALUE = {"STRAWBERRY": 5, "MELON": 5, "MILK": 4, "WOOL": 4, "TOMATO": 3, "FERTILIZER": 2, "EGG": 2, "CARROT": 1, "WHEAT": 0}
_VALUE_ORDER = sorted(_VALUE, key=lambda item: -_VALUE[item])
_SLACK = -3
_LIQUIDATE_FROM_STEP = 710
_LAST_STEP = 718
_LAND_FROM_STEP = 240
_LAND_PRICE = 4000
_TARGET = None
_LAND_PRICE_TABLE = None
_LAND_PRICE_PARAMS = (
    (25, 10000, 400, "sqrt", .8, "log", .2),
    (35, 10000, 450, "log", .2, "sqrt", .7),
    (60, 10000, 200, "linear", .4, "sqrt", .6),
    (120, 10000, 100, "sqrt", .7, "linear", 1.6),
    (250, 10000, 300, "log", .2, "sq", 3.6),
    (50, 10000, 332, "linear", .4, "log", .2),
    (160, 10000, 122, "sqrt", .6, "linear", 1.6),
    (200, 10000, 105, "log", .2, "sq", 3.2),
    (100, 10000, 200, "linear", .4, "linear", .4),
)


def _todict(value):
    if isinstance(value, dict):
        return {key: _todict(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {key: _todict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_todict(item) for item in value]
    return value


def _act(obs):
    tiles, own_units, opponent_units, global_features, legal_mask = featurize(obs)
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value[None], dtype=dtype)
    hidden, padding = _NET.encode(
        tensor(tiles).float(), tensor(own_units), tensor(opponent_units), tensor(global_features)
    )
    logits, quantity = _NET.unit_logits(hidden, tensor(own_units), tensor(legal_mask, torch.uint8))
    logits = logits[0]
    live = [unit for unit in range(20) if own_units[unit, 0] > 0]
    seeds = np.rint(global_features[27:32] * 20).astype(int)
    used = np.zeros(5, int)
    unit_labels = np.full(20, -1, np.int64)
    for unit in live:
        for candidate in torch.argsort(logits[unit], descending=True).tolist():
            if logits[unit, candidate] < -1e3:
                continue
            if 15 <= candidate < 20 and used[candidate - 15] >= seeds[candidate - 15]:
                continue
            unit_labels[unit] = candidate
            break
        if 15 <= unit_labels[unit] < 20:
            used[unit_labels[unit] - 15] += 1
    unit_quantities = _NET.unit_qty(quantity, torch.as_tensor(unit_labels[None])).argmax(-1)[0].numpy()

    previous = [NM]
    market_labels = []
    market_quantities = []
    for index in range(NS):
        market_logits, market_hidden = _NET.market(hidden, padding, torch.as_tensor([previous]))
        candidate = int(market_logits[0, -1].argmax()) if index < NS - 1 else M_STOP
        market_labels.append(candidate)
        market_quantities.append(
            int(_NET.market_qty(market_hidden[:, -1:], torch.as_tensor([[candidate]])).argmax(-1)[0, 0])
            if candidate < 19
            else 0
        )
        if candidate == M_STOP:
            break
        previous.append(candidate)
    market_labels += [M_STOP] * (NS - len(market_labels))
    market_quantities += [0] * (NS - len(market_quantities))
    return to_action(unit_labels, unit_quantities, market_labels, market_quantities, len(live))


def _step(obs):
    value = obs.get("step")
    return int(value) if value is not None else int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))


def _target():
    global _TARGET
    if _TARGET is None:
        with open(os.path.join(_D, "stock_target.json"), encoding="utf-8") as target_file:
            data = json.load(target_file)
        products = data["products"]
        _TARGET = np.asarray(
            [[row[products.index(item)] if item in products else 0.0 for item in _PRODUCTS] for row in data["75"]],
            dtype=np.float32,
        )
    return _TARGET


def _pace(action, obs):
    """Match pace_fast.apply_codes: inventory repair, then staged liquidation."""
    try:
        step = _step(obs)
        if step > _LAST_STEP:
            return action
        shed = dict((obs.get("private") or {}).get("shed") or {})
        market = [list(order) for order in (action.get("market") or []) if isinstance(order, list)]
        if step >= _LIQUIDATE_FROM_STEP:
            turns_left = _LAST_STEP - step + 1
            if turns_left <= 1:
                sells = [["SELL", item, 1000] for item in _VALUE_ORDER]
            else:
                sells = [
                    ["SELL", item, -(-int(shed.get(item, 0)) // turns_left)]
                    for item in _VALUE_ORDER
                    if int(shed.get(item, 0)) > 0
                ]
            keep = [order for order in market if order and order[0] != "SELL" and not str(order[0]).startswith("BUY")]
            result = dict(action)
            result["market"] = (sells + keep)[:10]
            return result

        target = _target()
        row = np.rint(target[min(step, len(target) - 1)]).astype(np.int64)
        want = {
            item: int(shed.get(item, 0)) - int(row[_PRODUCTS.index(item)]) - _SLACK
            for item in _OUTPUTS
        }
        want = {item: quantity for item, quantity in want.items() if quantity > 0}
        if not want:
            return action
        for order in market:
            if order and order[0] == "SELL" and len(order) > 2 and order[1] in want:
                order[2] = max(int(order[2]), want.pop(order[1]))
        for item in _VALUE_ORDER:
            if item not in want or len(market) >= 10:
                continue
            market.append(["SELL", item, want[item]])
        result = dict(action)
        result["market"] = market[:10]
        return result
    except Exception:
        return action


def _land_prices():
    global _LAND_PRICE_TABLE
    if _LAND_PRICE_TABLE is not None:
        return _LAND_PRICE_TABLE
    inventory = np.arange(6000, 14001, dtype=np.float64)
    shape = {"linear": lambda value: value, "sq": lambda value: value * value, "sqrt": np.sqrt, "log": np.log1p}
    prices = np.empty((len(_LAND_PRICE_PARAMS), len(inventory)), dtype=np.float32)
    for item, (base, equilibrium, scale, below_name, below_target, above_name, above_target) in enumerate(_LAND_PRICE_PARAMS):
        below = shape[below_name]
        above = shape[above_name]
        low_amplitude = below_target * base / below(np.asarray(scale, dtype=np.float64))
        high_amplitude = above_target * base / above(np.asarray(scale, dtype=np.float64))
        raw = np.where(
            inventory < equilibrium,
            base + low_amplitude * below(np.maximum(equilibrium - inventory, 0)),
            base - high_amplitude * above(np.maximum(inventory - equilibrium, 0)),
        )
        prices[item] = np.maximum(1, np.rint(raw))
    cumulative = np.zeros((len(prices), len(inventory) + 1), dtype=np.float64)
    cumulative[:, 1:] = np.cumsum(prices.astype(np.float64), axis=1)
    _LAND_PRICE_TABLE = cumulative
    return cumulative


def _projected_sell_revenue(market, obs):
    cumulative = _land_prices()
    inventory = dict((obs.get("market") or {}).get("inventory") or {})
    sold = {item: 0 for item in _PRODUCTS}
    revenue = 0.0
    for order in market:
        if not order or order[0] != "SELL" or len(order) < 3 or order[1] not in sold:
            continue
        item = order[1]
        quantity = max(0, int(order[2]))
        index = _PRODUCTS.index(item)
        start = int(np.clip(int(inventory.get(item, 0)) + sold[item] - 6000, 0, 8000))
        end = int(np.clip(start + quantity, 0, 8001))
        revenue += cumulative[index, end] - cumulative[index, start]
        sold[item] += quantity
    return revenue


def _land(action, obs):
    """Match pace_fast.apply_land for the candidate-only fourth-quadrant guard."""
    try:
        step = _step(obs)
        if step < _LAND_FROM_STEP or step >= _LIQUIDATE_FROM_STEP:
            return action
        me = int(obs["player"])
        farm = obs["farms"][me]
        if len(farm.get("unlocked_quadrants") or []) >= 4:
            return action
        market = [list(order) for order in (action.get("market") or []) if isinstance(order, list)]
        if any(order and order[0] == "BUY_LAND" for order in market) or len(market) >= 10:
            return action
        available = float(farm.get("money", 0)) + _projected_sell_revenue(market, obs)
        if available < _LAND_PRICE:
            return action
        market.append(["BUY_LAND"])
        result = dict(action)
        result["market"] = market
        return result
    except Exception:
        return action


def _postprocess(action, obs):
    return _land(_pace(action, obs), obs)


def agent(obs, config=None):
    with torch.no_grad():
        converted = _todict(obs)
        return _MREP.apply(converted, _MELON16.apply(converted, _CP_PLAN(converted, _D23S.apply(converted, config, _C1P_HOOK.apply(converted, config, _postprocess(_act(converted), converted))))))   # 網 → gen35 の _postprocess → 土台 → D23S_p1 → 回収計画
