"""Submission adapter for the existing training MELON16 overlay.
Official zero-based day16, steps384..389, target3, cash reserve1000.
Counts held seeds, today's plants and existing BUY_SEED orders; appends only.
"""
def apply(obs, action):
    step = int(obs['step'])
    if not 384 <= step < 390:
        return action
    farm = obs['farms'][int(obs['player'])]
    market = list(action.get('market') or [])
    if len(market) >= 10:
        return action
    seeds = int(obs['private']['seeds'].get('MELON', 0))
    planted = sum(1 for row in farm['tiles'] for tile in row if isinstance(tile, dict) and tile.get('crop') == 'MELON' and tile.get('planted_day') == step // 24)
    bought = sum(max(0, int(order[2])) for order in market if len(order) >= 3 and order[0] == 'BUY_SEED' and order[1] == 'MELON')
    affordable = int((float(farm['money']) - 1000) // 80)
    quantity = min(3 - seeds - planted - bought, affordable)
    if quantity <= 0:
        return action
    result = dict(action)
    result['market'] = market + [['BUY_SEED', 'MELON', quantity]]
    return result
