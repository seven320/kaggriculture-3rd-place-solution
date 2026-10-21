"""09-30 23:05: day16〜19、手持ちのメロンの種がある間は方策の PLANT WHEAT/CARROT を PLANT MELON に替える(状態なし)。"""
def apply(obs, action):
    try:
        day = int(obs['step']) // 24
        if not 16 <= day <= 19:
            return action
        left = int(((obs.get('private') or {}).get('seeds') or {}).get('MELON', 0))
        if left <= 0:
            return action
        units = [action.get('farmer')] + list(action.get('hands') or [])
        left -= sum(1 for u in units if isinstance(u, list) and len(u) >= 2 and u[0] == 'PLANT' and u[1] == 'MELON')
        changed = False; new = []
        for u in units:
            if left > 0 and isinstance(u, list) and len(u) >= 2 and u[0] == 'PLANT' and u[1] in ('WHEAT', 'CARROT'):
                u = ['PLANT', 'MELON']; left -= 1; changed = True
            new.append(u)
        if not changed:
            return action
        result = dict(action); result['farmer'] = new[0]; result['hands'] = new[1:]
        return result
    except Exception:
        return action
