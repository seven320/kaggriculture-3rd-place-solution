"""transformer 方策の提出エージェント。公式の観測だけから特徴と合法手を作り(obs_feat.py、エンジン由来と全手一致を検証済み)、
モデル(model_tf.py)で合法手を採点し、公式の行動の辞書に変換する(教師の手で公式残高の再現を検証済み)。"""
import os, sys
import numpy as np
import torch
# 公式の読み込み器はソースを exec するので __file__ が無いことがある。重みのある場所を順に探す
_C = ([os.path.dirname(os.path.abspath(__file__))] if "__file__" in globals() else []) + ["/kaggle_simulations/agent", os.getcwd()]
_D = next((d for d in _C if os.path.exists(os.path.join(d, "policy.pt"))), _C[-1])
sys.path.insert(0, _D)
import model_tf as M
from obs_feat import featurize, to_action
torch.set_num_threads(1)
_NET = M.Policy(); _NET.load_state_dict(torch.load(os.path.join(_D, "policy.pt"), map_location="cpu")); _NET.eval()
NM, NS, M_STOP = 23, 11, 22


def _todict(o):
    if isinstance(o, dict): return {k: _todict(v) for k, v in o.items()}
    if hasattr(o, "items"): return {k: _todict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_todict(v) for v in o]
    return o


def _act(obs):
    tiles, ou, pu, g, lm = featurize(obs)
    T = lambda a, dt=torch.float32: torch.as_tensor(a[None], dtype=dt)
    h, pad = _NET.encode(T(tiles).float(), T(ou), T(pu), T(g))
    lg, q = _NET.unit_logits(h, T(ou), T(lm, torch.uint8)); lg = lg[0]
    live = [u for u in range(20) if ou[u, 0] > 0]
    seeds = np.rint(g[27:32] * 20).astype(int); used = np.zeros(5, int)
    ul = np.full(20, -1, np.int64)
    for u in live:                         # 植え付けは種の数を超えると全部無効になるので順に割り当てる
        for c in torch.argsort(lg[u], descending=True).tolist():
            if lg[u, c] < -1e3: continue
            if 15 <= c < 20 and used[c - 15] >= seeds[c - 15]: continue
            ul[u] = c; break
        if 15 <= ul[u] < 20: used[ul[u] - 15] += 1
    uq = _NET.unit_qty(q, torch.as_tensor(ul[None])).argmax(-1)[0].numpy()
    prev = [NM]; ms = []; mq = []
    for j in range(NS):
        mo, y = _NET.market(h, pad, torch.as_tensor([prev])); c = int(mo[0, -1].argmax()) if j < NS - 1 else M_STOP
        ms.append(c); mq.append(int(_NET.market_qty(y[:, -1:], torch.as_tensor([[c]])).argmax(-1)[0, 0]) if c < 19 else 0)
        if c == M_STOP: break
        prev.append(c)
    ms += [M_STOP] * (NS - len(ms)); mq += [0] * (NS - len(mq))
    return to_action(ul, uq, ms, mq, len(live))


def agent(obs, config=None):
    # 公式の読み込み器は引数の数を調べて呼ぶので、デコレータで包まない(包むと引数0個で呼ばれた)
    with torch.no_grad():
        return _act(_todict(obs))
