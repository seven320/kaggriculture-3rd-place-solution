"""transformer 方策の提出エージェント。公式の観測だけから特徴と合法手を作り(obs_feat.py、エンジン由来と全手一致を検証済み)、
モデル(model_tf.py)で合法手を採点し、公式の行動の辞書に変換する(教師の手で公式残高の再現を検証済み)。"""
import os, sys
import numpy as np
import torch
# 公式の読み込み器はソースを exec するので __file__ が無いことがある。重みのある場所を順に探す
_C = ([os.path.dirname(os.path.abspath(__file__))] if "__file__" in globals() else []) + \
     [os.path.dirname(os.path.abspath(sys._getframe(0).f_code.co_filename))] + ["/kaggle_simulations/agent", os.getcwd()]   # 読み込み器はファイル名を残す
_D = next((d for d in _C if os.path.exists(os.path.join(d, "policy.pt"))), _C[-1])
sys.path.insert(0, _D)
import model_tf as M
from obs_feat import featurize, to_action
torch.set_num_threads(1)
import json
_CFG = json.load(open(os.path.join(_D, "cfg.json"))) if os.path.exists(os.path.join(_D, "cfg.json")) else {}
_P = _CFG.pop("ptr", False)
if _P:
    import model_ptr_infer as _MP
_NET = (_MP.PolicyPtrInfer if _P else M.Policy)(**_CFG); _NET.load_state_dict(torch.load(os.path.join(_D, "policy.pt"), map_location="cpu")); _NET.eval()
# ---- 09-30 C1' + W9'(W3+W5+W6+W9+W9'+WT)の後付け後処理: _NET の unit_logits を包んで agent 1 回の forward のスコアを控える ----
import postprocess_c1p_w9p as _C1P
_C1P_HOOK = _C1P.KitHook(_NET)
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



# ---- 09-29 最終手の売り切り(v106 と同じ): 最後の手(718)は方策の注文を捨てて9品目すべてを SELL 1000 にする ----
_LIQ_ITEMS = ["STRAWBERRY", "MELON", "MILK", "WOOL", "TOMATO", "EGG", "CARROT", "FERTILIZER", "WHEAT"]
def _liq(a, obs):
    try:
        t = obs.get("step")
        if t is None: t = int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
        if int(t) < 718: return a
        a = dict(a); a["market"] = [["SELL", it, 1000] for it in _LIQ_ITEMS]; return a
    except Exception:
        return a

# ---- exp222: 在庫の軌道を強い Majkel に合わせる修理層 + 最終盤の売り切り ----
# 同じ局で BC と Majkel を比べると、第12日まで完全に一致し、第12日から BC だけ納屋に在庫が溜まっていく(+26個、所持金 −12.9k)。
# 強い Majkel の実戦90局から「時刻 × 品目の在庫の75%点」を作り(stock_target.json)、BC の在庫がそれを SLACK 以上超えたら超過分を売る。
# 小麦・肥料は餌と畑の原料なので対象外。BC の注文は削らず、足りない売りを後ろに足す(雇用や種の注文を落とさないため)。
import json as _json
_TGT = None
_OUT = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
SLACK = -3; LIQ_START = 710; LAST_STEP = 718   # exp224: v61 から余裕 3 → −3 だけを変えた(claude-main)
_VAL = {"STRAWBERRY": 5, "MELON": 5, "MILK": 4, "WOOL": 4, "TOMATO": 3, "FERTILIZER": 2, "EGG": 2, "CARROT": 1, "WHEAT": 0}
def _target():
    global _TGT
    if _TGT is None:
        p = os.path.join(_D, "stock_target.json")
        _TGT = _json.load(open(p)) if os.path.exists(p) else {}
    return _TGT
def _pace(a, obs):
    return _liq(a, obs)   # 装置なし + 最終手の売り切り
    try:
        t = obs.get("step")
        if t is None: t = int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
        t = int(t)
        if t > LAST_STEP: return a
        shed = dict((obs.get("private") or {}).get("shed") or {})
        mk = [list(o) for o in (a.get("market") or []) if isinstance(o, list) and o]
        if t >= LIQ_START:                               # 最終盤: 在庫 ÷ 残り手数、最後の手で全部
            left = LAST_STEP - t + 1
            mine = [o for o in mk if o[0] != "SELL" and not o[0].startswith("BUY")]
            if left <= 1:   # 最後の手: 同じ手に運び込まれる分も売れるよう「全部(1000)」で全品目を出す(在庫が無い品は即座に止まるだけ)
                sells = [["SELL", it, 1000] for it in sorted(_VAL, key=lambda k: -_VAL[k])]
            else:
                sells = [["SELL", it, -(-q // left)] for it in sorted(_VAL, key=lambda k: -_VAL[k]) for q in [int(shed.get(it, 0))] if q > 0]
            a = dict(a); a["market"] = (sells + mine)[:10]; return a
        T = _target()
        if not T: return a
        P = T["products"]; row = T["75"][t]
        want = {}
        for it in _OUT:
            ex = int(shed.get(it, 0)) - int(round(row[P.index(it)])) - SLACK
            if ex > 0: want[it] = ex
        if not want: return a
        for o in mk:                                     # BC がすでに売っているなら数量を増やす
            if o[0] == "SELL" and len(o) > 2 and o[1] in want:
                o[2] = max(int(o[2]), want.pop(o[1]))
        for it in sorted(want, key=lambda k: -_VAL[k]):
            if len(mk) >= 10: break
            mk.append(["SELL", it, want[it]])
        a = dict(a); a["market"] = mk[:10]
    except Exception:
        pass
    return a

def agent(obs, config=None):
    # 公式の読み込み器は引数の数を調べて呼ぶので、デコレータで包まない(包むと引数0個で呼ばれた)
    # 読み込み器は「最後に定義された関数」をエージェントにするので、agent は必ずファイルの最後に置く
    with torch.no_grad():
        o = _todict(obs)
        return _C1P_HOOK.apply(o, config, _liq(_act(o), o))   # 装置なし + 最終手の売り切り + C1' + W9'(網 → _liq → C1' + W9')
