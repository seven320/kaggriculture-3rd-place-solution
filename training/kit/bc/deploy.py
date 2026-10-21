"""方策の行動を組み立ててエンジンに渡す経路。学習の前にここで恒等式を通す。

過去に「正解行動を入れても deploy 経路は69%失う」(実験3)を踏んでいるので、
**正解ラベルを流して残高がテープと一致すること**を先に確認してから学習する。
注意: codec.ITEMS は辞書順、kagpack.ITEMS は別順。必ず文字列経由で変換する。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
from codec import (enc_step, dec_unit, dec_mkt, UVERB, ITEMS, MOPS, QNONE,
                   UQ_MAX, MQ_MAX, q_class, q_value)
import kagpack
NU = 20; NSLOT = 10; NITEM = len(ITEMS)   # hands は実測最大16 = ユニット17体。余裕を見て20
ALL_Q = 9999


def wide_blank(n=720, nh=NU - 1):
    """幅広の空テープ。step_units が制御できるユニット数は**テープの宣言**で頭打ちになる。"""
    return kagpack.pack([{"farmer": ["PASS"], "hands": [["PASS"]]*nh, "market": []} for _ in range(n)])


def units_to_codes(uv, ui, uq, nu):
    """(動詞クラス, 品目クラス, 数量クラス) -> kagsim の (op, arg, n)。"""
    out = np.zeros((NU, 3), np.int32)
    out[:, 0] = 255                       # 255 = テープのまま(=触らない)
    for u in range(min(nu, NU)):
        a = dec_unit((int(uv[u]),
                      int(ui[u]) if int(ui[u]) < NITEM else 0,
                      q_value(int(uq[u]), UQ_MAX)))
        op = kagpack.OPI.get(a[0], 0)
        arg = kagpack.ITI.get(a[1], 0) if len(a) >= 2 else 0
        n = int(a[2]) if len(a) >= 3 else 1
        out[u] = (op, arg, n)
    return out


def mkt_to_codes(slot):
    """(操作クラス, 品目クラス, 数量クラス) x 10 -> kagsim の (op, item, n)。"""
    out = np.full((NSLOT, 3), 0, np.int32)
    out[:, 0] = 255                                     # 既定は終端
    for j in range(NSLOT):
        op = int(slot[j, 0])
        if op >= len(MOPS): break                       # 注文なし = ここで終端
        # market の "PASS" と [] はどちらも**枠を占める空注文**。
        # 終端(255)と取り違えると以降の注文を全部落とす(HIRE 9連発が消えて -45,544 になった)。
        if MOPS[op] in ("EMPTY", "PASS"):
            out[j] = (254, 0, 0); continue
        q = q_value(int(slot[j, 2]), MQ_MAX)
        # 売りは保有量で頭打ちになるので上限超えを「全部」にしてよい。
        # 買いは**所持金が尽きるまで買い続ける**ので、決して「全部」にしてはいけない。
        if q != QNONE and q > MQ_MAX:
            q = ALL_Q if MOPS[op] == "SELL" else MQ_MAX
        o = dec_mkt((op, int(slot[j, 1]) if int(slot[j, 1]) < NITEM else 0, q))
        if not o: continue
        mo = kagpack.MOPI.get(o[0], 0)
        if mo == 0:                                     # 未対応の操作も枠は占める
            out[j] = (254, 0, 0); continue
        if mo in (1, 2):                                # HIRE / BUY_LAND は引数なし
            out[j] = (mo, 0, 1); continue
        if len(o) < 3: continue
        out[j] = (mo, kagpack.ITI.get(o[1], 0), int(o[2]))
    return out


def encode_true(step_act):
    """テープの1手 -> (uv, ui, uq, slot)。抽出と同じ形。"""
    e = enc_step(step_act)
    us = [e["farmer"]] + list(e["hands"])
    uv = np.zeros(NU, np.int64); ui = np.full(NU, NITEM, np.int64); uq = np.zeros(NU, np.int64)
    for u in range(min(NU, len(us))):
        v, it, q = us[u]
        uv[u] = v; ui[u] = it if it >= 0 else NITEM; uq[u] = q_class(q, UQ_MAX)
    slot = np.zeros((NSLOT, 3), np.int64); slot[:, 0] = len(MOPS)
    for j, m in enumerate(e["market"][:NSLOT]):
        slot[j] = (m[0], m[1] if m[1] >= 0 else NITEM, q_class(m[2], MQ_MAX))
    return uv, ui, uq, slot, len(us)
