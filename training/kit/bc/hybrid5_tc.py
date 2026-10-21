"""sp_lib が import する tape_codes(exp175 の hybrid5.py から、その関数だけを取り出したもの)。
   中身は exp175 当時のまま(テープの再生では合法判定をしない — 同じ手番の中の受け渡しを合法手で表せないため)。"""
import numpy as np
from extract4 import ucand, mcand, codes, NU, NS, M_STOP
from codec import QNONE, UQ_MAX, MQ_MAX, q_class


def tape_codes(a, ou, lm):
    live = [u for u in range(NU) if ou[u, 0] > 0]
    us = [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])
    ul = np.full(NU, -1, np.int8); uq = np.zeros(NU, np.int16)
    for u in live:
        if u < len(us):
            c = ucand(us[u]); q = int(us[u][2]) if len(us[u]) > 2 else QNONE
            ul[u] = c; uq[u] = q_class(q, UQ_MAX) if c >= 20 else 0
        else:
            ul[u] = 0
    ms = np.full(NS, M_STOP, np.int8); mq = np.zeros(NS, np.int16)
    for j, o in enumerate((a.get("market") or [])[:10]):
        c = mcand(o); ms[j] = c
        if c < 19: mq[j] = q_class(int(o[2]), MQ_MAX)
    return codes(ul, uq, live, ms, mq)
