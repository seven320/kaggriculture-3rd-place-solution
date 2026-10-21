"""exp202: ppo_tf.to_codes(Python で1局・1ユニットずつ)と**完全に同じ出力**を、表引きの配列処理で出す。
   unit_code(c, qc)・mkt_code(c, qc) は (候補, 数量クラス) だけで決まる純粋な関数なので、全組み合わせの表を1回作れば済む。
   common/ は変えていない。check() で to_codes と一致するかを乱数の入力で確かめる。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract4 import unit_code, mkt_code, NU, NC, NM, NS, M_STOP
from codec import UQ_MAX, MQ_MAX

UQN_ = UQ_MAX + 2          # 数量クラスの取りうる値の上限(余裕をもって)
MQN_ = MQ_MAX + 2
LUT_U = np.zeros((NC, UQN_, 3), np.int32)
for c in range(NC):
    for q in range(UQN_):
        try: LUT_U[c, q] = unit_code(c, q)
        except Exception: LUT_U[c, q] = unit_code(c, 0)
LUT_M = np.zeros((NM, MQN_, 3), np.int32)
for c in range(NM):
    for q in range(MQN_):
        if c == M_STOP: LUT_M[c, q] = (255, 0, 0); continue
        try: LUT_M[c, q] = mkt_code(c, q)
        except Exception: LUT_M[c, q] = mkt_code(c, 0)


def to_codes_fast(ul, uq, ms, mq, ou):
    """ppo_tf.to_codes と同じ (U, O)。ul/uq: (B, NU)、ms/mq: (B, NS)、ou: (B, NU, >=1)。"""
    ul = np.asarray(ul); uq = np.asarray(uq); ms = np.asarray(ms); mq = np.asarray(mq)
    B = len(ul)
    live = np.asarray(ou)[:, :NU, 0] > 0
    U = LUT_U[np.clip(ul, 0, NC - 1), np.clip(uq, 0, UQN_ - 1)]
    U = np.where(live[..., None], U, np.array([255, 0, 0], np.int32))
    # 市場: 最初の終了(M_STOP)より後ろは 255、10枠まで
    m10 = ms[:, :10]; q10 = mq[:, :10]
    O = LUT_M[np.clip(m10, 0, NM - 1), np.clip(q10, 0, MQN_ - 1)]
    stop = (m10 == M_STOP)
    after = np.cumsum(stop, 1) > 0
    O = np.where(after[..., None], np.array([255, 0, 0], np.int32), O)
    return U.astype(np.int32), O.astype(np.int32)


def check(B=2000, seed=0):
    import ppo_tf as P
    rng = np.random.default_rng(seed)
    ou = np.zeros((B, NU, 1), np.float32); ou[..., 0] = rng.random((B, NU)) < 0.6
    ul = rng.integers(-1, NC, (B, NU)); uq = rng.integers(0, UQ_MAX + 1, (B, NU))
    ms = rng.integers(0, NM, (B, NS)); mq = rng.integers(0, MQ_MAX + 1, (B, NS))
    ul = np.where(ou[..., 0] > 0, np.clip(ul, 0, NC - 1), -1)
    import time
    t0 = time.time(); U0, O0 = P.to_codes(ul, uq, ms, mq, ou); t1 = time.time(); U1, O1 = to_codes_fast(ul, uq, ms, mq, ou); t2 = time.time()
    ok = np.array_equal(U0, U1) and np.array_equal(O0, O1)
    print("to_codes と一致: %s(ユニット不一致 %d・市場不一致 %d / %d局)。時間 %.1f ms → %.2f ms(%.0f倍)" % (
        ok, int((U0 != U1).any(-1).sum()), int((O0 != O1).any(-1).sum()), B, 1000 * (t1 - t0), 1000 * (t2 - t1), (t1 - t0) / max(t2 - t1, 1e-9)))
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
