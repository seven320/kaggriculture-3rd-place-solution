"""exp192: BC の方策に「終盤の清算」を載せる(v55 / ルーター系がやっている層)。
   BC はテープと違って最後まで納屋に物を残しがち。最終手で残っていると価値ゼロなので、
   step LIQ 以降、空いている注文枠に納屋の売れ残りの SELL を入れる。
   common/ は変えていない(sp_lib.PolicyOpp を継承した新しいクラス)。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagpack
import sp_lib as S

SELL = kagpack.MOPI["SELL"]
ALL_Q = 9999
SHED0, NIT = 18, 9          # g(obs_legal)の中の自分の納屋 9品目(V55Opp と同じ位置)


class PolicyLiqOpp(S.PolicyOpp):
    def __init__(self, net, greedy=True, cap=True, liq_from=700):
        S.PolicyOpp.__init__(self, net, greedy, cap); self.liq = int(liq_from)

    def act(self, st, t, idx, pool, seat):
        U, O = S.PolicyOpp.act(self, st, t, idx, pool, seat)
        if t >= self.liq:
            g = st[3]
            for j in range(len(O)):
                shed = np.rint(g[j, SHED0:SHED0 + NIT] * 100).astype(int)
                have = {int(o[1]) for o in O[j] if int(o[0]) == SELL}
                free = [k for k in range(10) if int(O[j][k][0]) == 255]
                for it in range(NIT):
                    if shed[it] > 0 and it not in have and free:
                        O[j][free.pop(0)] = (SELL, it, ALL_Q)
        return U, O


def shed_at_end(pool, seat):
    """終局時の納屋(12品目)を座席ごとに。"""
    out = []
    for env in pool.envs:
        env.set_view(int(seat)); out.append(np.asarray(env.shed_all())[0])
    return np.stack(out)
