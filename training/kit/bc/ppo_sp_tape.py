"""exp194: 勝敗を報酬にした仕上げ。`ppo_sp.py`(exp176 の自己対戦 PPO)の相手 v55 を
   **tape_ug(v57 の土台)**に差し替えて回す。v55 は Drive の資材が要るので クラスタ では動かせない。
   common/ は1文字も変えていない: `sp_lib.V55Opp` を `TapeRouterOpp(OPPDIR)` に差し替えてから ppo_sp.main() を呼ぶだけ。

   なぜやるか(exp191): d2xl は**コインでは tape_ug に並んだ**(差 −1,431±1,786)のに**勝率は 0.250**。
   平均では並んでいるのに競り負けている。最終順位は Bradley-Terry = 勝敗だけなので、**勝ちの数**を増やす必要がある。
   環境変数 WIN を上げて「最後に勝ったか」の重みを強くする(既定は 1.0 = 差 5万コインぶんと同じ重み)。

   使い方: OPPDIR=<tape_ug の提出フォルダ> WIN=5 NV55=24 NTAPE=8 NSELF=0 ND2=0 NSNAP=0 \
           python3 ppo_sp_tape.py <tag> /content/local_logs/tf_d2xl.pt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_lib as S

OPPDIR = os.environ["OPPDIR"]
_Tape = S.TapeRouterOpp
S.V55Opp = lambda: _Tape(OPPDIR)          # ppo_sp の「v55」枠が tape_ug になる

import ppo_sp
print("相手の v55 枠を %s に差し替えた(WIN=%s SCALE=%s)" % (OPPDIR, os.environ.get("WIN", "1.0"), os.environ.get("SCALE", "50000")), flush=True)
ppo_sp.main()
