"""exp192: 終盤の清算を載せた BC 対 テープ系。清算なしと同じシード・同じ相手で比べる(対応差)。
   おまけに「終局時に納屋に何が残っていたか」も出す(清算が効く理屈の確認)。
   使い方: OPPDIR=<提出フォルダ> SEEDS=32 LIQ=700 python3 vs_tape_liq.py <重み.pt> [...]
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_lib as S, model_tf as M, ppo_tf as P
from liq_lib import PolicyLiqOpp, shed_at_end

OPPDIR = os.environ["OPPDIR"]; NS = int(os.environ.get("SEEDS", 32))
LIQ = int(os.environ.get("LIQ", 700)); GREEDY = os.environ.get("GREEDY", "1") == "1"
SEEDS = [2_000_000 + 7919 * k for k in range(NS)]
IT = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]


def run(net, use_liq):
    """座席ごとに相手を作り直して(相手は局をまたいで状態を持つ)、両座席ぶんの差を返す。"""
    mg = []
    for s in (0, 1):
        me = PolicyLiqOpp(net, GREEDY, True, LIQ) if use_liq else S.PolicyOpp(net, GREEDY, True)
        m = S.play(SEEDS, {s: me, 1 - s: S.TapeRouterOpp(OPPDIR)})
        mg.append(m[:, s] - m[:, 1 - s])
    return np.concatenate(mg)


for w in sys.argv[1:]:
    net = M.load(w, P.DEV)
    torch.manual_seed(0); a = run(net, False)
    torch.manual_seed(0); b = run(net, True)
    n = len(a); d = (b > 0).astype(float) - (a > 0).astype(float); dm = b - a
    f = lambda x: ((x > 0).mean(), ((x > 0).mean() * (1 - (x > 0).mean()) / n) ** .5, x.mean(), x.std(ddof=1) / n ** .5)
    print("%s %s 対 %s  %d局" % (os.path.basename(w), "貪欲" if GREEDY else "確率的", os.path.basename(OPPDIR.rstrip("/")), n), flush=True)
    print("  清算なし      : 勝率 %.3f±%.3f 差 %+.0f±%.0f" % f(a), flush=True)
    print("  清算あり(%d〜): 勝率 %.3f±%.3f 差 %+.0f±%.0f" % ((LIQ,) + f(b)), flush=True)
    print("  対応差        : 勝率 %+.3f±%.3f (%.1fσ)  差 %+.0f±%.0f (%.1fσ)" % (
        d.mean(), d.std(ddof=1) / n ** .5, abs(d.mean()) / (d.std(ddof=1) / n ** .5 + 1e-12),
        dm.mean(), dm.std(ddof=1) / n ** .5, abs(dm.mean()) / (dm.std(ddof=1) / n ** .5 + 1e-12)), flush=True)
