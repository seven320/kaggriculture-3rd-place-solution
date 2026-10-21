"""exp191: BC の重み 対 テープ系エージェント(v57 の土台 tape_ug など)を**エンジンの中**で。
   exp183 の vs_v55_s.py と同じ形。v55 は Drive の資材が要るので、代わりに提出フォルダ(actions.json + main.py)を
   `sp_lib.TapeRouterOpp` に読ませる。32シード x 両座席 = 64局、乱数の種 0 で再現可能。
   使い方: OPPDIR=<提出フォルダ> SEEDS=32 python3 vs_tape_s.py <重み.pt> [...]
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_lib as S, model_tf as M, ppo_tf as P

OPPDIR = os.environ["OPPDIR"]
NS = int(os.environ.get("SEEDS", 32))
SEEDS = [2_000_000 + 7919 * k for k in range(NS)]
GREEDY = os.environ.get("GREEDY", "0") == "1"
for w in sys.argv[1:]:
    net = M.load(w, P.DEV); torch.manual_seed(0); mg = []
    for s in (0, 1):
        m = S.play(SEEDS, {s: S.PolicyOpp(net, greedy=GREEDY, cap=True), 1 - s: S.TapeRouterOpp(OPPDIR)})
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg); n = len(mg); wr = float((mg > 0).mean())
    out = os.environ.get("OUT")
    if out:
        import json; json.dump([float(x) for x in mg], open(out, "w")); print("  局ごとの差を保存 %s" % out, flush=True)
    q = np.percentile(mg, [10, 25, 50, 75, 90])
    near = int((np.abs(mg) < 5000).sum())
    print("  分位(10/25/50/75/90%%): %s | |差|<5,000 の接戦 %d/%d | 負けた局の差の中央値 %+.0f | 勝った局の差の中央値 %+.0f" % (
        " ".join("%+.0f" % v for v in q), near, n,
        np.median(mg[mg <= 0]) if (mg <= 0).any() else 0, np.median(mg[mg > 0]) if (mg > 0).any() else 0), flush=True)
    print("%s %s 対 %s %d局: 勝率 %.3f±%.3f 差 %+.0f±%.0f" % (
        os.path.basename(w), "貪欲" if GREEDY else "確率的", os.path.basename(OPPDIR.rstrip("/")), n,
        wr, (wr * (1 - wr) / n) ** .5, mg.mean(), mg.std(ddof=1) / n ** .5), flush=True)
