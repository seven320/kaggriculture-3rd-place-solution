"""eval58.py と同じ評価を、数値の設定を変えながら回す(exp186 の原因切り分け用)。
   eval58.py 本体は変えていない。環境変数:
     TF32=0/1        行列積の TF32(既定は torch まかせ)
     DETERM=1        torch.use_deterministic_algorithms(True)
     MODES=g,s       評価の種類
     REPEAT=2        同じ評価を何回か(実行ごとの再現性を見る)
   使い方: python3 eval58_diag.py <重み>
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tf32 = os.environ.get("TF32")
if tf32 is not None:
    torch.backends.cuda.matmul.allow_tf32 = (tf32 == "1")
    torch.backends.cudnn.allow_tf32 = (tf32 == "1")
if os.environ.get("DETERM") == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

import ppo_tf as P, model_tf as M

print("torch %s  cuda %s  gpu %s" % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)), flush=True)
print("torch のファイル: %s" % torch.__file__, flush=True)
print("allow_tf32 matmul=%s cudnn=%s" % (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), flush=True)

wpath = sys.argv[1]
hold = json.load(open("/content/local_logs/split_tf2.json"))["hold"]
net = M.load(wpath, P.DEV)

# 重みの指紋(重みそのものがずれていないかの確認)
h = 0.0
for k, v in sorted(net.state_dict().items()):
    h += float(v.double().sum())
print("重みの合計 %.6f  パラメータ %d" % (h, sum(p.numel() for p in net.parameters())), flush=True)

# 相手・シードの指紋(データがずれていないかの確認)
seeds = [P.load_meta(e)[0]["seed"] for e in hold]
seats = [P.load_meta(e)[0]["seat"] for e in hold]
print("hold 29局 シードの和 %d  座席の和 %d" % (sum(seeds), sum(seats)), flush=True)

for m in os.environ.get("MODES", "g").split(","):
    greedy = "g" in m; cap = "c" in m
    for r in range(int(os.environ.get("REPEAT", 1))):
        me, wr, mg = P.eval_policy(net, hold, greedy, 2, cap=cap)
        print("%s %s 回%d: 自分 %.0f 差 %.0f 勝率 %.3f" % (os.path.basename(wpath), m, r + 1, me, mg, wr), flush=True)
