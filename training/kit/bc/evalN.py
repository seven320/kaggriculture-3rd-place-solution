"""eval58 を一般化した評価: 未見29相手 x REPS シード(REPS=2 なら eval58 と同じ58局・同じ数字)。
   局ごとの結果も出せるので、標準誤差(= この指標でどこまで細かい差が見分けられるか)を測れる。
   環境変数: REPS(既定2)、MODE(g|s|gc|sc)、TF32(0/1)、OUT(局ごとの保存先.json)、SEED(既定0)
   使い方: python3 evalN.py <重み>
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tf32 = os.environ.get("TF32")
if tf32 is not None:
    torch.backends.cuda.matmul.allow_tf32 = (tf32 == "1")
    torch.backends.cudnn.allow_tf32 = (tf32 == "1")

import ppo_tf as P, model_tf as M

wpath = sys.argv[1]
REPS = int(os.environ.get("REPS", 2))
SEED = int(os.environ.get("SEED", 0))
mode = os.environ.get("MODE", "g")
greedy = "g" in mode; cap = "c" in mode
hold = json.load(open(os.environ.get("SPLITFILE", "/content/local_logs/split_tf2.json")))["hold"]
net = M.load(wpath, P.DEV)

t0 = time.time()
pool = P.Pool([(e, P.load_meta(e)[0]["seed"]) for e in hold], REPS)
torch.manual_seed(SEED)
for t in range(719):
    st = pool.state()
    ul, uq, ms, mq, _ = P.sample(net, st, greedy=greedy, cap=cap)
    U, O = P.to_codes(ul, uq, ms, mq, st[1])
    pool.step(U, O)
mg, me, op = pool.margin()
n = len(mg)
sd = float(np.std(mg, ddof=1)); sem = sd / n ** 0.5
wr = float((mg > 0).mean()); wsem = (wr * (1 - wr) / n) ** 0.5
print("%s mode=%s reps=%d n=%d tf32=%s: 自分 %.0f 差 %+.0f±%.0f 勝率 %.3f±%.3f (1局の標準偏差 %.0f, %.1f分)" % (
    os.path.basename(wpath), mode, REPS, n, torch.backends.cuda.matmul.allow_tf32,
    me.mean(), mg.mean(), sem, wr, wsem, sd, (time.time() - t0) / 60), flush=True)
out = os.environ.get("OUT")
if out:
    json.dump([dict(eid=hold[i // REPS], rep=i % REPS, me=float(me[i]), op=float(op[i]), mg=float(mg[i])) for i in range(n)],
              open(out, "w"))
    print("保存 %s" % out, flush=True)
