"""eval58 と同じ58局を、**局ごと**の最終所持金つきで出す(exp186: 機械が違うと何が起きるかの測定)。
   eval_policy と同じ手順(Pool・sample・to_codes・719手)。集計値は eval58.py と一致する。
   環境変数: TF32=0/1、MODE=g|s、OUT=<保存先.json>
   使い方: python3 eval58_pergame.py <重み>
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tf32 = os.environ.get("TF32")
if tf32 is not None:
    torch.backends.cuda.matmul.allow_tf32 = (tf32 == "1")
    torch.backends.cudnn.allow_tf32 = (tf32 == "1")

import ppo_tf as P, model_tf as M

wpath = sys.argv[1]
mode = os.environ.get("MODE", "g")
greedy = "g" in mode; cap = "c" in mode
hold = json.load(open("/content/local_logs/split_tf2.json"))["hold"]
net = M.load(wpath, P.DEV)

pool = P.Pool([(e, P.load_meta(e)[0]["seed"]) for e in hold], 2)
torch.manual_seed(0)
for t in range(719):
    st = pool.state()
    ul, uq, ms, mq, _ = P.sample(net, st, greedy=greedy, cap=cap)
    U, O = P.to_codes(ul, uq, ms, mq, st[1])
    pool.step(U, O)
mg, me, op = pool.margin()

rows = [dict(eid=hold[i // 2], rep=i % 2, me=float(me[i]), op=float(op[i]), mg=float(mg[i])) for i in range(len(mg))]
print("torch %s tf32=%s mode=%s" % (torch.__version__, torch.backends.cuda.matmul.allow_tf32, mode), flush=True)
print("集計: 自分 %.0f 差 %.0f 勝率 %.3f" % (me.mean(), mg.mean(), (mg > 0).mean()), flush=True)
out = os.environ.get("OUT")
if out:
    json.dump(rows, open(out, "w"))
    print("保存 %s" % out, flush=True)
for r in rows[:6]:
    print("  %s rep%d 自分 %9.0f 相手 %9.0f 差 %+9.0f" % (r["eid"], r["rep"], r["me"], r["op"], r["mg"]), flush=True)
