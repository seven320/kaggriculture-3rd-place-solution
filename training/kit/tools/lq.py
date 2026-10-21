"""exp292: リーグの常駐ワーカーに依頼を出す。192局×本数に分けて並べ、結果を合算して表示もできる。
  依頼: python3 tools/lq.py put <名前> "<相手名>:bc:<フォルダ>,<候補名>:bc:<フォルダ>" [本数=4] [局数=192]
  結果: python3 tools/lq.py get <名前>      (候補 = AGENTS の2番目。勝率 ± 誤差、z)"""
import sys, json, os, glob, math, re, random
W = "/workspace/kaggriculture/lab"; Q = W + "/leagueq"
cmd = sys.argv[1]
if cmd == "put":
    name, agents = sys.argv[2], sys.argv[3]; k = int(sys.argv[4]) if len(sys.argv) > 4 else 4; ns = int(sys.argv[5]) if len(sys.argv) > 5 else 192
    os.makedirs(Q + "/todo", exist_ok=True); base = random.randrange(10 ** 6, 9 * 10 ** 6)
    for i in range(k):
        json.dump({"agents": agents.replace(W, "/content/lab"), "nseed": ns, "seed0": base + 100003 * i, "out": "/content/lab/lq_%s_%d.json" % (name, i)},
                  open("%s/todo/%s_%d.json" % (Q, name, i), "w"))
    print("依頼 %d 本(%s)" % (k, name))
elif cmd == "get":
    name = sys.argv[2]; ws = []; mg = []
    for f in sorted(glob.glob("%s/lq_%s_*.json" % (W, name))):
        R = json.load(open(f)); key = [x for x in R if "|" in x and "#" not in x][0]; v = R[key]
        ws += [1.0 if b > a else (0.5 if a == b else 0.0) for a, b in v]; mg += [b - a for a, b in v]
    if not ws: print(name, "まだ"); sys.exit()
    n = len(ws); p = sum(ws) / n; se = math.sqrt(p * (1 - p) / n)
    print("%-24s %4d局 候補の勝率 %.3f ± %.3f z=%+.2f 差 %+.0f" % (name, n, p, se, (p - .5) / se if se else 0, sum(mg) / n))
