"""標準評価: 未見29相手 x 2シード(=58局)。確率的/貪欲、種の上限あり/なし。"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M
wpath = sys.argv[1]; modes = sys.argv[2].split(",")
hold = json.load(open("/content/local_logs/split_tf2.json"))["hold"]
net = M.load(wpath, P.DEV)
for m in modes:
    greedy = "g" in m; cap = "c" in m
    me, wr, mg = P.eval_policy(net, hold, greedy, 2, cap=cap)
    print("%s %s: 自分 %.0f 差 %.0f 勝率 %.3f" % (os.path.basename(wpath), m, me, mg, wr), flush=True)
