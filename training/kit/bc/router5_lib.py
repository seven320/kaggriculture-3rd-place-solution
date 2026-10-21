import json, glob, os, sys
import numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast"); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
import kagsim, kagpack
from extract3 import acts_of
def shops144(tape_acts, g):
    op = kagpack.pack(acts_of(json.load(open(glob.glob("/content/local_rp/our*/%s.json" % g["eid"])[0])), 1 - g["seat"]))
    me = kagpack.pack(tape_acts); A, Bt = (me, op) if g["seat"] == 0 else (op, me)
    env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [g["seed"]], 720, g["seat"]); env.keep_nonsell = False; env.reset()
    while env.step_index() < 144:
        env.step2(np.asarray(env.tape_orders())[None], np.asarray(env.opp_tape_orders()).reshape(1, 10, 3))
    return tuple(sorted(int(x) for x in np.asarray(env.shops())[0] if x >= 0))
