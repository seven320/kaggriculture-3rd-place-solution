"""Round robin over every route we met on the ladder today, plus the one we ship."""
import sys, os, glob, itertools, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim, kagpack
D = "/content/drive/MyDrive/kaggriculture/data/"
T = {"boat29(v22)": kagpack.load(D + "boatlee_v29-r1-adaptive-market_n0_d1_0.tape.json")}
for f in sorted(glob.glob(D + "nemesis/*.tape.json")):
    T[os.path.basename(f).rsplit("_", 1)[0][:20]] = kagpack.load(f)
names = list(T)
seeds = list(range(int(os.environ.get("S0", 4300000)), int(os.environ.get("S0", 4300000)) + int(os.environ.get("NSEED", 60))))
W = {k: [0, 0, 0] for k in names}
for a, b in itertools.combinations(names, 2):
    r = kagsim.run_tapes(T[a][0], T[a][1], T[b][0], T[b][1], seeds, 720, 2)
    wa = int((r[:, 0] > r[:, 1]).sum()); wb = int((r[:, 0] < r[:, 1]).sum()); ti = len(r) - wa - wb
    W[a][0] += wa; W[a][1] += wb; W[a][2] += ti
    W[b][0] += wb; W[b][1] += wa; W[b][2] += ti
rank = sorted(W.items(), key=lambda kv: -(kv[1][0] + 0.5*kv[1][2])/max(1, sum(kv[1])))
print(f"{len(names)} 経路の総当たり ({len(seeds)} シード x 2座席)")
print(f"{'tape':<24}{'勝':>7}{'負':>7}{'分':>5}  勝率")
for k, (w, l, t) in rank:
    print(f"{k:<24}{w:>7}{l:>7}{t:>5}  {(w+0.5*t)/max(1,w+l+t):.4f}")
json.dump([[k, (v[0]+0.5*v[2])/max(1,sum(v))] for k, v in rank], open("/content/drive/MyDrive/kaggriculture/runs/rr47.json", "w"))
