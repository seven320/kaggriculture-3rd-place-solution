"""extract4 で全局を抽出してローカルに保存。公式の残高と一致しない対局は捨てる。"""
import json, glob, os, sys, time, gc
import gzip
def _rj(p):
    """09-21: .json も .json.gz も読む。"""
    return __import__("json").load(gzip.open(p,"rt") if p.endswith(".gz") else open(p))

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract4 import extract_game
rdir, team, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
ok = bad = 0; t0 = time.time()
for f in sorted((glob.glob(rdir + "/*.json") + glob.glob(rdir + "/*.json.gz"))):
    eid = os.path.basename(f)[:-5]; p = "%s/%s.npz" % (out, eid)
    if os.path.exists(p): ok += 1; continue
    d = _rj(f); tn = d["info"]["TeamNames"]
    if team not in tn: continue
    seat = tn.index(team); rw = d["rewards"]
    X, fin, noops = extract_game(d, seat)
    if abs(fin[0]-rw[seat]) > .5 or abs(fin[1]-rw[1-seat]) > .5:
        print("  %s 不一致 %s vs %s -> 捨てる" % (eid, fin, rw)); bad += 1; continue
    meta = dict(seat=seat, seed=int(d["info"]["seed"]), teacher=rw[seat], opp=rw[1-seat], opp_name=tn[1-seat])
    np.savez(p, meta=json.dumps(meta), opp_acts=json.dumps(__import__("extract3").acts_of(d, 1-seat)), **X); ok += 1
    del d, X; gc.collect()
print("保存 %d 局 / 不一致で捨てた %d 局 (%.1f分)" % (ok, bad, (time.time()-t0)/60), flush=True)
