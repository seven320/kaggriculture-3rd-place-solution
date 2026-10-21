# exp312: legal_mask を緩めた版(LEGAL_OPT)で、無効手を待機に置換しても(MASKSUB=1)棋譜を再現できるか
import json, sys, glob, gzip, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract4 import extract_game

def rj(p):
    return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

TEAMS = (("redblackbst", "redblackbst_*"), ("Catalyst", "Catalyst_*"), ("mtmr_s1", "mtmr_s1_*"),
         ("leave you", "leave_you_*"), ("DSM", "DSM_*"), ("Majkel1337", "Majkel1337_*"))
NG = int(os.environ.get("NGAME", "4"))
tot_ok = tot_n = 0
for team, pat in TEAMS:
    ok = n = 0; ds = []
    for f in sorted(glob.glob("/content/data/rp/latest/%s/*.json*" % pat))[:NG]:
        d = rj(f); tn = d["info"]["TeamNames"]
        if team not in tn: continue
        seat = tn.index(team); rw = d["rewards"]; n += 1
        try:
            X, fin, no = extract_game(d, seat)
        except Exception:
            continue
        ds.append(fin[0] - rw[seat])
        if abs(fin[0] - rw[seat]) <= .5 and abs(fin[1] - rw[1 - seat]) <= .5: ok += 1
    if n:
        print("   %-13s 一致 %d/%d  平均の差 %+8.0f" % (team, ok, n, np.mean(ds) if ds else 0), flush=True)
        tot_ok += ok; tot_n += n
print("   合計 %d/%d" % (tot_ok, tot_n), flush=True)
