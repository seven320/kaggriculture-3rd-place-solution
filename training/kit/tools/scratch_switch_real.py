"""実戦の切り替え型(v105 sw2 / v106 rs05)で、切り替えが起きた局と起きなかった局の成績。店は後ろに足されるだけなので最終手の一覧の先頭 K 軒で判定する。"""
import json, gzip, glob, collections, statistics as S
SUBS = [("v105 sw2", "data/rp/our56646963", 3, 2), ("v106 rs05", "data/rp/our56661176", 4, 2), ("v107 rs21", "data/rp/our56664798", 4, 2)]
for nm, d, K, M in SUBS:
    rows = []
    for fp in glob.glob(d + "/*.json*"):
        try: R = json.load(gzip.open(fp, "rt") if fp.endswith(".gz") else open(fp))
        except Exception: continue
        tn = R["info"].get("TeamNames") or []
        if len(tn) != 2 or tn.count("DSM") != 1: continue
        me = tn.index("DSM"); opp = tn[1 - me]
        last = R["steps"][-1]; rw = [s.get("reward") for s in last]
        if None in rw: continue
        shops = (last[me]["observation"].get("town") or {}).get("unlocked_shops") or []
        n = sum(1 for s in shops[:K] if s in ("PIZZA_SHOP", "FARMERS_MARKET"))
        rows.append((n >= M, opp, rw[me] - rw[1 - me]))
    print("==", nm, "対戦", len(rows), "局")
    for lab, f in (("全相手", lambda o: True), ("PFN", lambda o: o == "M & M & P & Q"), ("PFN 以外", lambda o: o != "M & M & P & Q")):
        for sw in (True, False):
            v = [m for s, o, m in rows if s == sw and f(o)]
            if v: print("  %-8s %-10s %3d局 %2d勝 平均差 %+6.0f" % (lab, "切り替え" if sw else "切り替えなし", len(v), sum(x > 0 for x in v), S.mean(v)))
    c = collections.Counter(o for s, o, m in rows if s)
    print("  切り替えた局の相手:", dict(c.most_common(8)))
