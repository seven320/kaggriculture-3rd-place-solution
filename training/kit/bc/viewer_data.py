"""解説用の棋譜データ(1局)を作る。公式リプレイ → 盤面・お金・価格・行動を小さくまとめた JSON。
   使い方: python3 viewer_data.py <リプレイ.json> <我々のチーム名> <出力.json>"""
import json, sys
CROPS=["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"]; CH="WCTSM"
ITEMS=["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"]

def tile_code(t):
    if t is None: return ".", "0"
    if t=="LOCKED": return "#","0"
    k=t.get("kind")
    if k=="WEED": return "w","0"
    if k=="PLANT":
        return CH[CROPS.index(t["crop"])], str(min(9,int(t.get("yield_units",0) or 0)))
    a=t.get("animal")
    ch={"GOOSE":"g","COW":"c","SHEEP":"s"}.get(a, "o" if k=="COOP" else "p")
    return ch, str(min(9,int(t.get("yield_units",0) or 0)))

rp, team, out = sys.argv[1], sys.argv[2], sys.argv[3]
d=json.load(open(rp)); tn=d["info"]["TeamNames"]; me=tn.index(team); steps=d["steps"]
S=[]
for i,st in enumerate(steps):
    o=st[0].get("observation") or {}
    if not o.get("farms"): continue
    row={"m":[],"t":[],"y":[],"f":[],"h":[],"q":[]}
    for p in (me,1-me):
        fa=o["farms"][p]; tiles="".join(tile_code(fa["tiles"][yy][xx])[0] for yy in range(10) for xx in range(10))
        ylds="".join(tile_code(fa["tiles"][yy][xx])[1] for yy in range(10) for xx in range(10))
        row["m"].append(int(fa["money"])); row["t"].append(tiles); row["y"].append(ylds)
        row["f"].append(fa["farmer"]); row["h"].append(fa["hands"]); row["q"].append(len(fa["unlocked_quadrants"]))
    row["p"]=[int((o.get("market") or {}).get("prices",{}).get(k,0)) for k in ITEMS]
    row["s"]=list((o.get("town") or {}).get("unlocked_shops") or [])
    a=(steps[i][me].get("action") or {}) if i<len(steps) else {}
    row["a"]=[a.get("farmer") or ["PASS"]]+list(a.get("hands") or [])
    row["k"]=[x for x in (a.get("market") or []) if isinstance(x,list)][:10]
    # 納屋(shed)・種・各ユニットの持ち物。両プレイヤーぶん(相手の納屋は本来見えないが、解説用に見せる)
    row["d"]=[]; row["sd"]=[]; row["iv"]=[]
    for p2 in (me,1-me):
        pv=(steps[i][p2].get("observation") or {}).get("private") or {}
        sh=pv.get("shed") or {}; sd=pv.get("seeds") or {}
        row["d"].append([int(sh.get(k,0)) for k in ITEMS+["GOOSE","COW","SHEEP"]])
        row["sd"].append([int(sd.get(k,0)) for k in CROPS])
        row["iv"].append(sum(sum(x.values()) for x in (pv.get("inventories") or []) if isinstance(x,dict)))
    S.append(row)
json.dump({"eid":rp.split("/")[-1][:-5],"me":tn[me],"opp":tn[1-me],"final":[d["rewards"][me],d["rewards"][1-me]],"steps":S}, open(out,"w"), separators=(",",":"))
print("手数 %d、書き出し %s" % (len(S), out))
