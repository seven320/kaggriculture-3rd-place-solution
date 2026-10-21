"""exp320(終盤係): 終盤の損の内訳を出すための記録つき対局。
league.py と同じ経路(Rust arena batch2 + 提出形エージェント)で打ち、毎手の
  市場の在庫 / 両者の所持金・納屋・雇用・区画 / 実際に出した注文
を全部 npz に残す。売買の約定は記録から後で公式どおりに再現できる(_process_market と同じ規則)ので、
対局中は「出した注文」だけ残して、品目ごとの売上は解析側で復元する。
使い方: AGENTS="A:bc:...,B:bc:..." NSEED=192 SEED0=.. OUT=/content/lab/egd_x.npz python3 eg_diag.py
"""
import os, sys, json, time, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import league as L                     # BC / load_mod / sanitize / ARENA をそのまま使う

IT = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"]
PR = IT[:9]; CROP = IT[:5]; ANI = IT[9:]
SHOPS = ["BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"]
SNAP = [144, 300, 400, 500, 550, 600, 650, 700, 718]
T = 719

def census(farm, priv):
    """区画の中身を数える -> 長さ 24 の並び"""
    c = np.zeros(24, np.int32)
    for row in farm["tiles"]:
        for t in row:
            if t is None: c[0] += 1; continue
            if t == "LOCKED": c[1] += 1; continue
            k = t["kind"]
            if k == "WEED": c[2] += 1; continue
            if k == "PLANT":
                c[3 + CROP.index(t["crop"])] += 1        # 3..7 作物ごとのマス
                c[8] += int(t["yield_units"])            # 実っている数の合計
                c[9] += int(t["watered_today"])
                continue
            a = t.get("animal")
            if k == "COOP":
                c[10] += 1
                if a: c[11] += 1; c[14] += int(t["yield_units"]); c[15] += int(t["fed_today"])
            else:
                c[12] += 1
                if a == "COW": c[13] += 1; c[16] += int(t["yield_units"])
                elif a == "SHEEP": c[17] += 1; c[18] += int(t["yield_units"])
                if a: c[15] += int(t["fed_today"])
    c[19] = len(farm["hands"]); c[20] = len(farm["unlocked_quadrants"])
    c[21] = int(sum(priv["shed"].values()))
    c[22] = int(sum(sum(d.values()) for d in priv["inventories"]))
    c[23] = int(sum(priv["seeds"].values()))
    return c

OPC = {"SELL":0, "BUY_SEED":1, "BUY_PRODUCT":2, "BUY_ANIMAL":3, "HIRE":4, "BUY_LAND":5}
def orders(mk, sell, bseed, bprod, bani, hire, land, ordv, i, p):
    for j, o in enumerate(mk[:10]):                 # 枠の順番そのまま(公式は枠の添字で両者を突き合わせる)
        if isinstance(o, list) and o and o[0] in OPC:
            ordv[i, p, j] = (OPC[o[0]], IT.index(o[1]) if len(o) > 1 and o[1] in IT else -1,
                             min(int(o[2]), 2000) if len(o) > 2 and isinstance(o[2], int) else 0)
    for o in mk:
        if not isinstance(o, list) or not o: continue
        op = o[0]
        if op == "HIRE": hire[i, p] += 1; continue
        if op == "BUY_LAND": land[i, p] += 1; continue
        if len(o) < 3 or o[1] not in IT: continue
        q = min(int(o[2]), 2000)
        if op == "SELL" and o[1] in PR: sell[i, p, PR.index(o[1])] += q
        elif op == "BUY_SEED" and o[1] in CROP: bseed[i, p, CROP.index(o[1])] += q
        elif op == "BUY_PRODUCT" and o[1] in ("WHEAT", "FERTILIZER"): bprod[i, p, 0 if o[1] == "WHEAT" else 1] += q
        elif op == "BUY_ANIMAL" and o[1] in ANI: bani[i, p, ANI.index(o[1])] += q

def main():
    out = os.environ.get("OUT", "/content/lab/egd.npz")
    spec = [x.split(":", 2) for x in os.environ["AGENTS"].split(",")]
    AG = {n: (L.BC(f) if t == "bc" else L.PY(f)) for n, t, f in spec}
    A, B = spec[0][0], spec[1][0]
    N = int(os.environ.get("NSEED", 192)); s0 = int(os.environ.get("SEED0", 880001))
    seeds = [s0 + 7919 * k for k in range(N)]; seats = [k % 2 for k in range(N)]   # seats[i] = A が座る席
    a, b = AG[A], AG[B]; a.start(N); b.start(N)

    money = np.zeros((T, N, 2), np.float32)          # 絶対座席
    inv   = np.zeros((T, N, 9), np.int32)
    shed  = np.zeros((T, N, 2, 12), np.int16)
    sell  = np.zeros((T, N, 2, 9), np.int16)
    bseed = np.zeros((T, N, 2, 5), np.int16)
    bprod = np.zeros((T, N, 2, 2), np.int16)
    bani  = np.zeros((T, N, 2, 3), np.int16)
    hire  = np.zeros((T, N, 2), np.int8)
    land  = np.zeros((T, N, 2), np.int8)
    hires = np.zeros((T, N, 2), np.int8)             # その日すでに雇った数(値段に効く)
    nquad = np.zeros((T, N, 2), np.int8)
    ordv  = np.full((T, N, 2, 10, 3), -1, np.int16)   # 注文の並び(op, 品目, 数量)
    cen   = np.zeros((len(SNAP), N, 2, 24), np.int32)
    shopv = np.full((len(SNAP), N, 16), -1, np.int8)

    pr = subprocess.Popen([L.ARENA, "batch2"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    pr.stdin.write(json.dumps({"games": [dict(seed=s) for s in seeds]}) + "\n"); pr.stdin.flush()
    t0 = time.time(); tt = -1
    while True:
        msg = json.loads(pr.stdout.readline())
        if msg.get("done"): res = msg["results"]; break
        tt += 1
        o = (msg["obs0"], msg["obs1"]); act = ([None] * N, [None] * N)
        si = SNAP.index(tt) if tt in SNAP else -1
        for i in range(N):
            if o[0][i] is None: continue
            ob = o[0][i]
            mk = ob["market"]["inventory"]
            inv[tt, i] = [mk[k] for k in PR]
            for p in (0, 1):
                f = ob["farms"][p]
                money[tt, i, p] = f["money"]; hires[tt, i, p] = min(127, f["hires_today"]); nquad[tt, i, p] = len(f["unlocked_quadrants"])
                sd = o[p][i]["private"]["shed"]
                shed[tt, i, p] = [sd.get(k, 0) for k in IT]
            if si >= 0:
                sh = [SHOPS.index(s) for s in ob["town"]["unlocked_shops"]][:16]
                shopv[si, i, :len(sh)] = sh
                for p in (0, 1): cen[si, i, p] = census(ob["farms"][p], o[p][i]["private"])
        for who, ag in ((0, a), (1, b)):
            idx = [i for i in range(N) if o[0][i] is not None]
            if not idx: continue
            sidx = [(i, seats[i] if who == 0 else 1 - seats[i]) for i in idx]
            acts = ag.act([o[s][i] for i, s in sidx], idx)
            for (i, s), ac in zip(sidx, acts):
                ac = L.sanitize(ac); act[s][i] = ac
                orders(ac["market"], sell[tt], bseed[tt], bprod[tt], bani[tt], hire[tt], land[tt], ordv[tt], i, s)
        pr.stdin.write(json.dumps({"a0": act[0], "a1": act[1]}) + "\n"); pr.stdin.flush()
        if tt % 120 == 0: print("t=%d %.0fs" % (tt, time.time() - t0), flush=True)
    pr.wait()
    fin = np.array(res, np.float32)                  # 絶対座席の最終所持金
    np.savez_compressed(out, money=money, inv=inv, shed=shed, sell=sell, bseed=bseed, bprod=bprod,
                        bani=bani, hire=hire, land=land, ordv=ordv, hires=hires, nquad=nquad, cen=cen, shops=shopv,
                        fin=fin, seats=np.array(seats, np.int8), seeds=np.array(seeds, np.int64),
                        snap=np.array(SNAP), names=np.array([A, B]))
    d = np.array([fin[i, 1 - seats[i]] - fin[i, seats[i]] for i in range(N)])
    print("書き出した %s  %d局  候補(%s)の差 %+.0f 勝率 %.3f  %.0fs" % (out, N, B, d.mean(), (d > 0).mean(), time.time() - t0), flush=True)

main()
