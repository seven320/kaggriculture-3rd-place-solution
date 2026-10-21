#!/usr/bin/env python3
"""1局を回して内訳を出す。どこで金を取りこぼしているかを特定する。"""
import sys, os, json, collections
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE)
from kaggle_environments import make

def run(a_spec, b_spec="starter", seed=7, steps=720):
    env=make("kaggriculture",configuration={"episodeSteps":steps,"seed":seed})
    env.run([a_spec,b_spec])
    return env

def diag(env, p=0):
    steps=env.steps
    sold=collections.Counter(); revenue=collections.Counter()
    acts=collections.Counter(); market_ops=collections.Counter()
    hands_per_day={}; money_by_day={}; tiles_by_day={}
    prev_money=None
    for i,st in enumerate(steps):
        obs=st[0].observation
        day=obs.get("day",0); hour=obs.get("hour",0)
        farm=obs.farms[p]
        if hour==23 or i==len(steps)-1:
            money_by_day[day]=farm["money"]
            c=collections.Counter()
            for row in farm["tiles"]:
                for t in row:
                    if t is None: c["empty"]+=1
                    elif t=="LOCKED": c["locked"]+=1
                    elif isinstance(t,dict):
                        k=t.get("kind")
                        c[t.get("crop") if k=="PLANT" else (t.get("animal") or k)]+=1
            tiles_by_day[day]=dict(c)
        hands_per_day[day]=max(hands_per_day.get(day,0), len(farm.get("hands",[])))
        a=st[p].action
        if isinstance(a,dict):
            f=a.get("farmer") or ["PASS"]
            acts[f[0] if isinstance(f,list) and f else "PASS"]+=1
            for h in (a.get("hands") or []):
                acts[h[0] if isinstance(h,list) and h else "PASS"]+=1
            for m in (a.get("market") or []):
                if isinstance(m,list) and m:
                    market_ops[m[0]]+=1
                    if m[0]=="SELL" and len(m)>=3:
                        sold[m[1]]+=0  # 実約定は下で価格差から推定
    # 実約定: 各ステップの市場在庫の増分から売却量を推定(両者合算なので参考値)
    fin=steps[-1][0].observation
    print(f"  最終所持金 P{p} = ${fin.farms[p]['money']:,.0f}   相手 ${fin.farms[1-p]['money']:,.0f}")
    print(f"\n  --- 行動の内訳(全ユニット・全ターン) 合計 {sum(acts.values()):,} ---")
    tot=sum(acts.values())
    for k,v in acts.most_common():
        print(f"    {k:<20s} {v:>7,}  {v/tot*100:>5.1f}%")
    print(f"\n  --- 市場注文 ---")
    for k,v in market_ops.most_common(): print(f"    {k:<16s} {v:>6,}")
    print(f"\n  --- 1日の最大雇用人数 ---")
    print("   ", " ".join(f"d{d}:{n}" for d,n in sorted(hands_per_day.items())[:31]))
    print(f"\n  --- 所持金の推移(日終) ---")
    print("   ", " ".join(f"d{d}:{int(m):,}" for d,m in sorted(money_by_day.items())))
    print(f"\n  --- タイル構成 ---")
    for d in [4,9,14,19,24,29]:
        if d in tiles_by_day:
            print(f"    d{d:<3d} {tiles_by_day[d]}")
    m=steps[-1][0].observation.market
    print(f"\n  --- 市場の最終在庫と価格(I0=10000 からの差 = 売った量の目安)---")
    for it,inv in m["inventory"].items():
        print(f"    {it:<12s} 在庫 {inv:>7,} (I0比 {inv-10000:+6,})  価格 ${m['prices'][it]:>4}")

if __name__=="__main__":
    a=sys.argv[1] if len(sys.argv)>1 else "agent/main.py"
    b=sys.argv[2] if len(sys.argv)>2 else "starter"
    sd=int(sys.argv[3]) if len(sys.argv)>3 else 7
    env=run(os.path.join(BASE,a) if a.endswith(".py") else a, b, sd)
    diag(env,0)

def revenue_breakdown(a_spec, seed=7, steps=720):
    """相手を 'pass'(何も売らない)にして、市場在庫の増分から自分の売上を商品別に出す。"""
    env=make("kaggriculture",configuration={"episodeSteps":steps,"seed":seed})
    env.run([a_spec,"pass"])
    import collections
    units=collections.Counter(); rev=collections.Counter()
    prev=None
    for st in env.steps:
        m=st[0].observation.market
        inv=dict(m["inventory"]); pr=dict(m["prices"])
        if prev is not None:
            for it,v in inv.items():
                d=v-prev[0][it]
                if d>0:                       # 在庫増 = 我々が売った
                    units[it]+=d; rev[it]+=d*prev[1][it]
        prev=(inv,pr)
    fin=env.steps[-1][0].observation.farms[0]["money"]
    print(f"\n  === 商品別の売上(相手=pass、seed={seed})  最終所持金 ${fin:,.0f} ===")
    print(f"  {'商品':<12s} {'販売数':>7s} {'概算売上':>11s} {'平均単価':>8s}   {'プール上限':>10s}")
    CAP={"WHEAT":23843,"CARROT":11038,"TOMATO":11799,"STRAWBERRY":4947,"MELON":27527,
         "EGG":47399,"MILK":7305,"WOOL":9069,"FERTILIZER":25752}
    tot=0
    for it,n in units.most_common():
        tot+=rev[it]
        print(f"  {it:<12s} {n:>7,} ${rev[it]:>10,.0f} ${rev[it]/max(n,1):>7.0f}   "
              f"${CAP.get(it,0):>9,}  ({rev[it]/max(CAP.get(it,1),1)*100:>4.1f}% 到達)")
    print(f"  {'合計':<12s} {sum(units.values()):>7,} ${tot:>10,.0f}"
          f"          (全プール $168,679 の {tot/168679*100:.1f}%)")
