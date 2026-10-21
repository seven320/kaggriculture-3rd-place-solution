"""同じ局(同じシード・相手は実戦の手を再生)で、差し替えたエージェントと実戦の本人の打ち方を1手ずつ比べる。
   使い方: python3 lab_diverge.py <agent main.py> <出力.json>   環境変数: DIRS TEAM LIMIT PAR"""
import json, glob, os, sys, time, collections
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_real_env import index, acts_of
PAR = int(os.environ.get("PAR", 32)); LIMIT = int(os.environ.get("LIMIT", 64))
PROD = ["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"]
def summ(steps, seat):
    """1局の要約: 日ごとの所持金・区画・お手伝い・家畜・作物・雑草・納屋、行動の種類の回数、売った個数"""
    D = []
    for d in range(30):
        t = min(d * 24 + 23, len(steps) - 1)
        o = steps[t][0]["observation"]; fa = o["farms"][seat]
        c = collections.Counter()
        for row in fa["tiles"]:
            for x in row:
                if isinstance(x, dict):
                    c["家畜" if x.get("animal") else ("作物" if x.get("kind") == "PLANT" else ("雑草" if x.get("kind") == "WEED" else "小屋"))] += 1
        pv = steps[t][seat]["observation"].get("private") or {}
        D.append(dict(m=fa["money"], q=len(fa["unlocked_quadrants"]), h=len(steps[min(d*24+12, len(steps)-1)][0]["observation"]["farms"][seat]["hands"]),
                      a=c["家畜"], c=c["作物"], w=c["雑草"], shed=sum((pv.get("shed") or {}).values())))
    acts = collections.Counter(); byday = collections.defaultdict(collections.Counter)
    for t in range(1, len(steps)):
        a = steps[t][seat].get("action") or {}
        for u in [a.get("farmer") or ["PASS"]] + list(a.get("hands") or []):
            if isinstance(u, list) and u: acts[u[0]] += 1; byday[(t - 1) // 24][u[0]] += 1
        for o in (a.get("market") or [])[:10]:
            if isinstance(o, list) and o:
                k = "M_" + o[0] + ("_" + o[1] if o[0] in ("SELL", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT") and len(o) > 1 else "")
                acts[k] += 1; byday[(t - 1) // 24][k] += 1
    return dict(days=D, acts=dict(acts), byday={k: dict(v) for k, v in byday.items()})
def one(args):
    agent_path, g = args
    from kaggle_environments import make
    if os.environ.get("FAST") == "1":
        sys.path.insert(0, "/content/repo/experiments/exp211_fast_official_env/code"); import fast_env  # noqa
    d = json.load(open(g["path"])); opp = acts_of(d, 1 - g["seat"])
    real = summ(d["steps"], g["seat"]); del d
    def replay(obs, config=None):
        t = obs.get("step")
        if t is None: t = int(obs.get("day", 0) or 0) * 24 + int(obs.get("hour", 0) or 0)
        a = opp[min(int(t), len(opp) - 1)]
        return {"farmer": a.get("farmer") or ["PASS"], "hands": list(a.get("hands") or []), "market": list(a.get("market") or [])}
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": g["seed"]})
    env.run([agent_path, replay] if g["seat"] == 0 else [replay, agent_path])
    sim = summ([[{"observation": s[i].observation if hasattr(s[i], "observation") else s[i]["observation"], "action": s[i].action if hasattr(s[i], "action") else s[i].get("action")} for i in (0, 1)] for s in env.steps], g["seat"])
    money = [float(f["money"]) for f in env.steps[-1][0].observation.farms]
    return dict(eid=g["eid"], seat=g["seat"], me=money[g["seat"]], op=money[1 - g["seat"]], real_me=g["real_me"], real_op=g["real_op"], real=real, sim=sim)
if __name__ == "__main__":
    ag, out = sys.argv[1], sys.argv[2]
    G = index()[:LIMIT]; rows = []
    with ProcessPoolExecutor(max_workers=PAR) as ex:
        for i, r in enumerate(ex.map(one, [(ag, g) for g in G]), 1):
            rows.append(r)
            if i % 16 == 0 or i == len(G): json.dump(rows, open(out, "w")); print("  %d/%d" % (i, len(G)), flush=True)
    print("保存 %s" % out)
