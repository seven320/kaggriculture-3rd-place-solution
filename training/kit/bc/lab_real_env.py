"""exp190: 我々の実戦局を、公式環境で「元のシード・元の座席・相手は実際の手を再生」で打ち直す。
   exp176/exp181 の real_cf_env.py と同じ考え方だが、**対局の一覧を実戦リプレイから自前で作る**
   (Drive の runs/r3_meta.json が無くても回る。しかも 311局より多く取れる)。
   自分の側に v57(mix_ug)などの main.py を入れる。v57 の自分の実戦局を入れれば、実戦と一致するはず(恒等式)。
   環境変数: PAR(並列数)、LIMIT(局数の上限)、DIRS(リプレイdirをカンマ区切り)、TEAM(我々の名前)
   使い方: python3 lab_real_env.py <agent main.py> <出力.json>
"""
import json, glob, os, sys, time
import gzip
def _rj(p):
    """09-21: 棋譜は容量のため gzip で置いてある(.json も .json.gz も読む)。"""
    return __import__("json").load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

from concurrent.futures import ProcessPoolExecutor

TEAMS = os.environ.get("TEAM", "DSM,masspeaks,denden12").split(",")   # チームマージで masspeaks → denden12 → DSM と変わっている(09-20 修正: DSM を既定に追加。これが無いと新しいリプレイで 0 局になる)
DIRS = os.environ.get("DIRS", "/content/local_rp/our56136844,/content/local_rp/our56139356").split(",")
PAR = int(os.environ.get("PAR", 32))
LIMIT = int(os.environ.get("LIMIT", 10 ** 9))


def acts_of(d, seat):
    out = []
    for i in range(1, len(d["steps"])):
        a = d["steps"][i][seat].get("action") or {}
        out.append({"farmer": a.get("farmer") or ["PASS"], "hands": a.get("hands") or [],
                    "market": a.get("market") or []})
    return out


def index():
    """実戦局の一覧(自分の座席・シード・相手・実戦の結果)。自己対戦(検証局)は除く。"""
    G = []
    for d_ in DIRS:
        for f in sorted(glob.glob(d_ + "/*.json") + glob.glob(d_ + "/*.json.gz")):
            try:
                d = _rj(f)
                tn = d["info"]["TeamNames"]
                mine = [t for t in TEAMS if t in tn]
                if not mine or tn[0] == tn[1]:
                    continue
                seat = tn.index(mine[0]); rw = d["rewards"]
                G.append(dict(eid=os.path.basename(f)[:-5], path=f, seat=seat, seed=int(d["info"]["seed"]),
                              opp=tn[1 - seat], real_me=float(rw[seat]), real_op=float(rw[1 - seat])))
                del d
            except Exception as ex:
                print("  読めない %s %s" % (f, str(ex)[:50]), flush=True)
    return G


def one(args):
    agent_path, g = args
    from kaggle_environments import make
    if os.environ.get("FAST") == "1":   # exp211: 公式環境の複製を速くする差し替え(規則は不変、恒等式は確認済み)
        sys.path.insert(0, "/content/repo/experiments/exp211_fast_official_env/code"); import fast_env  # noqa: F401
    d = _rj(g["path"]); opp = acts_of(d, 1 - g["seat"]); del d

    def replay(obs, config=None):
        t = obs.get("step")
        if t is None:
            t = int(obs.get("day", 0) or 0) * 24 + int(obs.get("hour", 0) or 0)
        a = opp[min(int(t), len(opp) - 1)]
        return {"farmer": a.get("farmer") or ["PASS"], "hands": list(a.get("hands") or []),
                "market": list(a.get("market") or [])}

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": g["seed"]})
    agents = [agent_path, replay] if g["seat"] == 0 else [replay, agent_path]
    t0 = time.time(); env.run(agents)
    money = [float(f["money"]) for f in env.steps[-1][0].observation.farms]
    return dict(eid=g["eid"], seat=g["seat"], opp=g["opp"], me=money[g["seat"]], op=money[1 - g["seat"]],
                real_me=g["real_me"], real_op=g["real_op"], secs=time.time() - t0)


if __name__ == "__main__":
    ag, out = sys.argv[1], sys.argv[2]
    G = index()[:LIMIT]
    NSH, SH = int(os.environ.get("NSHARD", 1)), int(os.environ.get("SHARD", 0))   # exp286: 局を分けて複数の GPU ジョブで並べる
    if NSH > 1: G = G[SH::NSH]
    print("対局 %d 局(%s)" % (len(G), ", ".join(os.path.basename(x) for x in DIRS)), flush=True)
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=PAR) as ex:
        for i, r in enumerate(ex.map(one, [(ag, g) for g in G]), 1):
            rows.append(r)
            if i % 40 == 0 or i == len(G):
                json.dump(rows, open(out, "w"))
                mg = [x["me"] - x["op"] for x in rows]; rm = [x["real_me"] - x["real_op"] for x in rows]
                n = len(rows)
                sd = (sum((m - sum(mg) / n) ** 2 for m in mg) / max(1, n - 1)) ** .5
                wr = sum(m > 0 for m in mg) / n; rwr = sum(m > 0 for m in rm) / n
                print("  %d/%d  打ち直し 勝率 %.3f±%.3f 差 %+.0f | 実戦 勝率 %.3f 差 %+.0f | 実戦と完全一致 %d (%.1f分)" % (
                    i, len(G), wr, (wr * (1 - wr) / n) ** .5, sum(mg) / n, rwr, sum(rm) / n,
                    sum(abs(x["me"] - x["real_me"]) < 1 and abs(x["op"] - x["real_op"]) < 1 for x in rows),
                    (time.time() - t0) / 60), flush=True)
    json.dump(rows, open(out, "w"))
    print("保存 %s" % out, flush=True)
