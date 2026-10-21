"""exp191: 公式環境の直接対戦(同じシードで両座席 = CRN)。
   使い方: SEEDS=16 PAR=32 python3 lab_h2h.py <A の main.py> <B の main.py> <出力.json>
   出力: A から見た勝率・差(両座席をまとめたもの)と、座席ごとの内訳。
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

NS = int(os.environ.get("SEEDS", 16)); PAR = int(os.environ.get("PAR", 32))
BASE = int(os.environ.get("SEEDBASE", 1000))


def one(args):
    A, B, seed, seat = args
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    agents = [A, B] if seat == 0 else [B, A]
    t0 = time.time(); env.run(agents)
    m = [float(f["money"]) for f in env.steps[-1][0].observation.farms]
    return dict(seed=seed, seat=seat, a=m[seat], b=m[1 - seat], secs=time.time() - t0,
                status=[s.status for s in env.steps[-1]])


if __name__ == "__main__":
    A, B, out = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = [(A, B, BASE + 7919 * i, s) for i in range(NS) for s in (0, 1)]
    print("A=%s\nB=%s\n%d シード x 両座席 = %d 局" % (A, B, NS, len(jobs)), flush=True)
    rows = []; t0 = time.time()
    with ProcessPoolExecutor(max_workers=PAR) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            rows.append(r)
            if i % 8 == 0 or i == len(jobs):
                d = [x["a"] - x["b"] for x in rows]; n = len(d)
                mu = sum(d) / n; sd = (sum((x - mu) ** 2 for x in d) / max(1, n - 1)) ** .5
                wr = sum(x > 0 for x in d) / n
                print("  %d/%d  A の勝率 %.3f±%.3f  差 %+.0f±%.0f  (%.1f分)" % (
                    i, len(jobs), wr, (wr * (1 - wr) / n) ** .5, mu, sd / n ** .5, (time.time() - t0) / 60), flush=True)
    json.dump(rows, open(out, "w"))
    bad = [x for x in rows if any(s != "DONE" for s in x["status"])]
    if bad:
        print("  注意: 正常終了しなかった局 %d 件 %s" % (len(bad), bad[0]["status"]), flush=True)
    for s in (0, 1):
        r = [x for x in rows if x["seat"] == s]
        d = [x["a"] - x["b"] for x in r]
        print("  座席%d: A の勝率 %.3f 差 %+.0f" % (s, sum(x > 0 for x in d) / len(d), sum(d) / len(d)), flush=True)
    print("保存 %s" % out, flush=True)
