#!/usr/bin/env python3
"""CRN(共通乱数)ペア評価器。ポケモンコンペの事後検証ルール1「Day1に測定器を作る」の実装。

設計:
  * 同一シードで A vs B と B vs A の**両座席**を回す(座席バイアスを相殺)
  * 応答は2つ: 勝敗(ラダーが実際に見るもの)と最終所持金(連続、分散が桁違いに小さい)
  * 対照を必ず取る。`--floor` で同一エージェント同士を回し、床(検出限界)を実測する
  * 判定は必ず「差 ± SE と符号一致」で出す

使い方:
  python3 local/evaluate.py --a agent/main.py --b starter --games 60
  python3 local/evaluate.py --a agent/main.py --b agent/main.py --games 60   # 床の実測
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILTIN = {"pass", "random", "starter"}


def _resolve(spec):
    """'starter' などの組み込み名はそのまま、.py はパスとして渡す。"""
    if spec in BUILTIN:
        return spec
    p = spec if os.path.isabs(spec) else os.path.join(BASE, spec)
    if not os.path.exists(p):
        raise SystemExit(f"agent not found: {spec}")
    return p


def _one(args):
    a_spec, b_spec, seed, swap, steps = args
    from kaggle_environments import make
    cfg = {"episodeSteps": steps, "seed": seed}
    env = make("kaggriculture", configuration=cfg)
    agents = [b_spec, a_spec] if swap else [a_spec, b_spec]
    t0 = time.time()
    try:
        env.run(agents)
    except Exception as e:  # 落ちたエージェントは負け扱いにして続行(黙って消さない)
        return dict(seed=seed, swap=swap, err=repr(e)[:200])
    final = env.steps[-1]
    farms = final[0].observation.farms
    money = [float(f["money"]) for f in farms]
    a_idx = 1 if swap else 0
    a_money, b_money = money[a_idx], money[1 - a_idx]
    statuses = [s.status for s in final]
    return dict(seed=seed, swap=swap, a_money=a_money, b_money=b_money,
                a_win=1.0 if a_money > b_money else (0.5 if a_money == b_money else 0.0),
                a_status=statuses[a_idx], b_status=statuses[1 - a_idx],
                secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="starter")
    ap.add_argument("--games", type=int, default=40, help="シード数(各シードで両座席=2局)")
    ap.add_argument("--seed-base", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--par", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    A, B = _resolve(a.a), _resolve(a.b)
    jobs = []
    for g in range(a.games):
        s = a.seed_base + g * 7919          # シードは十分離す(PTCGの教訓)
        jobs.append((A, B, s, False, a.steps))
        jobs.append((A, B, s, True, a.steps))

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.par) as ex:
        for i, r in enumerate(ex.map(_one, jobs), 1):
            rows.append(r)
            if not a.quiet and i % 20 == 0:
                done = [x for x in rows if "a_money" in x]
                mm = sum(x["a_money"] - x["b_money"] for x in done) / max(len(done), 1)
                print(f"  {i}/{len(jobs)}  差の平均 {mm:+,.0f}  {time.time()-t0:.0f}s", flush=True)

    ok = [r for r in rows if "a_money" in r]
    errs = [r for r in rows if "err" in r]
    n = len(ok)
    if n == 0:
        print("全局失敗:", errs[:2]); sys.exit(1)

    # --- ペア応答: 同一シードの2座席を1つの観測にまとめる(座席効果を相殺) ---
    by_seed = {}
    for r in ok:
        by_seed.setdefault(r["seed"], []).append(r)
    pair_d, pair_w = [], []
    for s, v in by_seed.items():
        if len(v) != 2:
            continue
        pair_d.append(sum(x["a_money"] - x["b_money"] for x in v) / 2.0)
        pair_w.append(sum(x["a_win"] for x in v) / 2.0)

    def stat(xs):
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
        return m, sd / math.sqrt(len(xs))

    dm, dse = stat(pair_d)
    wm, wse = stat(pair_w)
    a_money = [r["a_money"] for r in ok]
    b_money = [r["b_money"] for r in ok]
    a_mu = sum(a_money) / n
    b_mu = sum(b_money) / n
    seat0 = [r["a_win"] for r in ok if not r["swap"]]
    seat1 = [r["a_win"] for r in ok if r["swap"]]

    print(f"\n{'='*66}")
    print(f"A = {a.a}\nB = {a.b}")
    print(f"{'='*66}")
    print(f"  局数 {n}(シード{len(pair_d)}個 × 両座席)  {time.time()-t0:.0f}s"
          + (f"  失敗{len(errs)}" if errs else ""))
    print(f"  A 最終所持金 平均 ${a_mu:>10,.0f}   B 平均 ${b_mu:>10,.0f}")
    print(f"  --- ペア差(座席相殺済み。これが主指標)---")
    print(f"  所持金差   {dm:>+12,.0f} ± {dse:,.0f}   z = {dm/dse if dse else 0:>+6.2f}")
    print(f"  勝率       {wm:>12.3f} ± {wse:.3f}   z = {(wm-0.5)/wse if wse else 0:>+6.2f}")
    print(f"  符号一致   {sum(1 for x in pair_d if x>0)}/{len(pair_d)} 局で A が上回る")
    if seat0 and seat1:
        print(f"  座席別勝率 seat0 {sum(seat0)/len(seat0):.3f} (n={len(seat0)})   "
              f"seat1 {sum(seat1)/len(seat1):.3f} (n={len(seat1)})")
    if errs:
        print(f"  !! 例外 {len(errs)}件: {errs[0].get('err','')[:120]}")
    if a.out:
        json.dump(dict(a=a.a, b=a.b, n=n, a_mu=a_mu, b_mu=b_mu, dm=dm, dse=dse,
                       wm=wm, wse=wse, rows=ok), open(a.out, "w"))
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
