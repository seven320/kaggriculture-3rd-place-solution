"""Play one game in the official environment between two submission-format agent folders (CPU).

Checks that a package plays 720 steps with both agents DONE and prints the final money.
Usage: official_game.py AGENT_DIR_A AGENT_DIR_B [--seed 1] [--out result.json]
"""
import argparse, contextlib, io, json, os, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):   # the import prints the OpenSpiel game list
        from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.seed})
    t0 = time.time()
    env.run([str(args.a / "main.py"), str(args.b / "main.py")])
    last = env.steps[-1]
    result = dict(a=str(args.a), b=str(args.b), seed=args.seed, steps=len(env.steps),
                  statuses=[s.status for s in last], rewards=[s.reward for s in last],
                  money=[f["money"] for f in last[0].observation.farms], seconds=round(time.time() - t0, 1))
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
