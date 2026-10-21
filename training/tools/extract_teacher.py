"""Turn official replays into the kit's BC format (extract4 npz, one file per seat) and write a split.

Input is the official daily archive from the dataset kaggle/kaggriculture-episodes-index (zip of <episode>.json).
Conversion is the kit's extract4.extract_game(replay, seat). Like the kit's extract_teama.py:
  - a seat whose reconstructed final money differs from the official reward by 0.5 or more is dropped;
  - only the seats of --team are kept (omit --team to keep every seat).
Unlike extract_teama.py, a teacher unit action outside the legal mask (a no-op in the official
environment) gets ul = -1, so it is left out of the loss; the count is in the records file.
Games that are not 720 steps are dropped.

Output: OUT/<episode>_s<seat>.npz, RECORDS (one row per seat with status), and with --split NAME
ROOT/content/local_logs/bc_NAME.json = {"train": [...], "hold": [...]}. The hold-out is taken by
episode (both seats of a game go to the same side), --hold of the episodes, fixed by --seed.

CPU only. Usage:
  extract_teacher.py --zips Z.zip.. --out DIR --records R.json [--team "M & M & P & Q"] [--split NAME]
                     [--hold 0.1] [--limit N] [--np 16] [--root ROOT]
"""
import argparse, collections, json, os, random, sys, time, zipfile
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A = None


def one(job):
    import numpy as np
    from extract4 import extract_game
    import extract3
    zp, n = job
    eid = int(os.path.basename(n)[:-5])
    try:
        d = json.loads(zipfile.ZipFile(zp).read(n))
    except Exception as ex:
        return [dict(episode=eid, seat=-1, status="read_error", error=str(ex)[:200])]
    info = d.get("info", {})
    tn, rw = info.get("TeamNames"), d.get("rewards")
    if not tn or not rw or len(d.get("steps", [])) != 720 or any(r is None for r in rw):
        return [dict(episode=eid, seat=-1, status="short_or_incomplete", teams=tn)]
    res = []
    for seat in (0, 1):
        if A.team and tn[seat] != A.team:
            continue
        r = dict(episode=eid, seat=seat, team=tn[seat], opp_name=tn[1 - seat])
        try:
            X, fin, noops = extract_game(d, seat)
        except Exception as ex:
            res.append(dict(r, status="extract_error", error=str(ex)[:200]))
            continue
        if abs(fin[0] - rw[seat]) > .5 or abs(fin[1] - rw[1 - seat]) > .5:
            res.append(dict(r, status="mismatch", fin=list(fin)))
            continue
        ul = X["ul"].astype(int)
        live = ul >= 0
        bad = live & (np.take_along_axis(X["legal"], np.clip(ul, 0, None)[..., None], -1)[..., 0] == 0)
        X["ul"] = np.where(bad, -1, X["ul"]).astype(X["ul"].dtype)
        X["uq"] = np.where(bad, 0, X["uq"]).astype(X["uq"].dtype)
        meta = dict(seat=seat, seed=int(info["seed"]), teacher=rw[seat], opp=rw[1 - seat], team=tn[seat],
                    opp_name=tn[1 - seat], src=os.path.basename(zp), episode_id=eid)
        p = "%s/%d_s%d.npz" % (A.out, eid, seat)
        np.savez(p + ".tmp.npz", meta=json.dumps(meta), opp_acts=json.dumps(extract3.acts_of(d, 1 - seat)), **X)
        os.replace(p + ".tmp.npz", p)
        res.append(dict(r, status="ok", name="%d_s%d" % (eid, seat), seed=meta["seed"], money=rw[seat], opp_money=rw[1 - seat],
                        rows=int(len(X["g"])), noops=int(noops), illegal_labels_dropped=int(bad.sum())))
    return res


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--team")
    ap.add_argument("--split")
    ap.add_argument("--hold", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--np", type=int, default=16)
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("TEAM_BC_PPO_ROOT", REPO / "training/work")))
    A = ap.parse_args()
    kagg = A.root.resolve() / "content/drive/MyDrive/kaggriculture"
    sys.path[:0] = [str(kagg / "bc"), str(kagg / "fast")]   # before the Pool forks: the workers import extract4
    os.makedirs(A.out, exist_ok=True)
    t0 = time.time()
    jobs = [(zp, n) for zp in A.zips for n in sorted(zipfile.ZipFile(zp).namelist()) if n.endswith(".json")]
    if A.limit:
        jobs = jobs[:A.limit]
    print("games %d (zips %d)" % (len(jobs), len(A.zips)), flush=True)
    out = []
    with Pool(A.np) as pool:
        for rs in pool.imap_unordered(one, jobs, chunksize=1):
            out.extend(rs)
    out.sort(key=lambda r: (r["episode"], r["seat"]))
    json.dump(dict(zips=A.zips, team=A.team, seconds=round(time.time() - t0, 1), records=out), open(A.records, "w"), indent=1, ensure_ascii=False)
    print(dict(collections.Counter(r["status"] for r in out)), "%.1f min" % ((time.time() - t0) / 60), flush=True)
    if A.split:
        ok = [r for r in out if r["status"] == "ok"]
        episodes = sorted({r["episode"] for r in ok})
        random.Random(A.seed).shuffle(episodes)
        hold = set(episodes[:max(1, round(len(episodes) * A.hold))])
        split = dict(train=[r["name"] for r in ok if r["episode"] not in hold], hold=[r["name"] for r in ok if r["episode"] in hold])
        path = A.root.resolve() / "content/local_logs" / ("bc_%s.json" % A.split)
        json.dump(split, open(path, "w"))
        print("split %s: train %d seats, hold %d seats" % (path, len(split["train"]), len(split["hold"])), flush=True)


if __name__ == "__main__":
    main()
