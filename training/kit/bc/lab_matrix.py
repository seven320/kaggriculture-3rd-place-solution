"""exp188: テープ x 対局 の総当たり R[t,g] = 最終所持金の差(自分 - 相手)。
   exp180 の router5_all.py と同じ測り方(元のシード・元の座席・相手は実際の手、`kagsim.run_tapes`)だが、
   32コアで並列に回す。テープも対局も lab_mine.py が作った npz から取る。
   環境変数: NT(使うテープ数、既定 全部)、NG(使う対局数、既定 全部)、NP(並列数、既定 16)
   使い方: python3 lab_matrix.py <npzのdir> <出力.npz>
"""
import json, glob, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")

D, OUT = sys.argv[1], sys.argv[2]
NT = int(os.environ.get("NT", 0)); NG = int(os.environ.get("NG", 0)); NP = int(os.environ.get("NP", 16))

files = sorted(glob.glob(D + "/*.npz"))
recs = []
for f in files:
    z = np.load(f, allow_pickle=False)
    m = json.loads(str(z["meta"]))
    recs.append(dict(m, f=f))
eids = [r["eid"] for r in recs]
T = list(range(len(recs)))[:NT] if NT else list(range(len(recs)))
G = list(range(len(recs)))[:NG] if NG else list(range(len(recs)))
print("テープ %d 本 x 対局 %d = %d 局" % (len(T), len(G), len(T) * len(G)), flush=True)

_cache = {}


def load(i):
    if i not in _cache:
        z = np.load(recs[i]["f"])
        _cache[i] = (np.ascontiguousarray(z["me"]), np.ascontiguousarray(z["op"]))
    return _cache[i]


def row(ti):
    import kagsim
    me, _ = load(ti)
    nme = recs[ti]["nme"]
    out = np.zeros(len(G), np.float32)
    for k, gi in enumerate(G):
        _, op = load(gi)
        g = recs[gi]
        r = np.asarray(kagsim.run_tapes(me, nme, op, g["nop"], [g["seed"]], 720, 1))[g["seat"]]
        out[k] = r[0] - r[1]
    return ti, out


if __name__ == "__main__":
    t0 = time.time()
    R = np.zeros((len(T), len(G)), np.float32)
    pos = {t: i for i, t in enumerate(T)}
    done = np.zeros(len(T), np.int8)
    with Pool(NP) as pool:
        for n, (ti, r) in enumerate(pool.imap_unordered(row, T, chunksize=1)):
            R[pos[ti]] = r
            done[pos[ti]] = 1
            if (n + 1) % 20 == 0:
                el = (time.time() - t0) / 60
                print("  %d/%d テープ (%.1f分, 残り %.1f分)" % (n + 1, len(T), el, el / (n + 1) * (len(T) - n - 1)), flush=True)
                np.savez_compressed(OUT + ".part.npz", R=R, done=done,
                                    tape_eids=np.array([eids[i] for i in T]), game_eids=np.array([eids[i] for i in G]),
                                    meta=json.dumps([{k: v for k, v in recs[i].items() if k != "f"} for i in G]))
    np.savez_compressed(OUT, R=R, tape_eids=np.array([eids[i] for i in T]), game_eids=np.array([eids[i] for i in G]),
                        meta=json.dumps([{k: v for k, v in recs[i].items() if k != "f"} for i in G]))
    print("保存 %s  R %s  (%.1f分)" % (OUT, R.shape, (time.time() - t0) / 60), flush=True)
