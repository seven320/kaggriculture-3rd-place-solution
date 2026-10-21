"""cache4.py と同じ抽出を、CPU コアで並列に回す(クラスタ は 32コアあるので 869局でも数分)。
   出力は cache4.py と同じ(<out>/<eid>.npz)。既にある局は飛ばす。
   使い方: python3 cache4_par.py <リプレイのdir> <チーム名> <出力dir> [並列数]
   注意: common/bc/cache4.py は変えていない。1局あたりの処理は extract4.extract_game をそのまま呼ぶ。"""
import json, glob, os, sys, time
import gzip
def _rj(p):
    """09-21: 棋譜は容量のため gzip で置いてあることがある(.json も .json.gz も読む)。"""
    return __import__("json").load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
rdir, team, out = sys.argv[1], sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else min(16, os.cpu_count())
os.makedirs(out, exist_ok=True)


def one(f):
    from extract4 import extract_game
    import extract3
    eid = os.path.basename(f)[:-5]
    p = "%s/%s.npz" % (out, eid)
    if os.path.exists(p):
        return ("skip", eid)
    try:
        d = _rj(f)
        tn = d["info"]["TeamNames"]
        if team not in tn:
            return ("notme", eid)
        seat = tn.index(team); rw = d["rewards"]
        X, fin, noops = extract_game(d, seat)
        if abs(fin[0] - rw[seat]) > .5 or abs(fin[1] - rw[1 - seat]) > .5:
            return ("mismatch", eid)
        meta = dict(seat=seat, seed=int(d["info"]["seed"]), teacher=rw[seat], opp=rw[1 - seat], opp_name=tn[1 - seat])
        np.savez(p + ".tmp.npz", meta=json.dumps(meta), opp_acts=json.dumps(extract3.acts_of(d, 1 - seat)), **X)
        os.replace(p + ".tmp.npz", p)
        return ("ok", eid)
    except Exception as ex:
        return ("err:" + str(ex)[:60], eid)


if __name__ == "__main__":
    fs = sorted((glob.glob(rdir + "/*.json") + glob.glob(rdir + "/*.json.gz")))
    t0 = time.time(); cnt = {}
    with Pool(NP) as pool:
        for i, (st, eid) in enumerate(pool.imap_unordered(one, fs, chunksize=1)):
            k = st.split(":")[0]; cnt[k] = cnt.get(k, 0) + 1
            if st.startswith("err") or st == "mismatch":
                print("  %s %s" % (st, eid), flush=True)
            if (i + 1) % 50 == 0:
                print("  %d/%d %s (%.1f分)" % (i + 1, len(fs), cnt, (time.time() - t0) / 60), flush=True)
    print("完了 %s npz=%d (%.1f分)" % (cnt, len(glob.glob(out + "/*.npz")), (time.time() - t0) / 60), flush=True)
