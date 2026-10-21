"""exp188: 公式リプレイから「テープ(その選手の全720手)」と「対局(相手の全手・シード・座席)」を取り出して詰める。
   kagsim(C++)は要らない(kagpack は純 Python)ので、コンテナの外でも動く。
   出力(1リプレイ1ファイル): <out>/<eid>.npz  … me/op(pack 済み int32)と meta
   環境変数: KAGG_FAST(kagpack.py のある場所。既定 /content/drive/MyDrive/kaggriculture/fast)
   使い方: python3 lab_mine.py <リプレイdir[,dir2]> <チーム名 or *> <出力dir> [並列数]
"""
import json, glob, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.environ.get("KAGG_FAST", "/content/drive/MyDrive/kaggriculture/fast"))

import re
rdirs, team, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 16
os.makedirs(out, exist_ok=True)
# 上位同士の対戦は同じ eid が2チームぶん出てくる。名前を <eid>__<チーム> にして両方とも残す
SUF = "__" + re.sub(r"[^A-Za-z0-9_.-]", "_", team)[:24] if team != "*" and os.environ.get("SUFFIX", "1") == "1" else ""


def acts_of(d, seat):
    """extract3.acts_of と同じ(kagsim を読み込まずに済むように写した)。"""
    out = []
    for i in range(1, len(d["steps"])):
        a = d["steps"][i][seat].get("action") or {}
        out.append({"farmer": a.get("farmer") or ["PASS"], "hands": a.get("hands") or [],
                    "market": a.get("market") or []})
    return out


def one(f):
    import kagpack
    eid = os.path.basename(f)[:-5]
    p = "%s/%s%s.npz" % (out, eid, SUF)
    if os.path.exists(p):
        return "skip"
    try:
        d = json.load(open(f))
        tn = d["info"]["TeamNames"]
        if team != "*" and team not in tn:
            return "notme"
        seat = tn.index(team) if team != "*" else 0
        me, nme = kagpack.pack(acts_of(d, seat))
        op, nop = kagpack.pack(acts_of(d, 1 - seat))
        rw = d.get("rewards") or [0, 0]
        np.savez_compressed(p + ".tmp.npz", me=me, op=op,
                            meta=json.dumps(dict(eid=eid, nme=int(nme), nop=int(nop),
                                                 seed=int(d["info"]["seed"]), seat=int(seat),
                                                 me_name=tn[seat], opp_name=tn[1 - seat],
                                                 reward_me=float(rw[seat]), reward_op=float(rw[1 - seat]))))
        os.replace(p + ".tmp.npz", p)
        return "ok"
    except Exception as ex:
        return "err:" + str(ex)[:70]


if __name__ == "__main__":
    fs = sorted(sum((glob.glob(r + "/*.json") for r in rdirs), []))
    t0 = time.time(); cnt = {}
    with Pool(NP) as pool:
        for i, st in enumerate(pool.imap_unordered(one, fs, chunksize=1)):
            k = st.split(":")[0]; cnt[k] = cnt.get(k, 0) + 1
            if st.startswith("err"):
                print("  " + st, flush=True)
            if (i + 1) % 100 == 0:
                print("  %d/%d %s (%.1f分)" % (i + 1, len(fs), cnt, (time.time() - t0) / 60), flush=True)
    print("完了 %s 出力 %d 本 (%.1f分)" % (cnt, len(glob.glob(out + "/*.npz")), (time.time() - t0) / 60), flush=True)
