"""exp301(MMPQ 単独 BC): M & M & P & Q の棋譜を、MMPQ が座った座席すべてで BC 形式(extract4)に抽出する。
cache4_par.py との違い: 自己対戦(両座席とも MMPQ)の局は両座席とも取る。出力名は <eid>_s<座席>.npz。
既に ext4p/mmpq/<eid>.npz に同じ座席で抽出済みの局は、その npz へのシンボリックリンクで済ませる(再抽出しない)。
最終残高が公式と一致しない局は捨てる(extract4 の恒等式)。ラベルは MMPQ 自身の手だけ。
使い方(容器内): python3 extract_teama.py <出力dir> <既存dir> <リプレイdir>... [--np 16]"""
import json, glob, os, sys, time
import gzip
def _rj(p):
    """09-21: 棋譜は容量のため gzip で置いてあることがある(.json も .json.gz も読む)。"""
    return __import__("json").load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

from multiprocessing import Pool
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
TEAM = "M & M & P & Q"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
NP = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 16
if "--np" in sys.argv: args.remove(str(NP))
OUT, OLD, RDIRS = args[0], args[1], args[2:]
os.makedirs(OUT, exist_ok=True)


def one(f):
    import numpy as np
    from extract4 import extract_game
    import extract3
    eid = os.path.basename(f)[:-5]; res = []
    try:
        d = _rj(f); tn = d["info"]["TeamNames"]; rw = d["rewards"]
    except Exception as ex:
        return [("err:" + str(ex)[:80], eid)]
    for seat in (0, 1):
        if tn[seat] != TEAM: continue
        p = "%s/%s_s%d.npz" % (OUT, eid, seat)
        if os.path.exists(p): res.append(("skip", eid)); continue
        try:
            old = "%s/%s.npz" % (OLD, eid)
            if os.path.exists(old) and tn.index(TEAM) == seat:
                os.symlink(os.path.realpath(old), p); res.append(("link", eid)); continue
            X, fin, noops = extract_game(d, seat)
            if abs(fin[0] - rw[seat]) > .5 or abs(fin[1] - rw[1 - seat]) > .5: res.append(("mismatch", eid)); continue
            meta = dict(seat=seat, seed=int(d["info"]["seed"]), teacher=rw[seat], opp=rw[1 - seat], opp_name=tn[1 - seat])
            np.savez(p + ".tmp.npz", meta=json.dumps(meta), opp_acts=json.dumps(extract3.acts_of(d, 1 - seat)), **X)
            os.replace(p + ".tmp.npz", p); res.append(("ok", eid))
        except Exception as ex:
            res.append(("err:" + str(ex)[:80], eid))
    return res


if __name__ == "__main__":
    fs = {}
    for r in RDIRS:
        for f in sorted(glob.glob(r + "/*.json") + glob.glob(r + "/*.json.gz")): fs.setdefault(os.path.basename(f), f)   # 同じ局は1回だけ
    jobs = list(fs.values()); print("局 %d" % len(jobs), flush=True)
    t0 = time.time(); cnt = {}
    with Pool(NP) as pool:
        for rs in pool.imap_unordered(one, jobs, chunksize=1):
            for st, eid in rs:
                k = st.split(":")[0]; cnt[k] = cnt.get(k, 0) + 1
                if st.startswith("err") or st == "mismatch": print("  %s %s" % (st, eid), flush=True)
    print("完了 %s npz=%d (%.1f分)" % (cnt, len(glob.glob(OUT + "/*.npz")), (time.time() - t0) / 60), flush=True)
