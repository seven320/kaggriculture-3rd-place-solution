"""exp302(教師1人の BC): 指定したチームの棋譜を、そのチームが座った座席すべてで BC 形式(extract4)に抽出する。
kagg/bc/extract_teama.py(MMPQ 専用)をチーム名で一般化したもの。cache4_par.py との違いは、自己対戦(両座席とも同じチーム)の
局を両座席とも取ること。出力名は <eid>_s<座席>.npz。最終残高が公式と一致しない局は捨てる(extract4 の恒等式)。
ラベルはそのチーム自身の手だけ。同じ局が複数のリプレイ dir にあるときは1回だけ扱う。

再抽出を避けるため、--old <dir> で渡した既存の抽出置き場(例 ext4 / ext4b / ext4c / ext4p/mkS)に <eid>.npz か
<eid>_s<座席>.npz があり、その meta の座席が一致すれば、実体へのリンクで済ませる(meta だけ読んで確かめる)。
使い方(容器内): python3 extract_team.py <出力dir> "<チーム名>" <リプレイdir>... [--old <既存dir>]... [--np 16]"""
import json, glob, os, sys, time
import gzip
def _rj(p):
    """09-21: 棋譜は容量のため gzip で置いてあることがある(.json も .json.gz も読む)。"""
    return __import__("json").load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

from multiprocessing import Pool
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
argv = sys.argv[1:]
NP = 16; OLD = []
while "--np" in argv: i = argv.index("--np"); NP = int(argv[i + 1]); del argv[i:i + 2]
while "--old" in argv: i = argv.index("--old"); OLD.append(argv[i + 1]); del argv[i:i + 2]
OUT, TEAM, RDIRS = argv[0], argv[1], argv[2:]
os.makedirs(OUT, exist_ok=True)
IDX = {}
for d in OLD:
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"): IDX.setdefault(f[:-4], os.path.realpath(os.path.join(d, f)))


def reuse(eid, seat, out):
    """既存の抽出があり座席が合えばリンクを張る(張れたら True)。"""
    import numpy as np
    for key in ("%s_s%d" % (eid, seat), eid):
        p = IDX.get(key)
        if not p or not os.path.exists(p): continue
        try:
            m = json.loads(str(np.load(p)["meta"]))
        except Exception:
            continue
        if int(m["seat"]) == seat:
            os.symlink(p, out); return True
    return False


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
            if reuse(eid, seat, p): res.append(("link", eid)); continue
            X, fin, noops = extract_game(d, seat)
            if abs(fin[0] - rw[seat]) > .5 or abs(fin[1] - rw[1 - seat]) > .5: res.append(("mismatch", eid)); continue
            meta = dict(seat=seat, seed=int(d["info"]["seed"]), teacher=rw[seat], opp=rw[1 - seat], opp_name=tn[1 - seat],
                        src=os.path.basename(os.path.dirname(f)))
            np.savez(p + ".tmp.npz", meta=json.dumps(meta), opp_acts=json.dumps(extract3.acts_of(d, 1 - seat)), **X)
            os.replace(p + ".tmp.npz", p); res.append(("ok", eid))
        except Exception as ex:
            res.append(("err:" + str(ex)[:80], eid))
    return res


if __name__ == "__main__":
    fs = {}
    for r in RDIRS:
        for f in sorted(glob.glob(r + "/*.json") + glob.glob(r + "/*.json.gz")): fs.setdefault(os.path.basename(f), f)   # 同じ局は1回だけ
    jobs = list(fs.values()); print("チーム %s / 局 %d / 既存の抽出 %d" % (TEAM, len(jobs), len(IDX)), flush=True)
    t0 = time.time(); cnt = {}
    with Pool(NP) as pool:
        for rs in pool.imap_unordered(one, jobs, chunksize=1):
            for st, eid in rs:
                k = st.split(":")[0]; cnt[k] = cnt.get(k, 0) + 1
                if st.startswith("err") or st == "mismatch": print("  %s %s" % (st, eid), flush=True)
    print("完了 %s npz=%d (%.1f分)" % (cnt, len(glob.glob(OUT + "/*.npz")), (time.time() - t0) / 60), flush=True)
