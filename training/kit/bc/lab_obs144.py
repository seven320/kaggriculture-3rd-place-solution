"""exp188(その3): **提出にそのまま載せられる形**の特徴を作る。
   家族の開幕(先頭 NPRE 手、家族の中で共通)を相手のテープにぶつけて NPRE 手まで進め、
   そこで**エンジンが見る相手の公開状態**(farm_pub / tiles / ユニット数 / 店の抽選)を読む。
   相手の行動列ではなく、実戦で本当に観測できるものだけを使う。
   使い方: SRC=<npzのdir> python3 lab_obs144.py <R.npz> <出力.npz> [NPRE=144] [家族の最小本数=8]
"""
import json, os, sys, hashlib, collections, time
import numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim

Z = np.load(sys.argv[1], allow_pickle=False)
OUT = sys.argv[2]
NPRE = int(sys.argv[3]) if len(sys.argv) > 3 else 144
MINF = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tape_eids = [str(x) for x in Z["tape_eids"]]; game_eids = [str(x) for x in Z["game_eids"]]
meta = json.loads(str(Z["meta"]))
src = os.environ.get("SRC", "/content/lab/all")

# lab_matrix は sorted(glob) の順で並べている。meta の eid にはチーム名の接尾辞が入っていないので、
# 名前ではなく**並び順**でファイルを引く(同じ eid が2チームぶんある場合に取り違えないため)。
import glob as _g
files = sorted(_g.glob(src + "/*.npz"))
assert len(files) == len(tape_eids) == len(game_eids), "npz の数が R.npz と合わない: %d vs %d" % (len(files), len(tape_eids))
D = []
for f in files:
    z = np.load(f); m = json.loads(str(z["meta"]))
    D.append((np.ascontiguousarray(z["me"]), m["nme"], np.ascontiguousarray(z["op"]), m["nop"], m))
tape_eids = [os.path.basename(f)[:-4] for f in files]; game_eids = list(tape_eids)


def walk(buf, nsteps, npre):
    i = 0
    for s in range(min(npre, nsteps)):
        nu = int(buf[i]); i += 1 + 3 * nu
        no = int(buf[i]); i += 1 + 3 * no
    return i


fam = {}
for i, e in enumerate(tape_eids):
    me, nme = D[i][0], D[i][1]
    fam[e] = hashlib.sha256(me[:walk(me, nme, NPRE)].tobytes()).hexdigest()[:12]
cnt = collections.Counter(fam.values())
fams = [f for f, n in cnt.most_common() if n >= MINF]
print("開幕%d手の家族 %d 種のうち、%d 本以上のもの %d 種" % (NPRE, len(cnt), MINF, len(fams)), flush=True)


def obs_at(me, nme, op, nop, seed, seat, npre):
    """自分のテープ me を相手のテープ op にぶつけて npre 手まで進め、相手の公開状態を読む。"""
    A, B = (me, nme), (op, nop)
    a, b = (A, B) if seat == 0 else (B, A)
    env = kagsim.VecMarket(a[0], a[1], b[0], b[1], [int(seed)], 720, int(seat))
    env.keep_nonsell = False; env.reset()
    while env.step_index() < npre:
        env.step2(np.asarray(env.tape_orders())[None], np.asarray(env.opp_tape_orders()).reshape(1, 10, 3))
    pub = np.asarray(env.farm_pub(1))[0]                       # 相手の 所持金/人数/区画/本日雇用/納屋計(納屋は -1)
    tb = np.asarray(env.tiles_base(1))[0].reshape(-1)           # 相手のタイル(静的)
    td = np.asarray(env.tiles_dyn(1))[0].reshape(-1)            # 相手のタイル(動的)
    up = np.asarray(env.unit_pos2(1))[0]
    sh = np.asarray(env.shops())[0]
    mny = np.asarray(env.money())[0]
    f = np.concatenate([pub, [float((up[:, 0] >= 0).sum())],
                        [tb.sum(), (tb > 0).sum(), td.sum(), (td > 0).sum()],
                        np.bincount(np.clip(sh[sh >= 0].astype(int), 0, 7), minlength=8).astype(float),
                        [mny[0], mny[1], mny[0] - mny[1]]])
    return f.astype(np.float32)


REPS = [next(i for i, e in enumerate(tape_eids) if fam[e] == f) for f in fams]   # 家族の代表(開幕は家族で共通)


def one(args):
    fi, gi = args
    me, nme = D[REPS[fi]][0], D[REPS[fi]][1]
    _, _, op, nop, m = D[gi]
    return fi, gi, obs_at(me, nme, op, nop, m["seed"], m["seat"], NPRE)


t0 = time.time()
NP = int(os.environ.get("NP", 32))
jobs = [(fi, gi) for fi in range(len(fams)) for gi in range(len(game_eids))]
X = np.zeros((len(fams), len(game_eids), 21), np.float32)
from multiprocessing import Pool
with Pool(NP) as pool:
    for n, (fi, gi, v) in enumerate(pool.imap_unordered(one, jobs, chunksize=32)):
        if X.shape[2] != len(v):
            X = np.zeros((len(fams), len(game_eids), len(v)), np.float32)
        X[fi, gi] = v
        if (n + 1) % 20000 == 0:
            print("  %d/%d (%.1f分)" % (n + 1, len(jobs), (time.time() - t0) / 60), flush=True)
np.savez_compressed(OUT, X=X, fams=np.array(fams), tape_fam=np.array([fam[e] for e in tape_eids]),
                    tape_eids=np.array(tape_eids), game_eids=np.array(game_eids), npre=NPRE)
print("保存 %s  X %s (%.1f分)" % (OUT, X.shape, (time.time() - t0) / 60), flush=True)
