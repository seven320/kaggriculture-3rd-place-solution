"""新マスク移行の蒸留データ(newmask-distill-plan.md 3節)。**新しいファイル。走っているジョブはこれを読まない。**

教師(旧マスクの最良版、例 orbit_g5g4_gen18.pt)に C++ エンジン(sp_fast.BatchSPPool)で打たせ、**教師の席の局面**を
extract4 と同じ形の npz(tiles ou pu g legal ul uq ms mq + meta)で書き出す。legal は**旧マスク(mask0)**。
新マスク(mask2)は特徴だけから決まるので(nm_lib.mask2_np)、ここには書かない = 既存の BC の読み込み器もそのまま読める。

局の組み(1束 NGAME 局):
  SELF_FRAC の局は自己対戦(両席とも教師、両席とも記録)。残りは OPPS の相手と(教師の席は局ごとに交互、教師の席だけ記録)。
  教師は温度 GEN_T の標本打ち(PPO の対局係と同じ分布)、相手は OPP_GREEDY=1 なら貪欲(リーグと同じ)、0 なら標本。
  相手は全員**旧マスク**で打つ(このプロセスでは LEGAL_OPT を 0 に固定。環境変数に LEGAL_OPT があれば止める)。

DAgger(任意): BEHAV=<生徒の重み> を渡すと、記録する席は**生徒が新マスク(opt=2)で**標本打ちする(相手は旧マスクのまま)。
  記録する legal は同じ局面の mask0(エンジンに opt=0 で聞き直す)なので、distill_nm はそのまま読める(教師の目標は distill_nm が作る)。
  生徒だけが行く局面(増えた手の質量の尾)を足すための2周目。opts つきの kagsim(容器の 3.10 版)が要る。
出力: OUT/<束>_<局>_s<席>.npz、OUT/_gen.json(設定・局ごとの最終所持金・対戦成績)。
分割: SPLIT を渡すと /content/local_logs/bc_<SPLIT>.json(局単位で HOLD_FRAC を未見に)を書く。

使い方(容器内、GPU 1枚):
  INIT=/content/local_logs/orbit_g5g4_gen18.pt OUT=/content/local_nm NGAME=256 NCHUNK=4 \
  OPPS=/content/local_logs/orbit_g5g4_gen16.pt,... SPLIT=nm18 python3 gen_nm.py
CPU の小さな確認: PPODEV=cpu DEV=cpu NGAME=4 NCHUNK=1 TMAX=719 python3 gen_nm.py
"""
import os, sys, json, time, random
if os.environ.get("LEGAL_OPT", "0") != "0":
    raise SystemExit("gen_nm: LEGAL_OPT=%s が入っている。蒸留データの legal は旧マスクで取る(相手も旧マスクで打たせる)ので外すこと" % os.environ["LEGAL_OPT"])
if os.environ.get("MASK_MIG") == "1":
    raise SystemExit("gen_nm: MASK_MIG=1 を外すこと(教師は素の旧マスクで打つ)")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
DEV = os.environ.get("DEV", os.environ.get("PPODEV", "cuda"))
os.environ.setdefault("PPODEV", DEV)
import model_tf as M
import ppo_tf as P
import sp_fast
from extract4 import NU, NS, M_STOP
assert P.DEV == DEV, "PPODEV と DEV をそろえること"

INIT = os.environ["INIT"]; OUT = os.environ["OUT"]
NGAME = int(os.environ.get("NGAME", "256")); NCHUNK = int(os.environ.get("NCHUNK", "4"))
SELF_FRAC = float(os.environ.get("SELF_FRAC", "0.4")); GEN_T = float(os.environ.get("GEN_T", "1.0"))
OPP_GREEDY = os.environ.get("OPP_GREEDY", "1") == "1"; SEED0 = int(os.environ.get("SEED0", "7100000"))
OPPS = [x for x in os.environ.get("OPPS", "").split(",") if x]
BEHAV = os.environ.get("BEHAV", "")
CH = int(os.environ.get("CH", "512")); TMAX = int(os.environ.get("TMAX", "719"))
SPLIT = os.environ.get("SPLIT", ""); HOLD_FRAC = float(os.environ.get("HOLD_FRAC", "0.05"))
AMP = DEV != "cpu" and os.environ.get("AMP", "1") == "1"


def load(p):
    return M.load(p, DEV).eval()


def act(net, st, rows, greedy, temp):
    """rows の局面だけ打たせる。-> ul uq ms mq(rows の順)"""
    sub = tuple(x[rows] for x in st)
    outs = []
    ctx = torch.autocast("cuda", dtype=torch.float16) if AMP else torch.autocast("cpu", enabled=False)
    with torch.no_grad(), ctx:
        for i in range(0, len(rows), CH):
            outs.append(P.sample(net, tuple(x[i:i + CH] for x in sub), greedy=greedy, cap=True, temp=temp)[:4])
    return [np.concatenate([o[j] for o in outs]) for j in range(4)]


def main():
    os.makedirs(OUT, exist_ok=True)
    T = load(BEHAV or INIT); opps = [load(p) for p in OPPS]
    if BEHAV: print("DAgger: 記録する席は %s が新マスクで打つ(記録する legal は mask0)" % BEHAV, flush=True)
    rng = random.Random(SEED0)
    log = dict(env={k: os.environ.get(k) for k in ("INIT", "BEHAV", "NGAME", "NCHUNK", "SELF_FRAC", "GEN_T", "OPP_GREEDY", "SEED0", "OPPS", "TMAX")}, games=[])
    names = []
    for k in range(NCHUNK):
        t0 = time.time()
        seeds = [SEED0 + 100003 * k + 7919 * i for i in range(NGAME)]
        nself = int(round(NGAME * SELF_FRAC)) if opps else NGAME
        # 局ごとの (席0の打ち手, 席1の打ち手)。-1 = 教師、j >= 0 = OPPS[j]
        who = []
        for i in range(NGAME):
            if i < nself: who.append((-1, -1))
            else:
                j = (i - nself) % len(opps); ts = (i - nself) // len(opps) % 2
                who.append((-1, j) if ts == 0 else (j, -1))
        pool = sp_fast.BatchSPPool(seeds)
        assert pool.legal_opt is None and not pool.mig_ok.any()
        if BEHAV:                                   # 生徒の席だけ opt=2、相手の席は -1(= 環境変数 = 0)
            pool.legal_opt = np.array([[2 if who[i][s] == -1 else -1 for s in (0, 1)] for i in range(NGAME)], np.int32)
        rec = {(i, s): {kk: [] for kk in ("tiles", "ou", "pu", "g", "legal", "ul", "uq", "ms", "mq")}
               for i in range(NGAME) for s in (0, 1) if who[i][s] == -1}
        for t in range(TMAX):
            UO = {}
            for s in (0, 1):
                st = pool.state(s)
                ul = np.full((NGAME, NU), -1, np.int64); uq = np.zeros((NGAME, NU), np.int64)
                ms = np.full((NGAME, NS), M_STOP, np.int64); mq = np.zeros((NGAME, NS), np.int64)
                groups = {}
                for i in range(NGAME): groups.setdefault(who[i][s], []).append(i)
                for w, rows in groups.items():
                    rows = np.asarray(rows)
                    if w == -1: a = act(T, st, rows, False, GEN_T)
                    else: a = act(opps[w], st, rows, OPP_GREEDY, 1.0)
                    ul[rows], uq[rows], ms[rows], mq[rows] = a
                L0 = np.asarray(pool.env.legal_mask(NU, np.zeros(NGAME, np.int32))) if BEHAV else st[4]   # 記録するのは常に mask0
                for i in groups.get(-1, []):
                    R = rec[(i, s)]
                    for kk, v in zip(("tiles", "ou", "pu", "g"), st[:4]): R[kk].append(v[i])
                    R["legal"].append(L0[i])
                    R["ul"].append(ul[i].astype(np.int8)); R["uq"].append(uq[i].astype(np.int16))
                    R["ms"].append(ms[i].astype(np.int8)); R["mq"].append(mq[i].astype(np.int16))
                UO[s] = P.to_codes(ul, uq, ms, mq, st[1])
            pool.step(UO[0][0], UO[0][1], UO[1][0], UO[1][1])
        mo = pool.money()
        for (i, s), R in rec.items():
            nm = "c%d_%d_s%d" % (k, i, s)
            opp = "self" if who[i][1 - s] == -1 else OPPS[who[i][1 - s]]
            meta = dict(seat=s, seed=seeds[i], opp=opp, money=[float(mo[i, s]), float(mo[i, 1 - s])], gen_t=GEN_T, teacher=INIT, legal_opt=0,
                        behav=BEHAV or INIT, behav_legal_opt=2 if BEHAV else 0)
            X = {kk: np.asarray(v) for kk, v in R.items()}
            X["tiles"] = X["tiles"].astype(np.float16); X["legal"] = X["legal"].astype(np.uint8)
            np.savez(os.path.join(OUT, nm + ".tmp.npz"), meta=json.dumps(meta), **X)
            os.replace(os.path.join(OUT, nm + ".tmp.npz"), os.path.join(OUT, nm + ".npz"))
            names.append(nm); log["games"].append(dict(name=nm, **meta))
        # 束ごとの対戦成績(教師から見た勝率)
        by = {}
        for i in range(NGAME):
            if who[i] == (-1, -1): continue
            s = 0 if who[i][0] == -1 else 1; o = OPPS[who[i][1 - s]]
            by.setdefault(os.path.basename(o), []).append(float(mo[i, s] - mo[i, 1 - s]))
        print("束 %d: %d局(自己対戦 %d)・記録 %d席・%.0f秒 | 教師の勝率 %s" % (k, NGAME, nself, len(rec), time.time() - t0,
              {o: "%.3f(差 %+.0f、%d局)" % (np.mean(np.array(v) > 0), np.mean(v), len(v)) for o, v in by.items()}), flush=True)
        json.dump(log, open(os.path.join(OUT, "_gen.json"), "w"), ensure_ascii=False)
    if SPLIT:
        games = sorted({n.rsplit("_s", 1)[0] for n in names}); r = random.Random(302); r.shuffle(games)
        nh = max(1, int(round(len(games) * HOLD_FRAC))); hold_g = set(games[:nh])
        sp = dict(train=[n for n in names if n.rsplit("_s", 1)[0] not in hold_g], hold=[n for n in names if n.rsplit("_s", 1)[0] in hold_g], cache=OUT)
        p = os.environ.get("SPLITDIR", "/content/local_logs") + "/bc_%s.json" % SPLIT
        json.dump(sp, open(p, "w")); print("分割 %s: 学習 %d席 / 未見 %d席" % (p, len(sp["train"]), len(sp["hold"])), flush=True)


if __name__ == "__main__":
    main()
