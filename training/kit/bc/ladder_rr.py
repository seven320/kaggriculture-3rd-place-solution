"""09-22: 段階表(ラダー)の相手どうしを総当たりで打たせ、学習と同じ条件で1局ずつの所持金を書き出す。
学習の相手を「レーティングから出した勝率」で抽選するための物差し(tools/ladder_fit.py で Elo にする)。
  条件は学習の対局と同じ: シミュレータ(kagsim)、両席とも PolicyOpp(標本 = greedy=False、種の上限あり)、
  PACE/LAND/LEGAL_OPT は環境変数のまま(学習係と同じ値を渡す)。貪欲で測るなら GREEDY=1。
  使い方: LADDER="名前=パス,..." PAIRS="a|b,c|d" NSEED=48 SEED0=... python3 ladder_rr.py 出力.json
  出力は lab/league*.json と同じ形 {"a|b": [[a の所持金, b の所持金], ...]}。途中で止められても、済んだ組は残る。"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
import sp_fast; S.SPPool = sp_fast.BatchSPPool
import fast_codes; P.to_codes = fast_codes.to_codes_fast

out = sys.argv[1]
lad = dict(x.split("=", 1) for x in os.environ["LADDER"].split(",") if x)
pairs = [tuple(x.split("|")) for x in os.environ["PAIRS"].split(",") if x]
NSEED = int(os.environ.get("NSEED", "48")); SEED0 = int(os.environ.get("SEED0", "7000000"))
GREEDY = os.environ.get("GREEDY", "0") == "1"
# 09-22: 相手ごとの合法判定の版(LMASK="a79=0,g117=0"、書かなければ環境変数 LEGAL_OPT)。旧マスクで学習した方策は旧マスクで打たせる
LMASK = {k: int(v) for k, v in (x.split("=") for x in os.environ.get("LMASK", "").split(",") if x)}
def play(seeds, na, nb, s, ma, mb):
    """a を席 s、b を席 1-s に置いて打つ。返り値は絶対座席の最終所持金 (局, 2)"""
    pool = S.SPPool(seeds)
    if ma >= 0 or mb >= 0:
        pool.legal_opt = np.full((pool.n, 2), -1, np.int32); pool.legal_opt[:, s] = ma; pool.legal_opt[:, 1 - s] = mb
    ag = {s: S.PolicyOpp(na, greedy=GREEDY, cap=True), 1 - s: S.PolicyOpp(nb, greedy=GREEDY, cap=True)}; idx = list(range(pool.n))
    for t in range(719):
        acts = {}
        for q in (0, 1):
            st = pool.state(q); acts[q] = ag[q].act(st, t, idx, pool, q)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    return pool.money()
R = json.load(open(out)) if os.path.exists(out) else {}
print("LEGAL_OPT", os.environ.get("LEGAL_OPT", "0"), "相手ごとの版", LMASK or "なし", "PACE", sp_fast.PACE, "LAND", sp_fast.LAND, "貪欲" if GREEDY else "標本", "組", len(pairs), "各", 2 * NSEED, "局", flush=True)
cache = {}
def net_of(nm):
    if nm not in cache:
        if len(cache) >= 6: cache.pop(next(iter(cache)))
        cache[nm] = M.load(lad[nm], P.DEV).eval()
    return cache[nm]
for k, (a, b) in enumerate(pairs):
    key = "%s|%s" % (a, b)
    if key in R and len(R[key]) >= 2 * NSEED: continue
    t0 = time.time(); seeds = [SEED0 + 7919 * j + 104729 * k for j in range(NSEED)]
    na, nb = net_of(a), net_of(b); rows = []
    with torch.no_grad():
        for s in (0, 1):
            m = play(seeds, na, nb, s, LMASK.get(a, -1), LMASK.get(b, -1))
            rows += [[float(m[i, s]), float(m[i, 1 - s])] for i in range(len(seeds))]
    R[key] = rows; json.dump(R, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    v = np.array(rows); w = float((v[:, 0] > v[:, 1]).mean() + 0.5 * (v[:, 0] == v[:, 1]).mean())
    print("%s  %s 対 %s  %d局 %s の勝率 %.3f 差 %+.0f  %.0f秒" % (time.strftime("%H:%M"), a, b, len(rows), a, w, (v[:, 0] - v[:, 1]).mean(), time.time() - t0), flush=True)
print("== 済 ==", flush=True)
