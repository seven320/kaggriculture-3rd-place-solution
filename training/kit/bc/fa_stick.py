"""exp320(失敗分析): 介入は方策に入ったのか / そもそも縛れていたのか。

OPEN(序盤のメロンの種を OPEN_MELCAP 個までに削り小麦へ振替)で学習した世代を **介入なし** で打たせ、
  (a) 狙った振る舞い(序盤のメロンを減らし小麦を増やす)が残っているか = 身につくか
  (b) そもそも方策が序盤に何個メロンの種を出しているか = 上限4は縛りになっていたのか
を測る。(b) が 4個未満なら、介入は「何もしていない」ことになる。

使い方: NSEED=96 python fa_stick.py <名前>=<重み> [...]
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf as M
import sp_lib as S
import kagpack

DEV = "cuda"
N = int(os.environ.get("NSEED", "96"))
T1 = int(os.environ.get("T1", "240"))        # OPEN_T1 と同じ「序盤」の定義
TMAX = int(os.environ.get("TMAX", "719"))
GREEDY = os.environ.get("GREEDY", "0") == "1"   # 学習中は標本なので既定は標本
CAP = int(os.environ.get("MELCAP", "4"))

BUYSEED = kagpack.MOPI["BUY_SEED"]
BUYLAND = kagpack.MOPI["BUY_LAND"]
BUYANI = kagpack.MOPI.get("BUY_ANIMAL")
IT = {k: kagpack.ITI[k] for k in ("MELON", "WHEAT", "STRAWBERRY", "TOMATO", "CARROT")}
ANI = {k: kagpack.ITI[k] for k in ("COW", "SHEEP", "GOOSE") if k in kagpack.ITI}
seeds = [3000017 + 7919 * k for k in range(N)]


def run(name, path, greedy):
    net = M.load(path, DEV); net.eval()
    pool = S.SPPool(seeds); idx = list(range(pool.n))
    ag = S.PolicyOpp(net, greedy=greedy, cap=True)
    qty = {k: np.zeros(N) for k in IT}        # 序盤(t<T1)に出した種の注文の「個数」の合計
    slots = {k: np.zeros(N) for k in IT}      # 同、注文の「枠数」
    ani = {k: np.zeros(N) for k in ANI}
    mel_cum = np.zeros(N)                     # メロンの累積(上限の当たり方を見る)
    over = np.zeros(N)                        # 上限 CAP を超えて出した個数(= OPEN が削った量)
    q_step = {2: np.full(N, -1), 3: np.full(N, -1), 4: np.full(N, -1)}
    with torch.no_grad():
        for t in range(TMAX):
            acts = {}
            for s in (0, 1):
                st = pool.state(s)
                U, O = ag.act(st, t, idx, pool, s)
                acts[s] = (U, O)
                if s == 0 and t < T1:
                    A = np.asarray(O)
                    op, it_, q = A[..., 0], A[..., 1], A[..., 2].astype(np.float64)
                    for k, code in IT.items():
                        m = (op == BUYSEED) & (it_ == code)
                        qty[k] += np.where(m, q, 0).sum(1)
                        slots[k] += m.sum(1)
                    mm = (op == BUYSEED) & (it_ == IT["MELON"])
                    mq = np.where(mm, q, 0).sum(1)
                    # OPEN と同じ順で上限をかけ、削られる量を数える
                    take = np.minimum(mq, np.maximum(0.0, CAP - mel_cum))
                    over += mq - take; mel_cum += take
                    if BUYANI is not None:
                        for k, code in ANI.items():
                            ani[k] += np.where((op == BUYANI) & (it_ == code), q, 0).sum(1)
            pool.envs[0].set_view(0)
            nq = np.array([np.rint(np.asarray(e.farm_pub(0))[0, 2] * 4) for e in pool.envs]).reshape(-1)
            if nq.size == 1 and N > 1:        # BatchSPPool 形式でない場合の保険
                nq = np.full(N, nq[0])
            for qn in (2, 3, 4):
                hit = (nq >= qn) & (q_step[qn] < 0); q_step[qn][hit] = t
            pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    money = pool.money()[:, 0]
    out = dict(name=name, greedy=bool(greedy), money=float(money.mean()), money_sd=float(money.std(ddof=1) / np.sqrt(N)))
    for k in IT:
        out["個数_" + k] = float(qty[k].mean()); out["個数sd_" + k] = float(qty[k].std(ddof=1) / np.sqrt(N))
        out["枠数_" + k] = float(slots[k].mean())
    out["メロン超過"] = float(over.mean())                    # OPEN が実際に削る個数
    out["上限に当たった局"] = float((over > 0).mean())          # 上限が縛った局の割合
    out["メロン分布"] = np.percentile(qty["MELON"], [0, 25, 50, 75, 100]).tolist()
    for k in ANI: out["家畜_" + k] = float(ani[k].mean())
    for qn in (2, 3, 4):
        v = q_step[qn]
        out["区画%d手番" % qn] = float(v[v >= 0].mean()) if (v >= 0).any() else -1.0
        out["区画%d到達" % qn] = float((v >= 0).mean())
    del net; torch.cuda.empty_cache()
    return out


rows = []
for a in sys.argv[1:]:
    nm, p = a.split("=", 1)
    for g in ([True, False] if os.environ.get("BOTH", "1") == "1" else [GREEDY]):
        r = run(nm, p, g); rows.append(r)
        print("%-14s %s 所持金 %7.0f±%4.0f | 序盤の個数 メロン %5.2f±%.2f 小麦 %5.2f±%.2f イチゴ %5.2f | "
              "OPEN が削る %.2f個(縛った局 %.0f%%) メロン四分位 %s | 区画2 %.0f手 区画3 %.0f手 区画4 %.0f手(到達 %.2f)"
              % (r["name"], "貪欲" if g else "標本", r["money"], r["money_sd"],
                 r["個数_MELON"], r["個数sd_MELON"], r["個数_WHEAT"], r["個数sd_WHEAT"], r["個数_STRAWBERRY"],
                 r["メロン超過"], 100 * r["上限に当たった局"], r["メロン分布"],
                 r["区画2手番"], r["区画3手番"], r["区画4手番"], r["区画4到達"]), flush=True)
json.dump(rows, open("/content/lab/fa_stick.json", "w"), ensure_ascii=False)
print("書き出した /content/lab/fa_stick.json", flush=True)
