"""exp320(失敗分析): 優位度(学習の信号)が局のどこにあるか。
序盤(t<240)に当てた介入(OPEN / PFNW / 家畜の開局)に、そもそも乗れる勾配があったのか。

やること(局 = <ASYNC_DIR>/buf/*.pt を読むだけ):
  1. 保存された V と最終所持金から GAE を組み直し(GAMMA=1, LAM=0.95, 終端だけ ±1)、
     |優位度| を手番の帯ごとに出す。「介入が当たる帯」に信号があるか。
  2. 市場の行動 ms/mq から「序盤のメロンの種・小麦の種の個数」を数え、
     その局の勝敗(z)との相関を出す。序盤の作付けが勝敗を説明するか。
  3. 方策が序盤にメロン/小麦の種の枠へ置く確率(= 介入が要求する行動の元の確率)を測る。

使い方: python fa_adv.py <名前>=<局.pt> [...]  (NET=... で確率の測定に使う重み)
"""
import os, sys, json
import numpy as np, torch

sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
from codec import q_value, MQ_MAX

GAMMA = float(os.environ.get("GAMMA", "1.0")); LAM = float(os.environ.get("LAM", "0.95"))
T1 = int(os.environ.get("T1", "240"))
NETP = os.environ.get("NET", "")
BUCK = [0, 120, 240, 360, 480, 600, 720]
C_WHEAT, C_MELON = 9 + 0, 9 + 4           # extract4.mkt_code: BUY_SEED item = 9 + IT.index
QV = np.array([q_value(c, MQ_MAX) for c in range(1 + MQ_MAX + 1 + 2)], np.float64)
QV[QV < 0] = 0.0; QV[QV > 1000] = 0.0     # QNONE と番兵(全部売り)は数えない


def gae(V, z):
    T_, N = V.shape
    Rw = np.zeros_like(V); Rw[-1] = z
    adv = np.zeros_like(V); last = np.zeros(N)
    for t in reversed(range(T_)):
        nv = V[t + 1] if t + 1 < T_ else np.zeros(N)
        delta = Rw[t] + GAMMA * nv - V[t]; last = delta + GAMMA * LAM * last; adv[t] = last
    return adv


def one(name, path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    V = np.asarray(d["V"], np.float64); fin = np.asarray(d["fin"], np.float64)
    T_, N = V.shape
    z = np.sign(fin[:, 0] - fin[:, 1])
    adv = gae(V, z)
    print("\n### %s  手%d x 軌道%d  版%s  勝率 %.3f" % (name, T_, N, d.get("ver"), float((z > 0).mean())), flush=True)
    tot = np.abs(adv).sum()
    print("  |優位度| の分布(帯ごと):")
    for a, b in zip(BUCK[:-1], BUCK[1:]):
        b = min(b, T_)
        if a >= b: continue
        seg = adv[a:b]
        print("    手%3d-%3d  平均|adv| %.4f  SD %.4f  全体に占める割合 %.3f(手数の割合 %.3f)"
              % (a, b, np.abs(seg).mean(), seg.std(), np.abs(seg).sum() / tot, (b - a) / T_), flush=True)

    ms = np.asarray(d["D"]["ms"]).reshape(T_, N, -1)
    mq = np.asarray(d["D"]["mq"]).reshape(T_, N, -1)
    q = QV[np.clip(mq, 0, len(QV) - 1)]
    mel = np.where(ms == C_MELON, q, 0.0); whe = np.where(ms == C_WHEAT, q, 0.0)
    mel_e = mel[:T1].sum((0, 2)); whe_e = whe[:T1].sum((0, 2))       # 軌道ごと、序盤の個数
    print("  序盤(t<%d)の種の個数: メロン %.1f ± %.1f / 小麦 %.1f ± %.1f" %
          (T1, mel_e.mean(), mel_e.std(), whe_e.mean(), whe_e.std()), flush=True)
    for nm_, x in (("メロン", mel_e), ("小麦", whe_e)):
        if x.std() > 1e-9:
            r = np.corrcoef(x, z)[0, 1]
            print("    %s の個数 と 勝敗 の相関 r=%+.3f (z=%+.2f)  四分位 %s"
                  % (nm_, r, r * np.sqrt(N - 1), np.percentile(x, [0, 25, 50, 75, 100]).round(1).tolist()), flush=True)
        else:
            print("    %s の個数 は一定(%.1f)" % (nm_, x.mean()), flush=True)

    hit = (ms[:T1] == C_MELON).any(-1)                                 # 序盤でメロンの種を注文した局面
    if hit.any() and (~hit).any():
        a1 = adv[:T1][hit]; a0 = adv[:T1][~hit]
        print("  序盤の局面の優位度: メロンを注文した手 %+.4f ± %.4f (%d手) / それ以外 %+.4f ± %.4f (%d手)  差 %+.4f"
              % (a1.mean(), a1.std() / np.sqrt(len(a1)), len(a1), a0.mean(), a0.std() / np.sqrt(len(a0)), len(a0),
                 a1.mean() - a0.mean()), flush=True)

    # 3. 元の確率(介入が要求する行動が、もともとどれだけの確率を持っていたか)
    if NETP:
        try:
            import model_tf as M, ppo_tf as P
            import torch.nn.functional as F
            net = M.load(NETP, "cuda"); net.eval()
            NM_ = 23
            rows = np.arange(T_ * N).reshape(T_, N)[:T1].reshape(-1)
            sel = rows[np.random.default_rng(0).choice(len(rows), size=min(4096, len(rows)), replace=False)]
            keys = ("tiles", "ou", "pu", "g")
            pm = {"小麦": 0.0, "メロン": 0.0, "終了": 0.0}; nb = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for i in range(0, len(sel), 256):
                    ix = torch.as_tensor(sel[i:i + 256])
                    b = [d["D"][k][ix].to("cuda") for k in keys]
                    b[0] = b[0].half()
                    h, pad = net.encode(b[0], b[1].float(), b[2].float(), b[3].float())
                    msb = d["D"]["ms"][ix].to("cuda").long()
                    prev = torch.cat([torch.full_like(msb[:, :1], NM_), msb[:, :-1]], 1).clamp(max=NM_)
                    mo, _ = net.market(h, pad, prev)
                    p = F.softmax(mo.float(), -1)[:, 0]              # 1枠目の分布
                    pm["小麦"] += float(p[:, C_WHEAT].sum()); pm["メロン"] += float(p[:, C_MELON].sum())
                    pm["終了"] += float(p[:, 22].sum()); nb += p.shape[0]
            print("  序盤の1枠目の確率(%d局面): 小麦の種 %.4f / メロンの種 %.4f / 終了 %.4f"
                  % (nb, pm["小麦"] / nb, pm["メロン"] / nb, pm["終了"] / nb), flush=True)
            del net; torch.cuda.empty_cache()
        except Exception as e:
            print("  [確率の測定は失敗] %r" % (e,), flush=True)
    del d


for a in sys.argv[1:]:
    nm, p = a.split("=", 1)
    one(nm, p)
print("== 終わり ==", flush=True)
