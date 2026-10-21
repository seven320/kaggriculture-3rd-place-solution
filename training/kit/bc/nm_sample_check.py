"""新マスクの生徒を PPO に渡す前の合格検査(newmask-distill-plan.md 5節の「標本打ちの関門」)。**新しいファイル。**

mask-migration-attempt.md の教訓: 貪欲の検査は全部通ったのに、標本打ち(学習の採取)で新しい手が 88% 選ばれ、2区画目が 702手になった。
そこで**PPO の対局係と同じ経路**(sp_fast.BatchSPPool の MASK_MIG=1 MIG_V=2 の印付き lm → ppo_tf.sample(cap=True, 温度 T))で
生徒の自己対戦を打たせ、同じ種で教師(旧マスク)の自己対戦と並べる。
  - 新しい手(lm == 3)を選んだ割合(ユニットの判断あたり)          … 合格: 1% 未満(apo の MIG_GUARD は 5% で止める)
  - 2区画目・3区画目に届いた手番(局の平均と中央値)               … 合格: 教師との差が ±10手以内(正常は 150 前後)
  - 最終所持金の平均(自己対戦なので勝敗ではなく経済の健全さ)       … 合格: 教師の 95% 以上
使い方: MASK_MIG=1 MIG_V=2 STUDENT=<重み> TEACHER=<重み> NGAME=64 T=1.0 python3 nm_sample_check.py
  (CPU の小さな確認: PPODEV=cpu DEV=cpu NGAME=2)
"""
import os, sys, json, time
if os.environ.get("MASK_MIG") != "1" or os.environ.get("MIG_V", "2") != "2":
    raise SystemExit("MASK_MIG=1 MIG_V=2 で起動すること(PPO の対局係と同じ印付きの合法マスクで打たせるため)")
if os.environ.get("LEGAL_OPT", "0") != "0":
    raise SystemExit("LEGAL_OPT は外すこと(基準の版は 0、生徒の席だけ 2)")
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
DEV = os.environ.get("DEV", os.environ.get("PPODEV", "cuda")); os.environ.setdefault("PPODEV", DEV)
import model_tf as M
import ppo_tf as P
import sp_fast
from extract4 import NU
assert M.MIG and sp_fast.MIG and P.DEV == DEV
M.mig_set(0.0)                     # 罰は 0(生徒は重みで学んでいる)。apo の MIG_PEN=0 と同じ
NG = int(os.environ.get("NGAME", "64")); TEMP = float(os.environ.get("T", "1.0")); SEED0 = int(os.environ.get("SEED0", "880000"))
CH = int(os.environ.get("CH", "256")); AMP = DEV != "cpu"


def run(net, mig):
    seeds = [SEED0 + 7919 * k for k in range(NG)]
    pool = sp_fast.BatchSPPool(seeds)
    if mig: pool.mig_ok[:] = True; pool.legal_opt = np.full((NG, 2), 2, np.int32)
    sel_new = n_units = 0; q2 = np.full((NG, 2), 999); q3 = np.full((NG, 2), 999)
    ctx = torch.autocast("cuda", dtype=torch.float16) if AMP else torch.autocast("cpu", enabled=False)
    t0 = time.time()
    for t in range(719):
        UO = []
        for s in (0, 1):
            st = pool.state(s)
            outs = []
            with torch.no_grad(), ctx:
                for i in range(0, NG, CH):
                    outs.append(P.sample(net, tuple(x[i:i + CH] for x in st), greedy=False, cap=True, temp=TEMP)[:4])
            ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
            lm = st[4]; live = ul >= 0
            ch = np.take_along_axis(lm, np.clip(ul, 0, None)[..., None], -1)[..., 0]
            sel_new += int(((ch == 3) & live).sum()); n_units += int(live.sum())
            nq = np.rint(st[3][:, 39] * 4)
            q2[:, s] = np.where((nq >= 2) & (q2[:, s] == 999), t, q2[:, s]); q3[:, s] = np.where((nq >= 3) & (q3[:, s] == 999), t, q3[:, s])
            UO.append(P.to_codes(ul, uq, ms, mq, st[1]))
        pool.step(UO[0][0], UO[0][1], UO[1][0], UO[1][1])
    mo = pool.money()
    return dict(sel_new=sel_new / max(n_units, 1), n_units=n_units, q2_mean=float(q2.mean()), q2_med=float(np.median(q2)), q3_mean=float(q3.mean()),
                q3_med=float(np.median(q3)), q2_miss=int((q2 == 999).sum()), money=float(mo.mean()), sec=time.time() - t0)


def main():
    S = M.load(os.environ["STUDENT"], DEV).eval(); T = M.load(os.environ["TEACHER"], DEV).eval()
    a = run(S, True); b = run(T, False)
    f = lambda r: "新しい手の選択率 %.4f%%(%d判断)| 2区画目 平均 %.1f 中央 %.0f(届かず %d席)3区画目 平均 %.1f 中央 %.0f | 所持金 %.0f | %.0f秒" % (
        100 * r["sel_new"], r["n_units"], r["q2_mean"], r["q2_med"], r["q2_miss"], r["q3_mean"], r["q3_med"], r["money"], r["sec"])
    print("生徒(新マスク・T=%.2f): %s" % (TEMP, f(a))); print("教師(旧マスク・T=%.2f): %s" % (TEMP, f(b)))
    ok = a["sel_new"] < 0.01 and abs(a["q2_med"] - b["q2_med"]) <= 10 and a["money"] >= 0.95 * b["money"]
    print("== 合格 ==" if ok else "!! 不合格(PPO に渡さない)")
    json.dump(dict(student=a, teacher=b, ok=ok, env={k: os.environ.get(k) for k in ("STUDENT", "TEACHER", "NGAME", "T", "SEED0")}),
              open(os.environ.get("OUTJSON", "/dev/null"), "w"))


if __name__ == "__main__":
    main()
