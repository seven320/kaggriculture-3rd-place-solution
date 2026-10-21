"""09-23: 新マスクで増えた手を**型ごと**に数え、そのうち実際に成立した件数を出す。

方針転換の根拠を作る道具。「増えた手のほぼ全部はエンジンがはねる no-op で、
本当に欲しい手番内の連携はごく一部」かどうかを、型(行動の種類)ごとの数字で示す。
選んだだけの件は成立に数えない。次の手番の**携行品**と**自分の農場のマス**を見て、
その行動が起こすはずの変化が実際に起きたものだけ数える。

  MASK_MIG=1 python3 mig_audit.py <重み> <penalty>   [AU_N=64 AU_T=719 AU_V=2(マスクの版)]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MASK_MIG"] = "1"; os.environ.setdefault("LEGAL_OPT", "0")
import numpy as np, torch
import ppo_tf as P, model_tf as M, sp_fast
import fast_codes; P.to_codes = fast_codes.to_codes_fast
DEV = P.DEV
W = sys.argv[1]; PEN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
NS = int(os.environ.get("AU_N", "64")); T = int(os.environ.get("AU_T", "719"))
SEED0 = int(os.environ.get("AU_SEED0", "7100000")); CH = int(os.environ.get("AU_CHUNK", "128"))
MV = int(os.environ.get("AU_V", str(sp_fast.MIG_V)))
sp_fast.MIG_V = MV; sp_fast.MIG_VENG = 2 if MV == 3 else MV      # mask2'(3)も試せるように
if os.environ.get("MIG_KEEP"): sp_fast.MIG_KEEP = set(x.strip() for x in os.environ["MIG_KEEP"].split(",") if x.strip())
IT = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
BASE = ["待機", "北", "南", "東", "西", "水やり", "収穫", "施肥", "掘る", "小屋", "牧草地", "給餌", "肥料採取", "世話", "全部置く"]
NAME = {c: BASE[c] for c in range(15)}
for c in range(5): NAME[15 + c] = "植える %s" % IT[c]
for c in range(12): NAME[20 + c] = "取り出す %s" % IT[c]; NAME[32 + c] = "置く %s" % IT[c]
# 成立の判定: ("inv", 品目, 向き) = 携行品がその向きに動いた / ("kind", 値) = 立っているマスの種類がその値になった
#            ("invsum", 向き) = 携行品の合計 / ("tile", 列, 向き) = マスのその列が動いた
JUDGE = {}
for i in range(12): JUDGE[20 + i] = ("inv", i, +1); JUDGE[32 + i] = ("inv", i, -1)
for c in range(5): JUDGE[15 + c] = ("kind", 5)
JUDGE.update({5: ("tile", 2, +1), 6: ("invsum", +1), 7: ("inv", 8, -1), 8: ("kind", 0), 9: ("kind", 3),
              10: ("kind", 4), 11: ("inv", 0, -1), 12: ("inv", 8, +1), 13: ("tile", 7, +1), 14: ("invsum", -1)})

net = M.load(W, DEV); M.mig_set(PEN)
torch.manual_seed(int(os.environ.get("AU_TSEED", "77")))
sp_fast._MIG_SEAT[0] = None
pool = sp_fast.BatchSPPool([SEED0 + 7919 * k for k in range(NS)])
pool.mig_ok[:] = True; pool.legal_opt = np.full((pool.n, 2), MV, np.int32)

nsel = np.zeros(44, np.int64); nreal = np.zeros(44, np.int64); navail = np.zeros(44, np.int64)
nlive = 0; nsel_tot = 0; ex = []
prev = {}; t0 = time.time()
for t in range(T):
    acts = {}
    for s in (0, 1):
        st = pool.state(s); lmc = st[4]; ou = st[1]
        inv = ou[..., 3:15]; ux = ou[..., 1].astype(np.int64).clip(0, 9); uy = ou[..., 2].astype(np.int64).clip(0, 9)
        own = np.asarray(st[0][:, 0], np.float32)                      # 自分の農場 (局,100,13)
        ti = (uy * 10 + ux)
        tl = np.take_along_axis(own, ti[..., None], 1)                 # (局,人手,13) 立っているマス
        kd = np.rint(tl[..., 0] * 5).astype(np.int64)
        if s in prev:                                                   # 1手前の「増えた手」が成立したか
            p_ul, p_xp, p_inv, p_kd, p_tl = prev[s]
            for c0 in np.unique(p_ul[p_xp]):
                c0 = int(c0)
                if c0 not in JUDGE: continue
                m = p_xp & (p_ul == c0); j = JUDGE[c0]
                if j[0] == "inv": d = inv[..., j[1]] - p_inv[..., j[1]]; ok = d * j[2] > 0.5
                elif j[0] == "invsum": d = inv.sum(-1) - p_inv.sum(-1); ok = d * j[1] > 0.5
                elif j[0] == "kind": ok = (kd == j[1]) & (p_kd != j[1])
                else: ok = (tl[..., j[1]] - p_tl[..., j[1]]) * j[2] > 1e-3
                nsel[c0] += int(m.sum()); nreal[c0] += int((m & ok).sum())
                if len(ex) < 8 and (m & ok).any():
                    g_, u_ = [int(x[0]) for x in np.nonzero(m & ok)]
                    ex.append("手%d 席%d 局%d 人手%d %s" % (t - 1, s, g_, u_, NAME[c0]))
        outs = [P.sample(net, tuple(x[i:i + CH] for x in st), greedy=False) for i in range(0, len(st[3]), CH)]
        ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
        live = ou[..., 0] > 0
        xp_sel = live & (np.take_along_axis((lmc & 2), np.clip(ul, 0, 43)[..., None], -1)[..., 0] != 0)
        navail += ((lmc & 2) != 0)[live].sum(0).astype(np.int64)        # 型ごとに「候補に出た人手」の数
        nlive += int(live.sum()); nsel_tot += int(xp_sel.sum())
        prev[s] = (ul.copy(), xp_sel.copy(), inv.copy(), kd.copy(), tl.copy())
        acts[s] = P.to_codes(ul, uq, ms, mq, ou)
    pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    if t % 240 == 0: print("  手%3d %.0fs 増えた手の選択 %d" % (t, time.time() - t0, nsel_tot), flush=True)

print("\n[増えた手の型ごとの集計] %s マスク版 %d 罰 %.1f %d局 %d手" % (os.path.basename(W), MV, PEN, NS, T))
print("  生きた人手の判断 %d 回 / 増えた手を選んだ %d 回 (%.4f%%)" % (nlive, nsel_tot, 100.0 * nsel_tot / max(nlive, 1)))
print("  %-22s %12s %10s %8s %10s" % ("型(なぜ旧マスクが禁じたか)", "候補に出た", "選んだ", "成立", "成立率"))
order = sorted(range(44), key=lambda c: (-nreal[c], -nsel[c]))
for c in order:
    if nsel[c] == 0 and navail[c] == 0: continue
    print("  %-22s %12d %10d %8d %10s" % (NAME.get(c, str(c)), navail[c], nsel[c], nreal[c],
          ("%.2e" % (nreal[c] / nsel[c])) if nsel[c] else "-"))
print("  合計 候補 %d 選択 %d **成立 %d**" % (navail.sum(), nsel.sum(), nreal.sum()))
for e in ex: print("    例: %s" % e)
print("  ※ 成立した型だけを残したマスク(opt=3)を作るのが目的。0 件の型は行動空間のゴミ")
