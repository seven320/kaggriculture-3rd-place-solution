"""09-23: マスク移行(mask0 → mask2)の penalty を較正する。

denden さんの tools/calibrate_mask2.py と同じ式を、我々の骨格(kagsim + sp_fast + model_tf)に移したもの。
  expanded = mask2 & ~mask0
  gap      = max(expanded の logit) - max(mask0 の logit)   (生きていて expanded がある人手だけ)
  odds     = logsumexp(expanded の logit) - logsumexp(mask0 の logit)
  P(増えた手) = sigmoid((odds - pen) / temp)          ← 標本打ちで実際に引く確率。temp = 採取の ROLL_T
  penalty  = これの平均が CAL_PTGT 以下、かつ 1局1席の期待選択回数が CAL_PICKS 以下になる最小の整数
  (参考として、貪欲で手が変わらない旧規則 ceil(max(0, gap の 99.9%点 + 2)) も併記する)
**自分の局(旧マスクで打った軌道)の上で測る**。新マスクで自由に打たせた分布で測ると、
それは移行前のモデルが実際に訪れる局面ではない(旧マスク第117世代をそのまま新マスクで打たせると Elo 1003→888)。
なので「手は旧マスクから選んで進め、測るときだけ新マスクの logit を見る」。

使い方(コンテナの中):
  MASK_MIG=1 python mig_calib.py <重み> <出力 json>   [CAL_N=32 CAL_T=719 CAL_STRIDE=4]
penalty はモデルごとに違う。他のモデルの値(land4=15、PFN=16)を流用しないこと。
"""
import os, sys, json, math, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MASK_MIG"] = "1"                      # sp_fast の2重マスクと model_tf の罰の道を開ける(罰は 0 のまま使う)
os.environ.setdefault("LEGAL_OPT", "0")
import numpy as np, torch
import ppo_tf as P, model_tf as M, sp_fast
import fast_codes; P.to_codes = fast_codes.to_codes_fast

DEV = P.DEV
W = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else ""
NS = int(os.environ.get("CAL_N", "32")); T = int(os.environ.get("CAL_T", "719"))
STRIDE = int(os.environ.get("CAL_STRIDE", "4")); SEED0 = int(os.environ.get("CAL_SEED0", "3000000"))
TEMP = float(os.environ.get("CAL_TEMP", "1.0")); CH = int(os.environ.get("CAL_CHUNK", "64"))   # CAL_TEMP = 系統の ROLL_T(初期値)

net = M.load(W, DEV); M.mig_set(0.0)
torch.manual_seed(int(os.environ.get("CAL_TSEED", "1234")))
seeds = [SEED0 + 7919 * k for k in range(NS)]
sp_fast._MIG_SEAT[0] = None
pool = sp_fast.BatchSPPool(seeds)
pool.mig_ok[:] = True                                          # 両席で2つのマスクを取る(打つのは旧マスク)
pool.legal_opt = np.full((pool.n, 2), sp_fast.MIG_V, np.int32)

gaps = []; odds = []; nobs = 0; nlive = 0; nxp_unit = 0
t0 = time.time()
for t in range(T):
    acts = {}
    for s in (0, 1):
        st = pool.state(s)
        lmc = st[4]
        lm2 = (lmc != 0); xp = (lmc & 2) != 0; lm0 = lm2 & ~xp   # mask0 = mask2 & ~expanded
        st0 = list(st); st0[4] = lm0.astype(np.uint8)            # 打つのは旧マスク = モデル本来の分布
        outs = [P.sample(net, tuple(x[i:i + CH] for x in st0), greedy=False, temp=TEMP) for i in range(0, len(st[3]), CH)]
        ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
        acts[s] = P.to_codes(ul, uq, ms, mq, st[1])
        if t % STRIDE: continue
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            h, _ = net.encode(P.tt(st[0], torch.float16), P.tt(st[1]), P.tt(st[2]), P.tt(st[3]))
            lg, _ = net.unit_logits(h, P.tt(st[1]), P.tt(lm2.astype(np.uint8), torch.uint8))   # 新マスクの生 logit(罰 0)
        lg = lg.float()
        o_t = torch.as_tensor(lm0, device=DEV); x_t = torch.as_tensor(xp, device=DEV)
        live = torch.as_tensor(st[1][..., 0] > 0, device=DEV)
        sel = live & x_t.any(-1) & o_t.any(-1)
        ol = lg.masked_fill(~o_t, -1e9); nl = lg.masked_fill(~x_t, -1e9)
        gaps.append((nl.max(-1).values - ol.max(-1).values)[sel].cpu())
        odds.append((nl.logsumexp(-1) - ol.logsumexp(-1))[sel].cpu())
        nobs += int(len(st[3])); nlive += int(live.sum().item()); nxp_unit += int(sel.sum().item())
    pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    if t % 120 == 0: print("手%3d %.0fs 人手 %d" % (t, time.time() - t0, nxp_unit), flush=True)

gap = torch.cat(gaps); odd = torch.cat(odds).double()
assert len(gap), "expanded のある生きた人手が1つも無い(マスクの版が効いていない)"
q999 = float(torch.quantile(gap.double(), 0.999))
pen_greedy = float(math.ceil(max(0.0, q999 + 2.0)))       # 09-23 まで使っていた「貪欲で手が変わらない」規則(参考)

# --- 09-23 改: 標本打ちの基準に差し替える -------------------------------------------------
# 1手番1ユニットの判断で増えた手に行く確率は  P = sigmoid((odds - pen) / temp)。
#   odds = logsumexp(増えた手の logit) - logsumexp(旧マスクの手の logit)
# 貪欲は最大値しか見ないので「最大 logit の差」で足りたが、標本打ちは**確率質量の尾**を引く。
# さらに学習の採取は温度 ROLL_T で割るので、罰は実効 pen/temp に薄まる(g5r1 は 1.3 = 実効 13/17)。
# そこで「1局1席あたりの期待選択回数 <= PICKS_TGT(既定 2回)」かつ「1判断あたり <= P_TGT(既定 1e-4)」
# を満たす**最小の整数 pen** を採る。
SCALE_G = STRIDE / float(NS * 2)          # 抜き取った手番 → 1局1席 719手 ぶんに戻す係数
def _stat(pen):
    pr = ((odd - pen) / max(TEMP, 1e-6)).sigmoid()
    return float(pr.mean()), float(pr.sum()) * SCALE_G, float(pr.max())
P_TGT = float(os.environ.get("CAL_PTGT", "1e-4")); PICKS_TGT = float(os.environ.get("CAL_PICKS", "2.0"))
table = []
pen_new = None
for pen in range(0, 201):
    m_, pk_, mx_ = _stat(pen)
    if pen % 5 == 0 or pen == pen_greedy: table.append(dict(penalty=pen, prob_mean=m_, picks_per_game=pk_, prob_max=mx_))
    if pen_new is None and m_ <= P_TGT and pk_ <= PICKS_TGT: pen_new = float(pen)
assert pen_new is not None, "罰 200 でも基準に届かない(較正の入力かマスクの版を疑う)"
m_, pk_, mx_ = _stat(pen_new); mg_, pkg_, mxg_ = _stat(pen_greedy)
penalty = pen_new
res = dict(method="min pen s.t. mean sigmoid((odds-pen)/temp) <= %g かつ 期待選択 <= %g回/局席 (09-23 改・標本打ち基準)" % (P_TGT, PICKS_TGT),
           weights=W, weights_sha256=hashlib.sha256(open(W, "rb").read()).hexdigest(),
           games=NS, steps=T, stride=STRIDE, seed0=SEED0, rollout_temp=TEMP,
           observations=nobs, live_workers=nlive, live_workers_with_expanded=int(len(gap)),
           penalty=penalty,
           prob_mean=m_, picks_per_game=pk_, prob_max=mx_,
           penalty_greedy_rule=pen_greedy, greedy_rule_prob_mean=mg_, greedy_rule_picks_per_game=pkg_,
           greedy_expanded_choices=int((gap > pen_greedy).sum()),
           odds_quantiles={str(q): float(torch.quantile(odd, q)) for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)},
           gap_quantiles={str(q): float(torch.quantile(gap.double(), q)) for q in (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)},
           penalty_table=table,
           note="**CAL_TEMP に系統の ROLL_T(初期値)を必ず渡すこと。** 温度で罰は pen/temp に薄まる。"
                "旧マスクで打った軌道の上での較正で、強さの非劣性は別途 roll/h2h で確かめる")
print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
if OUT:
    os.makedirs(os.path.dirname(OUT), exist_ok=True) if os.path.dirname(OUT) else None
    open(OUT, "w").write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print("→ %s" % OUT, flush=True)
