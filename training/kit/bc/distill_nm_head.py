"""新マスクの生徒を「ユニットの頭だけ」蒸留する(newmask-distill-plan.md 8節)。**新しいファイル。走っているジョブはこれを読まない。**

8節の診断: tf_nm21d(全体を蒸留)の弱さは**ユニットの側**だけにあった(混成の探り: 生徒のユニット+教師の市場 0.310、
教師のユニット+生徒の市場 0.500)。未見での貪欲一致 0.9888 は gen20/gen22(0.994)より低く、しかも教師の top1−top2 が
1 以上の「自信のある」手を取り違える(不一致の 20%、gen20 は 3%)。符号器ごと動かすと、ゴミを下げる学習と τ の縮めが
mask0 の中の順位まで崩す。

ここでは**教師の符号器・市場の頭・価値の頭をそのまま**使い、ユニットの頭(uq_in / u_score / cand / u_qty)だけを学ぶ。
  入力は教師の unit_query(h, ou)(= 自ユニットのトークン + そのマスのトークン、(B,20,2d))を先に一度だけ計算して持つ。
  目標は distill_nm と同じ q(教師の mask0 の中の softmax/τ_u + K への ε)、生徒は mask2 の softmax。
  + GREEDYW·hinge(教師の mask0 の argmax を他のどの手より GMARGIN 以上高く)、+ WQ·KL(数量、教師の頭が目標)。
出力の重みは教師の全重み + 学んだユニットの頭(形は model_tf.Policy そのもの。PPO・league・mkagent_nm がそのまま読む)。

環境変数:
  TEACHER(必須) TAG SPLIT SPLITDIR(/content/local_logs) CACHE(分割 json の cache を上書き) NTR(学習に使う席の数、0=全部) NHO(未見の席、8)
  TAU_U(1.0) EPS(0.01) GREEDYW(1.0) GMARGIN(0.5) WQ(0.5) LR(1e-4) WARM(100) STEPS(4000) BATCH(256) EVAL_EVERY(500)
  SEED(0) DEV(cuda) STORE(= DEV、持つ場所) LOGDIR(/content/local_logs) ENC_BS(256 = 教師の符号器を回す塊)
"""
import os, sys, json, time, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tf as M
import nm_lib
E = os.environ
DEV = E.get("DEV", "cuda")
TAU_U = float(E.get("TAU_U", "1.0")); EPS = float(E.get("EPS", "0.01"))
GREEDYW = float(E.get("GREEDYW", "1.0")); GMARGIN = float(E.get("GMARGIN", "0.5")); WQ = float(E.get("WQ", "0.5"))
KEYS = ("tiles", "ou", "pu", "g", "legal", "ul")
STORE = E.get("STORE", DEV)


@torch.no_grad()
def build(T, names, cache):
    """席ごとに教師の符号器を回し、ユニットの頭の入力と目標を持つ(置き場は STORE。既定 = DEV。1局面 ≈ 30KB、500席 ≈ 11GB)。"""
    X, LT, LQT, M0, M2, KK, LV, UL = [], [], [], [], [], [], [], []
    bs = int(E.get("ENC_BS", "256")); t0 = time.time()
    for si, n in enumerate(names):
        with np.load("%s/%s.npz" % (cache, n)) as z:
            d = {k: z[k] for k in KEYS}
        for i in range(0, len(d["g"]), bs):
            b = {k: torch.as_tensor(v[i:i + bs], device=DEV) for k, v in d.items()}
            tiles, ou, pu, g = b["tiles"].float(), b["ou"].float(), b["pu"].float(), b["g"].float()
            lm0 = b["legal"] != 0; lm2 = nm_lib.mask2_t(ou, g); ul = b["ul"].long()
            h, _ = T.encode(tiles, ou, pu, g)                  # fp32(教師の目標を丸めない)
            x = T.unit_query(h, ou)
            lg, q = T.unit_logits(h, ou, lm0)
            lq = T.unit_qty(q, ul)
            X.append(x.to(STORE, torch.float16)); LT.append(lg.float().to(STORE)); LQT.append(lq.to(STORE, torch.float16))
            M0.append(lm0.to(STORE)); M2.append(lm2.to(STORE)); KK.append((lm2 & ~lm0 & nm_lib.keep_t(tiles, ou)).to(STORE))
            LV.append((ou[..., 0] > 0).to(STORE)); UL.append(ul.to(STORE))
        if si % 10 == 9: print("  符号器 %d/%d 席 %.1f分" % (si + 1, len(names), (time.time() - t0) / 60), flush=True)
    C = lambda L: torch.cat(L)
    return dict(x=C(X), lt=C(LT), lqt=C(LQT), m0=C(M0), m2=C(M2), K=C(KK), live=C(LV), ul=C(UL))


def head(S, x, lm2, ul):
    """unit_logits / unit_qty と同じ式(model_tf の FAST の書き方 = 足す順だけ違う)。x は unit_query の出力。"""
    q = S.uq_in(x)
    l0 = S.u_score[0]; l2 = S.u_score[2]; d = S.d
    a = F.linear(q, l0.weight[:, :2 * d], l0.bias)
    c = F.linear(S.cand.weight, l0.weight[:, 2 * d:])
    s = F.linear(F.relu(a.unsqueeze(2) + c.view(1, 1, M.NC, d)), l2.weight, l2.bias).squeeze(-1)
    return s.masked_fill(~lm2, -1e4), S.unit_qty(q, ul)


def kl_rows(p, logq):
    return torch.where(p > 0, p * (torch.log(p.clamp_min(1e-30)) - logq), torch.zeros_like(p)).sum(-1)


def terms(S, D, j, stats=False):
    b = {k: v[j].to(DEV) for k, v in D.items()}
    x = b["x"].float(); lm0, lm2, K, live, ul = b["m0"], b["m2"], b["K"], b["live"], b["ul"]
    lgT = b["lt"]
    pT = torch.softmax(lgT / TAU_U, -1) * lm0; pT = pT / pT.sum(-1, keepdim=True).clamp_min(1e-30)
    nk = K.sum(-1, keepdim=True); e = EPS * (nk > 0).float()
    q = (1 - e) * pT + e * K.float() / nk.clamp(min=1).float()
    lgS, lqS = head(S, x, lm2, ul)
    ls = torch.log_softmax(lgS, -1); nl = live.sum().clamp(min=1)
    Lu = (kl_rows(q, ls) * live).sum() / nl
    qm = live & (ul >= 20)
    Lq = (kl_rows(torch.softmax(b["lqt"].float(), -1), torch.log_softmax(lqS.float(), -1)) * qm).sum() / qm.sum().clamp(min=1)
    aT = lgT.argmax(-1, keepdim=True)
    la = ls.gather(-1, aT).squeeze(-1); oth = ls.scatter(-1, aT, -1e9).max(-1).values
    Lg = (F.relu(oth - la + GMARGIN) * live).sum() / nl
    L = Lu + WQ * Lq + GREEDYW * Lg
    if not stats: return L, None
    with torch.no_grad():
        p = ls.exp(); new = lm2 & ~lm0
        top2 = lgT.topk(2, -1).values; gap = top2[..., 0] - top2[..., 1]
        aS = lgS.argmax(-1, keepdim=True)
        dis = live & (aS.squeeze(-1) != aT.squeeze(-1))
        st = dict(Lu=float(Lu), Lg=float(Lg), n=int(live.sum()), agree=int((live & ~dis).sum()),
                  dis_big=int((dis & (gap > 1)).sum()), newsel=int((dis & new.gather(-1, aS).squeeze(-1)).sum()),
                  qagree=int(((lqS.argmax(-1) == b["lqt"].float().argmax(-1)) & qm).sum()), nq=int(qm.sum()),
                  entS=float((-(p * ls.masked_fill(~lm2, 0)).sum(-1) * live).sum()), entQ=float((-(q * torch.log(q.clamp_min(1e-30))).sum(-1) * live).sum()),
                  mnew=(p * new).sum(-1)[live].cpu().numpy(), mgar=(p * (new & ~K)).sum(-1)[live].cpu().numpy(), nstate=len(x))
    return L, st


def evaluate(S, D, bs=512):
    A = {}; mn = []; mg = []
    for i in range(0, len(D["x"]), bs):
        j = torch.arange(i, min(i + bs, len(D["x"])), device=STORE)
        with torch.no_grad(): _, st = terms(S, D, j, stats=True)
        for k, v in st.items():
            if k in ("mnew", "mgar"): continue
            A[k] = A.get(k, 0) + (v * st["n"] if k in ("Lu", "Lg") else v)
        mn.append(st["mnew"]); mg.append(st["mgar"])
    mn = np.concatenate(mn); mg = np.concatenate(mg); n = A["n"]
    return dict(Lu=A["Lu"] / n, Lg=A["Lg"] / n, agree=A["agree"] / n, dis_big=A["dis_big"] / n, newsel=A["newsel"], qagree=A["qagree"] / max(A["nq"], 1),
                entS=A["entS"] / A["nstate"], entQ=A["entQ"] / A["nstate"], new_mean=float(mn.mean()), new_p99=float(np.quantile(mn, .99)),
                new_p999=float(np.quantile(mn, .999)), gar_mean=float(mg.mean()), n=n)


def fmt(e):
    return ("KL(ユニット)%.4f hinge %.4f | 貪欲一致 ユニット %.4f(教師の差>1 の取り違え %.2e、旧マスク外を選ぶ %d)数量 %.4f | ent 生徒 %.3f 目標 %.3f | "
            "増えた手の質量 平均 %.2e p99 %.2e p99.9 %.2e ゴミ 平均 %.2e" % (e["Lu"], e["Lg"], e["agree"], e["dis_big"], e["newsel"], e["qagree"],
                                                                     e["entS"], e["entQ"], e["new_mean"], e["new_p99"], e["new_p999"], e["gar_mean"]))


def main():
    tag = E["TAG"]; LOGDIR = E.get("LOGDIR", "/content/local_logs")
    logf = open("%s/tf_%s.log" % (LOGDIR, tag), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    torch.manual_seed(int(E.get("SEED", 0))); rs = np.random.RandomState(int(E.get("SEED", 0)))
    TP = E["TEACHER"]
    T = M.load(TP, DEV).eval()
    for p in T.parameters(): p.requires_grad_(False)
    S = copy.deepcopy(T)
    params = [p for m in (S.uq_in, S.u_score, S.cand, S.u_qty) for p in m.parameters()]
    for p in params: p.requires_grad_(True)
    sp = json.load(open("%s/bc_%s.json" % (E.get("SPLITDIR", "/content/local_logs"), E["SPLIT"])))
    cache = E.get("CACHE") or sp["cache"]
    NTR = int(E.get("NTR", "0")); NHO = int(E.get("NHO", "8"))
    tr = sp["train"] if NTR <= 0 else [sp["train"][i] for i in sorted(rs.choice(len(sp["train"]), NTR, replace=False))]
    ho = sp["hold"][:NHO]
    log("頭だけの蒸留 %s: 教師 %s | 学習 %d席 未見 %d席 | τ_u %.2f ε %.3f GREEDYW %.2f 余裕 %.2f WQ %.2f | 学ぶのは uq_in/u_score/cand/u_qty(%d 変数)DEV %s" % (
        tag, TP, len(tr), len(ho), TAU_U, EPS, GREEDYW, GMARGIN, WQ, sum(p.numel() for p in params), DEV))
    Ho = build(T, ho, cache); Tr = build(T, tr, cache)
    ntr = len(Tr["x"]); log("  局面 学習 %d 未見 %d" % (ntr, len(Ho["x"])))
    B = int(E.get("BATCH", "256")); STEPS = int(E.get("STEPS", "4000")); LR = float(E.get("LR", "1e-4")); WARM = int(E.get("WARM", "100"))
    EVE = int(E.get("EVAL_EVERY", "500"))
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(WARM, 1)) * (0.5 * (1 + np.cos(np.pi * min(1.0, s / max(STEPS, 1))))))
    e0 = evaluate(S, Ho); log("  起点(= 教師の重みを新マスクで) " + fmt(e0))
    R = dict(env={k: E.get(k) for k in ("TEACHER", "SPLIT", "NTR", "TAU_U", "EPS", "GREEDYW", "GMARGIN", "WQ", "LR", "WARM", "STEPS", "BATCH")}, init=e0, evals=[])
    cfg = json.load(open(TP + ".cfg.json")) if os.path.exists(TP + ".cfg.json") else {}
    best = None; t0 = time.time(); tl = []
    for step in range(1, STEPS + 1):
        j = torch.as_tensor(rs.randint(0, ntr, B), device=STORE)
        L, _ = terms(S, Tr, j)
        opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); tl.append(float(L))
        if step % max(1, EVE // 5) == 0:
            log("  歩 %5d 学習 %.4f lr %.2e %.1f分" % (step, np.mean(tl[-max(1, EVE // 5):]), sch.get_last_lr()[0], (time.time() - t0) / 60))
        if step % EVE == 0 or step == STEPS:
            e = evaluate(S, Ho); e["step"] = step; R["evals"].append(e)
            log("  歩 %5d 未見 %s" % (step, fmt(e)))
            sd = {k: v.detach().clone() for k, v in S.state_dict().items()}
            torch.save(sd, "%s/tf_%s_s%d.pt" % (LOGDIR, tag, step)); json.dump(cfg, open("%s/tf_%s_s%d.pt.cfg.json" % (LOGDIR, tag, step), "w"))
            score = e["Lu"] + e["Lg"]
            if best is None or score < best[0]:
                best = (score, step); torch.save(sd, "%s/tf_%s.pt" % (LOGDIR, tag)); json.dump(cfg, open("%s/tf_%s.pt.cfg.json" % (LOGDIR, tag), "w"))
            json.dump(R, open("%s/tf_%s.json" % (LOGDIR, tag), "w"), ensure_ascii=False, default=float)
    log("未見の KL+hinge が最小: 歩 %d → %s/tf_%s.pt" % (best[1], LOGDIR, tag))


if __name__ == "__main__":
    main()
