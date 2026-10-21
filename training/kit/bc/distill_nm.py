"""旧マスク(mask0)の最良版から、新マスク(mask2 = LEGAL_OPT=2)で打つ方策を蒸留する(newmask-distill-plan.md 3節)。
**新しいファイル。走っているジョブはこれを読まない。** model_tf / ppo_tf / apo / sp_fast は読むだけで変えない。

なぜ蒸留か(mask-migration-attempt.md の失敗から):
  旧マスクの方策は mask2 でだけ増える手(96% は no-op のゴミ)の logit を一度も学んでいない。罰(MIG_PEN)で押さえても
  尾(0.1% の局面)で確率 0.997 まで上がり、標本打ちで経済が止まった。ここでは**重みそのものを**「新マスクの softmax の中で
  ゴミの確率質量が 0 になる」ように学ばせる。罰も混合も使わないので、採取と学習の式のずれ(FASTINF の差し替え)が起きようがない。

目標分布(ユニットごと):
    q = (1 − ε·[K≠∅]) · softmax(教師の logit / τ_u  … 教師自身の mask0 の中だけ)  +  ε·[K≠∅] · 一様(K)
    K = (mask2 − mask0) ∩ nm_lib.keep_t(手番内の連携で本当に成立しうる型: 納屋の隣の取り出し・置く、先に動くユニットと同じマスへの手)
  **教師の logit を mask2 で正規化し直さない**(教師の増えた手の logit は未学習の値。そのまま使うと 91% がゴミに行く)。
  生徒は mask2 の softmax。損失は KL(q ‖ 生徒)。数量・市場は合法の定義が変わらないので教師強制で KL(温度 τ_q, τ_m)。
  τ_u > 1 で生徒を教師より広くする(本線の ent 1.20 → 1.04 の低下を戻す)。

生徒の起点は**教師の重みそのもの**(同じ構造)。mask0 の中の分布は最初から教師と一致し、学ぶのは「増えた手を下げる」ことと τ の分だけ。

データ: gen_nm.py の npz(legal = mask0)。mask2 は特徴(ou, g)から作る(nm_lib.mask2_t。C++ の opt=2 と一致を nm_check.py で確認)。

環境変数:
  TEACHER(必須) INIT(既定 = TEACHER) TAG SPLIT(/content/local_logs/bc_<SPLIT>.json、cache を含む) CACHE(split に無ければ)
  TAU_U(1.0) TAU_Q(1.0) TAU_M(1.0) EPS(0.01) HARDW(0 = 教師の実際の手への交差エントロピーの重み)
  GREEDYW(0 = 教師の貪欲の手を守る hinge の重み) GMARGIN(0.5 = その余裕、log 確率で)  … 8節
  WQ(0.5) WM(1.0) WMQ(0.5) LR(3e-5) WARM(200) EPOCHS(2) MAXSTEPS(0 = 無制限) BATCH(256) MICRO(1) EVALN(20000) EVAL_EVERY(500)
  SEED(0) DEV(cuda) LOGDIR(/content/local_logs) AMP(bf16 / 空)
出力: LOGDIR/tf_<TAG>.pt(未見の KL が最小)、tf_<TAG>_s<step>.pt、.cfg.json、tf_<TAG>.log、tf_<TAG>.json
"""
import os, sys, json, time, contextlib
if os.environ.get("MASK_MIG") == "1":
    raise SystemExit("distill_nm: MASK_MIG=1 を外すこと(罰は使わない。model_tf の mig_apply は採取側の関数)")
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tf as M
import nm_lib
NU, NC, NM, NS = M.NU, M.NC, M.NM, M.NS
M_STOP = 22
KEYS = ("tiles", "ou", "pu", "g", "legal", "ul", "uq", "ms", "mq")
E = os.environ
DEV = E.get("DEV", "cuda")
TAU_U = float(E.get("TAU_U", "1.0")); TAU_Q = float(E.get("TAU_Q", "1.0")); TAU_M = float(E.get("TAU_M", "1.0"))
EPS = float(E.get("EPS", "0.01")); HARDW = float(E.get("HARDW", "0"))
# 09-24(8節): 貪欲の手を守る項。教師の mask0 の中の argmax a* について、生徒の log π(a*) が他のどの手より GMARGIN 以上
#   高くなるよう hinge で押す(一致しているユニットでは 0 = 分布は KL のまま。食い違う/際どいユニットだけに効く)。既定 0 = 従来と同じ。
GREEDYW = float(E.get("GREEDYW", "0")); GMARGIN = float(E.get("GMARGIN", "0.5"))
WQ = float(E.get("WQ", "0.5")); WM = float(E.get("WM", "1.0")); WMQ = float(E.get("WMQ", "0.5"))
AMP = E.get("AMP", "bf16" if DEV != "cpu" else "")
actx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if AMP == "bf16" else contextlib.nullcontext


def load_split():
    sp = json.load(open("%s/bc_%s.json" % (E.get("SPLITDIR", "/content/local_logs"), E["SPLIT"])))
    cache = E.get("CACHE") or sp.get("cache")
    return sp["train"], sp["hold"], cache


def load_data(names, cache):
    """1席ずつ開いて閉じる(1,400席を同時に開くとファイル記述子の上限 1024 を超える)。"""
    lens = []
    for n in names:
        with np.load("%s/%s.npz" % (cache, n)) as z: lens.append(z["g"].shape[0])
    with np.load("%s/%s.npz" % (cache, names[0])) as z0:
        T = {k: torch.empty((sum(lens),) + z0[k].shape[1:], dtype=torch.float16 if k == "tiles" else torch.from_numpy(z0[k][:1]).dtype, device=DEV) for k in KEYS}
    i = 0
    for n, L in zip(names, lens):
        with np.load("%s/%s.npz" % (cache, n)) as z:
            for k in KEYS: T[k][i:i + L] = torch.as_tensor(z[k], device=DEV)
        i += L
    return T


def kl_rows(p, logq, mask=None):
    """sum_c p (log p − log q)(p=0 の枠は 0)。p は確率、logq は対数確率。"""
    t = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-30)) - logq), torch.zeros_like(p))
    return t.sum(-1)


def batch_terms(S, T, b, stats=False):
    """1バッチの損失(と統計)。b は KEYS の辞書(DEV 上)。"""
    tiles, ou, pu, g = b["tiles"], b["ou"].float(), b["pu"].float(), b["g"].float()
    lm0 = b["legal"] != 0
    ul, uq, ms, mq = b["ul"].long(), b["uq"].long(), b["ms"].long(), b["mq"].long()
    lm2 = nm_lib.mask2_t(ou, g)
    live = ou[..., 0] > 0
    viol = (lm0 & ~lm2).sum()                                          # mask0 ⊆ mask2 のはず(0 でなければデータか定義が壊れている)
    new = lm2 & ~lm0
    K = new & nm_lib.keep_t(tiles, ou)
    prev = torch.cat([torch.full_like(ms[:, :1], NM), ms[:, :-1]], 1).clamp(max=NM)
    first = torch.where(ms == M_STOP, torch.arange(NS, device=ms.device).expand_as(ms), torch.full_like(ms, NS)).min(1).values
    valid = torch.arange(NS, device=ms.device).unsqueeze(0) <= first.unsqueeze(1)
    vm = valid.clone(); vm[:, NS - 1] = False                           # 11枠目は終了のみ(両者とも同じ)なので KL から外す
    with torch.no_grad(), actx():
        hT, padT = T.encode(tiles, ou, pu, g)
        lgT, qT = T.unit_logits(hT, ou, lm0)
        moT, yT = T.market(hT, padT, prev)
        lqT = T.unit_qty(qT, ul); lmqT = T.market_qty(yT, ms)
    lgT = lgT.float()
    pT = torch.softmax(lgT / TAU_U, -1) * lm0                           # 教師自身の mask0 の中だけ
    pT = pT / pT.sum(-1, keepdim=True).clamp_min(1e-30)
    nk = K.sum(-1, keepdim=True)
    e = EPS * (nk > 0).float()
    q = (1 - e) * pT + e * K.float() / nk.clamp(min=1).float()
    with actx():
        hS, padS = S.encode(tiles, ou, pu, g)
        lgS, qS = S.unit_logits(hS, ou, lm2)
        moS, yS = S.market(hS, padS, prev)
        lqS = S.unit_qty(qS, ul); lmqS = S.market_qty(yS, ms)
    lsS = torch.log_softmax(lgS.float(), -1)
    nl = live.sum().clamp(min=1)
    Lu = (kl_rows(q, lsS) * live).sum() / nl
    qm = live & (ul >= 20)
    Lq = (kl_rows(torch.softmax(lqT.float() / TAU_Q, -1), torch.log_softmax(lqS.float(), -1)) * qm).sum() / qm.sum().clamp(min=1)
    Lm = (kl_rows(torch.softmax(moT.float() / TAU_M, -1), torch.log_softmax(moS.float(), -1)) * vm).sum() / vm.sum().clamp(min=1)
    mqm = vm & (ms < 19)
    Lmq = (kl_rows(torch.softmax(lmqT.float() / TAU_Q, -1), torch.log_softmax(lmqS.float(), -1)) * mqm).sum() / mqm.sum().clamp(min=1)
    L = Lu + WQ * Lq + WM * Lm + WMQ * Lmq
    if HARDW > 0:
        hu = live & (ul >= 0)
        L = L + HARDW * (-(lsS.gather(-1, ul.clamp(min=0).unsqueeze(-1)).squeeze(-1)) * hu).sum() / hu.sum().clamp(min=1)
    if GREEDYW > 0:
        aT = lgT.argmax(-1, keepdim=True)                                   # lgT は ~lm0 が -1e4 なので mask0 の中の argmax
        la = lsS.gather(-1, aT).squeeze(-1)
        oth = lsS.scatter(-1, aT, -1e9).max(-1).values
        L = L + GREEDYW * (F.relu(oth - la + GMARGIN) * live).sum() / nl
    if not stats: return L, None
    with torch.no_grad():
        pS = lsS.exp()
        mass_new = (pS * new).sum(-1)[live]; mass_gar = (pS * (new & ~K)).sum(-1)[live]
        lsT1 = torch.log_softmax(lgT, -1)                                   # 教師 τ=1(旧マスク)= 本線の ent と同じ定義
        entT = (-(lsT1.exp() * lsT1.masked_fill(~lm0, 0)).sum(-1) * live).sum(1)
        entQ = (-(q * torch.log(q.clamp_min(1e-30))).sum(-1) * live).sum(1)
        entS = (-(pS * lsS.masked_fill(~lm2, 0)).sum(-1) * live).sum(1)
        agree = ((lgS.argmax(-1) == lgT.argmax(-1)) & live).sum()
        magree = ((moS.argmax(-1) == moT.argmax(-1)) & vm).sum()
        st = dict(L=float(L), Lu=float(Lu), Lq=float(Lq), Lm=float(Lm), Lmq=float(Lmq), viol=int(viol),
                  n_live=int(live.sum()), agree=int(agree), n_vm=int(vm.sum()), magree=int(magree), n=len(ou),
                  entT=float(entT.sum()), entQ=float(entQ.sum()), entS=float(entS.sum()),
                  mass_new=mass_new.float().cpu().numpy(), mass_gar=mass_gar.float().cpu().numpy(),
                  keep_units=int(((nk.squeeze(-1) > 0) & live).sum()))
    return L, st


def evaluate(S, T, H, idx, bs):
    S.eval(); acc = {}; mn = []; mg = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], device=DEV)
        with torch.no_grad(): _, st = batch_terms(S, T, {k: H[k][j] for k in KEYS}, stats=True)
        w = st["n"]
        for k in ("L", "Lu", "Lq", "Lm", "Lmq"): acc[k] = acc.get(k, 0.0) + st[k] * w
        for k in ("viol", "n_live", "agree", "n_vm", "magree", "n", "entT", "entQ", "entS", "keep_units"): acc[k] = acc.get(k, 0) + st[k]
        mn.append(st["mass_new"]); mg.append(st["mass_gar"])
    n = acc["n"]; mn = np.concatenate(mn); mg = np.concatenate(mg)
    return dict(L=acc["L"] / n, Lu=acc["Lu"] / n, Lq=acc["Lq"] / n, Lm=acc["Lm"] / n, Lmq=acc["Lmq"] / n,
                agree=acc["agree"] / max(acc["n_live"], 1), magree=acc["magree"] / max(acc["n_vm"], 1),
                entT=acc["entT"] / n, entQ=acc["entQ"] / n, entS=acc["entS"] / n, viol=acc["viol"],
                new_mean=float(mn.mean()), new_p99=float(np.quantile(mn, 0.99)), new_p999=float(np.quantile(mn, 0.999)), new_max=float(mn.max()),
                gar_mean=float(mg.mean()), gar_max=float(mg.max()), keep_frac=acc["keep_units"] / max(acc["n_live"], 1), n=n)


def fmt(e):
    return ("KL %.4f(ユニット %.4f 数量 %.4f 市場 %.4f 市場数量 %.4f)| 貪欲一致 ユニット %.4f 市場 %.4f | ent 生徒 %.3f 目標 %.3f 教師 %.3f | "
            "増えた手の質量 平均 %.2e p99 %.2e p99.9 %.2e 最大 %.3f(ゴミ 平均 %.2e 最大 %.3f)| K のある判断 %.3f | mask0⊄mask2 %d" % (
                e["L"], e["Lu"], e["Lq"], e["Lm"], e["Lmq"], e["agree"], e["magree"], e["entS"], e["entQ"], e["entT"],
                e["new_mean"], e["new_p99"], e["new_p999"], e["new_max"], e["gar_mean"], e["gar_max"], e["keep_frac"], e["viol"]))


def main():
    tag = E["TAG"]; LOGDIR = E.get("LOGDIR", "/content/local_logs")
    logf = open("%s/tf_%s.log" % (LOGDIR, tag), "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    torch.manual_seed(int(E.get("SEED", 0))); np.random.seed(int(E.get("SEED", 0)))
    TP = E["TEACHER"]; IP = E.get("INIT", TP)
    cfg = json.load(open(TP + ".cfg.json")) if os.path.exists(TP + ".cfg.json") else {}
    T = M.load(TP, DEV).eval()
    for p in T.parameters(): p.requires_grad_(False)
    S = M.load(IP, DEV)
    tr, ho, cache = load_split()
    Tr = load_data(tr, cache); Ho = load_data(ho, cache)
    ntr, nho = len(Tr["g"]), len(Ho["g"])
    EVALN = int(E.get("EVALN", "20000")); rs = np.random.RandomState(1)
    hidx = np.sort(rs.choice(nho, min(EVALN, nho), replace=False))
    B = int(E.get("BATCH", "256")); MICRO = int(E.get("MICRO", "1")); EP = int(E.get("EPOCHS", "2"))
    MAXSTEPS = int(E.get("MAXSTEPS", "0")); EVE = int(E.get("EVAL_EVERY", "500"))
    steps = EP * (ntr // B)
    if MAXSTEPS: steps = min(steps, MAXSTEPS)
    LR = float(E.get("LR", "3e-5")); WARM = int(E.get("WARM", "200"))
    opt = torch.optim.AdamW(S.parameters(), lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(WARM, 1)) * (0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1))))))
    log("蒸留 %s: 教師 %s 起点 %s | 学習 %d局面(%d席) 未見 %d局面(%d席、測るのは %d) | τ_u %.2f τ_q %.2f τ_m %.2f ε %.3f HARDW %.2f GREEDYW %.2f(余裕 %.2f)| "
        "LR %.1e WARM %d BATCH %d MICRO %d 歩数 %d DEV %s AMP %s" % (tag, TP, IP, ntr, len(tr), nho, len(ho), len(hidx), TAU_U, TAU_Q, TAU_M, EPS, HARDW, GREEDYW, GMARGIN,
                                                             LR, WARM, B, MICRO, steps, DEV, AMP or "fp32"))
    json.dump(cfg, open("%s/tf_%s.pt.cfg.json" % (LOGDIR, tag), "w"))
    ebs = int(E.get("EVAL_BS", str(max(B // MICRO, 32))))
    e0 = evaluate(S, T, Ho, hidx, ebs); log("  起点 " + fmt(e0))
    if steps <= 0:                      # EPOCHS=0: 測るだけ(τ・ε を変えたときの目標の ent を見る)
        json.dump(dict(init=e0), open("%s/tf_%s.json" % (LOGDIR, tag), "w"), ensure_ascii=False, default=float); return
    R = dict(env={k: E.get(k) for k in ("TEACHER", "INIT", "SPLIT", "TAU_U", "TAU_Q", "TAU_M", "EPS", "HARDW", "GREEDYW", "GMARGIN", "WQ", "WM", "WMQ", "LR", "WARM", "EPOCHS", "MAXSTEPS", "BATCH", "MICRO")},
             init=e0, evals=[])
    best = None; t0 = time.time(); perm = np.random.permutation(ntr); pi = 0; tl = []
    S.eval()   # ドロップアウトを切る(PPO も eval で打つ。同じ分布を学ばせる)
    for step in range(1, steps + 1):
        if pi + B > ntr: perm = np.random.permutation(ntr); pi = 0
        idx = perm[pi:pi + B]; pi += B
        opt.zero_grad(set_to_none=True)
        mb = B // MICRO
        for m in range(MICRO):
            j = torch.as_tensor(idx[m * mb:(m + 1) * mb], device=DEV)
            L, _ = batch_terms(S, T, {k: Tr[k][j] for k in KEYS})
            (L / MICRO).backward(); tl.append(float(L.detach()))
        torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0); opt.step(); sch.step()
        if step % max(1, EVE // 10) == 0:
            log("  歩 %5d 学習 KL %.4f(直近の平均)lr %.2e %.1f分" % (step, float(np.mean(tl[-max(1, EVE // 10) * MICRO:])), sch.get_last_lr()[0], (time.time() - t0) / 60))
        if step % EVE == 0 or step == steps:
            e = evaluate(S, T, Ho, hidx, ebs); e["step"] = step; R["evals"].append(e)
            log("  歩 %5d 未見 %s(%.1f分)" % (step, fmt(e), (time.time() - t0) / 60))
            sd = {k: v.detach().clone() for k, v in S.state_dict().items()}
            torch.save(sd, "%s/tf_%s_s%d.pt" % (LOGDIR, tag, step)); json.dump(cfg, open("%s/tf_%s_s%d.pt.cfg.json" % (LOGDIR, tag, step), "w"))
            if best is None or e["L"] < best[0]:
                best = (e["L"], step); torch.save(sd, "%s/tf_%s.pt" % (LOGDIR, tag))
            json.dump(R, open("%s/tf_%s.json" % (LOGDIR, tag), "w"), ensure_ascii=False, default=float)
    log("未見の KL が最小: 歩 %d(%.4f)→ %s/tf_%s.pt" % (best[1], best[0], LOGDIR, tag))


if __name__ == "__main__":
    main()
