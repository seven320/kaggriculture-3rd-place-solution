"""exp301: M & M & P & Q(MMPQ)だけを教師にした BC(ID 条件なし、普通の model_tf.Policy)。
狙い: 4区画目の購入(市場の候補 20 = BUY_LAND)を写す。BUY_LAND は1局に2〜3手しかないので、普通の交差エントロピーでは
ほとんど重みが無い → 市場の列で正解が BUY_LAND の枠に LANDW 倍の重みを掛ける。納屋が満杯で止まる崩れ方の対策に、
SELLT 手目以降の売り(候補 0..8)に SELLW 倍の重みも掛けられる。
強い既存の BC(INIT、例 tf_idtop3_id0.pt = ID 0 焼き込み済みの普通の重み)から MMPQ の局だけで微調整する。

分割: /content/local_logs/bc_<SPLIT>.json = {train: [npz名], hold: [npz名]}(npz名は CACHE/<名>.npz)。
未見の局で毎エポック: 損失(重みなし)・ユニット一致・市場列一致・BUY_LAND の再現率(教師強制で枠ごと argmax == 20)、
4区画目(区画 3 → 4)の BUY_LAND の再現率、誤って BUY_LAND を出した枠の数(1局あたり)を記録する。
毎エポックの重みを tf_<TAG>_ep<k>.pt に残し(再現率で選べるように)、未見の損失が最小のものを tf_<TAG>.pt に置く。
混ぜ込み(任意): MIXSPLIT=<分割名>(bc_<名>.json の train)から MIXN 局を種固定で選び、MIXCACHE の npz を学習に足す。
その局の手の重みは MIXW(ユニット・市場とも)、MIXMKT=0 なら混ぜた局の市場の損失は数えない(市場は MMPQ だけから学ぶ)。
MIXIDS=0,3 のように分割の ids で教師を絞れる。
土地だけ MMPQ から(任意): MMUNIT=0 で MMPQ の局のユニットの損失を消し、MMLANDONLY=1 で MMPQ の局の市場の損失を BUY_LAND の枠だけにする
(= 盤上と売買は混ぜた教師(Majkel)から、4区画目を買う判断だけ MMPQ から学ぶ)。未見の評価は MMPQ の局だけ。
使い方(容器内): TAG=mmpq_a SPLIT=mmpq CACHE=/content/local_mmpq INIT=/content/local_logs/tf_idtop3_id0.pt LANDW=20 python3 bc_teama.py

---- bc_teama_nm.py(09-25、pfn-bc-rebuild.md 8節): bc_teama.py の写し + 新マスク(mask2)の BC。**既定(NM 未設定)は bc_teama.py と同じ式・同じ出力**。
NM=rel … 学習の lm を PPO の対局係(sp_fast._lm、MASK_MIG=1 MIG_V=2 MIG_BASE=0)と同じ印(0 非合法 / 1 旧マスクでも合法 / 3 新マスクでだけ増えた手)で作り、
         unit_logits の後に nmb_lib.rel_apply(増えた手 = min(logit, 旧マスクの最大) − NMB_PEN)を掛けた分布で交差エントロピー。
         = apo_nmbrel.py(学習・採取)と mkagent_nmb.py の rel(提出・リーグ)と同じ式。旧マスクは npz の legal、新マスクは nm_lib.mask2_t(ou, g)。
NM=m2  … 対照(案A): 新マスクの softmax そのまま(下駄なし)。
NM_OUT2=drop(既定)… 新マスクでも非合法な教師の手(種0の PLANT・盤外への移動。公式では no-op)を損失から外す。pass で待機に置き換える。
NMB_PEN(既定 6)。未見の評価で、旧マスク内/増えた手の正解率・増えた手の質量(平均・p99・p99.9・最大)・貪欲が増えた手を選ぶ率を足して記録する。
CPU の確認用: DEV=cpu LOGDIR=<dir> NT=<スレッド> NTR=<学習の席数> NHO=<未見の席数> MAXSTEPS=<1エポックの歩数の上限> EVAL_EVERY=<歩>。EPOCHS=0 は測るだけ。"""
import json, os, sys, time, copy, contextlib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf as M
from bc_tf import load_dev, KEYS
from extract4 import NM, NS, M_STOP
DEV = os.environ.get("DEV", "cuda")
LOGDIR = os.environ.get("LOGDIR", "/content/local_logs")
import bc_tf as _bt; _bt.DEV = DEV
if DEV == "cpu": torch.set_num_threads(int(os.environ.get("NT", "3")))
NMODE = os.environ.get("NM", "")            # "" = bc_teama と同じ / rel / m2
NMB_PEN = float(os.environ.get("NMB_PEN", 6.0)); NM_OUT2 = os.environ.get("NM_OUT2", "drop")
assert NMODE in ("", "rel", "m2") and NM_OUT2 in ("drop", "pass")
if NMODE:
    import nm_lib, nmb_lib
LANDW = float(os.environ.get("LANDW", 1.0)); SELLW = float(os.environ.get("SELLW", 1.0)); SELLT = int(os.environ.get("SELLT", 480))
AMP = os.environ.get("AMP") == "bf16"
if AMP: torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
actx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if AMP else contextlib.nullcontext
LAND = 20
LS = float(os.environ.get("LS", 0.0))   # 09-21(PTCG 9・10位): ラベル平滑化。0 で従来どおり


def ce_ls(logits, target, ls):
    """合法な手の中だけで平滑化した交差エントロピー(要素ごと)。
    torch の label_smoothing は非合法(logit -1e4)にも質量を配るので、そのまま使うと損失が壊れる
    (09-21 実測: ユニット一致 0.92 → 0.08)。合法かどうかは logit が -1e3 より上かで判定する。"""
    if ls <= 0: return F.cross_entropy(logits, target, reduction="none")
    lp = F.log_softmax(logits, -1)
    ok = logits > -1e3                                  # 合法な枠
    n = ok.sum(-1, keepdim=True).clamp(min=1).float()
    tgt = F.one_hot(target, logits.shape[-1]).float() * (1 - ls) + ok.float() * (ls / n)
    return -(tgt * lp.masked_fill(~ok, 0.0)).sum(-1)
MMUNIT = float(os.environ.get("MMUNIT", 1.0)); MMLANDONLY = os.environ.get("MMLANDONLY") == "1"


def load_dev_pairs(pairs):
    """bc_tf.load_dev と同じ(1局ずつ GPU の確保済み領域に詰める)。置き場の違う局を混ぜられるように (dir, 名) の組で受ける。"""
    lens = [np.load("%s/%s.npz" % pe)["g"].shape[0] for pe in pairs]; n = sum(lens)
    z0 = np.load("%s/%s.npz" % pairs[0])
    T = {k: torch.empty((n,) + z0[k].shape[1:], dtype=torch.float16 if k == "tiles" else torch.from_numpy(z0[k][:1]).dtype, device=DEV) for k in KEYS}
    i = 0
    for pe, L in zip(pairs, lens):
        z = np.load("%s/%s.npz" % pe)
        for k in KEYS: T[k][i:i + L] = torch.as_tensor(z[k], device=DEV)
        i += L
    return T, None


def mweights(ms, step):
    """市場の各枠の重み(正解が BUY_LAND なら LANDW、SELLT 手目以降の売りなら SELLW、それ以外 1)。"""
    w = torch.ones_like(ms, dtype=torch.float32)
    w = torch.where(ms == LAND, torch.full_like(w, LANDW), w)
    late = (step >= SELLT).view(-1, 1).expand_as(ms)
    return torch.where(late & (ms < 9), torch.full_like(w, SELLW), w)


def losses(net, T, idx, train=True):
    b = torch.as_tensor(idx, device=DEV)
    tiles, ou, pu, g, legal = T["tiles"][b], T["ou"][b], T["pu"][b], T["g"][b], T["legal"][b]
    ul, uq, ms, mq = T["ul"][b].long(), T["uq"][b].long(), T["ms"][b].long(), T["mq"][b].long()
    h, pad = net.encode(tiles, ou, pu, g)
    if not NMODE:
        lg, q = net.unit_logits(h, ou, legal)
        live = ul >= 0
    else:
        old = legal != 0
        xp = nm_lib.mask2_t(ou, g) & ~old                                    # 新マスクでだけ増えた手
        lm = old.to(torch.uint8) | (xp.to(torch.uint8) * 3)                  # sp_fast._lm と同じ印 0/1/3
        lg, q = net.unit_logits(h, ou, lm)
        if NMODE == "rel": lg = nmb_lib.rel_apply(lg, lm, NMB_PEN)
        live0 = ul >= 0
        inset = lm.gather(-1, ul.clamp(min=0).unsqueeze(-1)).squeeze(-1) != 0
        if NM_OUT2 == "pass":
            bad = live0 & ~inset; ul = torch.where(bad, torch.zeros_like(ul), ul); uq = torch.where(bad, torch.zeros_like(uq), uq); live = live0
        else:
            live = live0 & inset
    ws = T["w"][b] if train else torch.ones(len(idx), device=DEV)       # 局面ごとの重み(MMPQ 1、混ぜた局 MIXW)
    wmk = T["wm"][b] if train else ws                                   # 市場の重み(MIXMKT=0 なら混ぜた局は 0)
    wu = ws.view(-1, 1).expand_as(ul)
    lu = (ce_ls(lg[live].float(), ul[live], LS) * wu[live]).sum() / wu[live].sum().clamp(min=1e-6)
    qm = live & (ul >= 20)
    lq = (ce_ls(net.unit_qty(q, ul)[qm].float(), uq[qm], LS) * wu[qm]).sum() / wu[qm].sum().clamp(min=1e-6) if qm.any() else lg.sum() * 0
    prev = torch.cat([torch.full_like(ms[:, :1], NM), ms[:, :-1]], 1).clamp(max=NM)
    mo, y = net.market(h, pad, prev)
    first_stop = torch.where(ms == M_STOP, torch.arange(NS, device=DEV).expand_as(ms), torch.full_like(ms, NS)).min(1).values
    valid = torch.arange(NS, device=DEV).unsqueeze(0) <= first_stop.unsqueeze(1)
    ce = ce_ls(mo[valid].float(), ms[valid], LS)
    if train:
        wv = wmk.view(-1, 1).expand_as(ms)
        mw = mweights(ms, T["step"][b])
        if MMLANDONLY: mw = torch.where(T["mm"][b].view(-1, 1).expand_as(ms) & (ms != LAND), torch.zeros_like(mw), mw)
        w = (mw * wv)[valid]; lm = (ce * w).sum() / wv[valid].sum().clamp(min=1e-6)   # 平均は枠の数(×局面の重み)で割る(LANDW では割らない = 稀な手を本当に重くする)
    else:
        lm = ce.mean()
    mqm = valid & (ms < 19)
    wq = wmk.view(-1, 1).expand_as(ms)
    lmq = (ce_ls(net.market_qty(y, ms)[mqm].float(), mq[mqm], LS) * wq[mqm]).sum() / wq[mqm].sum().clamp(min=1e-6) if mqm.any() else lm * 0
    st = dict()
    with torch.no_grad():
        pred = mo.argmax(-1)
        st["ua"] = (lg.argmax(-1) == ul)[live].float().sum(); st["un"] = live.sum()
        st["mseq"] = ((pred == ms) | ~valid).all(1).float().sum(); st["n"] = torch.tensor(len(idx), device=DEV)
        isl = valid & (ms == LAND); st["land_n"] = isl.sum(); st["land_hit"] = (isl & (pred == LAND)).sum()
        st["land_p"] = torch.softmax(mo.float(), -1)[..., LAND][isl].sum()
        q4 = (torch.round(g[:, 39] * 4) == 3).view(-1, 1).expand_as(ms)     # 区画 3 のときの購入 = 4区画目
        st["l4_n"] = (isl & q4).sum(); st["l4_hit"] = (isl & q4 & (pred == LAND)).sum()
        st["land_fp"] = (valid & (ms != LAND) & (pred == LAND)).sum()
        if NMODE and not train:      # 評価のときだけ(学習の歩ごとに .cpu() で同期させない)
            ua = lg.argmax(-1); lab = ul.clamp(min=0)
            in0 = live0 & old.gather(-1, lab.unsqueeze(-1)).squeeze(-1); inx = live0 & xp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            st["a0"] = ((ua == ul) & in0).sum(); st["n0"] = in0.sum(); st["ax"] = ((ua == ul) & inx).sum(); st["nx"] = inx.sum()
            st["nout"] = (live0 & ~inset).sum(); st["nall"] = live0.sum()
            ms_ = (torch.softmax(lg.float(), -1) * xp.float()).sum(-1)[live0]
            st["_mass"] = ms_.cpu(); st["gnew"] = (xp.gather(-1, ua.unsqueeze(-1)).squeeze(-1) & live0).sum()
    return lu + 0.5 * lq + lm + 0.5 * lmq, st


def main():
    tag = os.environ["TAG"]; cache = os.environ["CACHE"]; EP = int(os.environ.get("EPOCHS", 8))
    logf = open(LOGDIR + "/tf_%s.log" % tag, "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    S = json.load(open(LOGDIR + "/bc_%s.json" % os.environ["SPLIT"])); tr_e, ho_e = S["train"], S["hold"]
    if os.environ.get("NTR"): tr_e = tr_e[:int(os.environ["NTR"])]
    if os.environ.get("NHO"): ho_e = ho_e[:int(os.environ["NHO"])]
    def prep(es):
        T, G = load_dev(es, cache)
        lens = [np.load("%s/%s.npz" % (cache, e))["g"].shape[0] for e in es]
        T["step"] = torch.as_tensor(np.concatenate([np.arange(L) for L in lens]), device=DEV)
        return T
    MIX = os.environ.get("MIXSPLIT"); mix_e = []
    if MIX:
        import random
        SM = json.load(open(LOGDIR + "/bc_%s.json" % MIX)); ids = [int(x) for x in os.environ.get("MIXIDS", "").split(",") if x]
        pool = sorted(e for e in SM["train"] if (not ids or SM["ids"][e] in ids) and "mmpq" not in e)
        random.Random(301).shuffle(pool); mix_e = pool[:int(os.environ.get("MIXN", 400))]
    allc = [(cache, e) for e in tr_e] + [(os.environ.get("MIXCACHE", "/content/local_ext4all"), e) for e in mix_e]
    def prep2(pairs):
        T, _ = load_dev_pairs(pairs)
        lens = [np.load("%s/%s.npz" % pe)["g"].shape[0] for pe in pairs]
        T["step"] = torch.as_tensor(np.concatenate([np.arange(L) for L in lens]), device=DEV)
        MIXW = float(os.environ.get("MIXW", 0.5)); mm = os.environ.get("MIXMKT", "1") == "1"
        T["mm"] = torch.as_tensor(np.concatenate([np.full(L, c == cache, bool) for (c, e), L in zip(pairs, lens)]), device=DEV)
        T["w"] = torch.as_tensor(np.concatenate([np.full(L, MMUNIT if c == cache else MIXW, np.float32) for (c, e), L in zip(pairs, lens)]), device=DEV)
        T["wm"] = torch.as_tensor(np.concatenate([np.full(L, 1.0 if c == cache else (MIXW if mm else 0.0), np.float32) for (c, e), L in zip(pairs, lens)]), device=DEV)
        return T
    T = prep2(allc); Th = prep(ho_e)
    if mix_e: print("混ぜた局 %d(%s、重み %s、市場 %s)" % (len(mix_e), MIX, os.environ.get("MIXW", 0.5), os.environ.get("MIXMKT", "1")), flush=True)
    log("学習 %d 局 + 混ぜた %d 局(%d手) / 未見 %d 局(%d手) LANDW %.1f SELLW %.1f(%d手目以降)" % (len(tr_e), len(mix_e), len(T["g"]), len(ho_e), len(Th["g"]), LANDW, SELLW, SELLT))
    INIT = os.environ.get("INIT"); CFG = json.load(open(INIT + ".cfg.json")) if INIT else json.loads(os.environ.get("CFG", "{}"))
    torch.manual_seed(int(os.environ.get("SEED", 0))); np.random.seed(int(os.environ.get("SEED", 0)))
    net = M.Policy(**CFG).to(DEV)
    if INIT: net.load_state_dict(torch.load(INIT, map_location=DEV)); log("続きから: %s" % INIT)
    json.dump(CFG, open(LOGDIR + "/tf_%s.pt.cfg.json" % tag, "w"))
    LRMAX = float(os.environ.get("LRMAX", 1e-4)); B = int(os.environ.get("BATCH", 64))
    S_ = len(T["g"]); Sh = len(Th["g"]); nb = S_ // B
    if os.environ.get("MAXSTEPS"): nb = min(nb, int(os.environ["MAXSTEPS"]))
    if NMODE: log("新マスク NM=%s NMB_PEN %g NM_OUT2 %s(lm = 旧マスク 1 / 増えた手 3、新マスクは nm_lib.mask2_t)" % (NMODE, NMB_PEN, NM_OUT2))
    if os.environ.get("OPT") == "muon":      # 09-21(PTCG 3・10位): 2次元の重みは直交化した運動量(Muon)、それ以外は AdamW
        from bc_tf_id import Muon
        mp_ = [q for q in net.parameters() if q.ndim == 2]; ap_ = [q for q in net.parameters() if q.ndim != 2]
        opt = Muon(mp_, ap_, lr=LRMAX, adam_lr=LRMAX)
        log("最適化: Muon(2次元 %d本 / AdamW %d本)" % (len(mp_), len(ap_)))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=LRMAX, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LRMAX, total_steps=max(EP * nb, 1), pct_start=0.05, cycle_momentum=os.environ.get("OPT") != "muon")   # Muon は運動量を自分で持つ

    def evaluate():
        net.eval(); acc = {}; tl = 0.0; mass = []
        with torch.no_grad():
            for k in range(0, Sh, 256):
                ix = np.arange(k, min(k + 256, Sh))
                with actx(): l, st = losses(net, Th, ix, train=False)
                tl += float(l) * len(ix)
                if "_mass" in st: mass.append(st.pop("_mass"))
                for kk, v in st.items(): acc[kk] = acc.get(kk, 0.0) + float(v)
        net.train()
        nx = {}
        if NMODE:
            mm = torch.cat(mass).double(); qs = torch.quantile(mm, torch.tensor([0.99, 0.999], dtype=torch.float64)) if len(mm) < 16000000 else torch.zeros(2)
            nx = dict(acc0=acc["a0"] / max(acc["n0"], 1), n0=acc["n0"], accx=acc["ax"] / max(acc["nx"], 1), nx=acc["nx"], nout=acc["nout"],
                      unit_all=acc["ua"] / max(acc["nall"], 1), xmass=float(mm.mean()), xmass99=float(qs[0]), xmass999=float(qs[1]), xmassmax=float(mm.max()),
                      gnew=acc["gnew"] / max(acc["nall"], 1))
        return dict(**nx,hloss=tl / Sh, hunit=acc["ua"] / acc["un"], hmseq=acc["mseq"] / acc["n"], land_rec=acc["land_hit"] / max(acc["land_n"], 1),
                    land_p=acc["land_p"] / max(acc["land_n"], 1), l4_rec=acc["l4_hit"] / max(acc["l4_n"], 1), l4_n=acc["l4_n"],
                    land_fp_per_game=acc["land_fp"] / len(ho_e), land_n=acc["land_n"])
    def fmt(e): return _fmt0(e) + ("" if not NMODE else "\n        新マスク: 旧マスク内の正解 %.4f(%d) 増えた手の正解 %.3f(%d) 範囲外 %d | 全教師手での一致 %.4f | 増えた手の質量 平均 %.2e p99 %.2e p99.9 %.2e 最大 %.3f | 貪欲が増えた手 %.2e" % (
        e["acc0"], e["n0"], e["accx"], e["nx"], e["nout"], e["unit_all"], e["xmass"], e["xmass99"], e["xmass999"], e["xmassmax"], e["gnew"]))
    def _fmt0(e): return "未見 損失 %.3f ユニット %.3f 市場列 %.3f | 土地 再現率 %.3f(p %.3f, %d枠) 4区画目 %.3f(%d枠) 誤発 %.2f/局" % (
        e["hloss"], e["hunit"], e["hmseq"], e["land_rec"], e["land_p"], e["land_n"], e["l4_rec"], e["l4_n"], e["land_fp_per_game"])
    e0 = evaluate(); log("  初期 " + fmt(e0))
    if EP == 0:
        json.dump(dict(hold=ho_e, init={k: float(v) for k, v in e0.items()}), open(LOGDIR + "/tf_%s.json" % tag, "w"), ensure_ascii=False); log("測るだけ(EPOCHS=0)"); return
    EVE = int(os.environ.get("EVAL_EVERY", 0))
    best = None; t0 = time.time(); R = dict(train=tr_e, hold=ho_e, init=e0, epochs=[], env={k: os.environ.get(k) for k in ("LANDW", "SELLW", "SELLT", "LRMAX", "INIT", "EPOCHS", "SPLIT", "MIXSPLIT", "MIXN", "MIXW", "MIXMKT", "MIXIDS", "MMUNIT", "MMLANDONLY", "NM", "NMB_PEN", "NM_OUT2")})
    for ep in range(EP):
        perm = np.random.permutation(S_); ls = []
        for k in range(nb):
            with actx(): l, _ = losses(net, T, perm[k * B:(k + 1) * B], True)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            ls.append(l.detach().float())
            if EVE and (k + 1) % EVE == 0 and k + 1 < nb: log("  ep%2d 歩 %d 学習 %.3f | %s" % (ep, k + 1, float(torch.stack(ls[-EVE:]).mean()), fmt(evaluate())))
        e = evaluate(); e["ep"] = ep; e["loss"] = float(torch.stack(ls).mean())
        R["epochs"].append({k: float(v) for k, v in e.items()})
        log("  ep%2d 学習 %.3f | %s (%.1f分)" % (ep, e["loss"], fmt(e), (time.time() - t0) / 60))
        sd = copy.deepcopy(net.state_dict()); torch.save(sd, LOGDIR + "/tf_%s_ep%d.pt" % (tag, ep))
        json.dump(CFG, open(LOGDIR + "/tf_%s_ep%d.pt.cfg.json" % (tag, ep), "w"))
        if best is None or e["hloss"] < best[0]:
            best = (e["hloss"], ep); torch.save(sd, LOGDIR + "/tf_%s.pt" % tag)
    log("未見の損失が最小: ep%d (%.3f)" % (best[1], best[0]))
    json.dump(R, open(LOGDIR + "/tf_%s.json" % tag, "w"), ensure_ascii=False, default=float)
    log("保存")


if __name__ == "__main__":
    main()
