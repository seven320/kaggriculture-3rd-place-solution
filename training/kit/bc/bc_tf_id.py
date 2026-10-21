"""exp227: プレイヤー ID 条件つきの BC。全教師の棋譜を1つの網で、局ごとの ID を大域トークンに足して学ぶ。
分割は /content/local_logs/bc_<SPLIT>.json = {train, hold, ids: {eid: k}, names: [..], bake: [k, ...]}。
未見の損失は hold の局をそれぞれの ID で測る(早期停止もこれ)。終わったら bake の各 ID を焼き込んだ
普通の重み tf_<tag>_id<k>.pt(+ cfg)を書く。学習の既定は bc_tf.py と同じ(B 64、AdamW、OneCycle、勾配の上限 1.0)。"""
import json, os, sys, time, copy, glob
for _d in ("/.singularity.d/libs", "/usr/local/cuda/compat/lib"):     # exp254: torch.compile(Triton)が libcuda.so を見つけられるように
    if glob.glob(_d + "/libcuda.so*"): os.environ.setdefault("TRITON_LIBCUDA_PATH", _d); break
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf_id as MI
MKT = os.environ.get("MKT") == "1"          # exp233: 市場の特徴を足す(ext4mkt/<局>.npz)
AMP = os.environ.get("AMP") == "bf16"          # exp254: bf16 の自動混合精度 + TF32(A100 のテンソルコア)。未指定なら従来どおり fp32
if AMP: torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
import contextlib
actx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if AMP else contextlib.nullcontext
WLOSS = float(os.environ.get("WLOSS", 1.0))   # exp236(PTCG 9・10位): 教師が負けた局の手の重み(勝った局は 1.0)。1.0 なら従来と同じ
LS = float(os.environ.get("LS", 0.0))         # exp236: ラベル平滑化(0 なら従来と同じ)
if MKT:
    import model_tf_mkt as MM
PTR = os.environ.get("PTR") == "1"        # exp264: 市場の頭をポインタ頭に(品目トークンとの内積)
if PTR:
    import model_tf_ptr as MP
from bc_tf import load, load_dev, to_dev, KEYS
from extract4 import NM, NS, M_STOP
DEV = "cuda"


def wce(logits, target, w, ls=0.0):
    """重みつきの交差エントロピーの平均(w が全部1・ls=0 なら F.cross_entropy と同じ)。"""
    return (F.cross_entropy(logits, target, reduction="none", label_smoothing=ls) * w).sum() / w.sum().clamp(min=1e-8)


def losses(net, T, idx, train=True, sync=True):
    b = torch.as_tensor(idx, device=DEV)
    wr = T["w"][b] if (train and "w" in T) else torch.ones(len(idx), device=DEV); ls = LS if train else 0.0
    tiles, ou, pu, g, legal = T["tiles"][b], T["ou"][b], T["pu"][b], T["g"][b], T["legal"][b]
    ul, uq, ms, mq = T["ul"][b].long(), T["uq"][b].long(), T["ms"][b].long(), T["mq"][b].long()
    h, pad = net.encode(tiles, ou, pu, g, T["pid"][b], T["mkt"][b]) if MKT else net.encode(tiles, ou, pu, g, T["pid"][b])
    lg, q = net.unit_logits(h, ou, legal)
    live = ul >= 0
    wu = wr.view(-1, *([1] * (ul.dim() - 1))).expand_as(ul)
    lu = wce(lg[live], ul[live], wu[live], ls)
    qm = live & (ul >= 20)
    lq = wce(net.unit_qty(q, ul)[qm], uq[qm], wu[qm]) if qm.any() else lg.sum() * 0
    prev = torch.cat([torch.full_like(ms[:, :1], NM), ms[:, :-1]], 1).clamp(max=NM)
    mo, y = net.market(h, pad, prev)
    first_stop = torch.where(ms == M_STOP, torch.arange(NS, device=DEV).expand_as(ms), torch.full_like(ms, NS)).min(1).values
    valid = torch.arange(NS, device=DEV).unsqueeze(0) <= first_stop.unsqueeze(1)
    wm = wr.view(-1, 1).expand_as(ms)
    lm = wce(mo[valid], ms[valid], wm[valid], ls)
    mqm = valid & (ms < 19)
    lmq = wce(net.market_qty(y, ms)[mqm], mq[mqm], wm[mqm]) if mqm.any() else lm * 0
    with torch.no_grad():
        ua = (lg.argmax(-1) == ul)[live].float().mean()
        mseq = ((mo.argmax(-1) == ms) | ~valid).all(1).float().mean()
    if not sync: return lu + 0.5 * lq + lm + 0.5 * lmq, ua, mseq      # exp254: 1歩ごとに GPU を待たない(記録はまとめて)
    return lu + 0.5 * lq + lm + 0.5 * lmq, float(ua), float(mseq)


class Muon(torch.optim.Optimizer):
    """PTCG 3・10位が使った Muon の簡易版。2次元の重みは運動量を Newton-Schulz で直交化してから足す。
    1次元(LayerNorm・バイアス)と埋め込みは AdamW(3位は1次元に減衰が掛からない不具合で崩壊したので分けて扱う)。"""
    def __init__(self, mp, ap, lr=6e-4, adam_lr=6e-4, momentum=0.95, wd=0.01):
        self.mp = list(mp); self.lr = lr; self.mom = momentum; self.wd = wd
        self.buf = [torch.zeros_like(p) for p in self.mp]
        self.adam = torch.optim.AdamW(ap, lr=adam_lr, weight_decay=wd)
        super().__init__([{"params": self.mp, "lr": lr}], dict(lr=lr))   # スケジューラ用に Optimizer を継承
        self.param_groups = self.adam.param_groups + self.param_groups

    @staticmethod
    def _ns(G, steps=5, eps=1e-7):
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16(); X = X / (X.norm() + eps)
        if X.shape[0] > X.shape[1]: X = X.T; tr = True
        else: tr = False
        for _ in range(steps):
            A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
        return (X.T if tr else X).to(G.dtype)

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        for p in self.mp: p.grad = None

    def step(self):
        self.lr = self.param_groups[-1]["lr"]      # OneCycle が学習率を動かすので毎回読む
        for p, buf in zip(self.mp, self.buf):
            if p.grad is None: continue
            buf.mul_(self.mom).add_(p.grad)
            g = self._ns(buf)
            p.data.mul_(1 - self.lr * self.wd).add_(g, alpha=-self.lr * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
        self.adam.step()

    def state_dict(self): return dict(adam=self.adam.state_dict(), buf=self.buf)
    def load_state_dict(self, d): self.adam.load_state_dict(d["adam"]); self.buf = d["buf"]


def pid_tensor(eids, lens, ids):
    return torch.as_tensor(np.concatenate([np.full(L, ids[e], np.int64) for e, L in zip(eids, lens)]), device=DEV)


def main():
    tag = sys.argv[1]; cache = os.environ.get("CACHE", "/content/local_ext4all"); EP = int(os.environ.get("EPOCHS", 12))
    logf = open("/content/local_logs/tf_%s.log" % tag, "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    S = json.load(open("/content/local_logs/bc_%s.json" % os.environ["SPLIT"])); tr_e, ho_e, ids = S["train"], S["hold"], S["ids"]
    T, tr = load_dev(tr_e, cache)
    lens = [int(np.load("%s/%s.npz" % (cache, e))["g"].shape[0]) for e in tr_e]
    T["pid"] = pid_tensor(tr_e, lens, ids)
    RECENCY = float(os.environ.get("RECENCY", 1.0))   # exp274: 新しい局を重く見る。最古を 1.0、最新を RECENCY として線形に
    if RECENCY != 1.0:
        import re as _re
        def _num(e):
            m = _re.findall(r"\d+", str(e)); return int(m[-1]) if m else 0
        v = np.array([_num(e) for e in tr_e], np.float64)
        r = (v - v.min()) / max(v.max() - v.min(), 1.0)
        wv = (1.0 + (RECENCY - 1.0) * r).astype(np.float32)
        T["w"] = torch.as_tensor(np.concatenate([np.full(L, float(w), np.float32) for w, L in zip(wv, lens)]), device=DEV)
        print("重み: 新しさで %.2f 〜 %.2f(最古 → 最新)" % (wv.min(), wv.max()), flush=True)
    if WLOSS != 1.0:                                  # 局ごとの勝敗(教師の所持金 > 相手)で手の重みを決める
        won = {e: (lambda m: m["teacher"] > m["opp"])(json.loads(str(np.load("%s/%s.npz" % (cache, e))["meta"]))) for e in tr_e}
        T["w"] = torch.as_tensor(np.concatenate([np.full(L, 1.0 if won[e] else WLOSS, np.float32) for e, L in zip(tr_e, lens)]), device=DEV)
        print("重み: 教師の負けた局 %d / %d に %.2f" % (sum(not v for v in won.values()), len(won), WLOSS), flush=True)
    ho = load(ho_e, cache); Th = to_dev(ho); Th["pid"] = pid_tensor(ho_e, [g["X"]["g"].shape[0] for g in ho], ids)
    if MKT:
        md = os.environ.get("MKTDIR", "/content/local_ext4mkt")
        mk = lambda es: torch.as_tensor(np.concatenate([np.load("%s/%s.npz" % (md, e))["mkt"] for e in es]), device=DEV)
        _nf = int(os.environ.get("MKTNF", 8))          # exp270: 16特徴の npz から先頭 nf 個だけ使えるようにする
        T["mkt"] = mk(tr_e)[..., :_nf]; Th["mkt"] = mk(ho_e)[..., :_nf]
        assert T["mkt"].shape[0] == T["g"].shape[0] and Th["mkt"].shape[0] == Th["g"].shape[0], "市場の特徴の行数が合わない"
    cnt = np.bincount([ids[e] for e in tr_e], minlength=len(S["names"]))
    log("学習 %d 局(%d手) / 未見 %d 局(%d手)。ID ごとの学習局数: %s" % (len(tr_e), len(T["g"]), len(ho_e), len(Th["g"]),
        ", ".join("%s %d" % (n, c) for n, c in zip(S["names"], cnt))))
    INIT = os.environ.get("INIT"); CFG = json.load(open(INIT + ".cfg.json")) if INIT else json.loads(os.environ.get("CFG", "{}"))
    nf = int(os.environ.get("MKTNF", 8))
    net = ((MM.PolicyMktPtr if PTR else MM.PolicyMkt)(nid=len(S["names"]), nf=nf, **CFG) if MKT else (MP.PolicyPtr if PTR else MI.PolicyID)(nid=len(S["names"]), **CFG)).to(DEV)
    if INIT:
        sd = torch.load(INIT, map_location=DEV); miss = net.load_state_dict(sd, strict=False)
        log("続きから: %s(足りない重み %s)" % (INIT, miss.missing_keys))
    json.dump(CFG, open("/content/local_logs/tf_%s.pt.cfg.json" % tag, "w"))
    log("構成 %s パラメータ %d" % (CFG, sum(p.numel() for p in net.parameters())))
    LRMAX = float(os.environ.get("LRMAX", 3e-4)); PAT = int(os.environ.get("PATIENCE", 3))
    S_ = len(T["g"]); Sh = len(Th["g"]); B = int(os.environ.get("BATCH", 64)); nb = S_ // B   # exp254: BATCH で1歩の局面数(既定 64 は従来どおり)
    if os.environ.get("OPT") == "muon":      # exp264(PTCG 3・10位): 2次元の重みは直交化した運動量(Muon)、それ以外は AdamW
        mp_ = [p for n, p in net.named_parameters() if p.ndim == 2 and "emb" not in n and "pid" not in n]
        ap_ = [p for n, p in net.named_parameters() if not (p.ndim == 2 and "emb" not in n and "pid" not in n)]
        opt = Muon(mp_, ap_, lr=LRMAX, adam_lr=LRMAX)
        log("最適化: Muon(2次元 %d本 / AdamW %d本)" % (len(mp_), len(ap_)))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=LRMAX, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LRMAX, total_steps=EP * nb, pct_start=0.1 if not INIT else 0.02,
                                              cycle_momentum=os.environ.get("OPT") != "muon")   # Muon は運動量を自分で持つ
    train_losses = torch.compile(losses) if os.environ.get("COMPILE") == "1" else losses   # exp254: 学習の1歩だけ compile(形が一定)。未見の評価は素のまま
    best = None; t0 = time.time(); R = dict(train=tr_e, hold=ho_e, epochs=[])
    CK = "/content/local_logs/tf_%s.ckpt" % tag; ep0 = 0     # 共用キューの割り込み(preemption)で打ち切られても、エポックの境目から続ける
    if os.path.exists(CK):
        c = torch.load(CK, map_location=DEV, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sch.load_state_dict(c["sch"]); ep0 = c["ep"] + 1; R = c["R"]
        best = (c["best_loss"], c["best_ep"], torch.load("/content/local_logs/tf_%s.pt" % tag, map_location=DEV))
        log("割り込みから再開: ep%d から(最良 ep%d %.3f)" % (ep0, best[1], best[0]))
    for ep in range(ep0, EP):
        perm = np.random.permutation(S_); ls = []; ua = []; ma = []
        for k in range(nb):
            with actx(): l, a, m = train_losses(net, T, perm[k*B:(k+1)*B], True, not AMP)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            ls.append(l.detach().float() if AMP else float(l)); ua.append(a); ma.append(m)
        if AMP: ls = [float(x) for x in torch.stack(ls).cpu()]; ua = [float(x) for x in torch.stack(ua).cpu()]; ma = [float(x) for x in torch.stack(ma).cpu()]
        net.eval(); hl = []; ha = []; hm = []; hw = []
        with torch.no_grad():
            for k in range(0, Sh, 256):
                ix = np.arange(k, min(k+256, Sh))
                with actx(): l, a, m = losses(net, Th, ix, train=False); hl.append(float(l) * len(ix)); ha.append(a * len(ix)); hm.append(m * len(ix)); hw.append(len(ix))
        net.train(); W = sum(hw)
        e = dict(ep=ep, loss=np.mean(ls), unit=np.mean(ua), mseq=np.mean(ma), hloss=sum(hl) / W, hunit=sum(ha) / W, hmseq=sum(hm) / W)
        R["epochs"].append({k: float(v) for k, v in e.items()})
        log("  ep%2d 学習 %.3f ユニット %.3f 市場列 %.3f | 未見 %.3f ユニット %.3f 市場列 %.3f (%.1f分)" % (ep, e["loss"], e["unit"], e["mseq"], e["hloss"], e["hunit"], e["hmseq"], (time.time()-t0)/60))
        if best is None or e["hloss"] < best[0]:
            best = (e["hloss"], ep, copy.deepcopy(net.state_dict())); torch.save(best[2], "/content/local_logs/tf_%s.pt" % tag)
        elif PAT and ep - best[1] >= PAT: log("未見の損失が %d エポック改善しないので停止(収束)" % PAT); break
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), ep=ep, R=R, best_loss=best[0], best_ep=best[1]), CK + ".tmp"); os.replace(CK + ".tmp", CK)
    log("未見の損失が最小: ep%d (%.3f)" % (best[1], best[0]))
    for k in ([] if MKT else S.get("bake", [])):     # 市場の特徴つきは焼き込めない(推論でも特徴を計算して渡す)
        out = "/content/local_logs/tf_%s_id%d.pt" % (tag, k)
        torch.save(MI.bake(best[2], k), out); json.dump(CFG, open(out + ".cfg.json", "w"))
        log("焼き込み: ID %d(%s)→ %s" % (k, S["names"][k], out))
    json.dump(R, open("/content/local_logs/tf_%s.json" % tag, "w"), ensure_ascii=False, default=float)
    if os.path.exists(CK): os.remove(CK)
    log("保存")


if __name__ == "__main__":
    main()
