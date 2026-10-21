"""エンティティ transformer の BC。学習 → 未見の対局で閉ループ(教師比)。
比較のため CNN(sp3a)と同じ対局の分割を使える(SPLIT=sp3a)。"""
import json, glob, os, sys, time, random, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf as M
from extract4 import state, codes, NU, NM, NS, M_STOP
from deploy import wide_blank
import kagsim, kagpack
DEV = "cuda"
KEYS = ("tiles", "ou", "pu", "g", "legal", "ul", "uq", "ms", "mq")
MKT_W = None

def load(eids, cache):
    G = []
    for e in eids:
        z = np.load("%s/%s.npz" % (cache, e)); X = {k: z[k] for k in KEYS}
        G.append(dict(eid=e, X=X, meta=json.loads(str(z["meta"])), opp_acts=json.loads(str(z["opp_acts"]))))
    return G

def load_dev(eids, cache):
    """1局ずつ読んで GPU の確保済みの領域に詰める(全局を一度に CPU に載せると 12GB を超える)。軽い情報だけ返す。"""
    lens = [np.load("%s/%s.npz" % (cache, e))["g"].shape[0] for e in eids]; n = sum(lens)
    z0 = np.load("%s/%s.npz" % (cache, eids[0]))
    T = {k: torch.empty((n,) + z0[k].shape[1:], dtype=torch.float16 if k == "tiles" else torch.from_numpy(z0[k][:1]).dtype, device=DEV) for k in KEYS}
    i = 0; G = []
    for e, L in zip(eids, lens):
        z = np.load("%s/%s.npz" % (cache, e))
        for k in KEYS: T[k][i:i + L] = torch.as_tensor(z[k], device=DEV)
        i += L; G.append(dict(eid=e, meta=json.loads(str(z["meta"]))))
    return T, G

def to_dev(G):
    cat = lambda k: torch.as_tensor(np.concatenate([g["X"][k] for g in G]), device=DEV)
    T = {k: cat(k) for k in KEYS}; T["tiles"] = T["tiles"].half(); return T

def losses(net, T, idx, train=True):
    b = torch.as_tensor(idx, device=DEV)
    tiles, ou, pu, g, legal = T["tiles"][b], T["ou"][b], T["pu"][b], T["g"][b], T["legal"][b]
    ul, uq, ms, mq = T["ul"][b].long(), T["uq"][b].long(), T["ms"][b].long(), T["mq"][b].long()
    h, pad = net.encode(tiles, ou, pu, g)
    lg, q = net.unit_logits(h, ou, legal)
    live = ul >= 0
    lu = F.cross_entropy(lg[live], ul[live])
    qm = live & (ul >= 20)
    lq = F.cross_entropy(net.unit_qty(q, ul)[qm], uq[qm]) if qm.any() else lg.sum() * 0
    prev = torch.cat([torch.full_like(ms[:, :1], NM), ms[:, :-1]], 1).clamp(max=NM)
    mo, y = net.market(h, pad, prev)
    first_stop = torch.where(ms == M_STOP, torch.arange(NS, device=DEV).expand_as(ms), torch.full_like(ms, NS)).min(1).values
    valid = torch.arange(NS, device=DEV).unsqueeze(0) <= first_stop.unsqueeze(1)
    lm = F.cross_entropy(mo[valid], ms[valid], weight=MKT_W)   # 稀な注文(土地・種)ほど重く(WMKT で指数)
    mqm = valid & (ms < 19)
    lmq = F.cross_entropy(net.market_qty(y, ms)[mqm], mq[mqm]) if mqm.any() else lm * 0
    with torch.no_grad():
        ua = (lg.argmax(-1) == ul)[live].float().mean()
        mseq = ((mo.argmax(-1) == ms) | ~valid).all(1).float().mean()
    return lu + 0.5 * lq + lm + 0.5 * lmq, float(ua), float(mseq)

@torch.no_grad()
def act(net, env):
    tiles, ou, pu, g, lm = state(env)
    t = lambda a, dt=torch.float32: torch.as_tensor(a[None], device=DEV, dtype=dt)
    h, pad = net.encode(t(tiles, torch.float16), t(ou), t(pu), t(g))
    lg, q = net.unit_logits(h, t(ou), t(lm, torch.uint8))
    lg = lg[0]; live = [u for u in range(NU) if ou[u, 0] > 0]
    seeds = np.rint(g[27:32] * 20).astype(int); used = np.zeros(5, int)
    ul = np.full(NU, -1, np.int64)
    for u in live:                                    # 植え付けは種の数を超えると全部無効になる(エンジン規則)ので順に割り当てる
        order = torch.argsort(lg[u], descending=True).tolist()
        for c in order:
            if lg[u, c] < -1e3: continue
            if 15 <= c < 20 and used[c - 15] >= seeds[c - 15]: continue
            ul[u] = c; break
        if 15 <= ul[u] < 20: used[ul[u] - 15] += 1
    uq = net.unit_qty(q, torch.as_tensor(ul[None], device=DEV)).argmax(-1)[0].cpu().numpy()
    prev = [NM]; ms = []; mq = []
    for j in range(NS):
        mo, y = net.market(h, pad, torch.as_tensor([prev], device=DEV))
        c = int(mo[0, -1].argmax()); ms.append(c)
        mq.append(int(net.market_qty(y[:, -1:], torch.as_tensor([[c]], device=DEV)).argmax(-1)[0, 0]) if c < 19 else 0)
        if c == M_STOP or j == 9: break
        prev.append(c)
    ms += [M_STOP] * (NS - len(ms)); mq += [0] * (NS - len(mq))
    return codes(ul, uq, live, np.asarray(ms), np.asarray(mq))

def closed_loop(net, gm):
    net.eval(); meta = gm["meta"]; seat = meta["seat"]
    op = kagpack.pack(gm["opp_acts"]); w = wide_blank(); A, Bt = (w, op) if seat == 0 else (op, w)
    env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], [meta["seed"]], 720, seat); env.keep_nonsell = False; env.reset()
    for i in range(719):
        U, O = act(net, env); env.step_units(U[None], O[None])
    m = env.money()[0]; net.train(); return float(m[0]), float(m[1])

def main():
    tag = sys.argv[1]; cache = os.environ.get("CACHE", "/content/local_ext4"); EP = int(os.environ.get("EPOCHS", 30))
    logf = open("/content/local_logs/tf_%s.log" % tag, "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    if os.environ.get("SPLIT"):
        S = json.load(open("/content/local_logs/bc_%s.json" % os.environ["SPLIT"])); tr_e, ho_e = S["train"], S["hold"]
    else:
        es = sorted(os.path.basename(f)[:-4] for f in glob.glob(cache + "/*.npz")); random.Random(0).shuffle(es)
        nh = int(os.environ.get("NHOLD", 29)); ho_e, tr_e = es[:nh], es[nh:]
    T, tr = load_dev(tr_e, cache); ho = load(ho_e, cache); Th = to_dev(ho)
    global MKT_W
    a = float(os.environ.get("WMKT", 0))
    if a > 0:
        ms = T["ms"].long(); first = torch.where(ms == M_STOP, torch.arange(NS, device=DEV).expand_as(ms), torch.full_like(ms, NS)).min(1).values
        valid = torch.arange(NS, device=DEV).unsqueeze(0) <= first.unsqueeze(1)
        cnt = torch.bincount(ms[valid], minlength=NM).float() + 1
        w = cnt.pow(-a); w = w / (w[ms[valid]].mean()); MKT_W = w
        log("市場の重み(指数 %.2f): 土地 %.2f 種(メロン) %.2f 雇用 %.2f 終了 %.2f" % (a, w[20], w[13], w[19], w[M_STOP]))
    S_ = len(T["g"]); Sh = len(Th["g"]); log("学習 %d 局(%d手) / 未見 %d 局(%d手)" % (len(tr), S_, len(ho), Sh))
    CFG = json.loads(os.environ.get("CFG", "{}"))   # 例 {"d":64,"layers":2,"ff":128,"enc":"tf","dec_layers":1}
    INIT = os.environ.get("INIT")                     # 続きから学習する重み(構成は INIT の .cfg.json を使う)
    if INIT and os.path.exists(INIT + ".cfg.json"): CFG = json.load(open(INIT + ".cfg.json"))
    SEED = int(os.environ.get("SEED", 0)); torch.manual_seed(SEED)   # SEED: 初期化の種。既定 0(従来と同じ)
    if "SEED" in os.environ: np.random.seed(SEED)   # 指定したときだけ並べ替えも固定する(従来は並べ替えが種なし)
    net = M.Policy(**CFG).to(DEV)
    if INIT: net.load_state_dict(torch.load(INIT, map_location=DEV)); log("続きから: %s" % INIT)
    json.dump(CFG, open("/content/local_logs/tf_%s.pt.cfg.json" % tag, "w"))
    log("構成 %s パラメータ %d" % (CFG, sum(p.numel() for p in net.parameters())))
    LRMAX = float(os.environ.get("LRMAX", 3e-4)); PAT = int(os.environ.get("PATIENCE", 0))   # PATIENCE>0: 未見の損失が P エポック続けて下がらなければ止める
    opt = torch.optim.AdamW(net.parameters(), lr=LRMAX, weight_decay=0.01); B = 64; nb = S_ // B
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LRMAX, total_steps=EP * nb, pct_start=0.1 if not INIT else 0.02)
    AMP = os.environ.get("AMP") == "1"; scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    if AMP: log("混合精度(AMP)で学習")
    best = None; t0 = time.time(); R = dict(train=tr_e, hold=ho_e, epochs=[])
    for ep in range(EP):
        perm = np.random.permutation(S_); ls = []; ua = []; ma = []
        for k in range(nb):
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):   # AMP=1: 混合精度(T4 のテンソルコア)。既定はオフ(従来と同じ)
                l, a, m = losses(net, T, perm[k*B:(k+1)*B])
            opt.zero_grad(); scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); ls.append(float(l)); ua.append(a); ma.append(m)
        net.eval(); hl = []; ha = []; hm = []
        with torch.no_grad():
            for k in range(0, Sh, 256):
                l, a, m = losses(net, Th, np.arange(k, min(k+256, Sh))); hl.append(float(l)); ha.append(a); hm.append(m)
        net.train()
        e = dict(ep=ep, loss=np.mean(ls), unit=np.mean(ua), mseq=np.mean(ma), hloss=np.mean(hl), hunit=np.mean(ha), hmseq=np.mean(hm))
        R["epochs"].append({k: float(v) for k, v in e.items()})
        log("  ep%2d 学習 %.3f ユニット %.3f 市場列 %.3f | 未見 %.3f ユニット %.3f 市場列 %.3f (%.1f分)" % (ep, e["loss"], e["unit"], e["mseq"], e["hloss"], e["hunit"], e["hmseq"], (time.time()-t0)/60))
        if best is None or e["hloss"] < best[0]:
            best = (e["hloss"], ep, copy.deepcopy(net.state_dict())); torch.save(best[2], "/content/local_logs/tf_%s.pt" % tag)   # 途中でも最良を残す
            json.dump(CFG, open("/content/local_logs/tf_%s.pt.cfg.json" % tag, "w"))
        elif PAT and ep - best[1] >= PAT: log("未見の損失が %d エポック改善しないので停止(収束)" % PAT); break
    net.load_state_dict(best[2]); torch.save(best[2], "/content/local_logs/tf_%s.pt" % tag)
    log("未見の損失が最小: ep%d (%.3f)" % (best[1], best[0]))
    res = []
    if os.environ.get("NOCLOSED"): log("閉ループ評価は省略(eval58.py で測る)"); return
    tr = load(tr_e[:6], cache)
    for kind, gs in (("未見", ho), ("学習", tr[:6])):
        for gm in gs:
            a, b = closed_loop(net, gm); t = gm["meta"]["teacher"]
            res.append(dict(kind=kind, game=gm["eid"], ours=a, opp=b, teacher=t, ratio=a/t, win=int(a > b)))
            log("  [%s] %s 自分 %9.0f 相手 %9.0f 教師 %9.0f 教師比 %.3f %s" % (kind, gm["eid"], a, b, t, a/t, "勝ち" if a > b else "負け"))
    for kind in ("未見", "学習"):
        rs = [r for r in res if r["kind"] == kind]
        log("%s: 教師比 平均 %.3f (最小 %.3f 最大 %.3f)  勝ち %d/%d" % (kind, np.mean([r["ratio"] for r in rs]), min(r["ratio"] for r in rs), max(r["ratio"] for r in rs), sum(r["win"] for r in rs), len(rs)))
    R["closed"] = res; json.dump(R, open("/content/local_logs/tf_%s.json" % tag, "w"), ensure_ascii=False, default=float)
    try: json.dump(R, open("/content/drive/MyDrive/kaggriculture/runs/tf_%s.json" % tag, "w"), ensure_ascii=False, default=float)
    except Exception: pass
    log("保存")

if __name__ == "__main__":
    main()
