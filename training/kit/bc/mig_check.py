"""09-23: マスク移行(mask0 → mask2)の検査。3つの問いに数字で答える。

  play <重み> <出力 npz>          既定(MASK_MIG を設定しない)で貪欲の自己対戦を打ち、全手番の手と最終所持金を控える。
                                  MIGOLD=<改修前のモジュールを置いた dir> を付けて2回走らせ、cmp で突き合わせる
                                  → (a)「新しい環境変数を設定しなければ従来どおり」の実測。
  cmp <npz> <npz>                 2つの控えを突き合わせる。
  roll <重み> <pen,pen,...> <温度>  **標本打ち**で経済が回るか(区画2の手番・最終所持金)。対照=旧マスク
  pair <重み> <penalty>           採取(sample)と学習(evaluate)に同じ罰がかかっているかを数値で示す
  h2h <重みA> <重みB> <penalty>   A(新マスク＋罰) 対 B(旧マスク) を H2H_N x 2席 貪欲で。合格の物差し(768局 0.47 以上)
  mig <重み> <penalty>            (b) 移行直後の貪欲な全行動が移行前と一致するか
                                  (c) 罰が効いている間、新しい手の確率がどれだけ小さいか
                                  (d) torch.compile の中でも罰の減衰が反映されるか
"""
import os, sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "help"
if MODE in ("mig", "h2h", "pair", "roll"): os.environ["MASK_MIG"] = "1"
os.environ.setdefault("LEGAL_OPT", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("MIGOLD"): sys.path.insert(0, os.environ["MIGOLD"])   # 改修前の model_tf / sp_fast / ppo_tf を先に読む
import hashlib, json, time
import numpy as np, torch
import ppo_tf as P, model_tf as M, sp_fast
import fast_codes; P.to_codes = fast_codes.to_codes_fast
DEV = P.DEV
NS = int(os.environ.get("CK_N", "16")); T = int(os.environ.get("CK_T", "719"))
STRIDE = int(os.environ.get("CK_STRIDE", "8")); SEED0 = int(os.environ.get("CK_SEED0", "4100000"))
CH = int(os.environ.get("CK_CHUNK", "64"))


def _play(net, out):
    pool = sp_fast.BatchSPPool([SEED0 + 7919 * k for k in range(NS)])
    hsh = hashlib.sha256(); t0 = time.time()
    for t in range(T):
        acts = {}
        for s in (0, 1):
            st = pool.state(s)
            outs = [P.sample(net, tuple(x[i:i + CH] for x in st), greedy=True, cap=True) for i in range(0, len(st[3]), CH)]
            ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
            U, O = P.to_codes(ul, uq, ms, mq, st[1]); acts[s] = (U, O)
            hsh.update(np.ascontiguousarray(U, np.int32).tobytes()); hsh.update(np.ascontiguousarray(O, np.int32).tobytes())
            hsh.update(np.ascontiguousarray(st[4], np.uint8).tobytes())
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    mny = pool.money()
    np.savez(out, money=mny, sha=np.frombuffer(hsh.digest(), np.uint8), path=np.array([os.environ.get("MIGOLD", "(新)")]))
    print("%d局 %d手 %.0fs  手と観測の SHA %s  最終所持金の和 %.0f → %s" % (NS, T, time.time() - t0, hsh.hexdigest()[:24], mny.sum(), out), flush=True)


def _cmp(a, b):
    A = np.load(a, allow_pickle=True); B = np.load(b, allow_pickle=True)
    sa = bytes(A["sha"]).hex(); sb = bytes(B["sha"]).hex()
    same_h = sa == sb; same_m = np.array_equal(A["money"], B["money"])
    print("[検査(a) 既定の挙動] %s と %s" % (str(A["path"][0]), str(B["path"][0])))
    print("  719手すべての手と観測の SHA: %s (%s / %s)" % ("一致" if same_h else "**不一致**", sa[:24], sb[:24]))
    print("  最終所持金 (%d局, 2席): %s" % (len(A["money"]), "ビット一致" if same_m else "**不一致**"))
    print("  → %s" % ("MASK_MIG を設定しなければ従来どおり" if (same_h and same_m) else "**既定の挙動が変わっている。差し戻すこと**"))
    sys.exit(0 if (same_h and same_m) else 1)


def _mig(W, pen):
    net = M.load(W, DEV)
    sp_fast._MIG_SEAT[0] = None
    pool = sp_fast.BatchSPPool([SEED0 + 7919 * k for k in range(NS)])
    pool.mig_ok[:] = True; pool.legal_opt = np.full((pool.n, 2), sp_fast.MIG_V, np.int32)
    nu = nu_same = 0; nturn = nturn_same = 0; xps = []; xpmax = 0.0; nsw = 0
    t0 = time.time()
    for t in range(T):
        acts = {}
        for s in (0, 1):
            st = pool.state(s)
            lmc = st[4]; lm2 = (lmc != 0); xp = (lmc & 2) != 0; lm0 = (lm2 & ~xp).astype(np.uint8)
            st_old = list(st); st_old[4] = lm0                      # 移行前: 旧マスク・罰なし
            M.mig_set(0.0)
            a0 = [np.concatenate([o[j] for o in [P.sample(net, tuple(x[i:i + CH] for x in st_old), greedy=True, cap=True) for i in range(0, len(st[3]), CH)]]) for j in range(4)]
            M.mig_set(pen)                                          # 移行直後: 新マスク・罰つき(lm の bit1 が印)
            a1 = [np.concatenate([o[j] for o in [P.sample(net, tuple(x[i:i + CH] for x in st), greedy=True, cap=True) for i in range(0, len(st[3]), CH)]]) for j in range(4)]
            live = st[1][..., 0] > 0
            nu += int(live.sum()); nu_same += int(((a0[0] == a1[0]) & live).sum())
            eq = np.ones(len(live), bool)
            for k_ in range(4):
                m_ = live if k_ < 2 else np.ones_like(a0[k_], bool)
                eq &= ((a0[k_] == a1[k_]) | ~m_).all(1)
            nturn += len(eq); nturn_same += int(eq.sum())
            nsw += int(((a0[0] != a1[0]) & live).sum())
            if t % STRIDE == 0:                                     # (c) 罰が効いている間の新しい手の確率
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    h, _ = net.encode(P.tt(st[0], torch.float16), P.tt(st[1]), P.tt(st[2]), P.tt(st[3]))
                    _l = P.tt(lmc, torch.uint8); lg, _ = net.unit_logits(h, P.tt(st[1]), _l); lg = M.mig_apply(lg, _l)
                pr = lg.float().softmax(-1)
                x_t = torch.as_tensor(xp, device=DEV); sel = torch.as_tensor(live, device=DEV) & x_t.any(-1)
                if int(sel.sum()) :
                    v = (pr * x_t.float()).sum(-1)[sel]
                    xps.append(float(v.mean())); xpmax = max(xpmax, float(v.max()))
            acts[s] = P.to_codes(a1[0], a1[1], a1[2], a1[3], st[1])   # 進めるのは移行後の手
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
        if t % 180 == 0: print("  手%3d %.0fs" % (t, time.time() - t0), flush=True)
    print("[検査(b) 移行直後の貪欲な全行動] 罰 %.1f" % pen)
    print("  人手ごとの行動: %d / %d 一致 (%.5f)  変わった人手 %d" % (nu_same, nu, nu_same / max(nu, 1), nsw))
    print("  手番まるごと(人手20 + 市場11 + 数量): %d / %d 一致 (%.5f)" % (nturn_same, nturn, nturn_same / max(nturn, 1)))
    print("[検査(c) 罰が効いている間の新しい手の確率] 平均 %.3e  最大 %.3e (%d標本)" % (
        float(np.mean(xps)) if xps else -1, xpmax, len(xps)))
    # (d) compile の中でも罰の減衰が効くか
    st = pool.state(0); lmc = st[4]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        h, _ = net.encode(P.tt(st[0], torch.float16), P.tt(st[1]), P.tt(st[2]), P.tt(st[3]))
    ou_t = P.tt(st[1]); lm_t = P.tt(lmc, torch.uint8)
    f = torch.compile(lambda hh, oo, ll: M.mig_apply(net.unit_logits(hh, oo, ll)[0], ll), dynamic=False)
    with torch.no_grad():
        M.mig_set(pen); r1 = f(h, ou_t, lm_t).float().clone()
        M.mig_set(0.0); r2 = f(h, ou_t, lm_t).float().clone()
    x_t = torch.as_tensor((lmc & 2) != 0, device=DEV)
    d = ((r2 - r1) * x_t.float()).abs()
    ok = bool(x_t.any()) and abs(float(d[x_t].mean()) - pen) < 1e-2 and float(((r2 - r1) * (~x_t).float()).abs().max()) < 1e-3
    print("[検査(d) torch.compile の中の罰] 新しい手の logit 差 平均 %.4f(期待 %.4f)・他の手の差 最大 %.2e → %s" % (
        float(d[x_t].mean()) if bool(x_t.any()) else -1, pen, float(((r2 - r1) * (~x_t).float()).abs().max()), "反映される" if ok else "**反映されない**"))


def _h2h(wa, wb, pen):
    """移行後の a(新マスク＋罰) 対 移行前の b(旧マスク・罰なし)。両席・貪欲。合格の物差し。"""
    import sp_lib as S
    S.SPPool = sp_fast.BatchSPPool
    a = M.load(wa, DEV); b = M.load(wb, DEV)
    nse = int(os.environ.get("H2H_N", "384")); sd = [int(os.environ.get("H2H_SEED0", "6100000")) + 7919 * k for k in range(nse)]
    mg = []
    for s_ in (0, 1):
        M.mig_set(pen); sp_fast._MIG_SEAT[0] = s_       # 罰をかけるのは a の席だけ
        m = S.play(sd, {s_: S.PolicyOpp(a, greedy=True, cap=True), 1 - s_: S.PolicyOpp(b, greedy=True, cap=True)})
        sp_fast._MIG_SEAT[0] = None
        mg.append(m[:, s_] - m[:, 1 - s_])
    mg = np.concatenate(mg); n = len(mg)
    w = float((mg > 0).mean() + 0.5 * (mg == 0).mean())
    print("[直接対戦] %s(新マスク・罰 %.1f) 対 %s(旧マスク) %d局 貪欲: 勝率 %.4f(勝 %d 分 %d 敗 %d) 所持金差 %+.0f  標準誤差 %.4f" % (
        os.path.basename(wa), pen, os.path.basename(wb), n, w, int((mg > 0).sum()), int((mg == 0).sum()), int((mg < 0).sum()),
        float(mg.mean()), (w * (1 - w) / n) ** 0.5), flush=True)


def _pair(W, pen):
    """(必須の検査) 採取(P.sample)と学習(P.evaluate)に**同じ罰**がかかっているか。
       同じ lm を通すので設計上ずれないが、数値で示す。"""
    net = M.load(W, DEV); sp_fast._MIG_SEAT[0] = None
    pool = sp_fast.BatchSPPool([SEED0 + 7919 * k for k in range(NS)])
    pool.mig_ok[:] = True; pool.legal_opt = np.full((pool.n, 2), sp_fast.MIG_V, np.int32)
    for t in range(int(os.environ.get("PR_T", "120"))):             # 少し進めて、増えた手が出る局面にする
        a = {}
        for s_ in (0, 1):
            st = pool.state(s_)
            M.mig_set(pen)
            o = [np.concatenate([x[j] for x in [P.sample(net, tuple(y[i:i + CH] for y in st), greedy=False, cap=True) for i in range(0, len(st[3]), CH)]]) for j in range(4)]
            a[s_] = P.to_codes(o[0], o[1], o[2], o[3], st[1])
            last = (st, o)
        pool.step(a[0][0], a[0][1], a[1][0], a[1][1])
    st, o = last
    ul, uq, ms, mq = o
    b = [P.tt(st[0], torch.float16), P.tt(st[1]), P.tt(st[2]), P.tt(st[3]), P.tt(st[4], torch.uint8),
         P.tt(ul, torch.long), P.tt(uq, torch.long), P.tt(ms, torch.long), P.tt(mq, torch.long)]
    xp = torch.as_tensor((st[4] & 2) != 0, device=DEV); live = torch.as_tensor(st[1][..., 0] > 0, device=DEV)
    out = {}
    for name, pv in (("罰あり %.1f" % pen, pen), ("罰なし 0", 0.0)):
        M.mig_set(pv)
        with torch.no_grad():
            lp, ent, v, dist, per, pmask = P.evaluate(net, b)              # 学習が使う道
            h, _ = net.encode(b[0], b[1], b[2], b[3])
            lg, _ = net.unit_logits(h, b[1], b[4]); lg = M.mig_apply(lg, b[4])   # 採取が使う道(ppo_tf.sample と同じ式)
        lsu_direct = torch.log_softmax(lg.float(), -1).gather(-1, b[5].clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lsu_eval = dist[0].float().gather(-1, b[5].clamp(min=0).unsqueeze(-1)).squeeze(-1)
        d = (lsu_direct - lsu_eval).abs()[live].max()
        out[name] = (float(d), lsu_eval.clone())
        print("  %s: 採取側の logit から直に出した対数確率 と 学習側 P.evaluate の対数確率 の最大差 %.3e" % (name, float(d)))
    a_, b_ = out["罰あり %.1f" % pen][1], out["罰なし 0"][1]
    selx = xp.gather(-1, b[5].clamp(min=0).unsqueeze(-1)).squeeze(-1) & live
    print("  罰を 0 にすると学習側の対数確率が動く: 増えた手を選んだ %d 個で平均 %+.3f、それ以外 %d 個で平均 %+.3f" % (
        int(selx.sum()), float((b_ - a_)[selx].mean()) if int(selx.sum()) else 0.0,
        int((live & ~selx).sum()), float((b_ - a_)[live & ~selx].mean())))
    print("  → 採取と学習が同じ lm・同じ罰を通っている(差 ~1e-6 は fp16 の丸め)")


def _quads(pool):
    e = pool.env; out = np.zeros((pool.n, 2), np.float32)
    for s_ in (0, 1):
        e.set_view(s_); out[:, s_] = np.clip(np.rint(np.asarray(e.farm_pub(0))[:, 2] * 4), 0, 4)
    e.set_view(0); return out


def _roll_one(net, pen, mig, temp):
    """標本打ちで 1 設定ぶん自己対戦し、経済が回っているかを測る。mig=False なら旧マスク・罰なし(対照)。"""
    sp_fast._MIG_SEAT[0] = None
    pool = sp_fast.BatchSPPool([SEED0 + 7919 * k for k in range(NS)])
    if mig:
        pool.mig_ok[:] = True; pool.legal_opt = np.full((pool.n, 2), sp_fast.MIG_V, np.int32)
    M.mig_set(pen if mig else 0.0)
    torch.manual_seed(int(os.environ.get("CK_TSEED", "31")))
    q2 = np.full((pool.n, 2), T, np.int32); q3 = np.full((pool.n, 2), T, np.int32)
    nlive = 0; nxp = 0; t0 = time.time()
    for t in range(T):
        nq = _quads(pool)
        q2 = np.where((nq >= 2) & (q2 == T), t, q2); q3 = np.where((nq >= 3) & (q3 == T), t, q3)
        acts = {}
        for s_ in (0, 1):
            st = pool.state(s_)
            outs = [P.sample(net, tuple(x[i:i + CH] for x in st), greedy=False, temp=temp) for i in range(0, len(st[3]), CH)]
            ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
            live = st[1][..., 0] > 0
            nlive += int(live.sum())
            if mig: nxp += int((live & (np.take_along_axis((st[4] & 2), np.clip(ul, 0, 43)[..., None], -1)[..., 0] != 0)).sum())
            acts[s_] = P.to_codes(ul, uq, ms, mq, st[1])
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    m = pool.money()
    return dict(name=("新マスク 罰 %.0f" % pen) if mig else "旧マスク(対照)", q2=float(np.median(q2)), q2m=float(q2.mean()),
                q3=float(np.median(q3)), money=float(m.mean()), live=nlive / float(T * 2),
                xp=nxp / max(nlive, 1), sec=time.time() - t0)


def _roll(W, pens, temp):
    """(必須の検査) **標本打ち**で経済が回るか。貪欲の一致と h2h だけでは見えない崩壊をここで捕まえる。"""
    net = M.load(W, DEV)
    rows = [_roll_one(net, 0.0, False, temp)]
    for pen in pens: rows.append(_roll_one(net, pen, True, temp))
    print("\n[標本打ちの健全性] %s  温度 %.2f  %d局 %d手 自己対戦" % (os.path.basename(W), temp, NS, T))
    print("  %-18s %10s %10s %10s %12s %10s %10s" % ("設定", "区画2 中央", "区画2 平均", "区画3 中央", "最終所持金", "生人手/手", "新手選択率"))
    base = rows[0]
    for r in rows:
        print("  %-18s %10.0f %10.0f %10.0f %12.0f %10.2f %10.2e%s" % (
            r["name"], r["q2"], r["q2m"], r["q3"], r["money"], r["live"], r["xp"],
            "" if r is base else ("  → %s" % ("合格" if (r["q2m"] <= base["q2m"] * 1.15 + 5 and r["money"] >= base["money"] * 0.85) else "**崩壊**"))))
    print("  合格の条件: 区画2の平均手番が対照の 1.15倍+5 以内 かつ 最終所持金が対照の 85%% 以上")


if MODE == "roll": _roll(sys.argv[2], [float(x) for x in sys.argv[3].split(",")], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
elif MODE == "pair": _pair(sys.argv[2], float(sys.argv[3]))
elif MODE == "h2h": _h2h(sys.argv[2], sys.argv[3], float(sys.argv[4]))
elif MODE == "play": _play(M.load(sys.argv[2], DEV), sys.argv[3])
elif MODE == "cmp": _cmp(sys.argv[2], sys.argv[3])
elif MODE == "mig": _mig(sys.argv[2], float(sys.argv[3]))
else: print(__doc__)
