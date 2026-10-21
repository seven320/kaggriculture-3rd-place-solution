"""exp298: PPO の更新(1ミニバッチ = 批評家の順逆 + 方策の順逆 + 基準の順 + Adam)の速さと、CUDAGRAPH=1 の版が同じ結果になるかを測る。
apo.py / ppo_orbit2.py の更新ループ(ANTI=1・FASTINF=1・CRITIC=1・COMPILE_UPD=1 の本番設定)を写したもの。
  base : いまの本番と同じ(COMPILE_UPD=1)
  base2: base をもう一度(同じ版どうしの揺れ = 逆伝播の atomicAdd などの非決定性の大きさ)
  cg   : 更新の compile を CUDA グラフで再生 + 1歩ごとの取り出しをやめる
  cgc  : cg + 市場デコーダの因果マスクの判定を省く(= CUDAGRAPH=1 の版)
  basec: base + 因果マスクの判定を省く(CUDA グラフなし)
  nosync: base + 1歩ごとの取り出しをやめるだけ
データは非同期の対局係が書き出した局(ASYNC_DIR/buf)の1つから取る。
環境変数: NSTEP(測る歩数、既定 60)WARM(捨てる歩数、既定 8)MB(既定 512)RUNS(既定 base,cg,base2)PROF(1 で base の数歩を profiler)"""
import os, sys, time, glob, json
os.environ.setdefault("FASTINF", "1")
DET = os.environ.get("DET") == "1"      # 1: 決定的なカーネルだけを使う(base どうしがビット一致するようにして、cg との差がゼロかを見る)
if DET: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M
DEV = "cuda"; CLIP = P.CLIP
NSTEP = int(os.environ.get("NSTEP", 60)); WARM = int(os.environ.get("WARM", 8)); MB = int(os.environ.get("MB", 512))
BETA, ENT = 0.1, 0.001
INIT = os.environ.get("INIT", "/content/local_logs/tf_idtop3_id0.pt"); CRITINIT = os.environ.get("CRITINIT", "/content/local_logs/vlab_vl4_anti_l1.pt")
KEYS = ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq"); OKEYS = ("otiles", "oou", "opu", "og")
RAW_EV = P.evaluate
if DET: torch.use_deterministic_algorithms(True, warn_only=True)
import cg_upd

# ---- データ(1局ぶん = 172,560局面のうち、使う分だけ GPU へ) ----
fs = sorted(glob.glob(os.environ.get("BUFGLOB", "/content/lab/async/apo2/buf/*.pt")), key=os.path.getmtime)
SH = None
for f in reversed(fs[:-1] or fs):
    try: SH = torch.load(f, map_location="cpu", weights_only=False); print("データ", os.path.basename(f), flush=True); break
    except Exception as e: print("読めない", f, str(e)[:80], flush=True)
T_, N_ = SH["T"], SH["N"]
n = MB * (NSTEP + WARM) + 16                 # 最後に 16局面の端数のミニバッチを1つ(本番と同じく再コンパイル/別グラフが要る形)
HOLD = int(os.environ.get("HOLD", 8192))    # 学習に使わない局面で、批評家の価値の当たり具合を測る
g = torch.Generator().manual_seed(0); _pm = torch.randperm(T_ * N_, generator=g); sel = _pm[:n]; hsel = _pm[n:n + HOLD]
D = {k: v[sel].to(DEV) for k, v in SH["D"].items()}; DO = {k: v[sel].to(DEV) for k, v in SH["DO"].items()}
PER0 = SH["PER0"][sel].to(DEV).float()
V = SH["V"]; fin = SH["fin"]; z = np.sign(fin[:, 0] - fin[:, 1])
Rw = np.zeros_like(V); Rw[-1] = z; adv = np.zeros_like(Rw); last = np.zeros(N_)
for t in reversed(range(T_)):
    nv = V[t + 1] if t + 1 < T_ else np.zeros(N_); delta = Rw[t] + nv - V[t]; last = delta + 0.95 * last; adv[t] = last
ret = adv + V
A = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)[sel].to(DEV); RT = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)[sel].to(DEV)
A = (A - A.mean()) / (A.std() + 1e-8)
DH = [SH["D"][k][hsel].to(DEV) for k in KEYS[:4]]; DOH = [SH["DO"][k][hsel].to(DEV) for k in OKEYS]
RTH = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)[hsel].to(DEV)
ZH = torch.as_tensor(np.broadcast_to(z.reshape(1, -1), V.shape).reshape(-1).copy(), dtype=torch.float32)[hsel].to(DEV)


@torch.no_grad()
def crit_quality(crit):
    """学習に使っていない HOLD 局面で、反対称の価値 (v(自分) − v(相手))/2 の 目標 ret への smooth_l1・説明率、勝敗 z への説明率。
    encode は差し替え前の元の式(Policy.encode)で計算する。"""
    vs = []
    for j in range(0, len(RTH), 1024):
        with torch.autocast("cuda", dtype=torch.float16):
            xs = [torch.cat([DH[q][j:j + 1024], DOH[q][j:j + 1024]], 0) for q in range(4)]
            hh, _ = M.Policy.encode(crit, *xs); vv = crit.value(hh[:, -1]).squeeze(-1).float(); m_ = len(xs[0]) // 2
            vs.append(0.5 * (vv[:m_] - vv[m_:]))
    v = torch.cat(vs)
    ev = lambda y: float(1 - (y - v).var() / (y.var() + 1e-8))
    return float(F.smooth_l1_loss(v, RTH)), ev(RTH), ev(ZH)
del SH
print("局面 %d(%d歩 + 端数16)、MB %d" % (n, NSTEP + WARM, MB), flush=True)


def run(mode, label):
    FL = dict((t_[:2], t_[2:]) for t_ in mode.split("+") if t_[:2] in ("ce", "cf", "cl"))   # ce2 = CRIT_EVERY=2、cf0.5 = CRIT_FRAC=0.5、cl = CRIT_LAST=1
    CE = int(FL.get("ce") or 1); CF = float(FL.get("cf") or 1); CL = "cl" in FL
    mode = mode.split("+")[0] if not FL else "base"
    CG = mode in ("cg", "cgc"); NOSYNC = CG or mode in ("nosync", "stn"); CAUSAL = mode in ("cgc", "basec"); ST = mode in ("st", "stn")
    ONLY = mode[4:] if mode.startswith("only") else ""   # only{crit,pol,ref}: その部分だけの時間(内訳用、値の比較はしない)
    torch._dynamo.reset()                     # 前の版の compile を持ち越さない(同じ関数の再コンパイル上限 8 に当たると eager に落ちる)
    s2 = torch.cuda.Stream() if ST else None  # st: 批評家の順逆を別の stream で方策の順逆と重ねる(同じカーネル・同じデータ、足し合わせの順も同じ)
    torch.manual_seed(0)
    net = M.load(INIT, DEV); crit = M.load(CRITINIT, DEV); ref = M.load(INIT, DEV)
    for p in ref.parameters(): p.requires_grad_(False)
    net.eval(); ref.eval(); crit.eval()
    q0 = crit_quality(crit)
    if CL:
        import crit_fast
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            xs = [torch.cat([DH[q][:512], DOH[q][:512]], 0) for q in range(4)]
            h_full = M.Policy.encode(crit, *xs)[0][:, -1].float(); crit_fast.install_last(crit); h_last = crit.encode(*xs)[0][:, -1].float()
        print("  CRIT_LAST: 最後のトークンの差 最大 %.3e(値の大きさ %.2f)" % (float((h_full - h_last).abs().max()), float(h_full.abs().max())), flush=True)
    # --- ここから ppo_orbit2.py / apo.py の compile の場所と同じ ---
    ev = torch.compile(RAW_EV, dynamic=False)
    crit.encode = torch.compile(crit.encode, dynamic=False)
    _evu, _cencu = ev, crit.encode
    if CG: _evu, _cencu = cg_upd.make(RAW_EV, crit)
    if CAUSAL: cg_upd.hint_causal(net, ref)
    cparams = list(crit.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=5e-5, eps=1e-5); opt_c = torch.optim.Adam(cparams, lr=1e-5, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda"); scaler_c = torch.amp.GradScaler("cuda")
    perm = torch.randperm(n, device=DEV)
    stats = []; vls = []; times = []
    prof = os.environ.get("PROF") == "1" and label == "base"
    for si, i in enumerate(range(0, n, MB)):
        if si == WARM: torch.cuda.synchronize(); t0 = time.time()
        if si == WARM + NSTEP: torch.cuda.synchronize(); times.append(time.time() - t0)
        if prof and si == WARM + 2:
            from torch.profiler import profile, ProfilerActivity
            _pr = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); _pr.__enter__(); torch.cuda.synchronize(); _pt = time.time()
        if prof and si == WARM + 7:
            torch.cuda.synchronize(); _w = time.time() - _pt; _pr.__exit__(None, None, None)
            ka = _pr.key_averages(); cu = sum(e.self_device_time_total for e in ka) / 1e3
            print("[profile] 5歩 壁時計 %.1f ms、GPU カーネル合計 %.1f ms(GPU 稼働率 %.0f%%)" % (_w * 1e3, cu, cu / (_w * 1e3) * 100), flush=True)
            print(ka.table(sort_by="self_device_time_total", row_limit=25), flush=True)
            print(ka.table(sort_by="self_cpu_time_total", row_limit=15), flush=True)
        # ---- ここから本番の更新ループの1ミニバッチ(apo.py 505〜531行)を写したもの ----
        if CG: cg_upd.step_begin()
        idx = perm[i:i + MB]; b = [D[k][idx] for k in KEYS]
        if ST: s2.wait_stream(torch.cuda.current_stream()); _sc = torch.cuda.stream(s2); _sc.__enter__()
        if ONLY == "ref":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): rd = _evu(ref, b)[3]
            stats.append((0., 0., 0., 0.)); vls.append(0.); continue
        if ONLY == "pol":        # 方策の順逆 + Adam(KL 項は ref が要るので外し、dist は 0 倍で勾配の経路だけ残す)
            with torch.autocast("cuda", dtype=torch.float16): lp, ent, v, dist, per, pmask = _evu(net, b)
            ent, per = ent.float(), per.float(); rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
            pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
            loss = pl + 0.0 * sum(x.float().sum() for x in dist if x.dtype.is_floating_point) - ENT * ent.mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
            stats.append((float(pl), 0., 0., 0.)); vls.append(0.); continue
        do_c = si % CE == 0; idx_, b_ = idx, b
        if CF < 1: idx = idx[:max(1, int(len(idx) * CF))]; b = [x[:len(idx)] for x in b]
        if do_c:
          with torch.autocast("cuda", dtype=torch.float16):
            ob = [DO[k][idx] for k in OKEYS]
            hh, _ = _cencu(*[torch.cat([b[q_], ob[q_]], 0) for q_ in range(4)])
            vv = crit.value(hh[:, -1]).squeeze(-1); nb_ = len(idx)
            vc = 0.5 * (vv[:nb_] - vv[nb_:])
          vl = F.smooth_l1_loss(vc.float(), RT[idx])
          opt_c.zero_grad(); scaler_c.scale(vl).backward()
        else: vl = torch.full((), float("nan"), device=DEV)
        idx, b = idx_, b_
        def _cstep():
            scaler_c.unscale_(opt_c); torch.nn.utils.clip_grad_norm_(cparams, 0.5); scaler_c.step(opt_c); scaler_c.update()
        if ST: _sc.__exit__(None, None, None)
        elif do_c: _cstep()
        if ONLY == "crit": stats.append((0., 0., 0., 0.)); vls.append(float(vl)); continue
        with torch.autocast("cuda", dtype=torch.float16):
            lp, ent, v, dist, per, pmask = _evu(net, b)
        ent, per = ent.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
        rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
        pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            rd = _evu(ref, b)[3]
        rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
        kl = P.kl_to_ref(dist, rd).mean()
        loss = pl + BETA * kl - ENT * ent.mean()
        opt.zero_grad(); scaler.scale(loss).backward()
        if ST:
            with torch.cuda.stream(s2):
                if do_c: _cstep()        # 批評家の Adam(同期 .item() を含む)は方策の逆伝播を積んだあとに
            torch.cuda.current_stream().wait_stream(s2)
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
        if NOSYNC:
            stats.append(cg_upd.stat4(pl, kl, ent, rr, pmask, CLIP)); vls.append(vl.detach())
        else:
            stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean()))); vls.append(float(vl))
    torch.cuda.synchronize()
    if NOSYNC: stats = cg_upd.fetch(stats); vls = torch.stack(vls).double().tolist()
    st = np.array(stats, np.float64); vls = np.array(vls, np.float64)
    ms = times[0] / NSTEP * 1e3
    print("[%s] 1歩 %.1f ms(%d歩)→ 1反復(337歩)%.0f 秒 | 最大メモリ %.1fGB | 平均 KL %.4f ent %.3f 切 %.3f | scale %g/%g" % (
        label, ms, NSTEP, ms * 337 / 1e3, torch.cuda.max_memory_allocated() / 2 ** 30, st[:, 1].mean(), st[:, 2].mean(), st[:, 3].mean(), scaler.get_scale(), scaler_c.get_scale()), flush=True)
    q1 = crit_quality(crit)
    print("  [%s] 批評家(学習外 %d 局面): smooth_l1 %.5f → %.5f、説明率(ret) %.4f → %.4f、説明率(勝敗) %.4f → %.4f | 学習中の価値損失 平均 %.5f" % (
        label, len(RTH), q0[0], q1[0], q0[1], q1[1], q0[2], q1[2], np.nanmean(vls)), flush=True)
    out = dict(ms=ms, q1=q1, st=st, vl=vls, net={k: v.detach().float().clone() for k, v in net.state_dict().items()},
               crit={k: v.detach().float().clone() for k, v in crit.state_dict().items()})
    del net, crit, ref, opt, opt_c; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return out


def cmp(a, b, na, nb):
    dfs = np.nonzero((np.abs(a["st"][:, :3] - b["st"][:, :3]).max(1) > 0) | (a["vl"] != b["vl"]))[0]
    print("[比較 %s 対 %s] 損失が最初に食い違う歩: %s(全 %d 歩)| 1歩目の差: 方策 %.3e KL %.3e ent %.3e 価値 %.3e" % (
        na, nb, dfs[0] if len(dfs) else "なし", len(a["vl"]), *np.abs(a["st"][0, :3] - b["st"][0, :3]), abs(a["vl"][0] - b["vl"][0])), flush=True)
    dp = max(float((a["net"][k] - b["net"][k]).abs().max()) for k in a["net"])
    dc = max(float((a["crit"][k] - b["crit"][k]).abs().max()) for k in a["crit"])
    bit = all(torch.equal(a["net"][k], b["net"][k]) for k in a["net"]) and all(torch.equal(a["crit"][k], b["crit"][k]) for k in a["crit"])
    ds = np.abs(a["st"] - b["st"]).max(0); dv = np.abs(a["vl"] - b["vl"]).max()
    same_loss = np.array_equal(a["st"][:, :3], b["st"][:, :3]) and np.array_equal(a["vl"], b["vl"])
    print("[比較 %s 対 %s] 重みの差 最大: 方策 %.3e 批評家 %.3e(ビット一致 %s)| 損失の差 最大: 方策 %.3e KL %.3e ent %.3e 切 %.3e 価値 %.3e(損失ビット一致 %s)| 速さ %.2f倍" % (
        na, nb, dp, dc, bit, ds[0], ds[1], ds[2], ds[3], dv, same_loss, a["ms"] / b["ms"]), flush=True)


print("torch", torch.__version__, torch.cuda.get_device_name(), flush=True)
R = {}
for mode in os.environ.get("RUNS", "base,cg,base2").split(","):
    tc = time.time(); R[mode] = run(mode.rstrip("0123456789") if mode.startswith("base") else mode, mode); R[mode]["wall"] = time.time() - tc
    print("  (%s は compile こみで %.0f 秒)" % (mode, R[mode]["wall"]), flush=True)
names = [x for x in R if not any(t_[:2] in ("ce", "cf", "cl") for t_ in x.split("+"))]
for x in R:
    if x not in names and names: print("[%s] 速さ %.3f倍(対 %s)| 学習外の smooth_l1 %.5f(%s は %.5f)説明率(ret) %.4f(%.4f)" % (
        x, R[names[0]]["ms"] / R[x]["ms"], names[0], R[x]["q1"][0], names[0], R[names[0]]["q1"][0], R[x]["q1"][1], R[names[0]]["q1"][1]), flush=True)
for j in range(1, len(names)): cmp(R[names[0]], R[names[j]], names[0], names[j])
for x in names[2:]:
    if "base2" in R and x != "base2": cmp(R["base2"], R[x], "base2", x)
