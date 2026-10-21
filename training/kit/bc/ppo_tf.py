"""BC で初期化した transformer 方策を PPO で仕上げる。

BC の限界(教師の軌道を1手外れると未学習の盤面に入り崩れる)を、自分の軌道で学ぶことで直す。
  環境   : 相手は学習用対局に出てきた強い選手の手をそのまま再生。K体 x Sシード を並列(VecMarket を相手ごとに1つ)
  行動   : ユニットは合法手(エンジン判定)から標本、市場は自己回帰で注文列を標本(数量も標本)
  報酬   : 毎手 = (自分-相手 の所持金差の増分)/SCALE、最後に勝ち負けで ±WIN。差分の和は最終差に一致する
  安定化 : BC 方策への KL 罰則(ポケカで3連続成功した klanchor の型)、最初の数回は価値だけ学習
  評価   : 未見の対局の相手に対して貪欲に打たせ、コイン・勝率を記録
"""
import os
import json, glob, os, sys, time, random, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim, kagpack
import model_tf as M
from extract4 import codes, NU, NM, NS, M_STOP
from deploy import wide_blank
DEV = os.environ.get("PPODEV", "cuda")
FASTINF = os.environ.get("FASTINF", "0") == "1"      # exp280: 結果を変えない高速化(ユニットの頭の分解・市場デコーダの逐次キャッシュ・Gumbel 標本)
if FASTINF:
    import fastinfer as _FI
    _FI.install()
CACHE = os.environ.get("CACHE", "/content/local_ext4")
SCALE = float(os.environ.get("SCALE", 50000)); WIN = float(os.environ.get("WIN", 1.0))
GAMMA, LAM, CLIP = 0.999, 0.95, 0.2
FACT = int(os.environ.get("FACT", 0))   # 1: 決定ごとに比を取りクリップする(約30個の決定の積の比だと1つが動くだけで全体が切られる)
BETA = float(os.environ.get("BETA", 0.1)); ENT = float(os.environ.get("ENT", 0.001)); LR = float(os.environ.get("LR", 5e-5))

def load_meta(e):
    z = np.load("%s/%s.npz" % (CACHE, e)); return json.loads(str(z["meta"])), json.loads(str(z["opp_acts"]))

class Pool:
    """相手ごとに VecMarket を1つ(相手のテープは VecMarket 内で共有される)。"""
    def __init__(self, games, seeds_per):
        self.envs = []
        for e, seed_base in games:
            meta, opp = load_meta(e); seat = meta["seat"]
            op = kagpack.pack(opp); w = wide_blank(); A, Bt = (w, op) if seat == 0 else (op, w)
            seeds = [seed_base + 104729 * k for k in range(seeds_per)]
            env = kagsim.VecMarket(A[0], A[1], Bt[0], Bt[1], seeds, 720, seat); env.keep_nonsell = False; env.reset()
            self.envs.append(env)
        self.S = seeds_per

    def state(self):
        T, OU, PU, G, L = [], [], [], [], []
        for env in self.envs:
            tiles = np.stack([np.concatenate([np.asarray(env.tiles_base(s)), np.asarray(env.tiles_dyn(s))], 3).reshape(-1, 100, 13) for s in (0, 1)], 1)
            up = np.asarray(env.unit_pos2(0))[:, :NU]; op = np.asarray(env.unit_pos2(1))[:, :NU]; inv = np.asarray(env.unit_inv(NU))
            OU.append(np.concatenate([(up[..., :1] >= 0), up, inv], 2).astype(np.float32))
            PU.append(np.concatenate([(op[..., :1] >= 0), op], 2).astype(np.float32))
            G.append(np.concatenate([np.asarray(env.obs_legal()), np.asarray(env.farm_pub(0)), np.asarray(env.farm_pub(1)),
                                     np.asarray(env.shops()).astype(np.float32), np.asarray(env.shed_all())], 1).astype(np.float32))
            T.append(tiles.astype(np.float16)); L.append(np.asarray(env.legal_mask(NU)))
        c = np.concatenate
        return c(T), c(OU), c(PU), c(G), c(L)

    def margin(self):
        m = np.concatenate([np.asarray(env.money()) for env in self.envs]); return m[:, 0] - m[:, 1], m[:, 0], m[:, 1]

    def step(self, U, O):
        k = 0
        for env in self.envs:
            env.step_units(U[k:k+self.S], O[k:k+self.S]); k += self.S

def tt(a, dt=torch.float32): return torch.as_tensor(a, device=DEV, dtype=dt)

def _gumbel_argmax(logits):
    """Categorical(logits).sample() と同じ分布。-inf や fp16 の最小値の行も扱えるよう float で計算する。"""
    l = logits.float()
    u = torch.rand_like(l).clamp_(1e-10, 1.0 - 1e-10)
    return (l - torch.log(-torch.log(u))).argmax(-1)


@torch.no_grad()
def sample(net, st, greedy=False, crit=None, cap=False, temp=1.0):
    # 09-21: temp>1 で「下の確率の手」も引く(探索の幅)。重要度比が狂わないよう、行動の対数確率(PER0)も
    #        同じ温度で測ること(apo.py が P.evaluate(..., temp=同じ値) を呼ぶ)。貪欲のときは温度は効かない。
    # exp270: 市場の先読み特徴つきのときは st が6つ(…, mkt)になる
    if len(st) == 6: tiles, ou, pu, g, lm, mk = st
    else: tiles, ou, pu, g, lm = st; mk = None
    B = len(g)
    _kw = {"mkt": tt(mk)} if (mk is not None and hasattr(net, "mnum")) else {}   # 素の Policy(相手・tape 経路)には渡さない
    h, pad = net.encode(tt(tiles, torch.float16), tt(ou), tt(pu), tt(g), **_kw)
    _lmt = tt(lm, torch.uint8)
    lg, q = net.unit_logits(h, tt(ou), _lmt)
    if M.MIG: lg = M.mig_apply(lg, _lmt)        # 09-23 マスク移行: 採取側の罰。FASTINF で unit_logits が差し替わっても効く
    live = tt(ou[..., 0] > 0, torch.bool)
    if cap:
        # 植え付けが種の数を超えるとその作物の植え付けが全部無効になる(エンジン規則)。deploy の act と同じく順に割り当てる
        seeds = torch.round(tt(g[:, 27:32]) * 20).long()
        if FASTINF:                      # exp280: 20人の逐次ループをまとめて計算(生きているユニットの結果は逐次版と完全一致)
            z = lg.float() / temp
            if not greedy:
                u_ = torch.rand_like(z).clamp_(1e-10, 1 - 1e-10); z = z - torch.log(-torch.log(u_))
            ul = _FI.choose_capped(z, live, seeds)
        else:
          used = torch.zeros_like(seeds); cols = []
          for u in range(lg.shape[1]):
              l = lg[:, u].clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
              c = l.argmax(-1) if greedy else torch.distributions.Categorical(logits=l / temp).sample()
              pl = (c >= 15) & (c < 20) & live[:, u]
              used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
          ul = torch.stack(cols, 1)
    else:
        ul = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg / temp).sample()
    ul = torch.where(live, ul, torch.full_like(ul, -1))
    ql = net.unit_qty(q, ul); uq = ql.argmax(-1) if greedy else (_gumbel_argmax(ql / temp) if FASTINF else torch.distributions.Categorical(logits=ql / temp).sample())
    uq = torch.where(ul >= 20, uq, torch.zeros_like(uq))
    prev = torch.full((B, 1), NM, device=DEV, dtype=torch.long); done = torch.zeros(B, dtype=torch.bool, device=DEV)
    ms = torch.full((B, NS), M_STOP, device=DEV, dtype=torch.long); mq = torch.zeros((B, NS), device=DEV, dtype=torch.long)
    _mc = _FI.MarketCache(net, h, pad) if (FASTINF and type(net).market is M.Policy.market) else None
    _tok = prev[:, 0]
    for j in range(NS):
        if _mc is not None:
            lj, y = _mc.step(_tok)
        else:
            mo, y = net.market(h, pad, prev); lj = mo[:, -1]
        # 11枠目は終了のみ。fp16 では -1e9 が表せない(half に変換できず落ちる)ので、その型で表せる最小値を使う
        if j == NS - 1: lj = torch.full_like(lj, torch.finfo(lj.dtype).min); lj[:, M_STOP] = 0
        cj = lj.argmax(-1) if greedy else (_gumbel_argmax(lj / temp) if FASTINF else torch.distributions.Categorical(logits=lj / temp).sample())
        cj = torch.where(done, torch.full_like(cj, M_STOP), cj); ms[:, j] = cj
        qj = net.market_qty(y[:, -1:], cj.view(-1, 1))[:, 0]
        qc = qj.argmax(-1) if greedy else (_gumbel_argmax(qj / temp) if FASTINF else torch.distributions.Categorical(logits=qj / temp).sample())
        mq[:, j] = torch.where((cj < 19) & ~done, qc, torch.zeros_like(qc))
        done = done | (cj == M_STOP); _tok = cj.clamp(max=NM)
        if _mc is None: prev = torch.cat([prev, _tok.view(-1, 1)], 1)
        if bool(done.all()): break
    if crit is not None:
        hc, _ = crit.encode(tt(tiles, torch.float16), tt(ou), tt(pu), tt(g)); v = crit.value(hc[:, -1]).squeeze(-1)
    else:
        v = net.value(h[:, -1]).squeeze(-1)
    if FASTINF:      # exp280: 取り出しを1回にまとめる(同期の回数を減らす)
        pack = torch.cat([ul.view(B, -1), uq.view(B, -1), ms.view(B, -1), mq.view(B, -1)], 1).to("cpu", non_blocking=False).numpy()
        nu = ul.shape[1]; nq = uq.shape[1]; nm = ms.shape[1]
        return pack[:, :nu], pack[:, nu:nu + nq], pack[:, nu + nq:nu + nq + nm], pack[:, nu + nq + nm:], v.float().cpu().numpy()
    return ul.cpu().numpy(), uq.cpu().numpy(), ms.cpu().numpy(), mq.cpu().numpy(), v.float().cpu().numpy()

def to_codes(ul, uq, ms, mq, ou):
    U = np.zeros((len(ul), NU, 3), np.int32); O = np.zeros((len(ul), 10, 3), np.int32)
    for b in range(len(ul)):
        live = [u for u in range(NU) if ou[b, u, 0] > 0]
        U[b], O[b] = codes(ul[b], uq[b], live, ms[b], mq[b])
    return U, O

def evaluate(net, batch, legal_live_only=True, temp=1.0):
    """保存した行動の対数確率・エントロピー・価値と、KL 用の分布を返す(教師強制)。
    temp≠1 は「実際に手を選んだときの分布」で測るため(対局係が PER0 を作るときだけ使う。学習係は既定の 1.0)。"""
    if len(batch) == 10: tiles, ou, pu, g, lm, mk, ul, uq, ms, mq = batch
    else: tiles, ou, pu, g, lm, ul, uq, ms, mq = batch; mk = None
    h, pad = net.encode(tiles, ou, pu, g, **({"mkt": mk} if (mk is not None and hasattr(net, "mnum")) else {}))
    lg, q = net.unit_logits(h, ou, lm)
    if M.MIG: lg = M.mig_apply(lg, lm)          # 09-23 マスク移行: 学習側の罰。**採取と必ず同じ式・同じ lm**
    live = ul >= 0
    lsu = F.log_softmax(lg / temp, -1)
    lp = (lsu.gather(-1, ul.clamp(min=0).unsqueeze(-1)).squeeze(-1) * live).sum(1)
    ent = -(lsu.exp() * lsu.masked_fill(lsu < -1e3, 0)).sum(-1)
    ent = (ent * live).sum(1)
    lsq = F.log_softmax(net.unit_qty(q, ul) / temp, -1)
    qm = live & (ul >= 20)
    lp = lp + (lsq.gather(-1, uq.unsqueeze(-1)).squeeze(-1) * qm).sum(1)
    prev = torch.cat([torch.full_like(ms[:, :1], NM), ms[:, :-1]], 1).clamp(max=NM)
    mo, y = net.market(h, pad, prev)
    last = torch.full_like(mo[:, NS - 1], -1e4); last[:, M_STOP] = 0            # 標本時と同じく11枠目は終了のみ
    mo = torch.cat([mo[:, :NS - 1], last.unsqueeze(1)], 1)
    first = torch.where(ms == M_STOP, torch.arange(NS, device=DEV).expand_as(ms), torch.full_like(ms, NS)).min(1).values
    valid = torch.arange(NS, device=DEV).unsqueeze(0) <= first.unsqueeze(1)
    lsm = F.log_softmax(mo / temp, -1)
    lp = lp + (lsm.gather(-1, ms.unsqueeze(-1)).squeeze(-1) * valid).sum(1)
    lsmq = F.log_softmax(net.market_qty(y, ms) / temp, -1)
    mqm = valid & (ms < 19)
    lp = lp + (lsmq.gather(-1, mq.unsqueeze(-1)).squeeze(-1) * mqm).sum(1)
    v = net.value(h[:, -1]).squeeze(-1)
    # 決定ごとの対数確率(ユニット20 + 市場11)。分解した PPO で使う
    lpu = lsu.gather(-1, ul.clamp(min=0).unsqueeze(-1)).squeeze(-1) + lsq.gather(-1, uq.unsqueeze(-1)).squeeze(-1) * qm
    lpm = lsm.gather(-1, ms.unsqueeze(-1)).squeeze(-1) + lsmq.gather(-1, mq.unsqueeze(-1)).squeeze(-1) * mqm
    per = torch.cat([lpu, lpm], 1); pmask = torch.cat([live, valid], 1)
    return lp, ent, v, (lsu, live, lsm, valid), per, pmask

def kl_to_ref(cur, ref):
    lsu, live, lsm, valid = cur; rsu, _, rsm, _ = ref
    pu = lsu.exp(); ku = (pu * (lsu - rsu).masked_fill(lsu < -1e3, 0)).sum(-1)
    pm = lsm.exp(); km = (pm * (lsm - rsm)).sum(-1)
    return (ku * live).sum(1) + (km * valid).sum(1)

def eval_policy(net, eids, greedy, reps=1, cap=False, seed=0):
    """未見の相手に対する成績。reps>1 は同じ相手をシードを変えて複数回(シードは毎回同じなので評価どうしで比べられる)。"""
    pool = Pool([(e, load_meta(e)[0]["seed"]) for e in eids], reps); torch.manual_seed(seed)
    for t in range(719):
        st = pool.state(); ul, uq, ms, mq, _ = sample(net, st, greedy=greedy, cap=cap)
        U, O = to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O)
    mg, me, op = pool.margin()
    return float(me.mean()), float((mg > 0).mean()), float(mg.mean())

def greedy_eval(net, eids):
    return eval_policy(net, eids, True)

def main():
    tag = sys.argv[1]; init = sys.argv[2]
    K = int(os.environ.get("K", 16)); S = int(os.environ.get("S", 4)); ITERS = int(os.environ.get("ITERS", 200)); VWARM = int(os.environ.get("VWARM", 2))
    sp = os.environ.get("SPLITFILE", "/content/local_logs/tf_%s.json" % init)
    R0 = json.load(open(sp)); train_e, hold_e = R0["train"], R0["hold"]
    logf = open("/content/local_logs/ppo_%s.log" % tag, "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    net = M.Policy().to(DEV); ref = M.Policy().to(DEV)
    sd = torch.load("/content/local_logs/tf_%s.pt" % init, map_location=DEV); net.load_state_dict(sd); ref.load_state_dict(sd); ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    # 評価役(価値)は別ネットワーク。共有の本体に価値の勾配が流れると方策が壊れた(p2: 価値のみ5回で 49k→25k)
    crit = M.Policy().to(DEV); crit.load_state_dict(sd); crit.eval()
    opt_c = torch.optim.Adam(crit.parameters(), lr=float(os.environ.get("LRC", 1e-4)), eps=1e-5)
    net.eval()   # ドロップアウトを切る。有効なままだと同じ手の確率が毎回変わり、更新前から比 r が 1 から大きくずれた(|r-1|=1.18)
    ck = "/content/local_logs/ppo_%s.pt" % tag; start = 0
    dck = "/content/drive/MyDrive/kaggriculture/runs/ppo_%s_ck.pt" % tag
    if not os.path.exists(ck) and os.path.exists(dck):
        import shutil; shutil.copy(dck, ck)       # 再起動後は Drive の控えから再開
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda")     # 更新を fp16 混合精度で(T4 で約2倍速)
    scaler_c = torch.amp.GradScaler("cuda")   # 評価役用(1つを共有すると1回の更新で調整が2回走る)
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); start = c["it"] + 1; log("再開 it%d" % start)
        if "crit" in c: crit.load_state_dict(c["crit"]); opt_c.load_state_dict(c["opt_c"])
        for g_ in opt.param_groups: g_["lr"] = LR          # 再開時は保存された学習率ではなく指定の値を使う
        log("学習率 %.1e" % LR)
    evset = hold_e[:8]
    if start == 0:
        me, wr, mg = eval_policy(net, hold_e, True, 2, cap=True); log("it  -1 [評価 未見29相手x2 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f (BC のまま)" % (me, wr, mg))
    rng = random.Random(1)
    for it in range(start, ITERS):
        t0 = time.time()
        games = [(e, rng.randrange(10**9)) for e in rng.sample(train_e, K)]
        pool = Pool(games, S); N = K * S
        buf = {k: [] for k in ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq", "v", "r")}
        mg_prev, _, _ = pool.margin()
        for t in range(719):
            st = pool.state(); ul, uq, ms, mq, v = sample(net, st, crit=crit)
            U, O = to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O)
            mg, me, op = pool.margin(); r = (mg - mg_prev) / SCALE; mg_prev = mg
            if t == 718: r = r + WIN * np.sign(mg)
            for k, a in zip(("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq", "v", "r"), (st[0], st[1], st[2], st[3], st[4], ul, uq, ms, mq, v, r)):
                buf[k].append(a)
        tr = time.time() - t0
        V = np.stack(buf["v"]); Rw = np.stack(buf["r"]); T_ = len(Rw)
        adv = np.zeros_like(Rw); last = np.zeros(N)
        for t in reversed(range(T_)):
            nv = V[t + 1] if t + 1 < T_ else np.zeros(N)
            delta = Rw[t] + GAMMA * nv - V[t]; last = delta + GAMMA * LAM * last; adv[t] = last
        ret = adv + V
        flat = lambda k, dt: torch.as_tensor(np.concatenate(buf[k]) if buf[k][0].ndim > 1 else np.stack(buf[k]).reshape(-1), device=DEV, dtype=dt)
        D = dict(tiles=flat("tiles", torch.float16), ou=flat("ou", torch.float32), pu=flat("pu", torch.float32), g=flat("g", torch.float32),
                 lm=flat("lm", torch.uint8), ul=flat("ul", torch.long), uq=flat("uq", torch.long), ms=flat("ms", torch.long), mq=flat("mq", torch.long))
        A = torch.as_tensor(adv.reshape(-1), device=DEV, dtype=torch.float32); RT = torch.as_tensor(ret.reshape(-1), device=DEV, dtype=torch.float32)
        A = (A - A.mean()) / (A.std() + 1e-8)
        n = len(A); mb = int(os.environ.get("MB", 256))   # 1024 では注意機構の行列で GPU が溢れた
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outs = [evaluate(net, [D[k][i:i+mb] for k in ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")]) for i in range(0, n, mb)]
            LP0 = torch.cat([o[0].float() for o in outs]); PER0 = torch.cat([o[4].float() for o in outs]); del outs
        vonly = it < VWARM; stats = []
        for ep in range(2):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, mb):
                idx = perm[i:i+mb]; b = [D[k][idx] for k in ("tiles", "ou", "pu", "g", "lm", "ul", "uq", "ms", "mq")]
                with torch.autocast("cuda", dtype=torch.float16):
                    lp, ent, v, dist, per, pmask = evaluate(net, b)
                lp, ent, v, per = lp.float(), ent.float(), v.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
                with torch.autocast("cuda", dtype=torch.float16):
                    hc, _ = crit.encode(b[0], b[1], b[2], b[3]); vc = crit.value(hc[:, -1]).squeeze(-1)
                vl = F.smooth_l1_loss(vc.float(), RT[idx])
                opt_c.zero_grad(); scaler_c.scale(vl).backward(); scaler_c.unscale_(opt_c)
                torch.nn.utils.clip_grad_norm_(crit.parameters(), 0.5); scaler_c.step(opt_c); scaler_c.update()
                if vonly: continue
                else:
                    a = A[idx]
                    if FACT:
                        r = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = a.unsqueeze(1)
                        term = torch.min(r * aa, r.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask
                        pl = -(term.sum(1) / pmask.sum(1).clamp(min=1)).mean()
                        ratio = r[pmask.bool()]; clipfrac = float(((r - 1).abs() > CLIP)[pmask.bool()].float().mean())
                    else:
                        ratio = torch.exp(lp - LP0[idx])
                        pl = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
                        clipfrac = float(((ratio - 1).abs() > CLIP).float().mean())
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        rd = evaluate(ref, b)[3]
                    rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
                    kl = kl_to_ref(dist, rd).mean()
                    loss = pl + BETA * kl - ENT * ent.mean()     # 価値の損失は方策に流さない
                    stats.append((float(pl), float(kl), float(ent.mean()), float((ratio - 1).abs().mean()), clipfrac))
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
        mgf, mef, opf = pool.margin()
        ev_ = 1 - np.var(ret.reshape(-1) - V.reshape(-1)) / (np.var(ret.reshape(-1)) + 1e-8)
        s_ = np.mean(stats, 0) if stats else [0, 0, 0, 0, 0]
        log("it%4d 自分 %8.0f 相手 %8.0f 差 %9.0f 勝率 %.3f | 方策 %.3f KL %.3f ent %.2f |r-1| %.3f 切 %.2f 価値EV %.2f %s| 収集 %.0fs 全体 %.0fs"
            % (it, mef.mean(), opf.mean(), mgf.mean(), (mgf > 0).mean(), s_[0], s_[1], s_[2], s_[3], s_[4], ev_, "価値のみ " if vonly else "", tr, time.time() - t0))
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), crit=crit.state_dict(), opt_c=opt_c.state_dict(), it=it), ck)
        try:   # VM が再起動するとローカルが消えるので Drive にも毎回控える
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), crit=crit.state_dict(), opt_c=opt_c.state_dict(), it=it), "/content/drive/MyDrive/kaggriculture/runs/ppo_%s_ck.pt" % tag)
        except Exception as ex: log("  Drive への保存失敗: %s" % str(ex)[:60])
        if it % 10 == 9:
            # 16局の評価は同じ方策でも ±7,000 ぶれた(BC で 52.5k/52.4k/45.1k)。未見29局x2シード=58局で、差を主に見る
            me, wr, mg = eval_policy(net, hold_e, False, 2); log("it%4d [評価 未見29相手x2 確率的] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))
            me, wr, mg = eval_policy(net, hold_e, True, 2, cap=True); log("it%4d [評価 未見29相手x2 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))
            try: torch.save(net.state_dict(), "/content/drive/MyDrive/kaggriculture/runs/ppo_%s_it%d.pt" % (tag, it))
            except Exception: pass

if __name__ == "__main__":
    main()
