"""fast_league.py — kagg/bc/league.py の速い版(同じ JSON を書く)。
  league.py は Rust arena(batch2)の観測辞書 → 各エージェントの obs_feat.featurize(Python、1観測ずつ)→ greedy_batch → to_action → JSON。
  ここは PPO の対局係と同じ C++ エンジン kagsim.VecMarket(全局を1本に入れる = sp_fast.BatchSPPool と同じ作り)から
  特徴をまとめて取り、推論は league.py と**同じ関数**(rust_eval2.greedy_batch / greedy_batch_ens)を同じ行の並び(局の番号順)で呼ぶ。
  行動は to_codes(fast_codes.to_codes_fast と同じ表引き。ただし番兵 999/1000 の数量まで表を広げた)。main.py の後処理(_pace / _land)は
  league.py と同じ順で呼ぶ(番兵の行動で「何もしない」と分かった局は省く。変える局だけ本物の to_action の辞書に掛けてから符号化)。
  エージェントの読み込みは league.BC をそのまま使う(フォルダごとの model_ptr_infer / obs_feat、切り替え型の a.* / b.* も同じ)。
  合法手の遮蔽は各フォルダの obs_feat に合わせる: NMB_MARK があれば 旧(版0)|新(版2)で増えた手に印 3、無ければ LEGAL_OPT の版。

  使い方(コンテナの中、cwd はどこでも):
    AGENTS="a:bc:/content/lab/agents/X,b:bc:/content/lab/agents/Y" NSEED=192 SEED0=880001 python3 fast_league.py 出力.json
  出力の鍵は league.py と同じ: "a|b" → [[A の所持金, B の所持金], ...]、"a|b#shops"、"a|b#money300"、"a|b#money500"、"a|b#seats"。
"""
import os, sys, json, time, atexit, itertools
os.environ.pop("LEGAL_OPT", None)          # league.py と同じ(版は各エージェントの既定で決める。kagsim も環境変数を読むので import 前に外す)
BC_DIR = os.environ.get("KAGG_BC", "/content/drive/MyDrive/kaggriculture/bc")
FAST_DIR = os.path.join(os.path.dirname(BC_DIR), "fast")
for p in (FAST_DIR, BC_DIR):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, torch
import league as LG                          # BC(読み込み)、sanitize、greedy_batch
import kagsim, kagpack, fast_codes
from deploy import wide_blank
NU = 20
IT = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"]
CROP = IT[:5]
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
MOPI = kagpack.MOPI; OPI = kagpack.OPI
_EMPTY_ROW = np.array([255, 0, 0], np.int32)

# ---- fast_codes.to_codes_fast と同じ表引きだが、数量クラスの表を番兵(999/1000)まで広げたもの。
#   fast_codes の表は MQ_MAX+2 / UQ_MAX+2 クラスまでしか無く、番兵のクラス(MQ 130=999, 131=1000)は 129 に切られて
#   「128 個」になる(to_action は 999/1000)。league.py と同じ手にするため、ここは codec の MQ_N / UQ_N まで作る。
from extract4 import unit_code, mkt_code, NC as _NC, NM as _NM, M_STOP as _M_STOP
from codec import UQ_N, MQ_N
_LUT_U = np.zeros((_NC, UQ_N, 3), np.int32)
for _c in range(_NC):
    for _q in range(UQ_N): _LUT_U[_c, _q] = unit_code(_c, _q)
_LUT_M = np.zeros((_NM, MQ_N, 3), np.int32)
for _c in range(_NM):
    for _q in range(MQ_N): _LUT_M[_c, _q] = (255, 0, 0) if _c == _M_STOP else mkt_code(_c, _q)


def to_codes(ul, uq, ms, mq, ou):
    live = np.asarray(ou)[:, :NU, 0] > 0
    U = _LUT_U[np.clip(ul, 0, _NC - 1), np.clip(uq, 0, UQ_N - 1)]
    U = np.where(live[..., None], U, _EMPTY_ROW)
    m10 = np.asarray(ms)[:, :10]; q10 = np.asarray(mq)[:, :10]
    O = _LUT_M[np.clip(m10, 0, _NM - 1), np.clip(q10, 0, MQ_N - 1)]
    O = np.where((np.cumsum(m10 == _M_STOP, 1) > 0)[..., None], _EMPTY_ROW, O)
    return U.astype(np.int32), O.astype(np.int32)


# ---------------------------------------------------------------- 行動の辞書 → kagsim のコード(フックが変えた局だけ)
def mkt_codes(market):
    """sanitize 済みの market(最大10件)→ (10,3)。空き枠は 255。数量は切り詰めずにそのまま(Rust と同じ)。"""
    O = np.tile(_EMPTY_ROW, (10, 1))
    for j, o in enumerate(market[:10]):
        op = o[0]
        if op in ("HIRE", "BUY_LAND"): O[j] = (MOPI[op], 0, 1); continue
        it, q = o[1], int(o[2])
        if op == "SELL": O[j] = (MOPI["SELL"], IT.index(it), q)
        elif op == "BUY_SEED": O[j] = (MOPI["BUY_SEED"], CROP.index(it), q)
        elif op == "BUY_PRODUCT": O[j] = (MOPI["BUY_PRODUCT"], IT.index(it), q)
        elif op == "BUY_ANIMAL": O[j] = (MOPI["BUY_ANIMAL"], IT.index(it), q)
        else: raise ValueError(o)
    return O


def unit_code_dict(u):
    op = u[0]
    if op == "PLANT": return (OPI["PLANT"], CROP.index(u[1]), 1)
    if op in ("PICKUP", "PLACE"): return (OPI[op], IT.index(u[1]), int(u[2]) if len(u) > 2 else 1)
    return (OPI[op], 0, 1)


_OBS_OK = {"step", "day", "hour", "player", "index", "private"}   # _proxy_obs が持つ鍵(private は shed だけ)
def hooks_supported(folder):
    """main.py の後処理が読む観測の鍵が _proxy_obs にあるものだけか(保守的に main.py 全体の obs["x"] / obs.get("x") を見る)。
    無いもの(例: exp290 の _land は obs["farms"])を読むフォルダは fast では打たない(leagueq_fast は league.py に回す)。"""
    import re
    src = open(os.path.join(folder, "main.py")).read()
    keys = set(re.findall(r'obs(?:\.get\(|\[)\s*["\'](\w+)["\']', src))
    priv = set(re.findall(r'get\(\s*"private"\s*\)\s*or\s*\{\}\)\.get\(\s*"(\w+)"', src)) | set(re.findall(r'\["private"\]\[\s*"(\w+)"', src))
    return keys <= _OBS_OK and priv <= {"shed"}, sorted(keys - _OBS_OK) + sorted(priv - {"shed"})


class Agent:
    def __init__(self, name, folder):
        self.name = name; self.bc = LG.BC(folder); m = self.bc.m
        if getattr(m, "_dump_fire", None) is not None:           # 切り替え型の main.py が終了時にフォルダへ書く記録は書かない(結果には無関係)
            try: atexit.unregister(m._dump_fire)
            except Exception: pass
        assert not self.bc.mkt, "市場16特徴つき(cfg mkt)のエージェントは未対応: %s" % folder
        ok, bad = hooks_supported(folder)
        assert ok, "main.py の後処理が観測の %s を読む(fast の代わりの観測に無い)。league.py で打つこと: %s" % (bad, folder)
        of = m._loc.get("obs_feat")
        self.nmb = bool(getattr(of, "NMB_MARK", False)); self.lopt = int(getattr(of, "LEGAL_OPT", 0))
        self.hooks = [h for h in (self.bc.pace, self.bc.land) if h is not None]
        self.to_action = m.to_action

        self.graphs = {}
        mp = m._loc.get("model_ptr_infer")
        self.use_b = getattr(mp, "use_b", None) if (mp is not None and hasattr(self.bc.net, "a") and hasattr(self.bc.net, "b")) else None

    def infer(self, st):
        b = self.bc
        if GRAPH and len(b.nets) == 1:
            B = len(st[3]); mode = None
            if self.use_b is not None:                # eager の _run と同じ分岐: 全行 A → a だけ、全行 B → b だけ、混在 → 両方
                m = self.use_b(torch.from_numpy(np.ascontiguousarray(st[3]))).numpy()
                mode = "a" if not m.any() else ("b" if m.all() else "mix")
            key = (B, mode)
            if key not in self.graphs:
                net = b.net if mode is None else (b.net.a if mode == "a" else (b.net.b if mode == "b" else b.net))
                self.graphs[key] = GraphedGreedy(net, self.use_b if mode == "mix" else None, st)
            return self.graphs[key](st)
        return LG.greedy_batch(b.net, st) if len(b.nets) == 1 else LG.greedy_batch_ens(b.nets, st)


# ---------------------------------------------------------------- CUDA graph で rust_eval2.greedy_batch を丸ごと再生する
#   同じカーネルを同じ形で呼ぶので値は eager と同じ(check で局ごとの所持金の一致を確かめる)。違いは3つだけ:
#   (1) 市場の11枠を途中で打ち切らない(終わった行は M_STOP / 0 に固定されるので出力は同じ)
#   (2) 切り替え型(a.*/b.*)は a と b を常に両方計算して use_b(g) で行を選ぶ(eager も混在時はそうする。全行 A のときの
#       a だけの計算と行ごとに同じ値)。FIRE の数えは同期になるので呼ばない
#   (3) TransformerDecoder の因果マスク判定(GPU→CPU 同期)を、eager と同じ結論(常に True)で置き換える
GRAPH = os.environ.get("FAST_GRAPH", "1") == "1"
FAST_INF = os.environ.get("FAST_INF", "0") == "1"   # 1 = 市場デコーダを逐次キャッシュ(fp16 の丸め程度の差が出る。既定は切 = league.py と同じ値)
if FAST_INF: import fastinfer as _FI
import torch.nn.modules.transformer as _TR
_detect_orig = _TR._detect_is_causal_mask
def _detect_const(mask, is_causal=None, size=None):
    if is_causal is None and mask is not None: return True      # market() の causal はいつも triu(…,1) = 真の因果マスク
    return is_causal is True


class _SwitchNoSync:
    def __init__(self, net, use_b): self.net, self.use_b = net, use_b
    def _run(self, name, *args):
        ra = getattr(self.net.a, name)(*args); rb = getattr(self.net.b, name)(*args); m = self._m
        def mix(x, y):
            if not torch.is_tensor(x): return x
            return torch.where(m.view(-1, *([1] * (x.dim() - 1))).to(x.device), y, x)
        if isinstance(ra, tuple): return tuple(mix(x, y) for x, y in zip(ra, rb))
        return mix(ra, rb)
    def encode(self, tiles, ou, pu, g): self._m = self.use_b(g); return self._run("encode", tiles, ou, pu, g)
    def unit_logits(self, h, ou, legal): return self._run("unit_logits", h, ou, legal)
    def unit_qty(self, q, cidx): return self._run("unit_qty", q, cidx)
    def market(self, h, pad, prev): return self._run("market", h, pad, prev)
    def market_qty(self, y, cidx): return self._run("market_qty", y, cidx)


_POOL = [None]


class GraphedGreedy:
    NM, NS, M_STOP = 23, 11, 22
    def __init__(self, net, use_b, st):
        dev = "cuda"
        self.net = _SwitchNoSync(net, use_b) if use_b is not None else net
        self.x = [torch.as_tensor(st[0], device=dev, dtype=torch.float16), torch.as_tensor(st[1], device=dev, dtype=torch.float32),
                  torch.as_tensor(st[2], device=dev, dtype=torch.float32), torch.as_tensor(st[3], device=dev, dtype=torch.float32),
                  torch.as_tensor(st[4], device=dev, dtype=torch.uint8)]
        self.h = [torch.empty(x.shape, dtype=x.dtype).pin_memory() for x in self.x]
        _TR._detect_is_causal_mask = _detect_const
        try:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2): self._body()
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            if _POOL[0] is None: _POOL[0] = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(self.g, pool=_POOL[0]): self.out = self._body()   # 全 graph で作業領域を共有(再生は1つずつ、出力はすぐ CPU へ写す)
        finally:
            _TR._detect_is_causal_mask = _detect_orig

    @torch.no_grad()
    def _body(self):
        net = self.net; tiles, ou, pu, g, lm = self.x; B = g.shape[0]; dev = g.device
        NM, NS, M_STOP = self.NM, self.NS, self.M_STOP
        with torch.autocast("cuda", dtype=torch.float16):
            h, pad = net.encode(tiles, ou, pu, g)
            lg, q = net.unit_logits(h, ou, lm)
            live = ou[..., 0] > 0
            seeds = torch.round(g[:, 27:32] * 20).long(); used = torch.zeros_like(seeds); cols = []
            for u in range(lg.shape[1]):
                l = lg[:, u].float().clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
                c = l.argmax(-1); pl = (c >= 15) & (c < 20) & live[:, u]
                used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
            ul = torch.stack(cols, 1); ul = torch.where(live, ul, torch.full_like(ul, -1))
            uq = net.unit_qty(q, ul).argmax(-1); uq = torch.where(ul >= 20, uq, torch.zeros_like(uq))
            prev = torch.full((B, 1), NM, device=dev, dtype=torch.long); done = torch.zeros(B, dtype=torch.bool, device=dev)
            ms = torch.full((B, NS), M_STOP, device=dev, dtype=torch.long); mq = torch.zeros((B, NS), device=dev, dtype=torch.long)
            if FAST_INF:                               # 近似モード: fastinfer.MarketCache(記憶の K/V を1回だけ射影、1枠ずつ進める)
                subs = [self.net.net.a, self.net.net.b] if isinstance(self.net, _SwitchNoSync) else [net]
                caches = [_FI.MarketCache(sn, h, pad) for sn in subs]; tok = prev[:, 0]
            for j in range(NS):
                if FAST_INF:
                    outs = [c.step(tok) for c in caches]
                    if len(outs) == 1: lj, y = outs[0]
                    else:
                        mm = self.net._m
                        lj = torch.where(mm[:, None], outs[1][0], outs[0][0]); y = torch.where(mm[:, None, None], outs[1][1], outs[0][1])
                    mo = lj[:, None]
                else:
                    mo, y = net.market(h, pad, prev)
                lj = mo[:, -1].float()
                if j == NS - 1: lj = torch.full_like(lj, torch.finfo(lj.dtype).min); lj[:, M_STOP] = 0
                cj = lj.argmax(-1); cj = torch.where(done, torch.full_like(cj, M_STOP), cj); ms[:, j] = cj
                qc = net.market_qty(y[:, -1:], cj.view(-1, 1))[:, 0].argmax(-1)
                mq[:, j] = torch.where((cj < 19) & ~done, qc, torch.zeros_like(qc))
                done = done | (cj == M_STOP); prev = torch.cat([prev, cj.clamp(max=NM).view(-1, 1)], 1); tok = cj.clamp(max=NM)
        return torch.cat([ul, uq, ms, mq], 1)

    def __call__(self, st):
        for k in range(5):
            self.h[k].numpy()[...] = st[k]
            self.x[k].copy_(self.h[k], non_blocking=True)
        self.g.replay()
        o = self.out.cpu().numpy(); nu = 20
        return o[:, :nu], o[:, nu:2 * nu], o[:, 2 * nu:2 * nu + self.NS], o[:, 2 * nu + self.NS:]


class Pool:
    """sp_fast.BatchSPPool.state と同じ特徴(MKTNF なし)。合法手は版0/版2 を両方取り、エージェントごとに組む。"""
    def __init__(self, seeds):
        w = wide_blank()
        self.env = kagsim.VecMarket(w[0], w[1], w[0], w[1], [int(s) for s in seeds], 720, 0)
        self.env.keep_nonsell = False; self.env.reset(); self.n = len(seeds)

    def state(self, seat, need):
        e = self.env; e.set_view(int(seat))
        tiles = np.stack([np.concatenate([np.asarray(e.tiles_base(s)), np.asarray(e.tiles_dyn(s))], 3).reshape(-1, 100, 13) for s in (0, 1)], 1)
        up = np.asarray(e.unit_pos2(0))[:, :NU]; op = np.asarray(e.unit_pos2(1))[:, :NU]; inv = np.asarray(e.unit_inv(NU))
        OU = np.concatenate([(up[..., :1] >= 0), up, inv], 2).astype(np.float32)
        PU = np.concatenate([(op[..., :1] >= 0), op], 2).astype(np.float32)
        G = np.concatenate([np.asarray(e.obs_legal()), np.asarray(e.farm_pub(0)), np.asarray(e.farm_pub(1)),
                            np.asarray(e.shops()).astype(np.float32), np.asarray(e.shed_all())], 1).astype(np.float32)
        L = {v: np.asarray(e.legal_mask(NU, np.full(self.n, v, np.int32))).astype(np.uint8, copy=False) for v in need}
        return tiles.astype(np.float16), OU, PU, G, L

    def money(self):
        self.env.set_view(0); return np.asarray(self.env.money()).copy()


def agent_lm(ag, L):
    if ag.nmb:
        L0, L2 = L[0], L[2]; xp = (L2 != 0) & (L0 == 0)
        return ((L0 != 0) | xp).astype(np.uint8) | (xp.astype(np.uint8) * np.uint8(2))
    return L[ag.lopt]


def _proxy_obs(t, seat, g):
    sa = np.rint(g[-12:]).astype(int)
    return {"step": t, "day": t // 24, "hour": t % 24, "player": seat, "index": seat,
            "private": {"shed": {IT[k]: int(sa[k]) for k in range(12) if sa[k]}}}


def act(ag, st, idx, seats_row, t, sentinel_cache):
    """ag が打つ局 idx(局の番号順)の行動コード (U, O)。st は ag の席から見た特徴(行 = idx)。"""
    ul, uq, ms, mq = ag.infer(st[:5])
    U, O = to_codes(ul, uq, ms, mq, st[1])
    if ag.hooks:
        for j in range(len(idx)):
            o = _proxy_obs(t, int(seats_row[j]), st[3][j])
            s0 = {"farmer": ["PASS"], "hands": [], "market": []}; r = s0
            for h in ag.hooks: r = h(r, o)
            if r is s0: continue                                # 番兵をそのまま返す = この手・この局では何もしない
            a0 = ag.to_action(ul[j], uq[j], ms[j], mq[j], int((st[1][j][:, 0] > 0).sum()))
            a = a0
            for h in ag.hooks: a = h(a, o)
            if a is a0: continue
            a = LG.sanitize(a); b = LG.sanitize(a0)
            O[j] = mkt_codes(a["market"])
            if a["farmer"] != b["farmer"] or a["hands"] != b["hands"]:
                us = [a["farmer"]] + a["hands"]; live = st[1][j][:, 0] > 0
                for u in range(NU):
                    if live[u]: U[j, u] = unit_code_dict(us[u]) if u < len(us) else (OPI["PASS"], 0, 1)
    return U, O


def play(A, B, seeds, seats, trace=None):
    """league.play と同じ返り値。play.shops / play.mid も同じ形で残す。"""
    n = len(seeds); seats = np.asarray(seats, np.int64); pool = Pool(seeds)
    need = sorted({v for ag in (A, B) for v in ((0, 2) if ag.nmb else (ag.lopt,))})
    MIDT = [int(x) for x in os.environ.get("MIDT", "300,500").split(",") if x]
    play.mid = {t: [None] * n for t in MIDT}; play.shops = [None] * n
    ar = np.arange(n); tm = dict(feat=0.0, inf=0.0, step=0.0)
    t = 0
    while not pool.env.done():
        x0 = time.time()
        S0 = pool.state(0, need); S1 = pool.state(1, need)
        if t in play.mid:
            m = pool.money(); play.mid[t] = [[float(m[i, 0]), float(m[i, 1])] for i in range(n)]
        tm["feat"] += time.time() - x0; x0 = time.time()
        UU = [np.zeros((n, NU, 3), np.int32), np.zeros((n, NU, 3), np.int32)]; OO = [np.zeros((n, 10, 3), np.int32), np.zeros((n, 10, 3), np.int32)]
        for who, ag in ((0, A), (1, B)):
            sr = seats if who == 0 else 1 - seats                 # この局でこのエージェントが座る席
            sel = lambda k: np.where((sr == 0).reshape((-1,) + (1,) * (S0[k].ndim - 1)), S0[k], S1[k])
            st = [sel(k) for k in range(4)]
            L = {v: np.where((sr == 0)[:, None, None], S0[4][v], S1[4][v]) for v in need}
            st.append(agent_lm(ag, L))
            U, O = act(ag, tuple(st), ar, sr, t, None)
            if trace is not None: trace(t, who, st, U, O)
            for s in (0, 1):
                r = sr == s; UU[s][r] = U[r]; OO[s][r] = O[r]
        tm["inf"] += time.time() - x0; x0 = time.time()
        pool.env.step_both(UU[0], OO[0], UU[1], OO[1]); t += 1
        tm["step"] += time.time() - x0
    sh = np.asarray(pool.env.shops())
    play.shops = [[SHOPS[int(v)] for v in sh[i] if v >= 0] for i in range(n)]   # 最後の状態の店(league.py は最後の観測の店。区画の開く時刻とは無関係)
    play.time = tm; play.steps = t
    m = pool.money()
    return [(float(m[i, seats[i]]), float(m[i, 1 - seats[i]])) for i in range(n)]


def main():
    out = sys.argv[1]; R = json.load(open(out)) if os.path.exists(out) else {}
    spec = [x.split(":", 2) for x in os.environ["AGENTS"].split(",")]
    for n_, t_, f_ in spec: assert t_ == "bc", "fast_league は bc 型だけ(py 型は league.py で)"
    AG = {n: Agent(n, f) for n, _, f in spec}
    NS = int(os.environ.get("NSEED", 48)); seeds = [int(os.environ.get("SEED0", 880001)) + 7919 * k for k in range(NS)]; seats = [k % 2 for k in range(NS)]
    for (a, _, _), (b, _, _) in itertools.combinations(spec, 2):
        key = "%s|%s" % (a, b)
        if key in R: continue
        t = time.time(); r = play(AG[a], AG[b], seeds, seats)
        R[key] = r; R[key + "#shops"] = play.shops
        for t_, v_ in play.mid.items(): R["%s#money%d" % (key, t_)] = v_
        R[key + "#seats"] = seats
        json.dump(R, open(out, "w"))
        AG[a].graphs.clear(); AG[b].graphs.clear()             # 組が変わるたびに graph を捨てる(GPU の記憶を増やさない)
        w = np.mean([x > y for x, y in r]); d = np.mean([x - y for x, y in r])
        print("%-14s 対 %-14s %d局  %s の勝率 %.3f 差 %+6.0f(%s %.0f / %s %.0f)%.0fs [特徴 %.0fs 推論 %.0fs エンジン %.0fs]" % (
            a, b, len(r), a, w, d, a, np.median([x for x, _ in r]), b, np.median([y for _, y in r]), time.time() - t,
            play.time["feat"], play.time["inf"], play.time["step"]) + " GPU 確保 %.1fGB" % (torch.cuda.max_memory_reserved() / 2**30), flush=True)


if __name__ == "__main__":
    main()
