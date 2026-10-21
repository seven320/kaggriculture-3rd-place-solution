"""自己対戦の土台(exp176)。両方の座席をエンジンの中で動かす。
  SPPool : 対局ごとに VecMarket(両座席とも幅の広い空テープ)。step_both で両座席の手を渡す、set_view で見る側を切り替える
  相手の種類:
    PolicyOpp(net, greedy)   方策(自分・過去の版・d2 など)
    TapeOpp(acts)            記録された手をそのまま(合法の判定なし。exp175 で一致を確認した経路)
    V55Opp()                 v55 そのもの: plan00 の開幕144手 → step144 に毛糸店があれば継続0、なければ継続1。step700 以降は納屋の売れ残りを売る
座席は絶対番号(0/1)。特徴は ppo_tf.Pool.state と同じ並び(見る側の座席から)。
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast"); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/bc")
import kagsim, kagpack, ppo_tf as P
from deploy import wide_blank
from hybrid5_tc import tape_codes
from extract4 import NU
B = "/content/drive/MyDrive/kaggriculture"
IT9 = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MOPI_SELL = kagpack.MOPI["SELL"]


# ---- exp270: 市場の先読み特徴(MKTNF=8 で有効)。値段の表はエンジンと完全一致を検証済み(mkt_fast.py)----
MKTNF = int(os.environ.get("MKTNF", "0") or 0)
_MF = _TGTROW = None; _NSHOP = 8
if MKTNF:
    import mkt_fast as _MF
    _t = json.load(open(os.environ.get("STOCKTGT", "/content/lab/agents/bc_dmftS2_pace/stock_target.json")))
    _tp = _t["products"]
    _TGTROW = np.array([[r[_tp.index(i)] if i in _tp else 0.0 for i in _MF.IT] for r in _t["75"]], np.float32)
    _NSHOP = 8

class SPPool:
    def __init__(self, seeds):
        w = wide_blank(); self.envs = []
        for sd in seeds:
            env = kagsim.VecMarket(w[0], w[1], w[0], w[1], [int(sd)], 720, 0); env.keep_nonsell = False; env.reset(); self.envs.append(env)
        self.n = len(self.envs)
    def state(self, seat):
        """seat の座席から見た特徴(全局)。"""
        T, OU, PU, G, L, MK = [], [], [], [], [], []
        for env in self.envs:
            env.set_view(int(seat))
            tiles = np.stack([np.concatenate([np.asarray(env.tiles_base(s)), np.asarray(env.tiles_dyn(s))], 3).reshape(-1, 100, 13) for s in (0, 1)], 1)
            up = np.asarray(env.unit_pos2(0))[:, :NU]; op = np.asarray(env.unit_pos2(1))[:, :NU]; inv = np.asarray(env.unit_inv(NU))
            OU.append(np.concatenate([(up[..., :1] >= 0), up, inv], 2).astype(np.float32)); PU.append(np.concatenate([(op[..., :1] >= 0), op], 2).astype(np.float32))
            G.append(np.concatenate([np.asarray(env.obs_legal()), np.asarray(env.farm_pub(0)), np.asarray(env.farm_pub(1)), np.asarray(env.shops()).astype(np.float32), np.asarray(env.shed_all())], 1).astype(np.float32))
            T.append(tiles.astype(np.float16)); L.append(np.asarray(env.legal_mask(NU)))
            if MKTNF:                                   # exp270: 市場の先読み特徴(lk_feat と同じ8個、値はエンジンと一致を検証済み)
                g = G[-1]
                inv = np.rint(g[:, 0:9] * 10000).astype(np.int64)
                shed = np.rint(g[:, 18:27] * 100).astype(np.int64)
                sh = np.asarray(env.shops())[:, :16]
                un = np.zeros((len(inv), _NSHOP), np.float32)
                for r in range(len(inv)):
                    for v in sh[r]:
                        if 0 <= v < _NSHOP: un[r, v] += 1.0       # 同じ店が複数開くので数える
                ti = int(env.step_index()); row = _TGTROW[min(ti, len(_TGTROW) - 1)]
                MK.append(_MF.feats(inv, shed, _MF.shop_demand(un), np.broadcast_to(row, inv.shape)))
        c = np.concatenate
        if MKTNF: return c(T), c(OU), c(PU), c(G), c(L), c(MK)
        return c(T), c(OU), c(PU), c(G), c(L)
    def step(self, U0, O0, U1, O1):
        for i, env in enumerate(self.envs): env.step_both(U0[i:i + 1], O0[i:i + 1], U1[i:i + 1], O1[i:i + 1])
    def money(self):
        """(局, 2) 絶対座席の所持金。"""
        out = []
        for env in self.envs: env.set_view(0); out.append(np.asarray(env.money())[0])
        return np.stack(out)
    def shops(self): return np.stack([np.asarray(env.shops())[0] for env in self.envs])
    def step_index(self): return self.envs[0].step_index()

class PolicyOpp:
    def __init__(self, net, greedy=True, cap=True): self.net, self.greedy, self.cap = net, greedy, cap
    def act(self, st, t, idx, pool, seat, ou_np=None):
        ch = int(os.environ.get("OPPCHUNK", "64"))          # exp280: 相手の推論をまとめる数(大きいほど速い)
        outs = [P.sample(self.net, tuple(x[i:i + ch] for x in st), greedy=self.greedy, cap=self.cap) for i in range(0, len(st[3]), ch)]
        ul, uq, ms, mq = [np.concatenate([o[j] for o in outs]) for j in range(4)]
        return P.to_codes(ul, uq, ms, mq, st[1] if ou_np is None else ou_np)   # exp280: 盤面が GPU 上のときは numpy の ou を使う

class TapeOpp:
    def __init__(self, acts_list): self.acts = acts_list          # 局ごとの手の列
    def act(self, st, t, idx, pool, seat):
        UO = [tape_codes(self.acts[k][min(t, len(self.acts[k]) - 1)], st[1][j], st[4][j]) for j, k in enumerate(idx)]
        return np.stack([u for u, _ in UO]), np.stack([o for _, o in UO])

class V55Opp:
    """v55 の main.py と同じ規則。テープは plan00[:144] + 継続[144:](main.py の _T と 719/719 一致を確認済み)。"""
    def __init__(self):
        plans = json.load(open(B + "/repro/shoprouter0909/actions.json"))
        ta = lambda e: json.load(open("%s/data/tapes_bulk/%s.json" % (B, e)))["acts"]
        self.T = [[dict(x) for x in plans[0][:144]] + ta("107125970")[144:], [dict(x) for x in plans[0][:144]] + ta("107130822")[144:]]
        self.pick = {}
    def act(self, st, t, idx, pool, seat):
        if t == 144:
            sh = pool.shops()
            for j, k in enumerate(idx): self.pick[k] = 0 if (sh[k] == 7).any() else 1
        U, O = [], []
        for j, k in enumerate(idx):
            a = json.loads(json.dumps(self.T[self.pick.get(k, 0) if t >= 144 else 0][min(t, 718)]))
            m = a.setdefault("market", []) or []; a["market"] = m
            if t >= 700:
                shed = np.rint(st[3][j, 18:27] * 100).astype(int)      # 自分の納屋(obs_legal の 9品目)
                have = {o[1] for o in m if isinstance(o, list) and len(o) >= 2 and o[0] == "SELL"}
                for it in ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER"):
                    q = int(shed[IT9.index(it)])
                    if q > 0 and it not in have and len(m) < 10: m.append(["SELL", it, q]); have.add(it)
            u, o = tape_codes(a, st[1][j], st[4][j]); U.append(u); O.append(o)
        return np.stack(U), np.stack(O)

def play(seeds, seat_agents, collect=None):
    """seat_agents: {0: 相手オブジェクト, 1: 相手オブジェクト}(全局同じ)。collect(t, seat, st, U, O) を呼べる。返り値: 絶対座席の最終所持金 (局, 2)。"""
    pool = SPPool(seeds); idx = list(range(pool.n))
    for t in range(719):
        acts = {}
        for s in (0, 1):
            st = pool.state(s); U, O = seat_agents[s].act(st, t, idx, pool, s); acts[s] = (U, O)
            if collect: collect(t, s, st, U, O)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    return pool.money()


class MixedOpp:
    """ユニットは unit_net(BC、貪欲・種の上限)、市場は mkt_net(貪欲 or 標本)。exp179 の「市場だけ RL」の打ち方。"""
    def __init__(self, unit_net, mkt_net, greedy=True): self.u, self.m, self.greedy = unit_net, mkt_net, greedy
    def act(self, st, t, idx, pool, seat):
        ul, uq, ms, mq = mixed_sample(self.u, self.m, st, self.greedy)[:4]
        return P.to_codes(ul, uq, ms, mq, st[1])

def mixed_sample(unit_net, mkt_net, st, greedy_mkt, crit=None):
    outs_u = [P.sample(unit_net, tuple(x[i:i + 64] for x in st), greedy=True, cap=True) for i in range(0, len(st[3]), 64)]
    outs_m = [P.sample(mkt_net, tuple(x[i:i + 64] for x in st), greedy=greedy_mkt, crit=crit) for i in range(0, len(st[3]), 64)]
    ul, uq = [np.concatenate([o[j] for o in outs_u]) for j in (0, 1)]
    ms, mq, v = [np.concatenate([o[j] for o in outs_m]) for j in (2, 3, 4)]
    return ul, uq, ms, mq, v


class TapeRouterOpp(V55Opp):
    """任意のテープの束 + step144 の店の組(並べ替えた2軒の名前を | でつないだもの) → テープ番号 の規則。終盤は v55 と同じ売り。
    exp180 の家族ルーター・固定テープ(規則なし)をエンジン内で動かす。main.py(router5_rule.py が作る)と同じ規則。"""
    SH = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
    def __init__(self, agent_dir):
        import re
        self.T = json.load(open(os.path.join(agent_dir, "actions.json")))
        src = open(os.path.join(agent_dir, "main.py")).read()
        m = re.search(r"^(?:FAM_RULE|_RULE) = (.*)$", src, re.M)   # 家族ルーター(v58 など)は FAM_RULE、router5 は _RULE
        self.rule = eval(m.group(1)) if m else {"YARN_STORE": 0}
        self.v55 = m is None; self.pick = {}
    def act(self, st, t, idx, pool, seat):
        if t == 144:
            sh = pool.shops()
            for j, k in enumerate(idx):
                names = sorted(self.SH[int(x)] for x in sh[k] if x >= 0)[:2]
                self.pick[k] = self.rule.get("|".join(names), 0)
        return V55Opp.act(self, st, t, idx, pool, seat) if False else self._act(st, t, idx)
    def _act(self, st, t, idx):
        U, O = [], []
        for j, k in enumerate(idx):
            tape = self.T[self.pick.get(k, 0) if t >= 144 else 0]
            a = json.loads(json.dumps(tape[min(t, len(tape) - 1)]))
            m = a.setdefault("market", []) or []; a["market"] = m
            if t >= 700:
                shed = np.rint(st[3][j, 18:27] * 100).astype(int)
                have = {o[1] for o in m if isinstance(o, list) and len(o) >= 2 and o[0] == "SELL"}
                for it in ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER"):
                    q = int(shed[IT9.index(it)])
                    if q > 0 and it not in have and len(m) < 10: m.append(["SELL", it, q]); have.add(it)
            u, o = tape_codes(a, st[1][j], st[4][j]); U.append(u); O.append(o)
        return np.stack(U), np.stack(O)
