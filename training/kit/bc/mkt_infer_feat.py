"""exp266: 推論時に市場の16特徴を作る(局ごとに1つの Feats を持ち、毎手 push する)。
学習時(lk2_feat.py)との違いは「自分の約定」を納屋の差から近似する点だけ(学習時はエンジンで正確に再現)。
相手の売り = 市場在庫の変化 + 店の消費 − 自分の約定。相手の納屋 = 相手の収穫の累計 − 相手の売りの累計。"""
import collections
IT = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
SHOPS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
         "YARN_STORE": ["WOOL"], "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
         "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
TOWN = [p for p in IT if p != "FERTILIZER"]
FLOOR = 1
DEFAULT_PARAMS = {"WHEAT": {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.8, "above_func": "log", "above_target": 0.2}, "CARROT": {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.0, "above_func": "sqrt", "above_target": 0.7}, "TOMATO": {"base": 60, "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.4, "above_func": "sqrt", "above_target": 0.6}, "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.7, "above_func": "linear", "above_target": 1.6}, "MELON": {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.2, "above_func": "sq", "above_target": 3.6}, "EGG": {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.4, "above_func": "log", "above_target": 0.2}, "MILK": {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.6, "above_func": "linear", "above_target": 1.6}, "WOOL": {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.2, "above_func": "sq", "above_target": 3.2}, "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.4, "above_func": "linear", "above_target": 0.4}}   # エンジンの既定値(obs の params が空のとき)


HINGE_GAIN = 8.0


def _shape(f, x, T=None):
    x = max(0.0, x)
    if f == "linear": return x
    if f == "sq": return x * x
    if f == "sqrt": return x ** 0.5
    if f == "log": import math; return math.log(1.0 + x)
    if f == "log10": import math; return math.log10(1.0 + x)
    if f == "hinge":
        if not T or T <= 0: return x
        u = x / T; return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def price(item, inv, params):
    """エンジンの market_price と同じ式(kaggriculture.py より)。params が無ければ既定値。"""
    p = (params or DEFAULT_PARAMS)[item]; base = p["base"]; I0 = p["I0"]; T = p["T"]
    if inv < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], T, T)
        v = base + amp * _shape(p["below_func"], I0 - inv, T)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], T, T)
        v = base - amp * _shape(p["above_func"], inv - I0, T)
    return max(FLOOR, int(round(v)))


class Feats:
    def __init__(self, target=None, products=None):
        self.prev = None; self.opp_cum = collections.Counter(); self.my_cum = collections.Counter()
        self.opp_prod = collections.Counter(); self.o24 = []; self.m24 = []
        self.hist_p = []; self.hist_i = []; self.tgt = target; self.tp = products

    @staticmethod
    def _crops(fa):
        c = collections.Counter(); ready = collections.Counter(); ani = collections.Counter()
        for row in (fa.get("tiles") or []):
            for t in (row or []):
                if not isinstance(t, dict): continue
                k = t.get("kind")
                if k == "PLANT" and t.get("crop"):
                    c[t["crop"]] += 1
                    if (t.get("yield_units") or 0) > 0: ready[t["crop"]] += 1
                elif k == "COOP": ani["EGG"] += 1
                elif k == "PASTURE": ani["MILK"] += 1; ani["WOOL"] += 1
        return c, ready, ani

    def push(self, obs, seat, t):
        mk = obs["market"]; inv = dict(mk["inventory"]); par = mk.get("params")
        shed = dict((obs.get("private") or {}).get("shed") or {})
        opp = obs["farms"][1 - seat]
        crops, ready, ani = self._crops(opp)
        if self.prev is not None:
            pc = self.prev["crops"]
            for c in set(pc) | set(crops):
                d = pc.get(c, 0) - crops.get(c, 0)
                if d > 0: self.opp_prod[c] += d
            cons = collections.Counter()
            st = t - 1
            if st % 4 == 0:
                for sh in obs["town"]["unlocked_shops"]:
                    pr = SHOPS.get(sh, []); m = 2 if len(pr) == 1 else 1
                    for i in pr: cons[i] += m
            if st % 24 == 0:
                for i in TOWN: cons[i] += 1
            mine = {}; oppn = {}
            for i in IT:
                mine[i] = max(self.prev["shed"].get(i, 0) - shed.get(i, 0), 0) if (st + 1) % 24 else 0   # 日の終わりは手持ちが入るので数えない
                oppn[i] = (inv[i] - self.prev["inv"][i]) + cons[i] - mine[i]
                if mine[i] > 0: self.my_cum[i] += mine[i]
                if oppn[i] > 0: self.opp_cum[i] += oppn[i]
            self.o24.append(oppn); self.m24.append(mine); self.o24 = self.o24[-24:]; self.m24 = self.m24[-24:]
        self.hist_p.append({i: price(i, inv[i], par) for i in IT}); self.hist_i.append(inv)
        self.hist_p = self.hist_p[-25:]; self.hist_i = self.hist_i[-25:]
        self.prev = dict(inv=inv, shed=shed, crops=crops)
        dem = collections.Counter()
        for sh in obs["town"]["unlocked_shops"]:
            pr = SHOPS.get(sh, []); m = 2 if len(pr) == 1 else 1
            for i in pr: dem[i] += m
        row = self.tgt["75"][min(t, len(self.tgt["75"]) - 1)] if self.tgt else None
        back_p = self.hist_p[0]; back_i = self.hist_i[0]
        o24 = collections.Counter(); m24 = collections.Counter()
        for h in self.o24:
            for i in IT:
                if h[i] > 0: o24[i] += h[i]
        for h in self.m24:
            for i in IT:
                if h[i] > 0: m24[i] += h[i]
        out = []
        for i in IT:
            iv = inv[i]; have = int(shed.get(i, 0))
            p0 = price(i, iv, par); rev = 0.0; q = iv
            for _ in range(min(have, 60)): rev += price(i, q, par); q += 1
            tg = row[self.tp.index(i)] if (row is not None and i in self.tp) else 0.0
            out.append([p0, price(i, iv + 10, par), rev / 100.0, have, have - tg, dem.get(i, 0), iv / 10.0, price(i, max(iv - 1, 0), par),
                        max(self.opp_prod.get(i, 0) - self.opp_cum.get(i, 0), 0), o24.get(i, 0), self.opp_cum.get(i, 0), m24.get(i, 0),
                        ready.get(i, 0), ani.get(i, 0), p0 - back_p[i], (iv - back_i[i]) / 10.0])
        return out
