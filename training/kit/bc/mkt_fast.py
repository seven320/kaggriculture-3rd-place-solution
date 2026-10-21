"""exp270: 市場の先読み特徴(exp262 の8個)を配列で一気に計算する。PPO の自己対戦でも使えるようにするため。
値段はエンジンの market_price を在庫の表に焼いておく(整数の在庫しか来ないので表引きで完全一致する)。
特徴(品目ごと8個)は lk_feat.feats と同じ順・同じ値:
 1 いまの値段、2 10個売った後の値段、3 納屋を全部売った手取り/100、4 自分の納屋、
 5 納屋 − 在庫の目標、6 この品を消費する店の1手あたりの量、7 在庫/10、8 1個買うときの値段"""
import json, os, sys, types, collections
import numpy as np

W = os.environ.get("KAGG_W", "/workspace/kaggriculture")
if "kaggle_environments" not in sys.modules:
    sys.modules["kaggle_environments"] = types.ModuleType("kaggle_environments")
    _u = types.ModuleType("kaggle_environments.utils"); _u.resolve_episode_seed = lambda *a, **k: 0
    sys.modules["kaggle_environments.utils"] = _u
K = types.ModuleType("kagg_engine"); K.__file__ = W + "/py310/kaggle_environments/envs/kaggriculture/kaggriculture.py"
exec(compile(open(K.__file__).read(), K.__file__, "exec"), K.__dict__)
IT = list(K.PRODUCTS)
LO, HI = K.MARKET_I0 - 4000, K.MARKET_I0 + 4000          # 実戦の在庫は I0 ± 200 程度。余裕を持って表にする

_TBL = None


def table():
    """PRICE[品目, 在庫 - LO] と、その前方累積 CUM。"""
    global _TBL
    if _TBL is None:
        n = HI - LO + 1
        pr = np.zeros((len(IT), n), np.float32)
        for j, it in enumerate(IT):
            pr[j] = [K.market_price(it, LO + k, None) for k in range(n)]
        cum = np.zeros((len(IT), n + 1), np.float64); cum[:, 1:] = np.cumsum(pr.astype(np.float64), 1)   # float32 の累積は桁落ちする
        _TBL = (pr, cum)
    return _TBL


def feats(inv, shed, dem, tgt):
    """inv/shed: (N, 9) 整数、dem: (N, 9)(店の1手あたりの消費)、tgt: (N, 9)(在庫の目標)
    → (N, 9, 8) float32。lk_feat.feats と同じ並び。"""
    pr, cum = table()
    inv = np.asarray(inv, np.int64); shed = np.asarray(shed, np.int64)
    k = np.clip(inv - LO, 0, HI - LO); j = np.arange(len(IT))[None, :]
    p0 = pr[j, k]
    p10 = pr[j, np.clip(k + 10, 0, HI - LO)]
    pbuy = pr[j, np.clip(k - 1, 0, HI - LO)]
    n = np.clip(shed, 0, 60)
    rev = (cum[j, np.clip(k + n, 0, HI - LO)] - cum[j, k]).astype(np.float32)
    out = np.stack([p0, p10, rev / 100.0, shed.astype(np.float32), (shed - tgt).astype(np.float32),
                    dem.astype(np.float32), inv.astype(np.float32) / 10.0, pbuy], -1)
    return out.astype(np.float32)


SHOP_DEM = None


def shop_demand(unlocked):
    """unlocked: (N, 店数) の 0/1 → (N, 9) の1手あたりの消費量。lk_feat と同じ規則(単品の店は2)。"""
    global SHOP_DEM
    if SHOP_DEM is None:
        names = list(K.SHOPS)
        M = np.zeros((len(names), len(IT)), np.float32)
        for a, sh in enumerate(names):
            ps = K.SHOPS[sh]; m = 2.0 if len(ps) == 1 else 1.0
            for p in ps:
                if p in IT: M[a, IT.index(p)] += m
        SHOP_DEM = M
    return np.asarray(unlocked, np.float32) @ SHOP_DEM


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    inv = K.MARKET_I0 + rng.integers(-300, 300, (2000, 9))
    shed = rng.integers(0, 80, (2000, 9))
    tgt = rng.random((2000, 9)).astype(np.float32) * 20
    dem = rng.integers(0, 4, (2000, 9)).astype(np.float32)
    F = feats(inv, shed, dem, tgt)
    bad = 0
    for i in range(0, 2000, 7):
        for jj, it in enumerate(IT):
            iv = int(inv[i, jj]); nn = int(min(shed[i, jj], 60))
            p0 = K.market_price(it, iv, None); p10 = K.market_price(it, iv + 10, None)
            rev = sum(K.market_price(it, iv + q, None) for q in range(nn))
            pb = K.market_price(it, max(iv - 1, 0), None)
            ok = (abs(F[i, jj, 0] - p0) < 1e-6 and abs(F[i, jj, 1] - p10) < 1e-6
                  and abs(F[i, jj, 2] - rev / 100.0) < 1e-3 and abs(F[i, jj, 7] - pb) < 1e-6)
            bad += 0 if ok else 1
    print("エンジンとの不一致:", bad, "/", len(range(0, 2000, 7)) * 9)


class Hist16:
    """exp274: 市場16特徴の後半8個(lk2_feat の 9〜16 列)を、局をまたいだ配列でまとめて持つ。
    push は1手に1回、観測 t の順に呼ぶ。lk2_feat と同じ規則:
      相手の収穫 = 前の手から消えた植え付け区画の数の累計、相手・自分の正味の売買 = 市場在庫に与えた効果の差分、
      直近24手の売り = 正の部分の和、値段と在庫の変化 = 24手前(25個前の記録、足りなければ最初の記録)との差。"""
    def __init__(self, n):
        self.n = n; self.t = 0
        self.prev_crop = None
        self.opp_prod = np.zeros((n, 9)); self.opp_cum = np.zeros((n, 9))
        self.prev_es = None; self.prev_eo = None
        self.opp_hist = []; self.my_hist = []
        self.pr_hist = []; self.inv_hist = []

    def push(self, inv, prices, eff_self, eff_opp, crop_opp, ready_opp, anim_opp):
        """inv/prices/eff_*/crop_opp/ready_opp/anim_opp: (N, 9)。戻り値 (N, 9, 8) float32。"""
        inv = np.array(inv, np.float64, copy=True); prices = np.array(prices, np.float64, copy=True)
        cur = np.array(crop_opp, np.float64, copy=True)
        if self.prev_crop is not None:
            self.opp_prod += np.maximum(self.prev_crop - cur, 0.0)
        self.prev_crop = cur
        es = np.array(eff_self, np.float64, copy=True); eo = np.array(eff_opp, np.float64, copy=True)   # 呼び出し側の配列を保持しない
        if self.prev_es is not None:
            a = es - self.prev_es; ob = eo - self.prev_eo
            self.opp_hist.append(ob); self.my_hist.append(a)
            self.opp_cum += np.maximum(ob, 0.0)
            if len(self.opp_hist) > 24: self.opp_hist.pop(0)
            if len(self.my_hist) > 24: self.my_hist.pop(0)
        self.prev_es, self.prev_eo = es, eo
        o24 = sum((np.maximum(h, 0.0) for h in self.opp_hist), np.zeros((self.n, 9)))
        m24 = sum((np.maximum(h, 0.0) for h in self.my_hist), np.zeros((self.n, 9)))
        self.pr_hist.append(prices); self.inv_hist.append(inv)
        if len(self.pr_hist) > 25:
            self.pr_hist.pop(0); self.inv_hist.pop(0)
        back = self.pr_hist[0]; binv = self.inv_hist[0]    # 25個溜まるまでは最初の記録、以降は25個前(= 24手前)
        est = np.maximum(self.opp_prod - self.opp_cum, 0.0)
        out = np.stack([est, o24, self.opp_cum, m24, np.asarray(ready_opp, np.float64), np.asarray(anim_opp, np.float64),
                        prices - back, (inv - binv) / 10.0], -1)
        self.t += 1
        return out.astype(np.float32)
