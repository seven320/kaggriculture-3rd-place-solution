"""Kaggriculture を「1日1決定」の強化学習環境として包む。

生の行動(720ステップ x 10ユニット x 18種 x 100タイル)で学習すると信用割当が壊れる。
ここでは階層化して、PPO には**方針ベクトル11次元を1日1回**だけ出させる(=30決定/局)。
方針を受けた配分(LP)と実行(貪欲割当)は agent3 が厳密に解く。

報酬は金差。記憶の鉄則「勝敗が目的関数」はゲート側で守る:
現状は全相手に勝率0で ±1 報酬が定数になり勾配が消えるため、
Pr[勝ち]=Φ(μ/σ) の μ を上げる代理として金差を使い、評価は勝率で行う。
"""
import numpy as np, sys
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim
from a2_run import BLANK, E
from agent2 import BS, OP
from agent3 import Agent3, N_STRAT
from league import LEAGUE
from shopfeat import shop_feat, DIM as SHOP_DIM

OBS_DIM = 41 + SHOP_DIM      # 41 + 18 = 59。店の抽選を観測に入れた(2026-09-09、SHOPFEAT=0 で無効)
BASE_P = dict(fert_use=0, sell_floor=0.0, crop=0, anim_last=29, keeper_helps=1,
              early_harvest=1, n_keeper=2, anim_reserve=400.0, seed_rate=14,
              seed_buffer=4, wheat_feed=8, feed_buffer=6, harvest_slack=-1,
              target_live=0, pen_radius=4, melon_tiles=0, carrot_tiles=0,
              tomato_tiles=0, straw_tiles=0, anim_per_day=1, lp_animals=0,
              hire=12, hire_early=6, anim_target=8, mix_cow=3, mix_sheep=1,
              mix_goose=0, land_reserve=0.0)


class KagEnv:
    """N シードを同時に進める。step() は 1 日(24時間)ぶん進める。"""

    def __init__(self, seeds, opp_idx=0):
        self.seeds = list(seeds); self.n = len(seeds); self.opp_idx = opp_idx

    def reset(self, opp_idx=None):
        if opp_idx is not None: self.opp_idx = opp_idx
        opp = LEAGUE[self.opp_idx][4]
        self.env = kagsim.VecMarket(BLANK[0], BLANK[1], opp[0], opp[1],
                                    self.seeds, 720, 0)
        self.env.keep_nonsell = False; self.env.reset()
        self.A = [Agent3(BASE_P) for _ in range(self.n)]
        for a in self.A:
            a._nk = 2; a.strat = np.ones(N_STRAT)
        self.day = 0
        return self._obs()

    # ---------------- 観測 ----------------
    def _obs(self):
        ob = self.env.obs(); uo = self.env.unit_obs(14); sh = self.env.seeds_held()
        n = self.n; out = np.zeros((n, OBS_DIM), np.float32)
        for i in range(n):
            b = uo[i, 0, 8:].reshape(100, 6)
            kind = np.rint(b[:, 0]*5).astype(int)
            what = np.rint(b[:, 1]*12).astype(int)
            nu = max(1, int((uo[i, :, 6] > 0).sum()))
            pen = (kind == 3) | (kind == 4)
            out[i, 0] = self.day / 29.0
            out[i, 1] = ob[i, 36]                      # 所持金(既に /50000)
            out[i, 2] = nu / 14.0
            out[i, 3] = (pen & (what >= 9)).sum() / 20.0
            out[i, 4] = (kind == 0).sum() / 100.0      # 空き
            out[i, 5] = (kind == 1).sum() / 100.0      # 未購入
            out[i, 6] = (kind == 2).sum() / 100.0      # 雑草
            out[i, 7] = (kind == 5).sum() / 100.0      # 作付
            out[i, 8] = pen.sum() / 20.0
            out[i, 9:18] = ob[i, 0:9]                  # 市場在庫
            out[i, 18:27] = ob[i, 18:27]               # 納屋
            out[i, 27:36] = np.clip(ob[i, 9:18], -3, 3)  # 価格帯
            out[i, 36:41] = np.clip(sh[i][:5] / 20.0, 0, 3)
        if SHOP_DIM: out[:, 41:] = shop_feat(self.env.shops())
        return out

    # ---------------- 1日進める ----------------
    def step(self, strat):
        """strat: (N, 11) 今日の方針。24時間ぶん実行して次の観測を返す。"""
        for i, a in enumerate(self.A): a.strat = strat[i]
        for _ in range(24):
            if self.env.done(): break
            self._hour()
        self.day += 1
        done = self.env.done()
        if done:
            m = self.env.money()
            rew = (m[:, 0] - m[:, 1]) / 50000.0
            self.final = m
            return self._obs(), rew.astype(np.float32), True
        return self._obs(), np.zeros(self.n, np.float32), False

    def _hour(self):
        env = self.env; n = self.n
        ob = env.obs(); t = env.step_index(); day, hour = divmod(t, 24)
        uo = env.unit_obs(14); sh = env.seeds_held()
        U = np.zeros((n, 14, 3), np.int32); MK = np.zeros((n, 10, 3), np.int32)
        for i in range(n):
            b = uo[i, 0, 8:].reshape(100, 6)
            kind = np.rint(b[:, 0]*5).astype(int).reshape(BS, BS)
            what = np.rint(b[:, 1]*12).astype(int).reshape(BS, BS)
            watd = (b[:, 2] > .5).reshape(BS, BS); fedd = (b[:, 3] > .5).reshape(BS, BS)
            yld = np.rint(b[:, 4]*6).astype(int).reshape(BS, BS)
            nu = max(1, int((uo[i, :, 6] > 0).sum()))
            ch5 = uo[i, :nu, 8:].reshape(nu, 100, 6)[:, :, 5]
            fav = ((np.abs(ch5-0.5) < 0.1).any(0)).reshape(BS, BS)
            ux = np.rint(uo[i, :nu, 3]*9).astype(int); uy = np.rint(uo[i, :nu, 4]*9).astype(int)
            shed9 = np.rint(ob[i, 18:27]*100).astype(np.int64)
            shed = np.concatenate([shed9, np.zeros(3, np.int64)])
            A = self.A[i]
            A.mktinv = ob[i, 0:9]*10000.0
            A._nk = min(2 + int((((kind == 3) | (kind == 4)) & (what >= 9)).sum())//6,
                        max(1, nu-2)) if nu > 2 else 0
            invu = np.zeros((max(nu, 1), 12), np.int64)
            if hasattr(A, "inv_track") and A.inv_track.shape[0] == max(nu, 1): invu = A.inv_track
            ops, mk = A.act(day, hour, kind, what, watd, fedd, yld, fav, ux, uy, shed,
                            float(ob[i, 36]*50000), sh[i], invu)
            for u in range(nu):
                o = int(ops[u, 0])
                if o == OP["PICKUP"]: invu[u, int(ops[u, 1])] += int(ops[u, 2])
                elif o == OP["FEED"]: invu[u, 0] = max(0, invu[u, 0]-1)
                elif o == OP["FERTILIZE"]: invu[u, 8] = max(0, invu[u, 8]-1)
                elif o == OP["PLACE"]: invu[u, int(ops[u, 1])] = max(0, invu[u, int(ops[u, 1])]-1)
                elif o == OP["COLLECT"]: invu[u, 8] += 1
            if hour == 23: invu[:] = 0
            A.inv_track = invu
            U[i, :nu] = ops[:nu]
            for j, o in enumerate(mk[:10]): MK[i, j] = o
        env.step_units(U, MK, E)
