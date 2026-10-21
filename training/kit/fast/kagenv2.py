"""ベクトル化エージェントを使う RL 環境。観測の抽出もシード方向にループしない。"""
import numpy as np, sys
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim
from a2_run import BLANK, E
from agent2 import BS
from vecagent import VecAgent
from league import LEAGUE
from shopfeat import shop_feat, DIM as SHOP_DIM

OBS_DIM = 41 + SHOP_DIM      # 41 + 18 = 59。店の抽選を観測に入れた(2026-09-09)
N_STRAT = 12


class KagEnv2:
    def __init__(self, seeds, opp_idx=0, nu_max=14):
        self.seeds = list(seeds); self.n = len(seeds)
        self.opp_idx = opp_idx; self.U = nu_max

    def reset(self, opp_idx=None):
        if opp_idx is not None: self.opp_idx = opp_idx
        opp = LEAGUE[self.opp_idx][4]
        self.env = kagsim.VecMarket(BLANK[0], BLANK[1], opp[0], opp[1], self.seeds, 720, 0)
        self.env.keep_nonsell = False; self.env.reset()
        self.ag = VecAgent(self.n, self.U)
        self.day = 0
        self._read()
        return self._obs()

    def _read(self):
        n, U = self.n, self.U
        ob = self.env.obs(); uo = self.env.unit_obs(U); sh = self.env.seeds_held()
        b = uo[:, 0, 8:].reshape(n, 100, 6)
        self.kind = np.rint(b[:, :, 0]*5).astype(np.int32).reshape(n, BS, BS)
        self.what = np.rint(b[:, :, 1]*12).astype(np.int32).reshape(n, BS, BS)
        self.watd = (b[:, :, 2] > .5).reshape(n, BS, BS)
        self.fedd = (b[:, :, 3] > .5).reshape(n, BS, BS)
        self.yld = np.rint(b[:, :, 4]*6).astype(np.int32).reshape(n, BS, BS)
        ch5 = uo[:, :, 8:].reshape(n, U, 100, 6)[:, :, :, 5]
        self.fav = (np.abs(ch5 - 0.5) < 0.1).any(1).reshape(n, BS, BS)
        self.umask = uo[:, :, 6] > 0
        self.umask[:, 0] = True
        self.ux = np.rint(uo[:, :, 3]*9).astype(np.int32)
        self.uy = np.rint(uo[:, :, 4]*9).astype(np.int32)
        shed9 = np.rint(ob[:, 18:27]*100).astype(np.int32)
        self.shed = np.concatenate([shed9, np.zeros((n, 3), np.int32)], 1)
        self.money = ob[:, 36]*50000.0
        self.seeds_h = sh
        self.mktinv = ob[:, 0:9]*10000.0
        self.ob_raw = ob

    def _obs(self):
        n = self.n; ob = self.ob_raw
        out = np.zeros((n, OBS_DIM), np.float32)
        pen = (self.kind == 3) | (self.kind == 4)
        out[:, 0] = self.day / 29.0
        out[:, 1] = ob[:, 36]
        out[:, 2] = self.umask.sum(1) / 14.0
        out[:, 3] = (pen & (self.what >= 9)).sum((1, 2)) / 20.0
        out[:, 4] = (self.kind == 0).sum((1, 2)) / 100.0
        out[:, 5] = (self.kind == 1).sum((1, 2)) / 100.0
        out[:, 6] = (self.kind == 2).sum((1, 2)) / 100.0
        out[:, 7] = (self.kind == 5).sum((1, 2)) / 100.0
        out[:, 8] = pen.sum((1, 2)) / 20.0
        out[:, 9:18] = ob[:, 0:9]
        out[:, 18:27] = ob[:, 18:27]
        out[:, 27:36] = np.clip(ob[:, 9:18], -3, 3)
        out[:, 36:41] = np.clip(self.seeds_h[:, :5] / 20.0, 0, 3)
        if SHOP_DIM: out[:, 41:] = shop_feat(self.env.shops())
        return out

    def step(self, strat):
        self.ag.strat = np.asarray(strat, np.float64)
        for _ in range(24):
            if self.env.done(): break
            t = self.env.step_index(); day, hour = divmod(t, 24)
            ops, mk = self.ag.act(day, hour, self.kind, self.what, self.watd, self.fedd,
                                  self.yld, self.fav, self.ux, self.uy, self.umask,
                                  self.shed, self.money, self.seeds_h, self.mktinv)
            self.env.step_units(ops.astype(np.int32), mk.astype(np.int32), E)
            if not self.env.done(): self._read()
        self.day += 1
        done = self.env.done()
        if done:
            m = self.env.money(); self.final = m
            return self._obs(), ((m[:, 0]-m[:, 1])/50000.0).astype(np.float32), True
        return self._obs(), np.zeros(self.n, np.float32), False
