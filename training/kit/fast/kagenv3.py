"""GPU 実行層を使う RL 環境。観測抽出も最小限にする。

unit_obs(14) は盤面を14体分複製するため N=1024 で 17.2ms/ステップ かかる。
盤面はユニット0の分だけで足りる(fav も全体のORと一致することを実測で確認)ので
unit_obs(1) + unit_pos() + tiles() に置き換える。
"""
import numpy as np, torch, sys
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim
from a2_run import BLANK, E
from agent2 import BS
from gpuexec import GpuAgent, DEV
from league import LEAGUE
from shopfeat import shop_feat, DIM as SHOP_DIM

OBS_DIM = 41 + SHOP_DIM      # 41 + 18 = 59。店の抽選を観測に入れた(2026-09-09)
N_STRAT = 12


class KagEnv3:
    def __init__(self, seeds, opp_idx=0, nu_max=14):
        self.seeds = list(seeds); self.n = len(seeds)
        self.opp_idx = opp_idx; self.U = nu_max

    def reset(self, opp_idx=None):
        if opp_idx is not None: self.opp_idx = opp_idx
        opp = LEAGUE[self.opp_idx][4]
        self.env = kagsim.VecMarket(BLANK[0], BLANK[1], opp[0], opp[1], self.seeds, 720, 0)
        self.env.keep_nonsell = False; self.env.reset()
        self.ag = GpuAgent(self.n, self.U)
        self.day = 0
        self._read()
        return self._obs()

    def _read(self):
        n, U = self.n, self.U
        ob = self.env.obs()
        # fav(肥料採取可能)はユニットごとの視界で、そのタイルに立つユニットからは見えない。
        # 先頭2体の OR で全14体の OR と完全一致することを実測で確認(30日x16シード)。
        uo2 = self.env.unit_obs(2)
        b = uo2[:, 0, 8:].reshape(n, 100, 6)
        ch5 = uo2[:, :, 8:].reshape(n, 2, 100, 6)[:, :, :, 5]
        up = np.asarray(self.env.unit_pos())[:, :U]
        T = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=DEV)
        self.kind = T(np.rint(b[:, :, 0]*5).reshape(n, BS, BS), torch.int32)
        self.what = T(np.rint(b[:, :, 1]*12).reshape(n, BS, BS), torch.int32)
        self.watd = T((b[:, :, 2] > .5).reshape(n, BS, BS), torch.bool)
        self.fedd = T((b[:, :, 3] > .5).reshape(n, BS, BS), torch.bool)
        self.yld = T(np.rint(b[:, :, 4]*6).reshape(n, BS, BS), torch.int32)
        self.fav = T((np.abs(ch5-0.5) < 0.1).any(1).reshape(n, BS, BS), torch.bool)
        live = up[:, :, 0] >= 0
        live[:, 0] = True
        self.umask = T(live, torch.bool)
        # unit_pos は (x, y) の順。unit_obs の (uy, ux) と逆なので取り違えないこと。
        self.ux = T(np.where(live, up[:, :, 0], 0), torch.int32)
        self.uy = T(np.where(live, up[:, :, 1], 0), torch.int32)
        shed9 = np.rint(ob[:, 18:27]*100)
        self.shed = T(np.concatenate([shed9, np.zeros((n, 3))], 1), torch.int32)
        self.money = T(ob[:, 36]*50000.0, torch.float32)
        self.seeds_h = T(self.env.seeds_held(), torch.int32)
        self.mktinv = T(ob[:, 0:9]*10000.0, torch.float32)
        self.ob_raw = ob

    def _obs(self):
        n = self.n; ob = self.ob_raw
        out = np.zeros((n, OBS_DIM), np.float32)
        k = self.kind.cpu().numpy(); w = self.what.cpu().numpy()
        pen = (k == 3) | (k == 4)
        out[:, 0] = self.day / 29.0
        out[:, 1] = ob[:, 36]
        out[:, 2] = self.umask.sum(1).cpu().numpy() / 14.0
        out[:, 3] = (pen & (w >= 9)).sum((1, 2)) / 20.0
        out[:, 4] = (k == 0).sum((1, 2)) / 100.0
        out[:, 5] = (k == 1).sum((1, 2)) / 100.0
        out[:, 6] = (k == 2).sum((1, 2)) / 100.0
        out[:, 7] = (k == 5).sum((1, 2)) / 100.0
        out[:, 8] = pen.sum((1, 2)) / 20.0
        out[:, 9:18] = ob[:, 0:9]
        out[:, 18:27] = ob[:, 18:27]
        out[:, 27:36] = np.clip(ob[:, 9:18], -3, 3)
        out[:, 36:41] = np.clip(self.env.seeds_held()[:, :5] / 20.0, 0, 3)
        if SHOP_DIM: out[:, 41:] = shop_feat(self.env.shops())
        return out

    def step(self, strat):
        self.ag.strat = torch.as_tensor(np.asarray(strat), dtype=torch.float32, device=DEV)
        for _ in range(24):
            if self.env.done(): break
            t = self.env.step_index(); day, hour = divmod(t, 24)
            ops, mk = self.ag.act(day, hour, self.kind, self.what, self.watd, self.fedd,
                                  self.yld, self.fav, self.ux, self.uy, self.umask,
                                  self.shed, self.money, self.seeds_h, self.mktinv)
            self.env.step_units(ops.cpu().numpy().astype(np.int32),
                                mk.cpu().numpy().astype(np.int32), E)
            if not self.env.done(): self._read()
        self.day += 1
        done = self.env.done()
        if done:
            m = self.env.money(); self.final = m
            return self._obs(), ((m[:, 0]-m[:, 1])/50000.0).astype(np.float32), True
        return self._obs(), np.zeros(self.n, np.float32), False
