"""シード方向に完全ベクトル化したエージェント。agent3 と同じ戦略を配列演算で行う。

目的は速度。実測でエンジンは 300局/秒 出せるのに、シードごとの Python 方策のせいで
1.5局/秒 しか出ておらず PPO が回せない。ここではシード方向のループを一切作らない。

方針ベクトル strat(N,11) は PPO が1日1回出す。中身は agent3 と同じ意味:
  0-4 作物の重み  5-7 牛/羊/鵞鳥の比  8 家畜目標  9 雇用  10 土地確保
"""
import numpy as np
import vecexec as V
from vecexec import (J_NONE, J_HARV, J_WATER, J_DIG, J_PLANT, J_PRI, J_OP,
                     J_FEED, J_CARE, J_COLL, J_AHARV, J_PLACE, J_BUILD, J_PICK,
                     JA_PRI, JA_OP, tile_jobs_vec, anim_jobs_vec, assign_pri)
from agent2 import (OP, M, BS, SEED, LASTD, ONGOING, CROP, ANIM, hire_cost,
                    WHEAT, CARROT, TOMATO, STRAW, MELON, EGG, MILK, WOOL, FERT,
                    GOOSE, COW, SHEEP, K_EMPTY, K_LOCKED, K_WEED, K_COOP, K_PASTURE, K_PLANT)
from vecpolicy import _price_np

NC = 5
C_SEED = np.array([SEED[c] for c in range(NC)], np.float64)
C_LAST = np.array([LASTD[c] for c in range(NC)], np.int32)
C_MAXY = np.array([(ONGOING[c][2] if c in ONGOING else CROP[c][1]) for c in range(NC)], np.int32)
A_COST = np.array([ANIM[a][0] for a in (COW, SHEEP, GOOSE)], np.float64)
A_ID = np.array([COW, SHEEP, GOOSE], np.int32)
_ACT = np.array([(1 + (LASTD[c]//2 + 1) + (ONGOING[c][2] if c in ONGOING else 1) + 1)
                 if c in ONGOING else (1 + (CROP[c][0] - (CROP[c][0]+1)//2 + 2) + 2)
                 for c in range(NC)], np.float64)
A_FY = np.array([ANIM[a][1] for a in (COW, SHEEP, GOOSE)], np.int32)
H = BS // 2


class VecAgent:
    def __init__(self, n, nu_max=14):
        self.n, self.U = n, nu_max
        self.pday = np.full((n, BS, BS), -1, np.int32)
        self.plot = np.full((n, BS, BS), 0, np.int32)
        self.cared = np.zeros((n, BS, BS), bool)
        self.collected = np.zeros((n, BS, BS), bool)
        self.carry_i = np.full((n, nu_max), -1, np.int32)
        self.carry_n = np.zeros((n, nu_max), np.int32)
        self.strat = np.ones((n, 12))
        self.nbuy = np.zeros((n, 3), np.int32)
        # 納屋の動物は観測に出ない(ob は9品目のみ)ので購入済み未設置を自前で持つ
        self.pending = np.zeros((n, 3), np.int32)

    # ---------- 1日1回: 作付け計画 ----------
    def plan(self, day, kind, mktinv, money, nu):
        """残り期間の総生産量を限界価格で評価し、価値の高い作物から空きタイルに割り当てる。
        限界価格は売るほど下がるので、同じ作物に寄せると自然に価値が落ちて分散する。"""
        n = self.n
        R = 29 - day
        cyc = np.maximum(0, R // (C_LAST + 1))
        upt = cyc * C_MAXY
        # 各作物について「t 枚目のタイル」の価値 (n, NC, 8)
        NSEG = 16   # 作物あたりの上限枚数。8だと空きタイルが余る(実測: 20日目に空き39/計画16)
        val = np.full((n, NC, NSEG), -1e9)
        for c in range(NC):
            if cyc[c] < 1: continue
            u = int(upt[c])
            if u <= 0: continue
            kk = np.arange(u * NSEG)
            pv = _price_np(c, mktinv[:, c][:, None] + kk[None, :])
            cs = np.cumsum(pv, 1)
            for t in range(NSEG):
                lo, hi = u * t, u * (t + 1) - 1
                rev = cs[:, hi] - (cs[:, lo - 1] if lo > 0 else 0.0)
                val[:, c, t] = (rev - cyc[c] * C_SEED[c]) * self.strat[:, c]
        # 価値の高い順に空きタイルへ
        free = (kind == K_EMPTY)
        nfree = free.sum((1, 2))
        flat = val.reshape(n, NC * NSEG)
        cid = np.repeat(np.arange(NC), NSEG)
        order = np.argsort(-flat, 1)
        want = np.zeros((n, NC), np.int32)
        budget = np.maximum(0.0, money.astype(np.float64))
        acts = np.zeros(n)
        labour = np.maximum(1.0, nu * 24.0 / 2.5)                  # 移動込みの実効手数
        a_per = np.array([_ACT[c] * max(1, cyc[c]) / max(1.0, R) for c in range(NC)])
        for r in range(NC * NSEG):
            idx = order[:, r]; c = cid[idx]
            v = np.take_along_axis(flat, idx[:, None], 1)[:, 0]
            cost = C_SEED[c] * np.maximum(1, cyc[c])
            room = ((want.sum(1) < nfree) & (v > 0) & (budget >= cost)
                    & (acts + a_per[c] <= labour))
            want[np.arange(n)[room], c[room]] += 1
            budget = np.where(room, budget - cost, budget)
            acts = np.where(room, acts + a_per[c], acts)
        self.want = want
        # 盤面へ塗る: 中心に近い順
        yy, xx = np.meshgrid(np.arange(BS), np.arange(BS), indexing="ij")
        d = (np.abs(yy - (H-1)) + np.abs(xx - (H-1))).reshape(-1)
        ordc = np.argsort(d)
        plot = np.zeros((n, BS*BS), np.int32)
        cum = np.cumsum(want, 1)
        rank = np.empty_like(ordc); rank[ordc] = np.arange(BS*BS)
        for c in range(NC):
            lo = cum[:, c-1] if c > 0 else np.zeros(n, np.int32)
            hit = (rank[None, :] >= lo[:, None]) & (rank[None, :] < cum[:, c][:, None])
            plot = np.where(hit, c, plot)
        self.plot = plot.reshape(n, BS, BS)

    # ---------- 毎時 ----------
    def act(self, day, hour, kind, what, watd, fedd, yld, fav, ux, uy, umask,
            shed, money, seeds, mktinv):
        n, U = self.n, self.U
        if hour == 0:
            self.cared[:] = False
            self.collected[:] = False
            self.apply_strategy()
            self.numax = np.maximum(getattr(self, 'numax', 0), umask.sum(1))
            self.plan(day, kind, mktinv, money, np.maximum(self.numax, 7))
        # 植えたタイルの植え付け日を記録(失敗しても自己修復する)
        newp = (kind == K_PLANT) & (self.pday < 0)
        self.pday = np.where(newp, day, self.pday)
        self.pday = np.where(kind != K_PLANT, -1, self.pday)

        nanim = ((kind == K_COOP) | (kind == K_PASTURE)).astype(np.int32)
        nanim = (((kind == K_COOP) | (kind == K_PASTURE)) & (what >= 9)).sum((1, 2))
        base_nk = 2 + nanim // 6
        ks = getattr(self, "kscale", np.ones(self.n))
        nk = np.where(umask.sum(1) > 2,
                      np.clip(np.round(base_nk * ks).astype(np.int32), 1,
                              np.maximum(1, umask.sum(1) - 2)), 0)

        shed_anim = self.pending.sum(1)
        pen_all = (kind == K_COOP) | (kind == K_PASTURE)
        fav_own = pen_all & (what >= 9) & ~self.collected
        aj = anim_jobs_vec(kind, what, fedd, yld, fav_own, self.cared,
                           self.carry_i, self.carry_n, shed_anim, day)
        cj, ca = tile_jobs_vec(kind, what, watd, yld, self.pday, self.plot, day)
        seed_left = seeds[:, :NC].astype(np.int32).copy()

        # 世話係(先頭 nk 体)は家畜、それ以外は作物
        uidx = np.arange(U)[None, :]
        keep = (uidx < nk[:, None]) & umask
        crop = (~keep) & umask
        ops = np.zeros((n, U, 3), np.int32)
        ty = np.full((n, U), -1, np.int32); tx = np.full((n, U), -1, np.int32)
        tj = np.zeros((n, U), np.int32)

        o1, y1, x1, j1, _ = assign_pri(aj, np.zeros_like(aj), JA_PRI, ux, uy, keep)
        take1 = (y1 >= 0)
        zone, uzone = self._zones(kind, nk, umask)
        o2, y2, x2, j2 = V.assign(cj, ca, ux, uy, umask & ~take1, 0, seed_left, H,
                                  zone, uzone)
        ty = np.where(take1, y1, y2); tx = np.where(take1, x1, x2)
        tj = np.where(take1, j1, j2)
        ops[:, :, 1] = np.where(take1, o1[:, :, 1], o2[:, :, 1])
        opmap = np.where(take1, JA_OP[np.clip(j1, 0, 11)], J_OP[np.clip(j2, 0, 4)])

        # 給餌には小麦の携行が要る。無ければ納屋(H-1,H-1)へ積みに行く。
        need_pick = (tj == J_FEED) & ((self.carry_n <= 0) | (self.carry_i != WHEAT))
        ty = np.where(need_pick, H-1, ty); tx = np.where(need_pick, H-1, tx)
        opmap = np.where(need_pick, OP["PICKUP"], opmap)
        ops[:, :, 1] = np.where(need_pick, WHEAT, ops[:, :, 1])
        # 設置も同様に納屋から動物を積む
        need_pa = (tj == J_PLACE) & (self.carry_i != 9) & (self.carry_i != 10) & (self.carry_i != 11)
        ty = np.where(need_pa, H-1, ty); tx = np.where(need_pa, H-1, tx)
        opmap = np.where(need_pa, OP["PICKUP"], opmap)
        first_anim = A_ID[np.argmax(self.pending > 0, 1)]
        ops[:, :, 1] = np.where(need_pa, first_anim[:, None], ops[:, :, 1])

        # PLACE は携行している動物の ID を引数に渡す(0 のままだと engine が拒否する)
        ops[:, :, 1] = np.where((tj == J_PLACE) & ~need_pa,
                                np.maximum(self.carry_i, 0), ops[:, :, 1])
        # 目標に居れば作業、居なければ1歩近づく
        has = (ty >= 0)
        dy = ty - uy; dx = tx - ux
        at = has & (dy == 0) & (dx == 0)
        step = np.where(dy != 0, np.where(dy > 0, OP["S"], OP["N"]),
                        np.where(dx > 0, OP["E"], OP["W"])).astype(np.int32)
        op = np.where(at, opmap, np.where(has, step, OP["PASS"])).astype(np.int32)
        ops[:, :, 0] = op
        ops[:, :, 1] = np.where(at, ops[:, :, 1], 0)
        pick_anim = at & (op == OP["PICKUP"]) & (ops[:, :, 1] >= 9)
        ops[:, :, 2] = np.where(at & (op == OP["PICKUP"]) & ~pick_anim, 4, 1)

        # 携行と世話済みの追跡
        did_pick = at & (op == OP["PICKUP"])
        self.carry_i = np.where(did_pick, ops[:, :, 1], self.carry_i)
        # 動物は1体、小麦は4個。動物に4を入れると「小麦を持っている」と誤認する。
        self.carry_n = np.where(did_pick, np.where(ops[:, :, 1] >= 9, 1, 4), self.carry_n)
        did_feed = at & (op == OP["FEED"])
        self.carry_n = np.where(did_feed, np.maximum(0, self.carry_n - 1), self.carry_n)
        didp = at & (op == OP["PLACE"])
        self.carry_i = np.where(didp, -1, self.carry_i)
        self.carry_n = np.where(didp, 0, self.carry_n)
        rp, up = np.nonzero(didp)
        if rp.size:
            k = np.clip(ops[rp, up, 1] - 9, 0, 2)
            np.subtract.at(self.pending, (rp, k), 1)
            np.maximum(self.pending, 0, out=self.pending)
        didc = at & (op == OP["CARE"])
        rows, us = np.nonzero(didc)
        if rows.size: self.cared[rows, ty[rows, us], tx[rows, us]] = True
        didf = at & (op == OP["COLLECT"])
        rf, uf = np.nonzero(didf)
        if rf.size: self.collected[rf, ty[rf, uf], tx[rf, uf]] = True
        return ops, self.market(day, hour, kind, what, shed, money, seeds, mktinv, nanim)

    def _zones(self, kind, nk, umask):
        """作物係にタイルの担当区画を割る。蛇行順に等分するので区画が連続し移動が減る。"""
        n, U = self.n, self.U
        yy, xx = np.meshgrid(np.arange(BS), np.arange(BS), indexing="ij")
        serp = (yy * BS + np.where(yy % 2 == 0, xx, BS - 1 - xx)).reshape(-1)
        rank0 = np.argsort(np.argsort(serp))                     # 蛇行順の順位 (100,)
        usable = (kind != K_LOCKED).reshape(n, -1)
        h = BS // 2
        blk = ((yy >= h-1) & (yy <= h) & (xx >= h-1) & (xx <= h)).reshape(-1)
        usable &= ~blk[None, :]
        ordr = np.argsort(rank0)
        u_sorted = np.take_along_axis(usable, ordr[None, :].repeat(n, 0), 1)
        rank = np.cumsum(u_sorted, 1) - 1
        cnt = np.maximum(1, usable.sum(1, keepdims=True))
        ncrop = np.maximum(1, umask.sum(1) - nk)[:, None]
        z_sorted = np.minimum(rank * ncrop // cnt, ncrop - 1)
        zone = np.zeros((n, BS*BS), np.int32)
        np.put_along_axis(zone, ordr[None, :].repeat(n, 0), z_sorted.astype(np.int32), 1)
        uidx = np.arange(U)[None, :]
        uzone = np.maximum(0, uidx - nk[:, None]).astype(np.int32)
        return zone, uzone

    # ---------- 市場注文 ----------
    def market(self, day, hour, kind, what, shed, money, seeds, mktinv, nanim):
        n = self.n
        MK = np.zeros((n, 10, 3), np.int32)
        slot = np.zeros(n, np.int32)

        def put(mask, op, item, cnt):
            m = mask & (slot < 10)
            r = np.nonzero(m)[0]
            if r.size == 0: return
            MK[r, slot[r], 0] = op
            MK[r, slot[r], 1] = item[r] if isinstance(item, np.ndarray) else item
            MK[r, slot[r], 2] = cnt[r] if isinstance(cnt, np.ndarray) else cnt
            slot[r] += 1

        mny = money.copy()
        if hour == 0:
            k = np.where(day < 4, self.hire_early, self.hire)
            for i in range(int(k.max())):
                put(i < k, M["HIRE"], 0, np.ones(n, np.int32))
        if hour == 1:
            # 支出の順序が要。種を後回しにすると初日に土地と家畜で使い切り、
            # 作物が回らず二度と資金が戻らない(実測: 8日目で残金0、最終1,246)。
            # 1) 種(成長エンジン)
            for c in range(NC):
                buf = max(1, int(40 // SEED[c]))
                need = np.where(self.want[:, c] > 0,
                                np.minimum(self.want[:, c], 14) + buf - seeds[:, c].astype(np.int32),
                                0)
                cnt = np.minimum(need, (np.maximum(0, mny) // SEED[c]).astype(np.int32))
                ok = cnt > 0
                put(ok, M["BUY_SEED"], c, cnt)
                mny = np.where(ok, mny - cnt * SEED[c], mny)
            # 2) 餌の小麦(切らすと動物が逃げて投資が丸ごと消える)
            nf = nanim * 2 + 6 - shed[:, WHEAT].astype(np.int32)
            ok = (nf > 0) & (mny > 30 * nf) & (nanim > 0)
            put(ok, M["BUY_PRODUCT"], WHEAT, np.maximum(0, nf))
            mny = np.where(ok, mny - 30 * np.maximum(0, nf), mny)
            # 3) 家畜(1日1頭。世話が追いつかないと餓死して丸損)
            held = nanim + self.pending.sum(1)
            mix = self.mix / np.maximum(1, self.mix.sum(1, keepdims=True))
            deficit = mix - self.nbuy / np.maximum(1, self.nbuy.sum(1, keepdims=True))
            deficit = np.where((self.mix > 0) & (29 - day - A_FY[None, :] >= 0), deficit, -1e9)
            pick = deficit.argmax(1)
            c = A_COST[pick]
            can = (held < self.atarget) & (mny > c + 800.0) & (deficit.max(1) > -1e8)
            put(can, M["BUY_ANIMAL"], A_ID[pick].astype(np.int32), np.ones(n, np.int32))
            r = np.nonzero(can)[0]
            if r.size:
                self.nbuy[r, pick[r]] += 1
                self.pending[r, pick[r]] += 1
            mny = np.where(can, mny - c, mny)
            # 4) 土地(余剰があるときだけ。運転資金を割ってはいけない)
            nlock = (kind == K_LOCKED).sum((1, 2))
            cost = np.select([nlock >= 75, nlock >= 50, nlock >= 25],
                             [1000., 2000., 4000.], 4000.)
            canl = (nlock > 0) & (mny > cost + 1500.0 + self.land_res)
            put(canl, M["BUY_LAND"], 0, np.ones(n, np.int32))
            mny = np.where(canl, mny - cost, mny)
        # 売却: 価格経路は単調(待っても戻らない)ので常に全部出す。小麦だけ餌を残す。
        order = [MELON, MILK, WOOL, STRAW, FERT, EGG, TOMATO, CARROT, WHEAT]
        for i in order:
            q = shed[:, i].astype(np.int32)
            if i == WHEAT:
                q = np.maximum(0, q - np.maximum(8, 2 * nanim + 4))
                q = np.where(day >= 29, shed[:, i].astype(np.int32), q)
            put(q > 0, M["SELL"], i, q)
        return MK

    def apply_strategy(self):
        w = self.strat
        self.atarget = np.clip(np.round(8 * w[:, 8]), 0, 24).astype(np.int32)
        self.hire = np.clip(np.round(12 * w[:, 9]), 1, 20).astype(np.int32)
        self.hire_early = np.clip(np.round(6 * w[:, 9]), 1, 12).astype(np.int32)
        self.land_res = np.clip(1000.0 * (w[:, 10] - 1.0), 0.0, 4000.0)
        mx = np.clip(np.round(w[:, 5:8] * 2), 0, 8)
        mx[mx.sum(1) == 0, 0] = 1
        self.mix = mx
        self.kscale = np.clip(w[:, 11], 0.25, 4.0)
