"""exp199: 自己対戦の環境を速くする。sp_lib.SPPool と同じ値を返す BatchSPPool。

   sp_lib.SPPool は**1局ごとに VecMarket を1つ**作り、毎手「局数 x 14回」エンジンを呼んで Python で連結している。
   エンジン(C++)は1コア 206万局面/秒なのに、この組み立てが 8,700局面/秒で頭打ちになっていた(エンジンの約240倍遅い)。
   VecMarket はもともと複数のシードを1つで持てる(ppo_tf.Pool はそう使っている)ので、**全局を1つの VecMarket に入れて**
   毎手の呼び出しを「14回」にする。値は SPPool と**ビット一致**することを check() で確かめる。
   common/ は変えていない。使うときは `sp_lib.SPPool = sp_fast.BatchSPPool` と差し替える。
"""
import os, sys
import random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ==== sp_fast_var(2026-09-24、rule-variant curriculum)====
#   sp_fast.py の写し。違いはエンジンを kagsim_var(ルール変種のつまみつきの写し、既定ではビット一致)にしたことと、
#   BatchSPPool に set_variants / perturb を足したことだけ。つまみを触らなければ sp_fast と同じ値を返す
#   (kagg/bc/test_var_equiv.py と test_var_play.py で確認)。走っているジョブの sp_fast.py / kagsim には触れない。
for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fast"), "/content/drive/MyDrive/kaggriculture/fast"):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.append(_p)
import kagsim_var as kagsim
from deploy import wide_blank
from extract4 import NU



# ---- exp270: 市場の先読み特徴(MKTNF=8 で有効)。sp_lib と同じ計算 ----
import json as _json
MKTNF = int(os.environ.get("MKTNF", "0") or 0)
_MF = _TGTROW = None; _NSHOP = 8
if MKTNF:
    import mkt_fast as _MF
    _t = _json.load(open(os.environ.get("STOCKTGT", "/content/lab/agents/bc_dmftS2_pace/stock_target.json")))
    _tp = _t["products"]
    _TGTROW = np.array([[r[_tp.index(i)] if i in _tp else 0.0 for i in _MF.IT] for r in _t["75"]], np.float32)

# 09-23(マスク移行 mask0→mask2): MASK_MIG=1 のとき、mig_ok[局, 席] が真の席だけ
#   「新マスク(MIG_V、既定 2)で打ち、新マスクでだけ増えた手に lm の bit1(値 3)で印を付ける」。
#   印は model_tf.unit_logits が読み、その手の logit から penalty を引く。印の無い席(相手・テープ・リーグ)は従来どおり。
#   lm は uint8 のままなので**記録の大きさは1バイトも増えない**(mask0 = mask2 & ~expanded で復元できる)。
#   _MIG_SEAT は h2h やものさしのように「呼び出し側が席を決める」場面で BatchSPPool の作成時に使う。
MIG = os.environ.get("MASK_MIG") == "1"
MIG_V = int(os.environ.get("MIG_V", "2"))          # 2 = 新マスクそのまま / **3 = mask2'(型で絞った新マスク)**
MIG_BASE = int(os.environ.get("MIG_BASE", "0"))    # 旧マスクの版(差を取る基準)
# 09-23(方針転換): mask2 が増やす手のほぼ全部は、手番内に何が起きてもエンジンがはねる no-op
#   (納屋に隣接していない取り出し、家畜のいないマスの給餌…)。行動空間のゴミを増やすだけなので、
#   MIG_V=3 では「**手番内に条件が変わりうる型だけ**」を残す = mask2' = mask0 ∪ {下の型}。
#     pickup … 納屋の隣にいる取り出し(先のユニットが同じ手番に入れた品を取る)
#     drop   … 納屋の隣で、その品を持っている置く(先のユニットの取り出しで納屋に空きができる)
#     plant  … 掘れるマスへの植え付け(DIG の後に同じマスへ PLANT)
#     water  … 空きマスへの水やり(先のユニットが植えた直後)  build … 掘れるマスへの小屋/牧草地
#   **エンジンの opt=3 とは別物**。エンジンには opt=2 を渡し、絞り込みは Python 側で行う(opt=0/2 は触らない)。
MIG_KEEP = set(x.strip() for x in os.environ.get("MIG_KEEP", "pickup,drop,plant").split(",") if x.strip())
MIG_VENG = 2 if MIG_V == 3 else MIG_V              # エンジンに渡す版
_SHED_TI = np.array([44, 45, 54, 55], np.int64)    # 納屋の隣 (x,y)=(4,4)(5,4)(4,5)(5,5) → y*10+x
_MIG_SEAT = [None]
PACE = os.environ.get("PACE", "0") == "1"     # exp286: 訓練中にも安全装置をかける
LAND = os.environ.get("LAND", "0") == "1"     # exp290: 4区画目のガードレール(M & M & P & Q の買い方)
_PACE_SKIP = [None]                              # かけない席(テープの相手)。呼び出し側が一時的に設定する
LAND3 = os.environ.get("LAND3") == "1"   # exp305: 3区画目のガードレール
LAND3_P0 = float(os.environ.get("LAND3_P0", "1")); LAND3_P1 = float(os.environ.get("LAND3_P1", os.environ.get("LAND3_P0", "1")))
_LAND_SKIP = [None]                              # exp290 修正: 4区画目のルールをかけない席(ものさしの相手)
OPEN = os.environ.get("OPEN") == "1"   # exp305: 序盤の作付けのガードレール(メロンの種を減らし、小麦に振り替える)
NOLAND4 = os.environ.get("NOLAND4") == "1"   # 09-23(exp318): 4区画目を買わせない(3区画で仕上げさせる)
NOLAND4_P0 = float(os.environ.get("NOLAND4_P0", "1")); NOLAND4_P1 = float(os.environ.get("NOLAND4_P1", os.environ.get("NOLAND4_P0", "1")))
PFNW = os.environ.get("PFNW") == "1"   # exp314: PFN の判断規則(日2-3の牛・日0の家畜構成・序盤の鵞鳥なし)をまとめてかける
PFNW_P0 = float(os.environ.get("PFNW_P0", "1")); PFNW_P1 = float(os.environ.get("PFNW_P1", os.environ.get("PFNW_P0", "1")))
_PACEMOD = None
if PACE or LAND or LAND3 or OPEN or PFNW or NOLAND4:
    import pace_fast as _PACEMOD


class BatchSPPool:
    def __init__(self, seeds):
        w = wide_blank()
        self.env = kagsim.VecMarket(w[0], w[1], w[0], w[1], [int(s) for s in seeds], 720, 0)
        self.env.keep_nonsell = False; self.env.reset()
        self.n = len(seeds)
        self.envs = [self]          # sp_lib の呼び出し側が envs を数えることがあるので形だけ合わせる
        # exp286: 訓練中にも在庫の安全装置(main.py の _pace)をかける。pace_ok[局, 席] が真の席の注文だけに適用。
        # テープ(実戦の上位チームの録画)の席には既定でかけない(_PACE_SKIP で呼び出し側が指定)。
        self.pace_ok = np.full((self.n, 2), PACE, bool)
        if PACE and _PACE_SKIP[0] is not None: self.pace_ok[:, _PACE_SKIP[0]] = False
        self.land_ok = np.full((self.n, 2), LAND, bool)          # exp290
        if LAND and _PACE_SKIP[0] is not None: self.land_ok[:, _PACE_SKIP[0]] = False
        if LAND and _LAND_SKIP[0] is not None: self.land_ok[:, _LAND_SKIP[0]] = False
        # exp305: 序盤のガードレール。open_ok[局, 席] が真の席だけにかける。mel/cred は局ごとの数え(pool は1反復ごとに作り直されるので自動で 0 に戻る)
        self.open_ok = np.full((self.n, 2), OPEN, bool)
        self.nl4_ok = np.full((self.n, 2), NOLAND4, bool)          # exp318: 4区画目を買わせない席
        if NOLAND4 and _PACE_SKIP[0] is not None: self.nl4_ok[:, _PACE_SKIP[0]] = False
        if NOLAND4 and _LAND_SKIP[0] is not None: self.nl4_ok[:, _LAND_SKIP[0]] = False
        if OPEN and _PACE_SKIP[0] is not None: self.open_ok[:, _PACE_SKIP[0]] = False
        if OPEN and _LAND_SKIP[0] is not None: self.open_ok[:, _LAND_SKIP[0]] = False
        self.open_mel = np.zeros((self.n, 2), np.int64); self.open_cred = np.zeros((self.n, 2), np.int64)
        # exp314: PFN の判断規則。pfnw_ok[局, 席] が真の席だけ。ncow は日2-3に買い足した牛の頭数
        self.pfnw_ok = np.full((self.n, 2), PFNW, bool)
        if PFNW and _PACE_SKIP[0] is not None: self.pfnw_ok[:, _PACE_SKIP[0]] = False
        if PFNW and _LAND_SKIP[0] is not None: self.pfnw_ok[:, _LAND_SKIP[0]] = False
        self.pfnw_ncow = np.zeros((self.n, 2), np.int64)
        self.pfnw_nwheat = np.zeros((self.n, 2), np.int64)   # 小麦の種を買った日数(1日1回)
        # 09-22: (局, 席) ごとの合法判定の版(負なら環境変数 LEGAL_OPT)。None なら従来どおり。旧マスクで学習した相手の席を 0 にする
        self.legal_opt = None
        # 09-23(マスク移行): 移行をかける席。既定は全部 False = 従来どおり
        self.mig_ok = np.zeros((self.n, 2), bool)
        if MIG and _MIG_SEAT[0] is not None: self.mig_ok[:, int(_MIG_SEAT[0])] = True

    # ---- rule variants(2026-09-24)----
    def set_variants(self, specs):
        """specs: 局ごとの dict か None。dict の鍵 money/quads/start/land/hire/tt/tt_n(省略は公式の値)。
        reset 直後(1手も進める前)にだけ呼ぶ。None の局には何もしない(= 公式のまま、ビット一致)。"""
        for i, sp in enumerate(specs):
            if not sp: continue
            if "tt" in sp:   # 09-25 tomtown: 最初の tt_n 店を確率 tt‰ でピザ屋/ファーマーズマーケットに(古い .so には無い引数なので鍵があるときだけ渡す)
                self.env.set_variant(int(i), int(sp.get("money", 3000)), int(sp.get("quads", 1)), int(sp.get("start", 0)),
                                     int(sp.get("land", 100)), int(sp.get("hire", 100)), int(sp["tt"]), int(sp.get("tt_n", 8)))
                continue
            self.env.set_variant(int(i), int(sp.get("money", 3000)), int(sp.get("quads", 1)), int(sp.get("start", 0)),
                                 int(sp.get("land", 100)), int(sp.get("hire", 100)))

    def perturb(self, i, sp):
        """局 i の今の局面を書き換える(規則は変えない)。sp の鍵:
           money=[座席0の加算, 座席1の加算] quad=[座席0の解放数, 座席1の解放数] mkt={品目: 在庫の加算}
           seeds={(座席, 作物): 個数} shed={(座席, 品目): 個数}"""
        e = self.env; i = int(i)
        for s_, d_ in enumerate(sp.get("money", ())):
            if d_: e.add_money(i, s_, float(d_))
        for s_, n_ in enumerate(sp.get("quad", ())):
            for _ in range(int(n_)): e.unlock_quad(i, s_)
        for it_, d_ in sp.get("mkt", {}).items():
            if d_: e.market_add(i, int(it_), int(d_))
        for (s_, c_), n_ in sp.get("seeds", {}).items(): e.add_seeds(i, int(s_), int(c_), int(n_))
        for (s_, it_), n_ in sp.get("shed", {}).items(): e.add_shed(i, int(s_), int(it_), int(n_))

    def state(self, seat):
        e = self.env; e.set_view(int(seat))
        tiles = np.stack([np.concatenate([np.asarray(e.tiles_base(s)), np.asarray(e.tiles_dyn(s))], 3).reshape(-1, 100, 13) for s in (0, 1)], 1)
        up = np.asarray(e.unit_pos2(0))[:, :NU]; op = np.asarray(e.unit_pos2(1))[:, :NU]; inv = np.asarray(e.unit_inv(NU)); uinv = inv   # 09-23: MKTNF 枝で inv が上書きされるので、人手の携行品は別名で保持
        OU = np.concatenate([(up[..., :1] >= 0), up, inv], 2).astype(np.float32)
        PU = np.concatenate([(op[..., :1] >= 0), op], 2).astype(np.float32)
        G = np.concatenate([np.asarray(e.obs_legal()), np.asarray(e.farm_pub(0)), np.asarray(e.farm_pub(1)),
                            np.asarray(e.shops()).astype(np.float32), np.asarray(e.shed_all())], 1).astype(np.float32)
        if MKTNF:                                   # exp270: 市場の先読み特徴(sp_lib.SPPool と同じ値)
            inv = np.rint(G[:, 0:9] * 10000).astype(np.int64)
            shed = np.rint(G[:, 18:27] * 100).astype(np.int64)
            sh = np.asarray(e.shops())[:, :16]
            un = np.zeros((len(inv), _NSHOP), np.float32)
            for r in range(len(inv)):
                for v in sh[r]:
                    if 0 <= v < _NSHOP: un[r, v] += 1.0      # 同じ店が複数開くので数える
            ti = int(e.step_index()); row = _TGTROW[min(ti, len(_TGTROW) - 1)]
            MK = _MF.feats(inv, shed, _MF.shop_demand(un), np.broadcast_to(row, inv.shape))
            if MKTNF >= 16:                          # exp274: 後半8特徴(相手の推定納屋・売りの履歴・相手の盤面・値段と在庫の24手変化)
                if not hasattr(self, "_h16"): self._h16 = {}; self._h16_step = {}; self._h16_out = {}
                if self._h16_step.get(seat) != ti:   # 1手に1回だけ履歴を進める(同じ手に2回呼ばれても二重に数えない)
                    if seat not in self._h16: self._h16[seat] = _MF.Hist16(len(inv))
                    ax_s = np.asarray(e.mkt_aux(0)); ax_o = np.asarray(e.mkt_aux(1))
                    pr_tbl = _MF.table()[0]
                    prices = pr_tbl[np.arange(9)[None, :], np.clip(inv - _MF.LO, 0, _MF.HI - _MF.LO)]
                    self._h16_out[seat] = self._h16[seat].push(inv, prices, ax_s[:, :, 0], ax_o[:, :, 0], ax_o[:, :, 1], ax_o[:, :, 2], ax_o[:, :, 3])
                    self._h16_step[seat] = ti
                MK = np.concatenate([MK, self._h16_out[seat]], -1)
            return tiles.astype(np.float16), OU, PU, G, self._lm(e, seat, tiles, up, uinv), MK
        return tiles.astype(np.float16), OU, PU, G, self._lm(e, seat, tiles, up, uinv)

    def _keep3(self, tiles, up, inv):
        """mask2' の絞り込み。増えた手のうち「手番内に条件が変わりうる型」だけ True。"""
        own = np.asarray(tiles[:, 0], np.float32)                     # (局,100,13) 自分の農場
        upa = np.asarray(up)
        ux = np.clip(upa[..., 0].astype(np.int64), 0, 9); uy = np.clip(upa[..., 1].astype(np.int64), 0, 9)
        ti = uy * 10 + ux
        tl = np.take_along_axis(own, ti[..., None], 1)                # (局,人手,13) 立っているマス
        kd = np.rint(tl[..., 0] * 5).astype(np.int64); has = tl[..., 6] > 0.5
        adj = np.isin(ti, _SHED_TI); digg = (kd != 1) & ~has          # 掘れるマス(DIG の前提)
        keep = np.zeros((own.shape[0], NU, 44), bool)
        if "pickup" in MIG_KEEP: keep[:, :, 20:32] |= adj[..., None]
        if "drop" in MIG_KEEP: keep[:, :, 32:44] |= adj[..., None] & (np.asarray(inv)[:, :NU, :12] > 0.5)
        if "plant" in MIG_KEEP: keep[:, :, 15:20] |= digg[..., None]
        if "water" in MIG_KEEP: keep[:, :, 5:6] |= (kd == 0)[..., None]
        if "build" in MIG_KEEP: keep[:, :, 9:11] |= digg[..., None]
        return keep

    def _lm(self, e, seat, tiles=None, up=None, inv=None):
        ok = self.mig_ok[:, int(seat)] if MIG else None
        if ok is None or not ok.any():
            if self.legal_opt is None: return np.asarray(e.legal_mask(NU))
            return np.asarray(e.legal_mask(NU, np.ascontiguousarray(self.legal_opt[:, seat], dtype=np.int32)))
        # マスク移行の席: 新マスクと旧マスクを両方取り、増えた手に bit1 を立てる(値 3)。他の席は従来の版のまま
        base = (np.full(self.n, -1, np.int32) if self.legal_opt is None
                else np.ascontiguousarray(self.legal_opt[:, seat], dtype=np.int32))
        lm2 = np.asarray(e.legal_mask(NU, np.ascontiguousarray(np.where(ok, MIG_VENG, base), dtype=np.int32))).astype(np.uint8, copy=False)
        lm0 = np.asarray(e.legal_mask(NU, np.ascontiguousarray(np.where(ok, MIG_BASE, base), dtype=np.int32))).astype(np.uint8, copy=False)
        xp = ok[:, None, None] & (lm2 != 0) & (lm0 == 0)
        if MIG_V == 3 and tiles is not None: xp &= self._keep3(tiles, up, inv)   # mask2': 落とした型は非合法に戻す
        allow = ((lm0 != 0) | xp).astype(np.uint8)
        return allow | (xp.astype(np.uint8) * np.uint8(2))  # 0=非合法 1=旧マスクでも合法 3=増えた手(残した型だけ)

    def step(self, U0, O0, U1, O1):
        if PACE and self.pace_ok.any():            # exp286: 方策の行動はそのまま(学習側に残る)、エンジンに渡す注文だけ後処理
            e = self.env; t = int(e.step_index())
            OO = [O0, O1]
            for s_ in (0, 1):
                rows = np.nonzero(self.pace_ok[:, s_])[0]
                if not len(rows): continue
                e.set_view(s_)
                g = np.concatenate([np.asarray(e.obs_legal()), np.asarray(e.farm_pub(0))], 1)
                shed = np.rint(g[:, 18:27] * 100).astype(np.int64)        # state() の G[:, 18:27] と同じ(自分の納屋)
                sub = np.array(OO[s_][rows], np.int32, copy=True)
                _PACEMOD.apply_codes(sub, shed[rows], t)
                OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
            O0, O1 = OO
        if LAND and self.land_ok.any():            # exp290: 4区画目を買う注文を足す(修理層の後。最終盤は触らない)
            e = self.env; t = int(e.step_index())
            if _PACEMOD.LAND_T <= t < _PACEMOD.LIQ_START:
                OO = [O0, O1]
                for s_ in (0, 1):
                    rows = np.nonzero(self.land_ok[:, s_])[0]
                    if not len(rows): continue
                    e.set_view(s_); fp = np.asarray(e.farm_pub(0))
                    nq = np.rint(fp[:, 2] * 4).astype(np.int64); mo = fp[:, 0] * 50000.0
                    sub = np.array(OO[s_][rows], np.int32, copy=True)
                    _PACEMOD.apply_land(sub, nq[rows], mo[rows], t)
                    OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
                O0, O1 = OO
        if LAND3 and self.land_ok.any():           # exp305: 3区画目を早める注文を足す(4区画目版と同じ作り)
            e = self.env; t = int(e.step_index())
            if _PACEMOD.LAND3_T <= t < _PACEMOD.LIQ_START:
                OO = [O0, O1]
                for s_ in (0, 1):
                    rows = np.nonzero(self.land_ok[:, s_])[0]
                    if not len(rows): continue
                    e.set_view(s_); fp = np.asarray(e.farm_pub(0))
                    nq = np.rint(fp[:, 2] * 4).astype(np.int64); mo = fp[:, 0] * 50000.0
                    sub = np.array(OO[s_][rows], np.int32, copy=True)
                    _PACEMOD.apply_land3(sub, nq[rows], mo[rows], t)
                    OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
                O0, O1 = OO
        if NOLAND4 and self.nl4_ok.any():          # exp318: 区画が3つある局の区画購入の注文を落とす
            e = self.env; t = int(e.step_index()); OO = [O0, O1]
            for s_ in (0, 1):
                rows = np.nonzero(self.nl4_ok[:, s_])[0]
                if not len(rows): continue
                e.set_view(s_); fp = np.asarray(e.farm_pub(0))
                nq_ = np.rint(fp[:, 2] * 4).astype(np.int64)
                sub = np.array(OO[s_][rows], np.int32, copy=True)
                _PACEMOD.apply_noland4(sub, nq_[rows], t)
                OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
            O0, O1 = OO
        if OPEN and self.open_ok.any():            # exp305: 序盤のメロンの種を上限まで削り、削った分を小麦の種に振り替える
            e = self.env; t = int(e.step_index())
            if t < _PACEMOD.OPEN_T1:
                OO = [O0, O1]
                for s_ in (0, 1):
                    rows = np.nonzero(self.open_ok[:, s_])[0]
                    if not len(rows): continue
                    sub = np.array(OO[s_][rows], np.int32, copy=True)
                    mel_ = self.open_mel[rows, s_].copy(); cred_ = self.open_cred[rows, s_].copy()
                    _PACEMOD.apply_open(sub, mel_, cred_, t)
                    self.open_mel[rows, s_] = mel_; self.open_cred[rows, s_] = cred_
                    OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
                O0, O1 = OO
        if PFNW and self.pfnw_ok.any():            # exp314: PFN の判断規則(まとめてかける)
            e = self.env; t = int(e.step_index())
            OO = [O0, O1]
            for s_ in (0, 1):
                rows = np.nonzero(self.pfnw_ok[:, s_])[0]
                if not len(rows): continue
                e.set_view(s_)
                gg = np.concatenate([np.asarray(e.obs_legal()), np.asarray(e.farm_pub(0))], 1)
                shed_ = np.rint(gg[:, 18:27] * 100).astype(np.int64)   # 自分の納屋(PACE と同じ取り方)
                fp = np.asarray(e.farm_pub(0)); mo = fp[:, 0] * 50000.0
                sub = np.array(OO[s_][rows], np.int32, copy=True)
                nc_ = self.pfnw_ncow[rows, s_].copy(); nw_ = self.pfnw_nwheat[rows, s_].copy()
                nq_ = np.rint(fp[:, 2] * 4).astype(np.int64)
                _PACEMOD.apply_pfnw(sub, mo[rows], nc_, t, shed_[rows], nw_, nq_[rows])
                self.pfnw_ncow[rows, s_] = nc_; self.pfnw_nwheat[rows, s_] = nw_
                OO[s_] = np.array(OO[s_], np.int32, copy=True); OO[s_][rows] = sub
            O0, O1 = OO
        self.env.step_both(U0, O0, U1, O1)

    def money(self):
        self.env.set_view(0); return np.asarray(self.env.money())

    def shops(self):
        return np.asarray(self.env.shops())

    def step_index(self):
        return self.env.step_index()


def check(N=16, T=200, seed0=12345):
    """同じシード・同じ(乱数の)行動で SPPool と BatchSPPool を進め、毎手の特徴と所持金が完全に同じか。"""
    import time
    import sp_lib as S
    seeds = [seed0 + 7919 * k for k in range(N)]
    a, b = S.SPPool(seeds), BatchSPPool(seeds)
    rng = np.random.default_rng(0); bad = 0; ta = tb = 0.0
    for t in range(T):
        for s in (0, 1):
            x0 = time.time(); sa = a.state(s); ta += time.time() - x0
            x0 = time.time(); sb = b.state(s); tb += time.time() - x0
            for k in range(5):
                if not np.array_equal(sa[k], sb[k]): bad += 1
        # 合法手の中から乱数で選んだ行動(ユニットは合法の候補、市場は空)を両方に同じく渡す
        import ppo_tf as P
        acts = []
        for s in (0, 1):
            st = b.state(s); lm = st[4]; ou = st[1]
            ul = np.full((N, NU), -1, np.int64)
            for i in range(N):
                for u in range(NU):
                    if ou[i, u, 0] > 0:
                        c = np.flatnonzero(lm[i, u]); ul[i, u] = rng.choice(c) if len(c) else 0
            uq = np.zeros((N, NU), np.int64); ms = np.full((N, 11), 22, np.int64); mq = np.zeros((N, 11), np.int64)
            acts.append(P.to_codes(ul, uq, ms, mq, ou))
        a.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1]); b.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    ma, mb = a.money(), b.money()
    print("恒等式: %d局 x %d手 x 両座席 の特徴5種で不一致 %d 件 / 最終の所持金 完全一致 %s" % (N, T, bad, np.array_equal(ma, mb)))
    print("特徴づくり: SPPool %.1f ms/手・座席 → BatchSPPool %.2f ms/手・座席(%.0f倍)" % (1000 * ta / (2 * T), 1000 * tb / (2 * T), ta / max(tb, 1e-9)))
    return bad == 0 and np.array_equal(ma, mb)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ok = check(N=N)
    sys.exit(0 if ok else 1)


TAPE_RAND = os.environ.get("TAPE_RAND") == "1"   # exp313: 局ごとにテープを乱択する(既定 0 = 従来どおり t=144 の店の規則)


class FastTapeRouterOpp:
    """sp_lib.TapeRouterOpp と同じ手を出す。毎手 json.loads(json.dumps(...)) で深く複製していたのを、
       変更する市場の注文の列だけ複製するようにした(中の注文は書き換えず、足すだけなので同じ結果になる)。"""
    def __init__(self, agent_dir):
        import sp_lib as S
        self._base = S.TapeRouterOpp(agent_dir)
        self.T, self.rule, self.pick = self._base.T, self._base.rule, {}
        self.SH = S.TapeRouterOpp.SH; self._IT9 = S.IT9
        from hybrid5_tc import tape_codes
        self._tc = tape_codes
        self._rng = random.Random(20260913)   # exp313: TAPE_RAND 用。種を固定して再現できるように

    def act(self, st, t, idx, pool, seat):
        if TAPE_RAND and t == 0:          # exp313: 局の頭でテープを引き直す(t<144 も T[0] 固定にしない)
            for k in idx: self.pick[k] = self._rng.randrange(len(self.T))
        if t == 144 and not TAPE_RAND:
            sh = pool.shops()
            for j, k in enumerate(idx):
                names = sorted(self.SH[int(x)] for x in sh[k] if x >= 0)[:2]
                self.pick[k] = self.rule.get("|".join(names), 0)
        U, O = [], []
        for j, k in enumerate(idx):
            tape = self.T[self.pick.get(k, 0) if (TAPE_RAND or t >= 144) else 0]
            src = tape[min(t, len(tape) - 1)]
            a = dict(src); m = list(src.get("market") or []); a["market"] = m
            if t >= 700:
                shed = np.rint(st[3][j, 18:27] * 100).astype(int)
                have = {o[1] for o in m if isinstance(o, list) and len(o) >= 2 and o[0] == "SELL"}
                for it in ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER"):
                    q = int(shed[self._IT9.index(it)])
                    if q > 0 and it not in have and len(m) < 10: m.append(["SELL", it, q]); have.add(it)
            u, o = self._tc(a, st[1][j], st[4][j]); U.append(u); O.append(o)
        return np.stack(U), np.stack(O)


def check_tape(oppdir, N=16):
    """TapeRouterOpp と FastTapeRouterOpp を同じ局面で動かし、719手すべてで出す手が同じか(相手側は同じテープで固定)。"""
    import sp_lib as S, time
    seeds = [777 + 7919 * k for k in range(N)]
    pa, pb = BatchSPPool(seeds), BatchSPPool(seeds)
    A0, B0 = S.TapeRouterOpp(oppdir), FastTapeRouterOpp(oppdir)
    A1, B1 = S.TapeRouterOpp(oppdir), FastTapeRouterOpp(oppdir)
    idx = list(range(N)); bad = 0; ta = tb = 0.0
    for t in range(719):
        sa0, sa1, sb0, sb1 = pa.state(0), pa.state(1), pb.state(0), pb.state(1)
        x = time.time(); ua0, oa0 = A0.act(sa0, t, idx, pa, 0); ua1, oa1 = A1.act(sa1, t, idx, pa, 1); ta += time.time() - x
        x = time.time(); ub0, ob0 = B0.act(sb0, t, idx, pb, 0); ub1, ob1 = B1.act(sb1, t, idx, pb, 1); tb += time.time() - x
        bad += int(not (np.array_equal(ua0, ub0) and np.array_equal(oa0, ob0) and np.array_equal(ua1, ub1) and np.array_equal(oa1, ob1)))
        pa.step(ua0, oa0, ua1, oa1); pb.step(ub0, ob0, ub1, ob1)
    same = np.array_equal(pa.money(), pb.money())
    print("TapeRouterOpp と同じ手: 不一致の手番 %d / 719、最終の所持金一致 %s。時間 %.2f s → %.2f s(%.1f倍)" % (bad, same, ta, tb, ta / max(tb, 1e-9)))
    return bad == 0 and same
