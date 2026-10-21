"""生成器: 毎ターン盤面を見て各ユニットにタスクを割り当てる。

上位問題(local/plan_lp.py)の解が「何をどれだけ作るか」を決め、ここは
「時間窓つき多台巡回」を優先度で解く下位問題。優先度は
    score = 限界コイン / (実行1手 + 距離)
で、限界コインは**その時点の市場在庫での価格**から取るので、店の抽選も
相手の売りも自動的に入る(固定計画に対して +12.4% と見積もった部分)。

engine の規約(sim.hpp):
  ・毎日の始めに全ユニットが消え、位置は (h-1,h-1)、手持ちは空。雇用はフィボナッチ価格で毎日リセット。
  ・SELL は小屋からのみ。小屋容量100を超えた分は日終わりに捨てられる。
  ・FEED は手持ちの小麦1、FERTILIZE は手持ちの肥料1、PLACE(動物)は手持ちの動物1が要る。
  ・PICKUP/DROP は小屋隣接マスのみ。
  ・非継続作物は齢0の水やり必須、以後1日おきで枯れない。収量増は w0..max_yield_day の水やりのみ。
"""
import sys, os, numpy as np
sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagsim
BS = 10
OP = dict(PASS=0, N=1, S=2, E=3, W=4, PICKUP=5, DROP=6, PLACE=7, PLANT=8, WATER=9,
          HARVEST=10, FERTILIZE=11, DIG=12, BUILD_COOP=13, BUILD_PASTURE=14,
          FEED=15, COLLECT=16, CARE=17)
M = dict(NONE=0, HIRE=1, BUY_LAND=2, BUY_SEED=3, BUY_PRODUCT=4, BUY_ANIMAL=5, SELL=6)
WHEAT, CARROT, TOMATO, STRAW, MELON, EGG, MILK, WOOL, FERT = range(9)
GOOSE, COW, SHEEP = 9, 10, 11
CROP = {WHEAT: (4, 6, 10), CARROT: (3, 4, 20), TOMATO: (8, 4, 50), STRAW: (10, 4, 100), MELON: (12, 6, 80)}
ONGOING = {TOMATO, STRAW}
ANIMAL = {GOOSE: (300, "COOP", 4, 1, 4, EGG), COW: (400, "PASTURE", 8, 2, 6, MILK),
          SHEEP: (500, "PASTURE", 6, 3, 6, WOOL)}
KIND_EMPTY, KIND_LOCKED, KIND_WEED, KIND_COOP, KIND_PASTURE, KIND_PLANT = 0, 1, 2, 3, 4, 5
# 価格 (sim.hpp:110 と同一)
import math
MK = [(25,400,"SQRT",.80,"LOG",.20),(35,450,"HINGE",1.,"SQRT",.70),(60,200,"HINGE",.40,"SQRT",.60),
      (120,100,"SQRT",.70,"LIN",1.60),(250,300,"LOG",.20,"SQ",3.60),(50,332,"HINGE",.40,"LOG",.20),
      (160,122,"SQRT",.60,"LIN",1.60),(200,105,"LOG",.20,"SQ",3.20),(100,200,"LIN",.40,"LIN",.40)]
def _sh(f,x,T):
    x=max(x,0.)
    return x if f=="LIN" else x*x if f=="SQ" else math.sqrt(x) if f=="SQRT" else math.log1p(x) if f=="LOG" \
        else (x/T)+8.*max(x/T-1.,0.)**2
AMP=[(bt*b/_sh(bf,T,T), at*b/_sh(af,T,T)) for b,T,bf,bt,af,at in MK]
def price(i, inv):
    b,T,bf,bt,af,at=MK[i]
    p = b+AMP[i][0]*_sh(bf,10000-inv,T) if inv<10000 else b-AMP[i][1]*_sh(af,inv-10000,T)
    return max(1., p)

DEF = dict(w_dist=0.5, w_far=0.3, n_keepers=2, anim_per_keeper=6, fert_carry=3, build_ahead=0, lane_mode="row", fert_ratio=1.0, keep_fert=10, feed_carry=1, release=1, tour=1, anim_px=0.6, anim_roi=0.0, anywhere=1, anim_per_day=2, animal_last_day=22, hire_reserve=60.0, solo_crop=1, route_collect=1, collect_px=0.0, kroute=0, ring_day=0, chain=10, sweep=1, seed_floor=12, land_empty=60, hire_early=5, hire_day=6,
           animal_day=2, w_place=1.0, open_cow=2, open_sheep=2, open_feed=12, feed_days=2, open_melon=10, open_wheat=7, zones=1, nzones=3, w_water=1.0, w_harv=1.0, w_plant=1.0, w_feed=1.0, w_care=1.0,
           w_fert=1.0, w_coll=1.0, w_dig=1.0, hire=13, sell_frac=1.0, keep_wheat=25.0,
           plant_mix=(1.0, .3, .0, .2, .6), n_sheep=6, n_cow=6, n_goose=3, animal_cash=1500.0, ring=2, w_surv=30.0, sell_gate=0.0,
           seed_k=1.2, melon_cap=30, reserve=200.0, buy_land=2, land_reserve=300.0)

class Dispatcher:
    """1シードぶんの状態。ベクトル化はせず、シードごとに1個持つ。"""
    def __init__(self, p):
        self.p = dict(DEF); self.p.update(p or {})
        self.crop = np.full((BS, BS), -1, np.int8)      # 自分が植えた作物
        self.pday = np.full((BS, BS), -1, np.int16)
        self.fert = np.full((BS, BS), -1, np.int16)
        self.inv = None                                  # [nu, 12]
        self.target = {}                                 # unit -> (y,x,op,arg)
        self.shed_anim = {GOOSE: 0, COW: 0, SHEEP: 0}   # 小屋にいる未設置の動物(自前追跡)
        self.cared = np.full((BS, BS), -1, np.int16)     # 最後に世話した日(cared_today は観測に無い)
        self.melon_planted = 0                           # メロンは町が吸わないので総量に上限

    def act(self, day, hour, kind, what, watd, fedd, yld, ux, uy, shed, money, seeds, ninv, favail=None):
        """1ターンぶん。返り値 (unit_ops [nu,3], market [<=10,3])"""
        p = self.p; nu = len(ux); ops = np.zeros((nu, 3), np.int32); mk = []; self._day = day
        inv_mkt = self.mktinv
        if not hasattr(self, "pending_buy"): self.pending_buy = {GOOSE: 0, COW: 0, SHEEP: 0}
        for a in (GOOSE, COW, SHEEP): self.shed_anim[a] += self.pending_buy[a]; self.pending_buy[a] = 0
        # (2) 前ターンの PLACE が通ったかを盤面で確認してから手持ちを減らす
        lp = getattr(self, "last_place", None)
        if lp:
            u0, y0, x0, a0 = lp
            if u0 < self.inv.shape[0] and what[y0, x0] == a0: self.inv[u0, a0] = max(0, self.inv[u0, a0] - 1)
            self.last_place = None
        gone = (self.crop >= 0) & (kind != KIND_PLANT)
        self.crop[gone] = -1; self.pday[gone] = -1; self.fert[gone] = -1
        fresh = (kind == KIND_PLANT) & (self.crop < 0)          # 知らない作物(通常は無い)
        self.crop[fresh] = what[fresh]; self.pday[fresh] = day
        if hour == 0 and self.inv is not None:
            for a in (GOOSE, COW, SHEEP): self.shed_anim[a] += int(self.inv[:, a].sum())
            self.inv[:] = 0; self.target = {}
        if self.inv is None or self.inv.shape[0] != nu:
            iv = np.zeros((max(nu, 1), 12), np.int32)
            if self.inv is not None: iv[:min(nu, self.inv.shape[0])] = self.inv[:min(nu, self.inv.shape[0])]
            self.inv = iv
        shed_anim = self.shed_anim
        # 翌朝の雇用に要る現金を実費で確保する(賃金はフィボナッチ)。
        # 固定値の床にしていたため、作付けや家畜を増やすと必ず資金が尽きて雇用が崩れていた。
        if p["hire_reserve"] < 0:
            k_ = int(p["hire_early"]) if day + 1 < p["hire_day"] else int(p["hire"])
            a_, b_, tot_ = 1, 1, 0
            for _ in range(min(k_, 13)): tot_ += a_; a_, b_ = b_, a_ + b_
            hire_res = float(tot_)
        else:
            hire_res = float(p["hire_reserve"])
        # --- 市場: 雇用・種・動物・売却
        nanim = int((((kind == KIND_PASTURE) | (kind == KIND_COOP)) & (what >= 9)).sum())
        if hour == 0:
            k = int(p["hire_early"]) if day < p["hire_day"] else int(p["hire"])
            for _ in range(min(k, 10)): mk.append((M["HIRE"], 0, 1))
            self._hire_more = max(0, k - 10)
            if day == 0:
                # 開幕はテープと同じ形: メロン12(日10-12に約72個=15k超)、牛2羊2、餌の小麦
                for a, n0 in ((COW, int(p["open_cow"])), (SHEEP, int(p["open_sheep"]))):
                    if n0 > 0: mk.append((M["BUY_ANIMAL"], a, n0)); self.pending_buy[a] += n0; money -= ANIMAL[a][0] * n0
                nm = int(p["open_melon"])
                while nm > 0 and money - 80 * nm - 10 * int(p["open_wheat"]) - 25 * int(p["open_feed"]) < hire_res: nm -= 1
                if nm > 0: mk.append((M["BUY_SEED"], MELON, nm)); money -= 80 * nm
                if p["open_wheat"] > 0: mk.append((M["BUY_SEED"], WHEAT, int(p["open_wheat"]))); money -= 10 * int(p["open_wheat"])
                if p["open_feed"] > 0: mk.append((M["BUY_PRODUCT"], WHEAT, int(p["open_feed"])))
        # 種の買い足し(hour 1: 雇用の後、資金を見て)。目標在庫は作付け候補の空きマス数に比例
        if hour == 1 and getattr(self, "_hire_more", 0) > 0:
            for _ in range(min(self._hire_more, 10 - len(mk))): mk.append((M["HIRE"], 0, 1))
            self._hire_more = 0
        if hour == 1 and day > 0:
            need_feed = (nanim + sum(self.shed_anim.values())) * int(p["feed_days"]) - int(shed[WHEAT])
            if need_feed > 0 and money > 25 * need_feed + hire_res and len(mk) < 10:
                mk.append((M["BUY_PRODUCT"], WHEAT, int(need_feed))); money -= 25 * need_feed
            empty = int((kind == KIND_EMPTY).sum()) + int((kind == KIND_WEED).sum())
            ripe = int(((kind == KIND_PLANT) & (yld > 0)).sum())
            mix = np.array(p["plant_mix"], float); mix = mix / max(1e-9, mix.sum())
            # 空き+収穫予定のマス全部に植えられる分を、安い作物から順に資金の許す範囲で
            slots = max(int(p["seed_floor"]), empty + ripe)
            for ci, c in sorted(enumerate((WHEAT, CARROT, TOMATO, STRAW, MELON)), key=lambda z: CROP[z[1]][2]):
                want = int(round(mix[ci] * slots * p["seed_k"]))
                if c == MELON: want = min(want, int(p["melon_cap"]) - self.melon_planted)
                need = want - int(seeds[c])
                cost = CROP[c][2]
                if need <= 0 or len(mk) >= 10: continue
                afford = int((money - p["reserve"] - hire_res) // cost)
                need = min(need, afford)
                if need > 0:
                    mk.append((M["BUY_SEED"], c, need)); money -= cost * need
            # 土地: 資金に余裕があれば開ける(1000/2000/4000)
            # 予約リングは day<ring_day のあいだ作物で埋まり、以後 EMPTY にならないので
            # そこだけを数えると牧場を建てる場所が永久に生まれない(実エンジンで free_slots=0 のまま動物0頭)。
            # 空の牧場/鶏舎 + 予約リングの空き + (足りなければ)任意の空きマス を数える。
            free_slots = sum(1 for y in range(BS) for x in range(BS)
                             if (kind[y, x] in (KIND_PASTURE, KIND_COOP) and what[y, x] == 0)
                             or (kind[y, x] == KIND_EMPTY and (self._reserved(y, x) or p["anywhere"])))
            pending_all = sum(shed_anim.values()) + int(self.inv[:, 9:12].sum()) + sum(self.pending_buy.values())
            bought_today = 0
            for a, tgt in ((COW, p["n_cow"]), (SHEEP, p["n_sheep"]), (GOOSE, p["n_goose"])):
                have = int((what == a).sum()) + int(shed_anim.get(a, 0)) + int(self.inv[:, a].sum())
                cost = ANIMAL[a][0]
                prod = ANIMAL[a][5]
                # 残り日数から期待生産を見積もり、投資として見合うかで判定する。
                # 「現在価格 >= 基準価格」にしていたため、自分が売って値が下がると恒久的に買えなくなっていた。
                _, _, fy, iv, mx, _ = ANIMAL[a]
                left = max(0, 29 - day - fy)
                exp_units = (left // max(1, iv)) * min(mx, 1 + iv)     # 世話込みの1回あたり
                px_ok = (exp_units * price(prod, inv_mkt[prod]) > cost * p["anim_roi"]) if p["anim_roi"] > 0 \
                        else (price(prod, inv_mkt[prod]) >= p["anim_px"] * float(MK[prod][0]))
                if day >= p["animal_day"] and have < tgt and money > cost + p["animal_cash"] + hire_res and len(mk) < 10 \
                        and pending_all < free_slots and shed_anim.get(a, 0) == 0 and px_ok \
                        and bought_today < int(p["anim_per_day"]) and day <= int(p["animal_last_day"]):
                    bought_today += 1
                    mk.append((M["BUY_ANIMAL"], a, 1)); money -= cost; self.pending_buy[a] += 1
            nlock = int((kind == KIND_LOCKED).sum()); nq = 4 - nlock // 25
            cost_land = (1000, 2000, 4000)[min(2, nq - 1)]
            if p["buy_land"] and nq < 1 + int(p["buy_land"]) and empty < p["land_empty"] \
                    and money > cost_land + p["land_reserve"] and len(mk) < 10:
                mk.append((M["BUY_LAND"], 0, 1))
        # 販売: 小屋にある物を、限界価格が高い順に少しずつ
        inv_mkt = self.mktinv
        if hour in (2, 8, 14, 20) or (day >= 29 and hour % 3 == 0) or (hour == 22 and money < hire_res):
            order = sorted(range(9), key=lambda i: -price(i, inv_mkt[i]))
            for i in order:
                if len(mk) >= 10: break
                q = int(shed[i])
                pend_anim = nanim + sum(self.shed_anim.values()) + int(self.inv[:, 9:12].sum())
                if i == WHEAT and day < 29: q = max(0, q - (int(p["keep_wheat"]) if pend_anim > 0 else 0))
                if i == FERT and day < 29 and p["fert_carry"] > 0 and price(FERT, inv_mkt[FERT]) < p["fert_ratio"] * 2 * price(WHEAT, inv_mkt[WHEAT]):
                    q = max(0, q - int(p["keep_fert"]))
                if q <= 0: continue
                if price(i, inv_mkt[i]) < p["sell_gate"]: continue
                mk.append((M["SELL"], i, max(1, int(q * p["sell_frac"]))))
        # --- タスクを作る
        tasks = []   # (score, y, x, op, arg, n)
        w0 = {c: (CROP[c][0] + 1) // 2 for c in CROP}
        for y in range(BS):
            for x in range(BS):
                k = kind[y, x]
                if k == KIND_LOCKED: continue
                if k == KIND_PLANT:
                    c = int(self.crop[y, x]);  c = c if c >= 0 else int(what[y, x])
                    if c not in CROP: continue
                    myd, mx, _ = CROP[c]; age = day - int(self.pday[y, x])
                    pr = price(c, inv_mkt[c])
                    if not watd[y, x]:
                        if c in ONGOING:
                            g = 1.0
                        else:
                            g = (2.0 if self.fert[y, x] >= day else 1.0) if w0[c] <= age <= myd else 0.0
                            if yld[y, x] >= mx: g = 0.0
                        surv = p["w_surv"] if (age % 2 == 0 or age == 0) else 0.0   # 乾き2日で雑草化
                        tasks.append((p["w_water"] * (g * pr + surv), y, x, OP["WATER"], 0, 1))
                    if yld[y, x] > 0 and (c in ONGOING or age >= myd or yld[y, x] >= mx):
                        tasks.append((p["w_harv"] * yld[y, x] * pr, y, x, OP["HARVEST"], 0, 1))
                    if c not in ONGOING and age == w0[c] and self.fert[y, x] < day \
                            and price(FERT, inv_mkt[FERT]) < p["fert_ratio"] * 2 * pr:
                        tasks.append((p["w_fert"] * 2 * pr, y, x, OP["FERTILIZE"], 0, 1))
                elif k == KIND_EMPTY:
                    if self._reserved(y, x): continue
                    best = None
                    for ci, c in enumerate((WHEAT, CARROT, TOMATO, STRAW, MELON)):
                        if seeds[c] <= 0: continue
                        myd, mx, _ = CROP[c]
                        v = p["plant_mix"][ci] * mx * price(c, inv_mkt[c]) / (myd + 1)
                        if best is None or v > best[0]: best = (v, c)
                    if best:
                        dsh = abs(y - (BS // 2 - 1)) + abs(x - (BS // 2 - 1))
                        tasks.append((p["w_plant"] * best[0] - p["w_far"] * dsh, y, x, OP["PLANT"], best[1], 1))
                elif k == KIND_WEED:
                    tasks.append((p["w_dig"] * 40, y, x, OP["DIG"], 0, 1))
                elif k in (KIND_COOP, KIND_PASTURE):
                    a = int(round(what[y, x]))
                    if a in ANIMAL:
                        _, _, fy, iv, mx, prod = ANIMAL[a]
                        pr = price(prod, inv_mkt[prod])
                        if not fedd[y, x]: tasks.append((p["w_feed"] * 200.0, y, x, OP["FEED"], 0, 1))
                        if yld[y, x] > 0: tasks.append((p["w_harv"] * yld[y, x] * pr, y, x, OP["HARVEST"], 0, 1))
                        if self.cared[y, x] < day and fedd[y, x]:
                            tasks.append((p["w_care"] * pr / max(1, iv), y, x, OP["CARE"], 0, 1))
                        if favail is None or favail[y, x]:
                            tasks.append((p["w_coll"] * max(60.0, price(FERT, inv_mkt[FERT])), y, x, OP["COLLECT"], 0, 1))
        # 小屋での補給: 餌の小麦、肥料、動物
        hy, hx = BS // 2 - 1, BS // 2 - 1
        unfed = int((((kind == KIND_PASTURE) | (kind == KIND_COOP)) & (what >= 9) & (~fedd)).sum())
        nk0 = max(1, min(int(p["n_keepers"]) + nanim // int(p["anim_per_keeper"]), nu - 2)) if nu > 1 else 1
        if unfed > 0 and shed[WHEAT] > 0 and int(self.inv[:nk0, WHEAT].sum()) < unfed:
            # 1日分をまとめて積む(小屋との往復が家畜係の移動の大半だった)
            tasks.append((p["w_feed"] * 95.0, hy, hx, OP["PICKUP"], WHEAT, min(int(shed[WHEAT]), max(3, int(p["feed_carry"]) * nanim // max(1, nk0)))))
        nfertop = int(((kind == KIND_PLANT) & (yld < 6)).sum())
        if shed[FERT] > 0 and nfertop > 0 and int(self.inv[:, FERT].sum()) == 0 and p["w_fert"] > 0:
            tasks.append((p["w_fert"] * 40.0, hy, hx, OP["PICKUP"], FERT, min(int(shed[FERT]), 4)))
        for a in (COW, SHEEP, GOOSE):
            heading = sum(1 for tg in self.target.values() if tg[2] == OP["PICKUP"] and tg[3] == a)
            for _ in range(int(self.shed_anim[a]) - heading):
                tasks.append((p["w_place"] * 140.0, hy, hx, OP["PICKUP"], a, 1))
        # 家畜: 小屋の動物 or 手持ちの動物があれば、空の牧場/鶏舎へ置く。無ければ建てる
        carried = {a: int(self.inv[:, a].sum()) for a in (GOOSE, COW, SHEEP)}
        # 建て先行: 目標頭数に届かず、空き牧場/鶏舎も無ければ、小屋に近い空きマスに建てておく
        for a, tgt in ((COW, p["n_cow"]), (SHEEP, p["n_sheep"]), (GOOSE, p["n_goose"])):
            want_kind = KIND_COOP if a == GOOSE else KIND_PASTURE
            have = int((what == a).sum()) + int(self.shed_anim[a]) + carried[a]
            empt = int(((kind == want_kind) & (what == 0)).sum())
            prod = ANIMAL[a][5]
            _, _, fy2, iv2, mx2, _ = ANIMAL[a]
            left2 = max(0, 29 - day - fy2); exp2 = (left2 // max(1, iv2)) * min(mx2, 1 + iv2)
            if p["build_ahead"] and day >= p["animal_day"] and have < tgt and empt == 0 and money > ANIMAL[a][0] + p["animal_cash"] \
                    and ((exp2 * price(prod, inv_mkt[prod]) > ANIMAL[a][0] * p["anim_roi"]) if p["anim_roi"] > 0
                         else (price(prod, inv_mkt[prod]) >= p["anim_px"] * float(MK[prod][0]))) and day <= int(p["animal_last_day"]):
                cand = sorted((abs(y - 4) + abs(x - 4), y, x) for y in range(BS) for x in range(BS)
                              if kind[y, x] == KIND_EMPTY and (self._reserved(y, x) or p["anywhere"] or not p["ring"]))
                if cand:
                    _, y, x = cand[0]
                    tasks.append((p["w_place"] * 120.0, y, x, OP["BUILD_COOP" if a == GOOSE else "BUILD_PASTURE"], 0, 1))
        for a in (COW, SHEEP, GOOSE):
            pend = carried[a] + int(self.shed_anim[a])
            if pend <= 0: continue
            want_kind = KIND_COOP if a == GOOSE else KIND_PASTURE
            empties = [(y, x) for y in range(BS) for x in range(BS) if kind[y, x] == want_kind and what[y, x] == 0]
            for (y, x) in empties[:pend]:
                tasks.append((p["w_place"] * 150.0, y, x, OP["PLACE"], a, 1))
            short = pend - len(empties)
            if short > 0:
                cand = [(abs(y - 4) + abs(x - 4), y, x) for y in range(BS) for x in range(BS)
                        if kind[y, x] == KIND_EMPTY and (self._reserved(y, x) or p["anywhere"])]
                if not cand:
                    cand = [(abs(y - 4) + abs(x - 4), y, x) for y in range(BS) for x in range(BS) if kind[y, x] == KIND_EMPTY]
                cand.sort()
                for _, y, x in cand[:short]:
                    tasks.append((p["w_place"] * 130.0, y, x, OP["BUILD_COOP" if a == GOOSE else "BUILD_PASTURE"], 0, 1))
        tasks.sort(key=lambda t: -t[0])
        nz = max(1, min(int(p["nzones"]), nu - nk0))
        cols = sorted(set(x for y in range(BS) for x in range(BS) if kind[y, x] != KIND_LOCKED))
        zone_of_x = {x: min(nz - 1, (i * nz) // max(1, len(cols))) for i, x in enumerate(cols)}
        nk = max(1, min(int(p["n_keepers"]) + nanim // int(p["anim_per_keeper"]), nu - 2)) if nu > 1 else 1
        ANIM_OPS = (OP["FEED"], OP["CARE"], OP["COLLECT"], OP["PLACE"], OP["BUILD_COOP"], OP["BUILD_PASTURE"])
        def zone_ok(u, y, x, op, arg=None):
            is_anim = op in ANIM_OPS or (op == OP["HARVEST"] and kind[y, x] != KIND_PLANT)
            if op == OP["PICKUP"]:
                if nu <= 1: return True
                if arg == WHEAT: return u < nk
                if arg == FERT: return u >= nk
                return True
            if nu > 1:
                if u < nk: return is_anim
                if u in getattr(self, "released", ()): return True
                if is_anim: return False
            if not p["zones"]: return True
            if op not in (OP["WATER"], OP["PLANT"], OP["FERTILIZE"], OP["DIG"]): return True
            return zone_of_x.get(x, 0) == (u - nk) % nz
        used = set(); assigned = set(); plant_left = {c: int(seeds[c]) for c in CROP}
        live = {(y, x, op, arg): (sc, n) for sc, y, x, op, arg, n in tasks}
        # --- 家畜係(巡回モード): 小屋で餌を積み、担当の家畜を固定順に回って 餌→世話→採取→収穫 を打ち、小屋へ戻る
        if p["kroute"] and nu > 1 and nanim > 0:
            anim_tiles = [(y, x) for y in range(BS) for x in (range(BS) if y % 2 == 0 else range(BS - 1, -1, -1))
                          if kind[y, x] in (KIND_PASTURE, KIND_COOP) and what[y, x] >= 9]
            if not hasattr(self, "kstate"): self.kstate = {}
            if hour <= 1: self.kstate = {}
            per = -(-len(anim_tiles) // nk)
            hy, hx = BS // 2 - 1, BS // 2 - 1
            chain_pending = sum(self.shed_anim.values()) + int(self.inv[:, 9:12].sum()) > 0
            for u in range(nk):
                if u in assigned: continue
                if u == 0 and chain_pending: continue      # 家畜係0 は設置の鎖(貪欲側)へ
                mine_t = anim_tiles[u * per:(u + 1) * per]
                if not mine_t: continue
                st = self.kstate.setdefault(u, {"i": 0, "phase": "load"})
                at_shed = (int(uy[u]) == hy and int(ux[u]) == hx)
                unfed_mine = sum(1 for (y, x) in mine_t if not fedd[y, x])
                if st["phase"] == "load":
                    need = max(0, unfed_mine - int(self.inv[u, WHEAT]))
                    if need <= 0 or shed[WHEAT] <= 0:
                        st["phase"] = "tour"; st["i"] = 0          # 餌が足りている(or 小屋に無い)なら戻らずに回る
                    elif at_shed:
                        self.target[u] = (hy, hx, OP["PICKUP"], WHEAT, min(int(shed[WHEAT]), need + 1)); assigned.add(u); continue
                    else:
                        self.target[u] = (hy, hx, OP["PASS"], 0, 1); assigned.add(u); continue
                if st["phase"] == "tour":
                    if st["i"] >= len(mine_t):
                        st["phase"] = "load"; st["i"] = 0
                        if unfed_mine - int(self.inv[u, WHEAT]) > 0 and shed[WHEAT] > 0:
                            self.target[u] = (hy, hx, OP["PASS"], 0, 1); assigned.add(u); continue
                        st["phase"] = "tour"
                        ny, nx = mine_t[0]; self.target[u] = (ny, nx, OP["PASS"], 0, 1); assigned.add(u); continue
                    y, x = mine_t[st["i"]]
                    here = (int(uy[u]) == y and int(ux[u]) == x)
                    if not here:
                        self.target[u] = (y, x, OP["PASS"], 0, 1); assigned.add(u); continue
                    a = int(what[y, x]); prod = ANIMAL.get(a, (0,) * 6)[5]
                    if (not fedd[y, x]) and self.inv[u, WHEAT] > 0: op = OP["FEED"]
                    elif fedd[y, x] and self.cared[y, x] < day: op = OP["CARE"]
                    elif favail is not None and favail[y, x] and p["route_collect"] \
                            and price(FERT, inv_mkt[FERT]) >= p["collect_px"]: op = OP["COLLECT"]
                    elif yld[y, x] > 0: op = OP["HARVEST"]
                    else:
                        st["i"] += 1
                        if st["i"] < len(mine_t):
                            ny, nx = mine_t[st["i"]]; self.target[u] = (ny, nx, OP["PASS"], 0, 1)
                        else:
                            st["phase"] = "load"; self.target[u] = (hy, hx, OP["PASS"], 0, 1)
                        assigned.add(u); continue
                    self.target[u] = (y, x, op, 0, 1); assigned.add(u); used.add((y, x))
        # --- 作物係(掃除機モード): 蛇行順に並べた担当マスを往復し、立ったマスで必要な手を打つ
        if p["sweep"] and (nu > 1 or p["solo_crop"]):
            nunl = int((kind != KIND_LOCKED).sum())
            if hour <= 1 or not hasattr(self, "lanes") or len(self.lanes) != max(1, nu - nk) or getattr(self, "lane_nunl", -1) != nunl:
                self.lane_nunl = nunl
                import math
                def usable(y, x):
                    if kind[y, x] in (KIND_LOCKED, KIND_PASTURE, KIND_COOP): return False
                    # 予約地は植えないが、雑草が生えたら掘る(掘らないと牧場も建たない)
                    if self._reserved(y, x) and kind[y, x] not in (KIND_PLANT, KIND_WEED): return False
                    return True
                h = BS // 2; rr = int(p["ring"]) if day >= int(p["ring_day"]) else 0
                lo, hi = h - rr, h - 1 + rr            # 予約ブロックの行/列の範囲(rr=0 なら小屋の 2x2 だけ)
                def serp(ys, xs_fn):
                    out = []
                    for i, y in enumerate(ys):
                        xs = list(xs_fn); xs = xs if i % 2 == 0 else xs[::-1]
                        out += [(y, x) for x in xs if usable(y, x)]
                    return out
                # 北(ブロックより上の行、全幅) → 西(ブロックの行、左側、列で蛇行) → 南 → 東
                order = serp(range(0, lo), range(BS))
                west = []
                for i, x in enumerate(range(0, lo)):
                    ys = list(range(lo, hi + 1)); ys = ys if i % 2 == 0 else ys[::-1]
                    west += [(y, x) for y in ys if usable(y, x)]
                east = []
                for i, x in enumerate(range(hi + 1, BS)):
                    ys = list(range(lo, hi + 1)); ys = ys if i % 2 == 0 else ys[::-1]
                    east += [(y, x) for y in ys if usable(y, x)]
                order += west + serp(range(hi + 1, BS), range(BS)) + east
                nw = max(1, nu - nk)
                if p["lane_mode"] == "block":
                    # 使えるマスを nw 個のコンパクトな区画に切る(縦横に分割してから区画内を蛇行)。
                    # 盤全体の蛇行を等分すると1人の担当が端から端まで散り、移動が作業の1.9倍になっていた。
                    import math as _m
                    cells = [(y, x) for y in range(BS) for x in range(BS) if usable(y, x)]
                    if not cells: cells = [(BS // 2 - 1, BS // 2 - 1)]
                    ncol = max(1, int(round(_m.sqrt(nw))));  nrow = -(-nw // ncol)
                    xs = sorted(set(x for _, x in cells)); ys = sorted(set(y for y, _ in cells))
                    def band(v, arr, n):
                        i = arr.index(v); return min(n - 1, i * n // max(1, len(arr)))
                    buckets = {}
                    for (y, x) in cells:
                        buckets.setdefault((band(y, ys, nrow), band(x, xs, ncol)), []).append((y, x))
                    keys = sorted(buckets)
                    self.lanes = []
                    for k_ in keys:
                        blk = buckets[k_]
                        rows_ = sorted(set(y for y, _ in blk))
                        seq = []
                        for i, y in enumerate(rows_):
                            xs_ = sorted(x for yy, x in blk if yy == y)
                            seq += [(y, x) for x in (xs_ if i % 2 == 0 else xs_[::-1])]
                        self.lanes.append(seq)
                    while len(self.lanes) < nw: self.lanes.append([])
                    self.lanes = self.lanes[:nw]
                elif p["lane_mode"] == "ring":
                    # 小屋を囲む同心の環を順に回る(環の中は角度順)。各レーンは1つの環の弧になり、通勤が環の半径で済む
                    cy, cx = BS / 2 - 0.5, BS / 2 - 0.5
                    def ringkey(t):
                        r = max(abs(t[0] - cy), abs(t[1] - cx))
                        a = math.atan2(t[0] - cy, t[1] - cx)
                        if int(round(r - 0.5)) % 2 == 1: a = -a          # 隣り合う環は逆回りにして継ぎ目の移動を短く
                        return (round(r, 1), a)
                    rg = sorted(order, key=ringkey)
                    per = -(-len(rg) // nw)
                    self.lanes = [rg[i * per:(i + 1) * per] for i in range(nw)]
                elif p["lane_mode"] == "wedge":
                    # 小屋を中心にした扇形。各レーンは小屋の近くから遠くへ向かうので朝の通勤が仕事になる
                    cy, cx = BS / 2 - 0.5, BS / 2 - 0.5
                    ang = sorted(order, key=lambda t: math.atan2(t[0] - cy, t[1] - cx))
                    per = -(-len(ang) // nw)
                    self.lanes = []
                    for i in range(nw):
                        w = ang[i * per:(i + 1) * per]
                        w.sort(key=lambda t: (abs(t[0] - cy) + abs(t[1] - cx), t[0], t[1]))
                        self.lanes.append(w)
                else:
                    per = -(-len(order) // nw)
                    self.lanes = [order[i * per:(i + 1) * per] for i in range(nw)]
                # 朝は各レーンの「小屋に最も近い端」から入り、遠い側へ向かう(通勤を仕事に変える)
                self.lane_i = {}; self.lane_dir = {}
                for u in range(nk, nu):
                    lane = self.lanes[u - nk]
                    if not lane: self.lane_i[u] = 0; self.lane_dir[u] = 1; continue
                    d0 = abs(lane[0][0] - 4) + abs(lane[0][1] - 4); d1 = abs(lane[-1][0] - 4) + abs(lane[-1][1] - 4)
                    if d0 <= d1: self.lane_i[u] = 0; self.lane_dir[u] = 1
                    else: self.lane_i[u] = len(lane) - 1; self.lane_dir[u] = -1
            tile_task = {}
            for sc, y, x, op, arg, n in tasks:
                if op in (OP["DIG"], OP["HARVEST"], OP["PLANT"], OP["FERTILIZE"], OP["WATER"]) and kind[y, x] != KIND_PASTURE and sc > 0:
                    # 水やりが施肥より先。逆にすると施肥がその日の水やりを押しのけ、乾いて枯れる(実測 -34k)
                    pri = {OP["DIG"]: 0, OP["HARVEST"]: 1, OP["PLANT"]: 2, OP["WATER"]: 3, OP["FERTILIZE"]: 4}[op]
                    tile_task.setdefault((y, x), []).append((pri, op, arg, n, sc))
            if not hasattr(self, "released"): self.released = set()
            if hour <= 1: self.released = set()
            # 1日の仕事を朝に確定し、必要なタイルだけを最近傍で繋いだ巡回路にする。
            # 何も要らないタイルを1マスずつ歩くのをやめる(作業率 34% -> ?)
            if p["tour"]:
                if not hasattr(self, "tour"): self.tour = {}
                if hour <= 1: self.tour = {}
                for u in (range(nk, nu) if nu > 1 else range(0, 1)):
                    lane = self.lanes[u - nk] if nu > 1 else (self.lanes[0] if self.lanes else [])
                    if not lane: continue
                    q = self.tour.get(u)
                    if q is None:
                        need = [(y, x) for (y, x) in lane if tile_task.get((y, x))]
                        cy, cx = int(uy[u]), int(ux[u]); seq = []
                        while need:
                            j = min(range(len(need)), key=lambda k2: abs(need[k2][0]-cy)+abs(need[k2][1]-cx))
                            cy, cx = need[j]; seq.append(need.pop(j))
                        q = seq; self.tour[u] = q
                    while q and not tile_task.get(q[0]): q.pop(0)
                    if not q:
                        # 仕事が尽きたら、その日の残りは巡回路を組み直す(新しく必要になったタイルを拾う)
                        need = [(y, x) for (y, x) in lane if tile_task.get((y, x))]
                        if need:
                            cy, cx = int(uy[u]), int(ux[u]); seq = []
                            while need:
                                j = min(range(len(need)), key=lambda k2: abs(need[k2][0]-cy)+abs(need[k2][1]-cx))
                                cy, cx = need[j]; seq.append(need.pop(j))
                            q = seq; self.tour[u] = q
                        else:
                            self.target.pop(u, None); assigned.add(u); ops[u] = (OP["PASS"], 0, 1); continue
                    y, x = q[0]
                    tt = None
                    for cand_t in sorted(tile_task.get((y, x), [])):
                        if cand_t[1] == OP["PLANT"] and plant_left[cand_t[2]] <= 0: continue
                        if cand_t[1] == OP["FERTILIZE"] and self.inv[u, FERT] <= 0: continue
                        tt = cand_t; break
                    if tt is None:
                        q.pop(0); self.target.pop(u, None); assigned.add(u); ops[u] = (OP["PASS"], 0, 1); continue
                    if int(uy[u]) == y and int(ux[u]) == x:
                        self.target[u] = (y, x, tt[1], tt[2], tt[3]); assigned.add(u); used.add((y, x))
                        if tt[1] == OP["PLANT"]: plant_left[tt[2]] -= 1
                    else:
                        self.target[u] = (y, x, OP["PASS"], 0, 1); assigned.add(u)
            for u in ([] if p["tour"] else (range(nk, nu) if nu > 1 else range(0, 1))):
                lane = self.lanes[u - nk] if nu > 1 else (self.lanes[0] if self.lanes else [])
                if not lane: continue
                if p["release"] and u not in self.released:
                    pending_any = False
                    for (ly, lx) in lane:
                        for cand_t in tile_task.get((ly, lx), []):
                            if cand_t[1] == OP["PLANT"] and plant_left[cand_t[2]] <= 0: continue
                            if cand_t[1] == OP["FERTILIZE"] and self.inv[u, FERT] <= 0: continue
                            pending_any = True; break
                        if pending_any: break
                    if not pending_any: self.released.add(u)
                if u in self.released: continue
                if hour <= 2 and int(uy[u]) == BS // 2 - 1 and int(ux[u]) == BS // 2 - 1 \
                        and self.inv[u, FERT] == 0 and shed[FERT] > 0 and p["w_fert"] > 0 and p["fert_carry"] > 0 \
                        and price(FERT, inv_mkt[FERT]) < p["fert_ratio"] * 2 * price(WHEAT, inv_mkt[WHEAT]):
                    self.target[u] = (BS // 2 - 1, BS // 2 - 1, OP["PICKUP"], FERT, int(p["fert_carry"])); assigned.add(u); continue
                i = self.lane_i.get(u, 0) % len(lane)
                y, x = lane[i]
                here = (int(uy[u]) == y and int(ux[u]) == x)
                tt = None
                for cand_t in sorted(tile_task.get((y, x), [])):
                    if cand_t[1] == OP["PLANT"] and plant_left[cand_t[2]] <= 0: continue
                    if cand_t[1] == OP["FERTILIZE"] and self.inv[u, FERT] <= 0: continue
                    tt = cand_t; break
                if here and tt:
                    self.target[u] = (y, x, tt[1], tt[2], tt[3]); assigned.add(u); used.add((y, x))
                    if tt[1] == OP["PLANT"]: plant_left[tt[2]] -= 1
                elif here:
                    # 何も要らないマス: 次へ(端で折り返す)
                    d = self.lane_dir.get(u, 1); j = i + d
                    if j >= len(lane) or j < 0: d = -d; j = i + d; self.lane_dir[u] = d
                    self.lane_i[u] = max(0, min(len(lane) - 1, j))
                    ny, nx = lane[self.lane_i[u]]
                    self.target[u] = (ny, nx, OP["PASS"], 0, 1); assigned.add(u)
                else:
                    self.target[u] = (y, x, OP["PASS"], 0, 1); assigned.add(u)
        # --- 作物係(鎖モード): 区域内の未請求タスクを最近傍で鎖にする
        if not hasattr(self, "queue"): self.queue = {}
        if hour == 0: self.queue = {}
        CROP_OPS = (OP["WATER"], OP["PLANT"], OP["FERTILIZE"], OP["DIG"])
        claimed = set()
        for u, q in self.queue.items():
            for key in q: claimed.add(key[:2])
        for u in (range(nk, nu) if (nu > 1 and not p["sweep"]) else []):
            q = self.queue.get(u, [])
            # 先頭が無効なら捨てる
            while q and (q[0] not in live or (q[0][2] == OP["PLANT"] and plant_left[q[0][3]] <= 0)):
                claimed.discard(q[0][:2]); q.pop(0)
            if not q:
                cand = [(sc, y, x, op, arg, n) for sc, y, x, op, arg, n in tasks
                        if (op in CROP_OPS or (op == OP["HARVEST"] and kind[y, x] == KIND_PLANT))
                        and zone_ok(u, y, x, op, arg) and (y, x) not in claimed and sc > 0
                        and not (op == OP["FERTILIZE"] and self.inv[u, FERT] <= 0)]
                cy, cx = int(uy[u]), int(ux[u]); chain = []
                for _ in range(int(p["chain"])):
                    if not cand: break
                    j = min(range(len(cand)), key=lambda k: (abs(cand[k][1] - cy) + abs(cand[k][2] - cx)) * 3 - cand[k][0] * 0.05)
                    sc, y, x, op, arg, n = cand.pop(j)
                    chain.append((y, x, op, arg)); claimed.add((y, x)); cy, cx = y, x
                q = chain
            self.queue[u] = q
            if q:
                y, x, op, arg = q[0]
                self.target[u] = (y, x, op, arg, live[q[0]][1]); assigned.add(u); used.add((y, x))
                if op == OP["PLANT"]: plant_left[arg] -= 1
            else:
                self.target.pop(u, None)
        for u in list(self.target.keys()):
            if u >= nu: self.target.pop(u); continue
            if u in assigned or (nu > 1 and u >= nk): continue
            y, x, op, arg, n = self.target[u]
            key = (y, x, op, arg)
            if key in live and (op != OP["PLANT"] or plant_left[arg] > 0):
                assigned.add(u)
                if op != OP["PICKUP"]: used.add((y, x))
                if op == OP["PLANT"]: plant_left[arg] -= 1
            else:
                self.target.pop(u)
        for sc, y, x, op, arg, n in tasks:
            if (y, x) in used and op != OP["PICKUP"]: continue
            if p["kroute"] and nu > 1 and op in (OP["FEED"], OP["CARE"], OP["COLLECT"]): continue
            if p["kroute"] and nu > 1 and op == OP["HARVEST"] and kind[y, x] != KIND_PLANT: continue
            if p["kroute"] and nu > 1 and op == OP["PICKUP"] and arg == WHEAT: continue
            if op == OP["PLANT"]:
                if plant_left[arg] <= 0: continue
            best = None
            for u in range(nu):
                if u in assigned: continue
                if op == OP["FEED"] and self.inv[u, WHEAT] <= 0: continue
                if op == OP["FERTILIZE"] and self.inv[u, FERT] <= 0: continue
                if op == OP["PLACE"] and self.inv[u, arg] <= 0: continue
                if op == OP["PICKUP"] and self.inv[u, arg] > 0: continue
                if not zone_ok(u, y, x, op, arg): continue
                d = abs(int(ux[u]) - x) + abs(int(uy[u]) - y)
                v = sc / (1.0 + p["w_dist"] * d)          # 限界コイン / (実行1手 + 距離)
                if best is None or v > best[0]: best = (v, u, d)
            if best is None or best[0] <= 0: continue
            _, u, d = best
            if op != OP["PICKUP"]: used.add((y, x))
            assigned.add(u)
            if op == OP["PLANT"]: plant_left[arg] -= 1

            self.target[u] = (y, x, op, arg, n)
        # --- 未割当のユニットは小屋で小麦/肥料を補充
        h = BS // 2
        for u in range(nu):
            if u in assigned: continue
            if nu > 1 and u >= nk: continue
            if nanim > 0 and self.inv[u, WHEAT] < 2 and shed[WHEAT] > 0:
                self.target[u] = (h - 1, h - 1, OP["PICKUP"], WHEAT, 5)
            elif self.inv[u, FERT] < 2 and shed[FERT] > 0:
                self.target[u] = (h - 1, h - 1, OP["PICKUP"], FERT, 3)
        # --- 実行 or 1歩寄る
        self._shed_now = shed.astype(np.int64).copy()
        for u in range(nu):
            t = self.target.get(u)
            if t is None: ops[u] = (OP["PASS"], 0, 1); continue
            y, x, op, arg, n = t
            if int(uy[u]) == y and int(ux[u]) == x and op == OP["PASS"]:
                ops[u] = (OP["PASS"], 0, 1); self.target.pop(u, None); continue
            if int(uy[u]) == y and int(ux[u]) == x:
                ops[u] = (op, arg, n)
                self._apply(u, y, x, op, arg, n, day, kind, what, yld)
                self.target.pop(u, None)
                q = self.queue.get(u) if hasattr(self, "queue") else None
                if q and q[0][:2] == (y, x): q.pop(0)
            else:
                dy = y - int(uy[u]); dx = x - int(ux[u])
                ops[u] = (OP["S"] if dy > 0 else OP["N"], 0, 1) if dy else (OP["E"] if dx > 0 else OP["W"], 0, 1)
        left = {c: int(seeds[c]) for c in CROP}
        for u in range(nu):
            if ops[u, 0] == OP["PLANT"]:
                c = int(ops[u, 1])
                if left.get(c, 0) <= 0: ops[u] = (OP["PASS"], 0, 1)
                else: left[c] -= 1
        return ops, mk[:10]

    def _reserved(self, y, x):
        """牧場・鶏舎の予約地(小屋アクセス4マスの外側の環)"""
        h = BS // 2
        d = max(abs(y - (h - 1)), abs(y - h)) if not (h - 1 <= y <= h) else 0
        e = max(abs(x - (h - 1)), abs(x - h)) if not (h - 1 <= x <= h) else 0
        r = max(d, e)
        if getattr(self, "_day", 0) < int(self.p["ring_day"]): return False
        return 1 <= r <= int(self.p["ring"])

    def _apply(self, u, y, x, op, arg, n, day, kind, what, yld):
        if op == OP["PLANT"]:
            self.crop[y, x] = arg; self.pday[y, x] = day; self.fert[y, x] = -1
            if arg == MELON: self.melon_planted += 1
        elif op == OP["FERTILIZE"]:
            self.fert[y, x] = day + 2; self.inv[u, FERT] -= 1
        elif op == OP["PLACE"]: self.last_place = (u, y, x, arg)
        elif op == OP["FEED"]:
            if self.inv[u, WHEAT] > 0: self.inv[u, WHEAT] -= 1
        elif op == OP["COLLECT"]: self.inv[u, FERT] += 1
        elif op == OP["PICKUP"]:
            if arg in self.shed_anim:
                if self.shed_anim[arg] > 0: self.inv[u, arg] += 1; self.shed_anim[arg] -= 1
            else:
                got = min(int(n), int(self._shed_now[arg])); self.inv[u, arg] += got; self._shed_now[arg] -= got
        elif op == OP["CARE"]: self.cared[y, x] = day
        elif op == OP["HARVEST"]:
            k = kind[y, x]
            it = int(self.crop[y, x]) if k == KIND_PLANT else ANIMAL.get(int(round(what[y, x])), (0,)*6)[5]
            if it is not None and 0 <= it < 12: self.inv[u, it] += int(yld[y, x])
        elif op == OP["DIG"]:
            self.crop[y, x] = -1; self.pday[y, x] = -1
