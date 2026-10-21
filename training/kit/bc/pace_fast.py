"""exp284: 在庫の安全装置(exp222 の main.py の `_pace`)を、訓練中の環境側にも入れる。

これまで: 訓練(自己対戦)は装置なしの生の方策、評価・提出は装置つき。
  → 育てる対象と出荷する対象がずれていた(装置が直す部分の伸びは出荷時に消える)。
ここでは同じ規則を「エンジンに渡す注文コードの配列」の上で一括に適用する。
方策が出した行動(ms・mq)はそのまま学習側に残り、環境が後処理する形なので、方策勾配は正しいまま
(環境 = 元の環境 ∘ 装置 と見なす)。

使い方: 環境変数 PACE=1。sp_fast.BatchSPPool.step が両席の注文にこれをかける。
検証: python3 pace_fast.py → エージェントの `_pace` と同じ注文列になるかを乱数の局面で確かめる。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import kagpack
from extract4 import IT, MQ_MAX

SELL = kagpack.MOPI["SELL"]
BUYOPS = {kagpack.MOPI[x] for x in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND")}
EMPTY, NONE = 254, 255          # 254 = 枠だけ占める注文、255 = 枠が空いている
IT9 = list(IT[:9])
OUTS = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]   # 小麦・肥料は餌と畑の原料なので対象外
VAL = {"STRAWBERRY": 5, "MELON": 5, "MILK": 4, "WOOL": 4, "TOMATO": 3, "FERTILIZER": 2, "EGG": 2, "CARROT": 1, "WHEAT": 0}
SLACK = -3; LIQ_START = 710; LAST_STEP = 718
_OUT_I = [IT9.index(x) for x in OUTS]
_VAL_ORDER = [IT9.index(x) for x in sorted(VAL, key=lambda k: -VAL[k])]      # 高い順の品目番号
_TGT = None


def target(path=None):
    """時刻 × 品目の在庫の75%点(強い Majkel の90局から作ったもの)を (T,9) で返す。"""
    global _TGT
    if _TGT is None:
        p = path or os.environ.get("STOCKTGT", "/content/lab/agents/mk16_g6/stock_target.json")
        t = json.load(open(p)); pr = t["products"]
        _TGT = np.array([[r[pr.index(i)] if i in pr else 0.0 for i in IT9] for r in t["75"]], np.float32)
    return _TGT


def apply_codes(O, shed, t, tgt=None):
    """O: (B,10,3) int32 の注文コード(書き換える)、shed: (B,9) の納屋、t: 手番。エージェントの `_pace` と同じ規則。"""
    t = int(t)
    if t > LAST_STEP: return O
    B = O.shape[0]
    shed = np.asarray(shed).astype(np.int64)
    if t >= LIQ_START:                                   # 最終盤: 在庫 ÷ 残り手数、最後の手で全部
        left = LAST_STEP - t + 1
        out = np.full_like(O, 0); out[:, :, 0] = NONE
        for b in range(B):
            if left <= 1:
                sells = [(SELL, c, 1000) for c in _VAL_ORDER]      # 在庫が無い品は即座に止まるだけ
            else:
                sells = [(SELL, c, -(-int(shed[b, c]) // left)) for c in _VAL_ORDER if shed[b, c] > 0]
            mine = [tuple(r) for r in O[b] if r[0] != NONE and r[0] != SELL and r[0] not in BUYOPS]
            for j, o in enumerate((sells + mine)[:10]): out[b, j] = o
        O[...] = out
        return O
    T = target() if tgt is None else tgt
    row = np.rint(T[min(t, len(T) - 1)]).astype(np.int64)
    want = shed - row[None, :] - SLACK                   # 在庫が目安を SLACK 以上超えた分を売る
    want = np.where(want > 0, want, 0)
    want[:, [c for c in range(9) if c not in _OUT_I]] = 0
    for b in np.nonzero(want.any(1))[0]:
        w = want[b]
        n = 0
        for j in range(10):
            op = O[b, j, 0]
            if op == NONE: break
            n = j + 1
            if op == SELL:                               # すでに売っているなら数量を増やす
                c = int(O[b, j, 1])
                if w[c] > 0:
                    O[b, j, 2] = max(int(O[b, j, 2]), int(w[c])); w[c] = 0
        for c in _VAL_ORDER:                             # 残りは後ろに足す(方策の注文は削らない)
            if w[c] <= 0: continue
            if n >= 10: break
            O[b, n] = (SELL, c, int(w[c])); n += 1
    return O


# ---- 検証用: エージェント main.py の _pace(辞書の注文列)をそのまま写したもの ----
def _pace_ref(market, shed_d, t, T):
    if t > LAST_STEP: return market
    mk = [list(o) for o in market]
    if t >= LIQ_START:
        left = LAST_STEP - t + 1
        mine = [o for o in mk if o[0] != "SELL" and not o[0].startswith("BUY")]
        if left <= 1:
            sells = [["SELL", it, 1000] for it in sorted(VAL, key=lambda k: -VAL[k])]
        else:
            sells = [["SELL", it, -(-q // left)] for it in sorted(VAL, key=lambda k: -VAL[k]) for q in [int(shed_d.get(it, 0))] if q > 0]
        return (sells + mine)[:10]
    row = T[min(t, len(T) - 1)]
    want = {}
    for it in OUTS:
        ex = int(shed_d.get(it, 0)) - int(round(float(row[IT9.index(it)]))) - SLACK
        if ex > 0: want[it] = ex
    if not want: return mk
    for o in mk:
        if o[0] == "SELL" and len(o) > 2 and o[1] in want:
            o[2] = max(int(o[2]), want.pop(o[1]))
    for it in sorted(want, key=lambda k: -VAL[k]):
        if len(mk) >= 10: break
        mk.append(["SELL", it, want[it]])
    return mk[:10]


def _to_dicts(O):
    """注文コード -> 辞書の注文列(検証用。_pace が見るのは種類・品目・数量だけ)。"""
    inv = {v: k for k, v in kagpack.MOPI.items()}
    out = []
    for r in O:
        op = int(r[0])
        if op == NONE: break
        if op == EMPTY: out.append(["SELL", "WHEAT", 0]); continue      # 枠だけ占める注文(リーグの NOOP と同じ)
        name = inv[op]
        out.append([name, IT9[int(r[1])] if name == "SELL" else IT[int(r[1])], int(r[2])] if name.startswith(("SELL", "BUY")) else [name])
    return out


def check(n=2000, seed=0):
    """乱数の局面で apply_codes と _pace_ref が同じ注文列になるか。"""
    rng = np.random.default_rng(seed); T = target(); bad = 0
    for k in range(n):
        t = int(rng.integers(0, 719))
        shed = rng.integers(0, 40, 9)
        nord = int(rng.integers(0, 11))
        O = np.full((1, 10, 3), 0, np.int32); O[0, :, 0] = NONE
        for j in range(nord):
            if rng.random() < 0.6:
                O[0, j] = (SELL, int(rng.integers(0, 9)), int(rng.integers(1, MQ_MAX)))
            elif rng.random() < 0.5:
                O[0, j] = (kagpack.MOPI["BUY_SEED"], int(rng.integers(0, 5)), int(rng.integers(1, 20)))
            else:
                O[0, j] = (kagpack.MOPI["HIRE"], 0, 1)
        ref = _pace_ref(_to_dicts(O[0]), {IT9[i]: int(shed[i]) for i in range(9)}, t, T)
        got = _to_dicts(apply_codes(O.copy(), shed[None, :], t)[0])
        if [x for x in got] != [x for x in ref]:
            bad += 1
            if bad <= 3: print("不一致 t=%d\n  期待 %s\n  実際 %s" % (t, ref, got))
    print("%d 件中 不一致 %d" % (n, bad))
    return bad == 0


if __name__ == "__main__":
    ok = check()
    raise SystemExit(0 if ok else 1)


# ---- exp290: 4区画目のガードレール(M & M & P & Q の買い方)----
# 1位チームは 82% の局で手番 290 前後に4区画目(4,000コイン)を買う。我々と Majkel は3区画で止まる。
# 手番 LAND_T 以降、区画が4未満・所持金が価格以上・まだ区画の注文が無い局に、区画を買う注文を後ろに足す。
BUYLAND = kagpack.MOPI["BUY_LAND"]
LAND_T = int(os.environ.get("LAND_T", 290)); LAND_PRICE4 = 4000


def apply_land(O, nq, money, t):
    """O: (B,10,3) 注文コード(書き換える)、nq: (B,) 解放済みの区画数、money: (B,) 所持金。"""
    t = int(t)
    if t < LAND_T or t >= LIQ_START: return O
    for b in np.nonzero((np.asarray(nq) < 4) & (np.asarray(money) >= LAND_PRICE4))[0]:
        ops = O[b, :, 0]
        if (ops == BUYLAND).any(): continue
        free = np.nonzero(ops == NONE)[0]
        if len(free): O[b, free[0]] = (BUYLAND, 0, 1)
    return O

# ---- exp305: 3区画目のガードレール(我々は221手、M & M & P & Q は198手。前倒しで対 PFN +7,257)----
LAND3_T = int(os.environ.get("LAND3_T", 198)); LAND3_PRICE = 3000
def apply_land3(O, nq, money, t):
    """3区画目を LAND3_T 以降に買わせる(4区画目版と同じ作り)。"""
    t = int(t)
    if t < LAND3_T or t >= LIQ_START: return O
    for b in np.nonzero((np.asarray(nq) < 3) & (np.asarray(money) >= LAND3_PRICE))[0]:
        ops = O[b, :, 0]
        if (ops == BUYLAND).any(): continue
        free = np.nonzero(ops == NONE)[0]
        if len(free): O[b, free[0]] = (BUYLAND, 0, 1)
    return O

# ---- 09-23(exp318): 4区画目を買わせない枠(NOLAND4=1)。ユーザーの仮説「3区画→4区画→3区画に矯正」の検証用 ----
# 4区画で稼ぎ方を覚えた方策を、3区画で仕上げさせる。方策の行動そのものは書き換えず、環境側で注文を落とすだけ
# (系統A = OPEN / apply_land と同じ扱い)。区画が3つ以上ある局の BUY_LAND 注文を EMPTY にする。
def apply_noland4(O, nq, t):
    """O: (B,10,3) 注文コード(書き換える)、nq: (B,) 解放済みの区画数。枠の位置は変えない(EMPTY で埋める)。"""
    ops = O[:, :, 0]
    for b in np.nonzero(np.asarray(nq) >= 3)[0]:
        for h in np.nonzero(ops[b] == BUYLAND)[0]: O[b, h] = (EMPTY, 0, 0)
    return O


# ---- exp305: 序盤の作付けのガードレール(OPEN=1)----
# 実戦の比較: メロンの種は 我々12個 / M & M & P & Q 9個(株数 = 種の個数、1株6個)。序盤(d0-9)の小麦の売上は 300 対 984。
# 反実仮想(今の方策のまま差し替え)では −1,770〜−3,853 で損だったが、**方策が作り直す機会が無かった**。
# ここでは訓練の環境側にかけて、PPO に「メロンが9個・小麦が多い序盤」で作り直させる(exp290 の apply_land と同じ作り)。
#   OPEN_MELCAP: 1局に出せるメロンの種の注文の合計個数の上限(既定 9)
#   OPEN_WHEAT=1: 削ったメロンの個数を、その後の小麦の種の注文に足す(既定 1)
#   OPEN_T1: この手番までを「序盤」として扱う(既定 240 = d0-9)
BUYSEED = kagpack.MOPI["BUY_SEED"]
_SEED_MELON = kagpack.ITI["MELON"]; _SEED_WHEAT = kagpack.ITI["WHEAT"]
OPEN_MELCAP = int(os.environ.get("OPEN_MELCAP", 9))
OPEN_WHEAT = int(os.environ.get("OPEN_WHEAT", 1))
OPEN_T1 = int(os.environ.get("OPEN_T1", 240))


def apply_open(O, mel, cred, t):
    """O: (B,10,3) 注文コード(書き換える)、mel: (B,) その局でこれまでに出したメロンの種の個数(更新する)、
    cred: (B,) 小麦に振り替える残り(更新する)。注文の枠の位置は変えない(空にするときは EMPTY)。"""
    t = int(t)
    if t >= OPEN_T1: return O
    ops = O[:, :, 0]; its = O[:, :, 1]
    for b in np.nonzero(((ops == BUYSEED) & (its == _SEED_MELON)).any(1))[0]:
        for j in np.nonzero((O[b, :, 0] == BUYSEED) & (O[b, :, 1] == _SEED_MELON))[0]:
            q = int(O[b, j, 2]); take = min(q, max(0, OPEN_MELCAP - int(mel[b])))
            mel[b] += take
            if OPEN_WHEAT: cred[b] += q - take
            if take > 0: O[b, j, 2] = take
            else: O[b, j] = (EMPTY, 0, 0)
    if OPEN_WHEAT:
        for b in np.nonzero((np.asarray(cred) > 0) & ((ops == BUYSEED) & (its == _SEED_WHEAT)).any(1))[0]:
            j = np.nonzero((O[b, :, 0] == BUYSEED) & (O[b, :, 1] == _SEED_WHEAT))[0][0]
            O[b, j, 2] = int(O[b, j, 2]) + int(cred[b]); cred[b] = 0
    return O


# ---- exp314: PFN(M & M & P & Q)の判断規則を、市場の注文側だけ移植する(PFNW=1) ----
# 棋譜40局ずつの実測(実際に約定した取引だけを再現、所持金の不一致0)で分かった差:
#   ・日2〜3に牛を1〜2頭買い足す … PFN 40/40局、我々 0/40局。牛400円・牧草地は無料で、肥料が翌日から日産+1個(約95円)
#   ・日0の家畜 … PFN 牛3+羊2(2,200円)、我々 牛2+羊3(2,300円)。
#     羊毛(日6/t=147)は −1,167円だが牛乳(日8/t=193)は +1,217円。3区画目の2,000円は日8の牛乳で作られるので、
#     牛が多いほうが21手早く3区画目に届く(PFN t=198 100% / 我々 t=219 60%)
#   ・日6の鵞鳥 … 我々は68%の局で鶏舎+鵞鳥2羽(600円)を買うが、t=220 までに卵は1個も売れていない。
#     同じ日6の資金で3区画目と競合する
# exp305 の教訓: 後づけの規則を**1つずつ**入れると全部リーグで負けた。PFN はこれらを同時にやっていて、
#   牛→肥料→現金→土地 と支え合っている。ここでは**まとめてかける**。既定は無効(PFNW=0)。
BUYANIMAL = kagpack.MOPI["BUY_ANIMAL"]
_A_GOOSE = kagpack.ITI["GOOSE"]; _A_COW = kagpack.ITI["COW"]; _A_SHEEP = kagpack.ITI["SHEEP"]
PFNW_COW_T0 = int(os.environ.get("PFNW_COW_T0", 48))    # 日2の頭
PFNW_COW_T1 = int(os.environ.get("PFNW_COW_T1", 96))    # 日4の頭(= 日2と日3のあいだ)
PFNW_COWS = int(os.environ.get("PFNW_COWS", 0))          # その窓で買い足す頭数(既定 0: 16局の切り分けで害と判明)
# 09-21 実測: PFN は t=48-95 の所持金が中央値 108〜303 円しかないのに牛(400円)を買っている。
#   注文は列の順に処理されるので、**同じ手番の売却で得た金**で買っている。手番開始時の所持金で判定すると
#   永久に発火しないので、既定では所持金を見ない(PFNW_COW_MONEY=0)。買えなければエンジンが黙って落とすだけ。
PFNW_COW_MONEY = float(os.environ.get("PFNW_COW_MONEY", "0"))
PFNW_D0 = int(os.environ.get("PFNW_D0", 0))              # 日0の家畜構成(既定 0: 羊毛と牛乳を入れ替えるだけで、方策が適応していないと害)
PFNW_D0_COW = int(os.environ.get("PFNW_D0_COW", 3)); PFNW_D0_SHEEP = int(os.environ.get("PFNW_D0_SHEEP", 2))
PFNW_NOGOOSE_T = int(os.environ.get("PFNW_NOGOOSE_T", 0))   # 鵞鳥を止める(既定 0=止めない。卵は日10から出るので、t=220 までの観測で切ったのは誤りだった)
# 09-21 実測(日3-5、1局あたり): 我々は市場注文を 535.5 出していて、そのうち約504(94%)が
#   「持っていない品の売り注文」(いちご/メロンは日10まで、牛乳は日8まで、卵は鵞鳥が居ないのでゼロ個)。
#   PFN は 45.1 注文で、うち 種7.4・牛1.1 を**買う**ことに使っている。我々はその窓で種を1つも買わない。
#   PFNW_NOSELL: 在庫ゼロの品の売り注文を落とす。**エンジンでは何も売れない no-op なので結果は変わらず**、
#     枠だけ空く(第117世代は t=60 で空き枠 0/10 まで埋まっていた)。
#   PFNW_WHEAT: その窓で1日1回、小麦の種を買わせる(PFN は日3-5 に 1.0/1.0/0.8 買う)。
# 09-21 重要: 公式エンジンは 1手番のなかで **_apply_unit_action(全ユニット) → _process_market** の順に処理する。
#   つまり「同じ手番に収穫した分」を売る注文が正当に成立する。手番**開始時**の納屋で判定して売り注文を消すと、
#   その正当な売りまで潰してしまう(実測: 16局で最終 -12,942、0/16勝。作物が実り始める日10から負け続けた)。
#   なので消すのは「その時期に構造上あり得ない品」だけに限る:
#     羊毛=置いた6日後 / 牛乳=8日後 / 卵=4日後(鵞鳥が要る) / メロン・いちご=植えた10日後。
#   日0の家畜は t=0-23 に置かれるので、t < PFNW_NOSELL_T(既定 144 = 日6)なら納屋には入り得ない。
PFNW_NOSELL = int(os.environ.get("PFNW_NOSELL", 1))
PFNW_NOSELL_T = int(os.environ.get("PFNW_NOSELL_T", 144))   # この手番までに限って消す
_NOSELL_ITEMS = [kagpack.ITI[x] for x in ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG")]
PFNW_WHEAT = int(os.environ.get("PFNW_WHEAT", 1))
PFNW_WHEAT_T0 = int(os.environ.get("PFNW_WHEAT_T0", 48)); PFNW_WHEAT_T1 = int(os.environ.get("PFNW_WHEAT_T1", 144))
PFNW_WHEAT_N = int(os.environ.get("PFNW_WHEAT_N", 1))     # 1日あたりの個数
SELL = kagpack.MOPI["SELL"]; BUYSEED_ = kagpack.MOPI["BUY_SEED"]
# 09-21 実測: PFN は BUY_LAND を1回出すのではなく、**t≈105 から毎手 注文列に入れっぱなし**にして
#   金が届いた瞬間に通している(1本目の注文の中央値 129手、3区画目の山 195-198手)。所持金は見ていない
#   (買えなければエンジンが黙って落とすだけ)。既存の apply_land3 は所持金 3000 以上を条件にしていて
#   (3区画目は2000円なのに)必要より遅くしか発火しない。
PFNW_LAND = int(os.environ.get("PFNW_LAND", 1))
PFNW_LAND_T0 = int(os.environ.get("PFNW_LAND_T0", 105))   # ここから出しっぱなしにする
PFNW_LAND_NQ = int(os.environ.get("PFNW_LAND_NQ", 3))     # この区画数に届くまで


def apply_pfnw(O, money, ncow, t, shed=None, nwh=None, nq=None):
    """O: (B,10,3) 注文コード(書き換える)、money: (B,) 所持金、ncow: (B,) 牛の買い足し数(更新する)、
       shed: (B,9) 納屋の在庫(PFNW_NOSELL に使う)、nwh: (B,) 小麦の種を買った日数(更新する)。"""
    t = int(t)
    if t >= LIQ_START: return O
    # 0) 在庫ゼロの品の売り注文を落とす(no-op なので結果は変わらない。枠だけ空く)
    if PFNW_NOSELL and shed is not None and t < PFNW_NOSELL_T:
        sh = np.asarray(shed).astype(np.int64)
        for b in range(O.shape[0]):
            for j in np.nonzero(O[b, :, 0] == SELL)[0]:
                it = int(O[b, j, 1])
                if it in _NOSELL_ITEMS and 0 <= it < sh.shape[1] and sh[b, it] <= 0: O[b, j] = (EMPTY, 0, 0)
    # 0b) 日1回、小麦の種を買わせる
    if PFNW_WHEAT and nwh is not None and PFNW_WHEAT_T0 <= t < PFNW_WHEAT_T1:
        day = t // 24
        for b in range(O.shape[0]):
            if int(nwh[b]) >= day + 1: continue            # その日はもう入れた
            if ((O[b, :, 0] == BUYSEED_) & (O[b, :, 1] == _SEED_WHEAT)).any():
                nwh[b] = day + 1; continue                  # 自分で買うなら任せる
            free = np.nonzero((O[b, :, 0] == NONE) | (O[b, :, 0] == EMPTY))[0]
            if len(free): O[b, free[0]] = (BUYSEED_, _SEED_WHEAT, PFNW_WHEAT_N); nwh[b] = day + 1
    # 0c) 土地の注文を出しっぱなしにする(PFN と同じ。所持金は見ない)
    if PFNW_LAND and t >= PFNW_LAND_T0 and nq is not None:
        for b in np.nonzero(np.asarray(nq) < PFNW_LAND_NQ)[0]:
            ops = O[b, :, 0]
            if (ops == BUYLAND).any(): continue
            free = np.nonzero((ops == NONE) | (ops == EMPTY))[0]
            if len(free): O[b, free[0]] = (BUYLAND, 0, 1)
    # 1) 日0の家畜構成を寄せる(牛を増やし、羊を減らす)
    if PFNW_D0 and t < 24:
        for b in range(O.shape[0]):
            ops = O[b, :, 0]; its = O[b, :, 1]
            js = np.nonzero((ops == BUYANIMAL) & (its == _A_SHEEP))[0]
            jc = np.nonzero((ops == BUYANIMAL) & (its == _A_COW))[0]
            nsh = int(sum(O[b, j, 2] for j in js)); nco = int(sum(O[b, j, 2] for j in jc))
            if nsh > PFNW_D0_SHEEP and len(js):
                cut = nsh - PFNW_D0_SHEEP
                for j in js:
                    take = min(cut, int(O[b, j, 2])); cut -= take
                    O[b, j, 2] = int(O[b, j, 2]) - take
                    if O[b, j, 2] <= 0: O[b, j] = (EMPTY, 0, 0)
                    if cut <= 0: break
                add = PFNW_D0_COW - nco
                if add > 0:
                    if len(jc): O[b, jc[0], 2] = int(O[b, jc[0], 2]) + add
                    else:
                        free = np.nonzero((O[b, :, 0] == NONE) | (O[b, :, 0] == EMPTY))[0]
                        if len(free): O[b, free[0]] = (BUYANIMAL, _A_COW, add)
    # 2) 日2〜3に牛を買い足す
    if PFNW_COW_T0 <= t < PFNW_COW_T1 and PFNW_COWS > 0:
        for b in np.nonzero((np.asarray(money) >= PFNW_COW_MONEY) & (np.asarray(ncow) < PFNW_COWS))[0]:
            ops = O[b, :, 0]
            if ((ops == BUYANIMAL) & (O[b, :, 1] == _A_COW)).any(): continue   # 自分で買うなら任せる
            ops = O[b, :, 0]   # NOSELL で空いた枠を見るため取り直す
            free = np.nonzero((ops == NONE) | (ops == EMPTY))[0]
            if len(free): O[b, free[-1]] = (BUYANIMAL, _A_COW, 1); ncow[b] += 1   # 売却の後に置く
    # 3) 序盤の鵞鳥を止める(t=220 までに卵が出ないので、3区画目の資金と競合するだけ)
    if PFNW_NOGOOSE_T and t < PFNW_NOGOOSE_T:
        for b in np.nonzero(((O[:, :, 0] == BUYANIMAL) & (O[:, :, 1] == _A_GOOSE)).any(1))[0]:
            for j in np.nonzero((O[b, :, 0] == BUYANIMAL) & (O[b, :, 1] == _A_GOOSE))[0]:
                O[b, j] = (EMPTY, 0, 0)
    return O
