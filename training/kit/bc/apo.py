"""exp202: ppo_orbit.py(exp198)の高速版。変更点:
     1. 行動の変換を表引きの配列処理に(fast_codes.to_codes_fast、元の to_codes と完全一致・7倍速)。相手の PolicyOpp にも効く
     2. 軌道の詰め直しを座席ごとの配列の一括取り出しに(Python の1軌道ずつのループをやめる)
     3. 価値の評価役を小さいネットワークに(CRITINIT、既定は d2 = 100万。方策の1/10の計算で済む)
     4. 更新は EPOCHS 周(既定1)・ミニバッチ MB(既定512)
     5. 区間ごとの時間を毎反復ログに出す
     6. 別の実行(RESUME_FROM)のチェックポイントから、評価役だけ作り直して再開できる
   以下は元の説明:
exp198: Orbit Wars 1位(Tufa Labs「Scaling Reinforcement Learning to the Stars」)の型を移した自己対戦 PPO。

   1位から持ってきたもの:
     - 報酬は**勝ち +1 / 負け −1 だけ**(形を付けない。コインの差は使わない)
     - **「直前の最良版」への関門**: GATE 反復ごとに 現行 対 最良版 を直接対戦させ、GATE_WR 以上勝てば最良版を置き換える
     - **KL 罰則は「直前の最良版」に対して**(BC の初期値に縛り続けない)
     - 並列の局数を増やす(1位は数千。ここは NSELF 等で指定)
   1位が「やればよかった」と書いていたもの:
     - **過去の最良版を少し混ぜる**(リーグ。戦略の堂々巡りと自己過適合を防ぐ)
   こちらの事情で足したもの:
     - 初期値は BC(d2xl)。計算量が1位の約1/8なので、ゼロからではなく BC から始める
     - tape_ug(v57 の土台)を少しだけ相手に混ぜ、同時にものさしにも使う

   common/ は変えていない(sp_lib・ppo_tf・model_tf を import して使うだけ)。
   使い方: python3 ppo_orbit.py <tag> <初期の重み.pt>
   環境変数: NSELF NBEST NLEAGUE NTAPEUG ITERS GATE GATE_WR EVAL LR BETA ENT MB OPPDIR SPLITFILE
"""
import copy, json, os, random, sys, time, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
FASTPOOL_ON = os.environ.get("FASTPOOL", "1") == "1"
if FASTPOOL_ON:          # exp199: 全局を1つの VecMarket に入れる版(値は SPPool とビット一致、環境側は約5倍)
    import sp_fast; S.SPPool = sp_fast.BatchSPPool
# exp286: 訓練中の安全装置(PACE=1 のとき)。局ごとに確率 PACE_P でかける。PACE_ANNEAL 反復かけて PACE_P0 → PACE_P1 に動かす
PACE_P0 = float(os.environ.get("PACE_P0", "1")); PACE_P1 = float(os.environ.get("PACE_P1", os.environ.get("PACE_P0", "1")))
PACE_ANNEAL = int(os.environ.get("PACE_ANNEAL", "0"))
# exp290: 4区画目のガードレール(LAND=1)。局ごとに確率 LAND_P でかけ、LAND_ANNEAL 反復かけて LAND_P0 → LAND_P1 に動かす
LAND_P0 = float(os.environ.get("LAND_P0", "1")); LAND_P1 = float(os.environ.get("LAND_P1", os.environ.get("LAND_P0", "1")))
LAND_ANNEAL = int(os.environ.get("LAND_ANNEAL", "0"))
ROLE = os.environ.get("ROLE", ""); ADIR = os.environ.get("ASYNC_DIR", "/content/lab/async/run"); MAXLAG = int(os.environ.get("MAXLAG", "6")); KSHARD = int(os.environ.get("KSHARD", "1")); MICRO = int(os.environ.get("MICRO", "1")); BUFMAX = int(os.environ.get("BUFMAX", "24"))   # exp296: 非同期
OWN_R0 = float(os.environ.get("OWN_R0", "0")); OWN_ANNEAL = int(os.environ.get("OWN_ANNEAL", "0"))   # exp294: 自分の所持金の報酬(係数と、0 に下げる反復数)
# exp305: 序盤の作付けのガードレール(OPEN=1、sp_fast/pace_fast 側で実装)。局ごとに確率 OPEN_P でかけ、OPEN_ANNEAL 反復かけて OPEN_P0 → OPEN_P1 に動かす
OPEN_P0 = float(os.environ.get("OPEN_P0", "1")); OPEN_P1 = float(os.environ.get("OPEN_P1", os.environ.get("OPEN_P0", "1")))
OPEN_ANNEAL = int(os.environ.get("OPEN_ANNEAL", "0")); OPEN_OPP = os.environ.get("OPEN_OPP", "0") == "1"
# 09-23(exp318): 4区画目を買わせない枠。「3区画→4区画→3区画に矯正」の検証用。局ごとに確率 NOLAND4_P でかける
NOLAND4_P0 = float(os.environ.get("NOLAND4_P0", "1")); NOLAND4_P1 = float(os.environ.get("NOLAND4_P1", os.environ.get("NOLAND4_P0", "1")))
NOLAND4_ANNEAL = int(os.environ.get("NOLAND4_ANNEAL", "0")); NOLAND4_OPP = os.environ.get("NOLAND4_OPP", "0") == "1"
# exp314: PFN(M & M & P & Q)の判断規則をまとめてかける(PFNW=1)。局ごとに確率 PFNW_P、PFNW_ANNEAL 反復で P0 → P1。
#   exp305 で後づけの規則を1つずつ入れて全部負けたので、ここは**まとめて**かけ、徐々に外して自力で再現させる。
PFNW_P0 = float(os.environ.get("PFNW_P0", "1")); PFNW_P1 = float(os.environ.get("PFNW_P1", os.environ.get("PFNW_P0", "1")))
PFNW_ANNEAL = int(os.environ.get("PFNW_ANNEAL", "0")); PFNW_OPP = os.environ.get("PFNW_OPP", "0") == "1"
LAND_OPP = os.environ.get("LAND_OPP", "0") == "1"   # exp290 修正: 相手にもルールをかけるか(既定しない。かけると相手も損をして、自分が4区画目を活かさなくても勝ててしまう)
import fast_codes; P.to_codes = fast_codes.to_codes_fast   # exp202: 元の to_codes と完全一致(乱数の入力2,000局で確認)
EPOCHS = int(os.environ.get("EPOCHS", 1)); CRITINIT = os.environ.get("CRITINIT", "/content/local_logs/tf_d2.pt")
RESUME_FROM = os.environ.get("RESUME_FROM", "")
AMP_SAMPLE = os.environ.get("AMP_SAMPLE", "1") == "1"   # 学習用の対局を集めるときだけ半精度(評価は従来どおり32bit のまま)
# exp298: 批評家の計算を減らす(既定はすべて従来どおり)。CRIT_EVERY=k: 批評家の更新を k ミニバッチに1回(価値のみの反復は毎回)。
#   CRIT_FRAC=f: 批評家の更新はミニバッチの先頭 f 割の行だけ。CRIT_LAST=1: 批評家の最後の層を読むトークン1つ分だけ計算(crit_fast.py、式は同じ)
CRIT_EVERY = int(os.environ.get("CRIT_EVERY", "1")); CRIT_FRAC = float(os.environ.get("CRIT_FRAC", "1")); CRIT_LAST = os.environ.get("CRIT_LAST") == "1"
import contextlib

# exp292: 同期データ並列(torchrun、1 GPU = 1プロセス)。WORLD=1 なら従来とまったく同じ
import torch.distributed as tdist
WORLD = int(os.environ.get("WORLD_SIZE", "1")); RANK = int(os.environ.get("RANK", "0")); LRANK = int(os.environ.get("LOCAL_RANK", "0"))
if WORLD > 1:
    import datetime
    torch.cuda.set_device(LRANK); tdist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
def _allsum(t):
    if WORLD > 1: tdist.all_reduce(t)
    return t
# 09-23: 「同期」の計り方。これまでは all_reduce の後に torch.cuda.synchronize() を入れて
#   time.time() の差を足していた。CUDA は非同期なので、_sync_grads に入った時点で GPU には
#   そのミニバッチの順伝播・逆伝播がまるごと積まれたままであり、synchronize はそれを全部待つ。
#   つまり旧「同期 80s」は通信ではなく**積み残した計算**を数えていた(4GPU 0.232ms/行、
#   8GPU 二ノード 0.182ms/行 と行数に比例。41.8MB の all_reduce なら回数に比例するはず)。
#   さらに synchronize 自体が 1反復に 338 回 CPU の先回りを止めるので、実際に遅くもしていた。
#   ここでは CUDA イベントで all_reduce だけを挟み、反復の終わりに1回だけ読む。
#   **足す値・通信する値は一切変わらない**(イベントは記録するだけ)。
SYNC_T = [0.0]          # 1反復ぶんの本当の通信時間(秒)
SYNC_N = [0, 0]         # [all_reduce の回数, 送った要素数]
_SYNCPROF = os.environ.get("SYNCPROF") == "1"
_SYNC_EV = []           # (開始, 終了) の CUDA イベント。反復の終わりにまとめて読む
def sync_drain():
    """溜めたイベントを読んで SYNC_T に足す。反復の終わりに1回だけ呼ぶ。
    学習の値には触れない(イベントの読み出しだけ)。"""
    if not _SYNC_EV: return
    _SYNC_EV[-1][1].synchronize()
    SYNC_T[0] += sum(a.elapsed_time(b) for a, b in _SYNC_EV) / 1000.0
    del _SYNC_EV[:]
def sync_bench(n_el, dt=torch.float32):
    """SYNCBENCH=1 のとき、学習を始める前に一度だけ NCCL の実力を測る。
    専用の緩衝を使うので勾配にも重みにも触れない。"""
    if WORLD == 1 or os.environ.get("SYNCBENCH") != "1": return None
    b = torch.empty(n_el, device=DEV, dtype=dt)
    for _ in range(5): tdist.all_reduce(b)          # 暖機(NCCL の経路確定)
    torch.cuda.synchronize(); tdist.barrier()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20): tdist.all_reduce(b)
    e1.record(); e1.synchronize()
    ms = e0.elapsed_time(e1) / 20.0; by = n_el * b.element_size()
    del b
    return (ms, by / 1e6, by * 2.0 * (WORLD - 1) / WORLD / (ms / 1e3) / 1e9)
_FLATBUF = {}   # 09-21: 平らな緩衝を使い回す(毎ミニバッチ torch.cat で 42MB 確保していた)
_FC = hasattr(torch, "_foreach_copy_")
# 09-21: 逆伝播と通信を重ねる(PyTorch の DDP が既定でやっていること)。
#   勾配が出来たパラメータから順にバケツに詰め、いっぱいになったら非同期で all_reduce を投げる。
#   逆伝播が終わる頃には通信の大半が済んでいる。**足す値は直列版と同一**なので結果は変わらない。
#   OVERLAP=0 で従来の直列版に戻せる。
# 09-21 実測: 速度の利得は「平らな器の使い回し」だけで得られた(更新 104s → 37s)。重ね合わせ自体は 36s 対 37s で差なし。
#   一方でフックの発火順は rank ごとに違いうるのに NCCL は全 rank で同じ順序の集団通信を要求するため、
#   発火した bucket と控えで送る bucket の順が食い違うとデッドロックする。効果が無い以上、既定は無効。
OVERLAP = os.environ.get("OVERLAP", "0") == "1"
BUCKET_MB = float(os.environ.get("BUCKET_MB", "25"))
class _Overlap:
    """パラメータ群に逆伝播フックを付け、バケツ単位で非同期 all_reduce する。"""
    def __init__(self, params):
        self.ps = [q for q in params if q.requires_grad]
        cap = int(BUCKET_MB * 1024 * 1024 / 4)
        self.buckets = []; cur = []; n = 0
        for q in reversed(self.ps):                      # 逆伝播は後ろの層から来る
            cur.append(q); n += q.numel()
            if n >= cap: self.buckets.append(cur); cur = []; n = 0
        if cur: self.buckets.append(cur)
        self.bufs = [torch.empty(sum(q.numel() for q in bk), device=self.ps[0].device, dtype=torch.float32) for bk in self.buckets]
        self.views = []
        for bk, bf in zip(self.buckets, self.bufs):
            v = []; o = 0
            for q in bk: n_ = q.numel(); v.append(bf[o:o + n_].view_as(q)); o += n_
            self.views.append(v)
        self.left = [len(bk) for bk in self.buckets]; self.work = [None] * len(self.buckets)
        self.of = {}
        for bi, bk in enumerate(self.buckets):
            for q in bk: self.of[q] = bi
        for q in self.ps:
            q.register_post_accumulate_grad_hook(self._hook)
    def _hook(self, q):
        bi = self.of[q]; self.left[bi] -= 1
        if self.left[bi] == 0:
            bk, bf, vs = self.buckets[bi], self.bufs[bi], self.views[bi]
            for v_, q_ in zip(vs, bk): v_.copy_(q_.grad)
            self.work[bi] = tdist.all_reduce(bf, async_op=True)
    def wait(self):
        # 発火しなかったバケツ(そのミニバッチで勾配が付かないパラメータを含む)は、ここで同期に流す。
        # これが無いと一部の勾配が平均されず、直列版と結果が変わってしまう。
        for bi, w in enumerate(self.work):
            if w is not None: continue
            bk, bf, vs = self.buckets[bi], self.bufs[bi], self.views[bi]
            if all(q_.grad is None for q_ in bk): continue     # 誰にも勾配が無いバケツは送らない
            for v_, q_ in zip(vs, bk): v_.copy_(q_.grad) if q_.grad is not None else v_.zero_()
            self.work[bi] = tdist.all_reduce(bf, async_op=True)
        for bi, w in enumerate(self.work):
            if w is None: continue
            w.wait(); self.bufs[bi] /= WORLD
            for v_, q_ in zip(self.views[bi], self.buckets[bi]):
                if q_.grad is not None: q_.grad.copy_(v_)
            self.work[bi] = None
        self.left = [len(bk) for bk in self.buckets]
_OVL = {}
def overlap_of(params, tag):
    if not (OVERLAP and WORLD > 1): return None
    if tag not in _OVL: _OVL[tag] = _Overlap(params)
    return _OVL[tag]

def _sync_grads(params, tag=""):
    """拡大率をかけたままの勾配を全 GPU で平均する(unscale の前に呼ぶので、オーバーフローの判定が全 GPU で揃う)。
    09-21: 1反復に 674 回呼ばれ、毎回 torch.cat で新しい平らな器を作っていた(約85GBのメモリ転送)。
      器を使い回し、コピーを _foreach でまとめる。**通信する値は前と完全に同じ**なので結果は変わらない。
    09-23: 器の鍵から data_ptr を外した。opt.zero_grad() は既定で勾配を None に落とすので
      毎ミニバッチ新しい番地が割り当たりうる(expandable_segments では特に)。番地が変わるたびに
      41.8MB の器を作って _FLATBUF に永久に貯め込んでいた。器は毎回まるごと上書きしてから送るので、
      どの器を使っても**送る値は同じ**。鍵は呼び元の名前(net / crit)と形だけで足りる。"""
    if WORLD == 1: return
    gs = [q.grad for q in params if q.grad is not None]
    n_tot = sum(g.numel() for g in gs)
    key = (tag, n_tot, len(gs))
    ent = _FLATBUF.get(key)
    if ent is None:
        buf = torch.empty(n_tot, device=gs[0].device, dtype=gs[0].dtype)
        views = []; o = 0
        for g in gs: n_ = g.numel(); views.append(buf[o:o + n_].view_as(g)); o += n_
        ent = (buf, views); _FLATBUF[key] = ent
    buf, views = ent
    e0 = e1 = None
    if _SYNCPROF: e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    if _FC:
        torch._foreach_copy_(views, gs)      # 勾配 → 平らな器
        if e0 is not None: e0.record()
        tdist.all_reduce(buf)
        if e1 is not None: e1.record()
        buf /= WORLD
        torch._foreach_copy_(gs, views)      # 平らな器 → 勾配
    else:                                     # 古い torch 用の控え(値は同じ)
        for v_, g_ in zip(views, gs): v_.copy_(g_)
        if e0 is not None: e0.record()
        tdist.all_reduce(buf)
        if e1 is not None: e1.record()
        buf /= WORLD
        for v_, g_ in zip(views, gs): g_.copy_(v_)
    if e0 is not None: _SYNC_EV.append((e0, e1)); SYNC_N[0] += 1; SYNC_N[1] += n_tot
DEV = P.DEV; CLIP = P.CLIP
# exp208: 報酬が最終手だけなのに GAMMA=0.999・LAM=0.95 だと、勝敗が伝わるのは実効 19.6手ぶんしかない
#   (手0 に入る係数は 4.7e-17)。720手の勝負では 12.4% の手にしか信号が届かない。
#   終端だけの報酬では GAMMA=LAM=1 が正しく、このとき adv[t] = z - V[t](全手に等しく勝敗が入る)。
GAMMA = float(os.environ.get("GAMMA", 1.0)); LAM = float(os.environ.get("LAM", 1.0))
E = lambda k, d: type(d)(os.environ.get(k, d))
NSELF, NBEST, NLEAGUE, NTAPEUG = E("NSELF", 48), E("NBEST", 24), E("NLEAGUE", 12), E("NTAPEUG", 12)
ITERS, GATE, GATE_WR, EVAL, VWARM = E("ITERS", 200), E("GATE", 5), E("GATE_WR", 0.70), E("EVAL", 5), E("VWARM", 3)
LR, BETA, ENT, MB = E("LR", 5e-5), E("BETA", 0.1), E("ENT", 0.001), E("MB", 512)
SHAPE = float(os.environ.get("SHAPE", 0))
# exp302(一から学ぶ用の形勢): ポテンシャル Φ = c(it)·[SHAPE_W·(t/T)^SHAPE_TP·tanh((log1p 自分 − log1p 相手)/SHAPE) + SHAPE_OWN·log1p(自分)/log1p(1e5)]
#   r_t += γΦ(s_{t+1}) − Φ(s_t)、終端は Φ=0(最適方策は変わらない)。LAM<1 のときだけ効く(GAMMA=LAM=1 では基準線がずれるだけ)。
#   c(it) = max(0, 1 − (it − SHAPE_IT0)/SHAPE_ANNEAL)(SHAPE_ANNEAL=0 なら常に 1)。非同期では局集め係が各手の所持金を局に書き、学習係が足す。
#   所持金は各手を打つ前(= 価値 V と同じ局面)で記録する。SHAPE=0 かつ SHAPE_OWN=0(既定)なら従来とまったく同じ。
SHAPE_W = float(os.environ.get("SHAPE_W", 1)); SHAPE_OWN = float(os.environ.get("SHAPE_OWN", 0)); SHAPE_TP = float(os.environ.get("SHAPE_TP", 2))
SHAPE_ANNEAL = int(os.environ.get("SHAPE_ANNEAL", 0)); SHAPE_IT0 = int(os.environ.get("SHAPE_IT0", 0))
SHAPE_ON = SHAPE > 0 or SHAPE_OWN > 0
MNY_REC = SHAPE_ON or os.environ.get("SHAPE_REC") == "1" or float(os.environ.get("AUX_LAND", "0")) > 0   # exp302: 形勢を使わなくても所持金の列を記録・ログに出す(対照実験用)
SHAPE_ASSET_TP = float(os.environ.get("SHAPE_ASSET_TP", "0"))   # exp303: >0 で所持金以外の資産の重みを 1 − (t/718)^p で終局に向け 0 へ(最後は売って現金にしないと形勢にならない)。0 で常に 1
SHAPE_ASSET = os.environ.get("SHAPE_ASSET") == "1"
# exp305: 区画のポテンシャル整形(既定で無効。QUAD_A=0 なら従来とまったく同じ)。
#   Φ_q = c_q(it) · QUAD_A · [(自分の区画数 − 1) + QUAD_B · (自分の区画数 − 相手の区画数)]、報酬に γΦ' − Φ、終端 Φ=0。
#   ポテンシャル型なので**最適方策は変わらない**(手番を固定する規則と違い、損なら方策は無視できる)。
#   c_q(it) = max(0, 1 − (it − QUAD_IT0)/QUAD_ANNEAL)(QUAD_ANNEAL=0 なら常に 1)。
#   局集め係が各手の前の区画数(自分・相手)を局に書き、学習係が足す(所持金 mny と同じ作り)。
# exp314: 区画を**早く**買うことへの直接の報酬(LANDR)。ポテンシャル整形は γ=1 だと総和が Φ(s_0) の定数に
#   telescoping してしまい「早さ」への誘因にならない(だから最適方策も変わらない)。ここは意図的に
#   最適方策を変える訓練の補助なので、LANDR_ZERO 反復で 0 に落として、外しても残るかを見る。
#   区画数が増えた手番 t に  LANDR · c(it) · max(0, (LANDR_REF − t)/LANDR_REF)  を足す。t が小さいほど大きい。
#   LANDR=0(既定)なら一切通らない。
LANDR = float(os.environ.get("LANDR", "0")); LANDR_REF = int(os.environ.get("LANDR_REF", "240"))
LANDR_ZERO = int(os.environ.get("LANDR_ZERO", "0")); LANDR_Q = int(os.environ.get("LANDR_Q", "3"))   # この区画数まで
# 09-22(ユーザー指示「早く買えたら買えるだけボーナス。2区画目・3区画目まで」):
#   ・対象は 1→2(2区画目)と 2→3(3区画目)だけ。3→4 は付けない。
#     以前の LANDR_Q=3 は「買う前の区画数が3以下」で判定していたので、3→4(4区画目)まで報酬が付いていた。
#   ・基準を区画ごとに分ける。以前は1本(LANDR_REF=240)で、3区画目は230〜250手で買うので
#     (240−237)/240 ≒ 0.01 しか付かず、早める勾配がほぼ無かった。
#     2区画目は LANDR_REF2(既定 220): PFN の129手で 0.41、我々の150手で 0.32
#     3区画目は LANDR_REF3(既定 320): PFN の198手で 0.38、我々の237手で 0.26
#   LANDR_REF2/REF3 を渡さなければ LANDR_REF2=220, LANDR_REF3=320。
LANDR_REF2 = int(os.environ.get("LANDR_REF2", "220")); LANDR_REF3 = int(os.environ.get("LANDR_REF3", "320"))
# 09-22(方針「区画の速さを目指してから徐々に外す。一旦は勝率は忘れる」): LANDR_AUTO=1 のとき、学習係が読んだ局の
#   自分の2区画目・3区画目の平均手番が LANDR_T2 / LANDR_T3 以下に LANDR_HOLD 反復続けて届いたら、その反復から
#   LANDR_ZERO 反復かけて LANDR を 0 まで線形に下げる(届くまでは LANDR のまま)。下げ始めた反復は途中保存(landr_t0)に残すので、
#   立て直しても下げ方が最初に戻らない。LANDR_IT0=<反復> を渡せば、判定なしでその反復から下げる。
LANDR_AUTO = os.environ.get("LANDR_AUTO") == "1"; LANDR_T2 = float(os.environ.get("LANDR_T2", "131")); LANDR_T3 = float(os.environ.get("LANDR_T3", "198"))
LANDR_HOLD = int(os.environ.get("LANDR_HOLD", "3")); LANDR_IT0 = os.environ.get("LANDR_IT0", "")
# 09-22(方針「2区画目を PFN の129手より早く」): 2区画目の手番は 130/132/135 と飛び飛びで、従来の報酬では 130 と 132 の差が
#   0.009 しかなかった。LANDR_B2>0 で「2区画目を LANDR_P2 手より1手早く取るごとに +LANDR_B2」を足す(手番は区画の記録と同じ数え方)。
#   LANDR_MED=1(既定): 下げ始めの判定を平均ではなく中央値で(150手などの外れ値で判定が揺れないように)。
#   LANDR_T0_RESET=1: 途中保存にある「下げ始めた反復」を捨てて、もう一度全力から(1回の立て直しでだけ渡す。lines.conf には書かない)。
LANDR_B2 = float(os.environ.get("LANDR_B2", "0")); LANDR_P2 = float(os.environ.get("LANDR_P2", "131"))
LANDR_MED = os.environ.get("LANDR_MED", "1") == "1"; LANDR_T0_RESET = os.environ.get("LANDR_T0_RESET") == "1"
# 09-22(分析係: 2区画目が130手で止まるのは資金。4日目(手110〜114)のイチゴの種 約235 で1000に届くのが129手になり、
#   しかも方策は区画の発注を手129に「時計で」出している): E2=1 で、対局係が学習側の手を選んだ直後に、選ばれた軌道(割合 E2_P)で
#   自分の区画が1つのまま E2_T0 ≤ 手 < E2_T1 の間、市場の手の列から E2_DROP の注文(既定 12 = イチゴの種)を外し、
#   区画の購入(20)を列の最後(売りの後ろ)に置く。書き換えた手を**そのまま記録**するので、方策はその手を選んだものとして学ぶ
#   (市場の手にはマスクが無く、確率は有限)。E2_P は学習係が pub/state.json で配り、制御ファイルで止めずに下げられる。
E2 = os.environ.get("E2") == "1"; E2_P = float(os.environ.get("E2_P", "0.5"))
E2_T0 = int(os.environ.get("E2_T0", "96")); E2_T1 = int(os.environ.get("E2_T1", "140"))
E2_DROP = set(int(x) for x in os.environ.get("E2_DROP", "12").split(",") if x)
# 09-22(方針「速さを覚えたら徐々に外す」): E2_AUTO=1 のとき学習係が、介入しなかった軌道の2区画目の中央値が E2_T2 以下に
#   E2_HOLD 反復続けて届いたら、その反復から E2_ZERO 反復かけて E2_P を 0 まで下げて配る(下げ始めた反復と値は途中保存に残す)。
# 09-22(第117世代から4区画目を自分で買えるように): E4=1 で、選ばれた軌道(割合 E4_P)の区画が3つのまま手 E4_T0 以降、
#   市場の手の列の最後に区画の購入(20)を足して**そのまま記録**する(所持金が4000に届けば同じ手番の売上込みで買える)。
#   実測: 第117世代は貪欲では4区画目を一度も買わず、「買えたらすぐ買う」規則で第117世代に 0.703、v82 に 0.844。
#   E4_AUTO=1: 介入しなかった軌道のうち4区画目を買った割合が E4_SHARE 以上に E4_HOLD 反復続けて届いたら、E4_ZERO 反復かけて E4_P を 0 へ。
E4 = os.environ.get("E4") == "1"; E4_P = float(os.environ.get("E4_P", "0.5")); E4_T0 = int(os.environ.get("E4_T0", "250"))
E4_AUTO = os.environ.get("E4_AUTO") == "1"; E4_SHARE = float(os.environ.get("E4_SHARE", "0.5"))
E4_HOLD = int(os.environ.get("E4_HOLD", "3")); E4_ZERO = int(os.environ.get("E4_ZERO", "60"))
# 09-22(3パターン: 4区画型 / 3区画型の速攻 / 4区画の速攻): 区画ごとに介入を始める手番を指定する。-1 で無効。
#   E4_T2 = 区画1つのとき(2区画目)、E4_T3 = 区画2つのとき(3区画目)、E4_T0 = 区画3つのとき(4区画目)。
#   自動で外す判定は「介入しなかった軌道のうち、手 E4_BY までに E4_QMAX 区画に届いた割合 ≥ E4_SHARE」。
# 09-22(旧マスクの第117世代を新マスクへ段階的に移す): LMIX=1 のとき、学習側の軌道のうち割合 LMIX_P を新マスク(版 LMIX_V、既定2)で
#   打たせる(残りは環境変数 LEGAL_OPT の版)。LMIX_P は LMIX_ANNEAL 反復かけて LMIX_P0 → LMIX_P1 に上げ、学習係が配る
#   (制御ファイルでも変えられる)。相手の席は環境変数の版のまま。記録される合法手は実際に打った版なので、学習と一致する。
LMIX = os.environ.get("LMIX") == "1"; LMIX_V = int(os.environ.get("LMIX_V", "2"))
LMIX_P0 = float(os.environ.get("LMIX_P0", "0.1")); LMIX_P1 = float(os.environ.get("LMIX_P1", "1.0")); LMIX_ANNEAL = int(os.environ.get("LMIX_ANNEAL", "200"))
LMIX_P = LMIX_P0
# 09-23(マスク移行 mask0→mask2、denden さんの方法の移植): MASK_MIG=1 で
#   ・採取: 学習側の軌道(自分の席)だけ新マスク(MIG_V、既定 2)で打つ。相手・テープ・リーグの席は従来の版のまま。
#   ・罰:   新マスクでだけ増えた手 expanded = mask2 & ~mask0 の logit から MIG_PEN を引く。**元の重みは変えない**。
#           罰は合法マスク lm の bit1 に載せて運ぶので、採取(P.sample)と学習(P.evaluate)が必ず同じ罰になる。
#   ・日程: MIG_HOLD 反復(既定 10)は初期値を保持し、そこから MIG_ZERO 反復(既定 60)で線形に 0 へ。
#   MIG_PEN は**モデルごとに較正**する(mig_calib.py。新旧 logit 差の 99.9%点 + 2 を切り上げ)。他のモデルの値を流用しない。
#   非同期では「その局を集めたときの罰」を局に書き込み、学習係はそれを使う(lag があっても採取と学習が一致する)。
MASK_MIG = os.environ.get("MASK_MIG") == "1"
MIG_PEN = float(os.environ.get("MIG_PEN", "0")); MIG_HOLD = int(os.environ.get("MIG_HOLD", "10")); MIG_ZERO = int(os.environ.get("MIG_ZERO", "60"))
MIG_IT0 = int(os.environ.get("MIG_IT0", "-1")); MIG_V = int(os.environ.get("MIG_V", "2"))   # -1 = 起動した反復を起点にする
# 09-23: 罰が届いていないまま4GPUを溶かさないための安全装置。保持のあいだ「増えた手を選んだ割合」が
#   MIG_GUARD を超えたら、対局係はその場で止まる(0 で無効)。FASTINF が unit_logits を差し替えて罰が
#   消えていた事故(g5r1m、選んだ割合 0.88)を二度と見逃さないため。
MIG_GUARD = float(os.environ.get("MIG_GUARD", "0.05"))
# 崩壊のいちばん分かりやすい印は「2区画目をいつまでも買えない」(正常 150手前後、崩壊すると 700手)。
#   選択率より直接的なので、これも見て止める(0 で無効)。
MIG_Q2MAX = float(os.environ.get("MIG_Q2MAX", "300"))


def mig_pen_at(it):
    """この反復で使う係数。罰の方式は 初期値 MIG_PEN → 0、混合(MIG_MIX=1)の方式は ε 初期値 → 1。
       どちらも MIG_HOLD 反復は初期値を保ち、そこから MIG_ZERO 反復で線形に動かす。"""
    if not MASK_MIG: return 0.0
    k = it - MIG_IT0; end = 1.0 if M.MIG_MIX else 0.0
    if k < MIG_HOLD: return MIG_PEN
    if k >= MIG_ZERO: return end
    return MIG_PEN + (end - MIG_PEN) * ((k - MIG_HOLD) / float(max(1, MIG_ZERO - MIG_HOLD)))
E4_T2 = int(os.environ.get("E4_T2", "-1")); E4_T3 = int(os.environ.get("E4_T3", "-1"))
E4_QMAX = int(os.environ.get("E4_QMAX", "4")); E4_BY = int(os.environ.get("E4_BY", "719"))
# 09-22(巡回係の指摘): 毎手 区画の購入を足すと、買えない手番でも「方策が選んだ手」として大量に記録される(3区画速攻で1軌道 約70手)。
#   E4_AFF=1 で、手番の初めの所持金が次の区画の値段(1000/2000/4000)以上のときだけ書き換える(同じ手番の売上は見込まない)。
E4_AFF = os.environ.get("E4_AFF") == "1"; E4_PRICE = {1: 1000.0, 2: 2000.0, 3: 4000.0}
E4_TS = {1: E4_T2, 2: E4_T3, 3: E4_T0}; E4_TMIN = min([v for v in E4_TS.values() if v >= 0] or [10 ** 9])
def e4_reach(QD0, BY=None, QM=None):
    """(T, N) の区画数 → 手 BY までに QM 区画に届いたか(軌道ごと)"""
    BY = E4_BY if BY is None else BY; QM = E4_QMAX if QM is None else QM
    return (QD0[:min(BY + 1, QD0.shape[0])] >= QM).any(0)
def e4_fix(m, q):
    """市場の手の列の最後(終了の前)に区画の購入を足す。すでに区画の注文があれば最後に移す"""
    ns_ = len(m); pre = []
    for c, x in zip(m, q):
        if c == P.M_STOP: break
        if c == 20: continue
        pre.append((int(c), int(x)))
    pre = pre[:ns_ - 2] + [(20, 0)]
    om = np.full(ns_, P.M_STOP, dtype=m.dtype); oq = np.zeros(ns_, dtype=q.dtype)
    for k_, (c, x) in enumerate(pre): om[k_] = c; oq[k_] = x
    return om, oq, not (np.array_equal(om, m) and np.array_equal(oq, q))
E2_AUTO = os.environ.get("E2_AUTO") == "1"; E2_T2 = float(os.environ.get("E2_T2", "128"))
E2_HOLD = int(os.environ.get("E2_HOLD", "3")); E2_ZERO = int(os.environ.get("E2_ZERO", "60"))
def e2_fix(m, q):
    """市場の手の列 m(候補)と q(数量クラス)を書き換える。戻り値 (m, q, 変えたか)"""
    ns_ = len(m); pre = []
    for c, x in zip(m, q):
        if c == P.M_STOP: break
        if c in E2_DROP or c == 20: continue
        pre.append((int(c), int(x)))
    pre = pre[:ns_ - 2] + [(20, 0)]
    om = np.full(ns_, P.M_STOP, dtype=m.dtype); oq = np.zeros(ns_, dtype=q.dtype)
    for k_, (c, x) in enumerate(pre): om[k_] = c; oq[k_] = x
    return om, oq, not (np.array_equal(om, m) and np.array_equal(oq, q))
# 09-23(exp320): 介入E5 = 序盤の家畜の買い増し。**系統B**(記録される行動 ms/mq を書き換えるので勾配に乗る)。
#   動機: 介入なしで打たせた実測(probe_stick / fa_stick)で、PFN 模倣は序盤に家畜を 59.0 頭買うのに我々は 20.5 頭。
#   これは「家畜を買う店が多い盤面で1軒あたり −2,100」「損の88%が500手以降」と噛み合う(終盤の店の需要に供給できていない)。
#   系統A(OPEN/PFNW)は pool.step の中で注文を書き換えるため記録に残らず、勾配に一度も現れなかった(failure-analysis.md)。
#   判定: ログの「介入なし」の頭数が上がれば方策が受け入れている。E4 は1反復で 0.00→1.00 に立った。E2 は 57反復 不動で拒否された。
# 09-23(exp321): SHOPW = 家畜を買う店が多い局の優位度の重みを上げる。
#   終盤係の診断: 店1軒あたり −2,256 の損は TOMATO(−1,728)と EGG(−1,028)の売上が消えること。
#   ところが**自己対戦では両席が同じ地図で同じだけ損をするので、この勾配は完全に打ち消える**
#   (実測: 身内どうしの店傾き −2±58、対 PFN −1,937±157)。だから何十世代回しても学べない。
#   そこで「相手が違う局(self 以外)」だけ、家畜店の多い局の優位度を重くする。自己対戦は等倍のまま。
SHOPW = float(os.environ.get("SHOPW", "0")); SHOPW_MAX = float(os.environ.get("SHOPW_MAX", "3"))
SHOPW_SELF = os.environ.get("SHOPW_SELF", "0") == "1"   # 1 で自己対戦にもかける
_LIVE_SHOP = {3, 6, 7}    # ICE_CREAM / SMOOTHIE / YARN(obs_feat の SHOPS の並び、アルファベット順)
E5 = os.environ.get("E5") == "1"; E5_P = float(os.environ.get("E5_P", "0.5"))
E5_T0 = int(os.environ.get("E5_T0", "0")); E5_T1 = int(os.environ.get("E5_T1", "240"))
E5_CLASS = int(os.environ.get("E5_CLASS", "17"))     # 16=GOOSE 17=COW 18=SHEEP(obs_feat の市場の候補)
E5_QTY = int(os.environ.get("E5_QTY", "1"))          # 数量クラス
E5_EVERY = int(os.environ.get("E5_EVERY", "12"))     # この手番おきに1回だけ足す(毎手だと買いすぎる)
E5_MONEY = float(os.environ.get("E5_MONEY", "1500")) # 所持金がこれ以上のときだけ
E5_ZERO = int(os.environ.get("E5_ZERO", "0"))        # >0 ならこの反復数で E5_P を 0 に落とす


# 09-23(exp322): 介入E6 = 支出を絞る。**系統B**(記録される行動 ms/mq を書き換えるので勾配に乗る)。
#   解析係が我々 対 PFN 56371729 の実戦16局を収支に分解した結果:
#     売上は同じ(+185 ± 1,212)、**支出だけ +5,064 ± 391(t≈13)** 多い。これで負けの −4,880 が全部説明できる。
#     内訳: 市場での小麦の買い +3,005 / 雇用(11人目)+1,116 / 家畜 +800 / 種 +143。
#     177局の回帰では **1円使っても 0.915 ± 0.021 円しか戻らない**(1.0 を 4σ 下回る)。支出は純損。
#   B1 市場の小麦を日10から1日 E6_WHEAT_N 件まで / B2 日11から止める / B3 雇用を1日 E6_HIRE_N 人まで。
M_BUYLAND = 20    # obs_feat: c==20 → BUY_LAND
M_BUYWHEAT = 14   # obs_feat: c==14 → BUY_PRODUCT WHEAT
M_HIRE = 19       # obs_feat: c==19 → HIRE
E6 = os.environ.get("E6") == "1"; E6_P = float(os.environ.get("E6_P", "0.5"))
E6_WHEAT_T0 = int(os.environ.get("E6_WHEAT_T0", "240"))   # ここから1日の件数を絞る
E6_WHEAT_N = int(os.environ.get("E6_WHEAT_N", "2"))       # 1日に通す小麦買いの件数
E6_WHEAT_STOP = int(os.environ.get("E6_WHEAT_STOP", "264"))  # ここから全部落とす
E6_LAND4 = os.environ.get("E6_LAND4") == "1"          # 4区画目を買わせない(系統B版。NOLAND4 は系統Aで勾配に乗らなかった)
E6_LAND4_NQ = int(os.environ.get("E6_LAND4_NQ", "3"))  # この区画数以上なら区画購入の注文を落とす
E6_HIRE_T0 = int(os.environ.get("E6_HIRE_T0", "264"))
E6_HIRE_N = int(os.environ.get("E6_HIRE_N", "10"))        # 1日に通す雇用の件数


def e6_fix(m, q, t, nw, nh, nq=0):
    """支出の注文を落とす。nw/nh はその軌道のその日の通過件数(更新する)。戻り値 (m, q, 変えたか, nw, nh)"""
    ns_ = len(m); pre = []; ch = False
    for c, x in zip(m, q):
        c = int(c); x = int(x)
        if c == P.M_STOP: break
        if c == M_BUYWHEAT and t >= E6_WHEAT_T0:
            if t >= E6_WHEAT_STOP or nw >= E6_WHEAT_N: ch = True; continue
            nw += 1
        elif c == M_BUYLAND and E6_LAND4 and nq >= E6_LAND4_NQ:
            ch = True; continue
        elif c == M_HIRE and t >= E6_HIRE_T0:
            if nh >= E6_HIRE_N: ch = True; continue
            nh += 1
        pre.append((c, x))
    if not ch: return m, q, False, nw, nh
    om = np.full(ns_, P.M_STOP, dtype=m.dtype); oq = np.zeros(ns_, dtype=q.dtype)
    for k_, (c, x) in enumerate(pre[:ns_]): om[k_] = c; oq[k_] = x
    return om, oq, True, nw, nh


def e5_fix(m, q):
    """市場の手の列の最後(終了の前)に家畜の購入を足す。すでに同じ候補があれば何もしない。e4_fix と同じ作り。"""
    ns_ = len(m); pre = []
    for c, x in zip(m, q):
        if c == P.M_STOP: break
        pre.append((int(c), int(x)))
    if any(c == E5_CLASS for c, _ in pre) or len(pre) >= ns_ - 1:
        return m, q, False
    pre = pre[:ns_ - 2] + [(E5_CLASS, E5_QTY)]
    om = np.full(ns_, P.M_STOP, dtype=m.dtype); oq = np.zeros(ns_, dtype=q.dtype)
    for k_, (c, x) in enumerate(pre): om[k_] = c; oq[k_] = x
    return om, oq, not (np.array_equal(om, m) and np.array_equal(oq, q))


LANDR_W3 = float(os.environ.get("LANDR_W3", "1"))   # 09-22: 3区画目の報酬の重み(0 で「2区画目だけに報酬」)
# 09-22: 走ったまま設定を変える制御ファイル <ASYNC_DIR>/ctl.json(学習係だけが毎反復読む)。共有の区画は満杯で、立て直すと
#   再開が約20時間後になったため。{"LANDR_W3": 0, "LANDR_B2": 0.2, ..., "landr_reset": "r1"} のように書く。
#   landr_reset は値が変わったときだけ「区画報酬の減衰」を取り消して全力に戻す(値は途中保存に残るので、立て直しても二重にかからない)。
CTL_KEYS = {"LANDR": float, "LANDR_B2": float, "LANDR_P2": float, "LANDR_W3": float, "LANDR_T2": float, "LANDR_T3": float,
            "LANDR_ZERO": int, "LANDR_HOLD": int, "LANDR_AUTO": bool, "LANDR_MED": bool, "BETA": float, "E2_P": float, "E4_P": float, "LMIX_P": float, "AUX_BC": float}
QUAD_A = float(os.environ.get("QUAD_A", "0")); QUAD_B = float(os.environ.get("QUAD_B", "1"))
QUAD_ANNEAL = int(os.environ.get("QUAD_ANNEAL", "0")); QUAD_IT0 = os.environ.get("QUAD_IT0", "")   # 空ならこの実行の開始反復(引き継ぎで最初から 0 にならないように)
QUAD_ON = QUAD_A != 0
QUAD_REC = QUAD_ON or os.environ.get("QUAD_REC") == "1" or float(os.environ.get("AUX_LAND", "0")) > 0 or LANDR > 0   # 使わなくても区画数の列を記録・ログに出す(対照実験用)   # exp303: 形勢の「所持金」を資産(所持金 + 種・植えた作物は種の値段、納屋と手持ちの品は基準価格、家畜は買値、土地は買値)にする
# exp308: 「土地を買う」行動の対数確率を直接上げる補助損失(denden さんの land4 v3 の写し)。
#   報酬ボーナスは**自発購入にしか付かない**ので、一度も買わない方策には一度も発生せず不発に終わる
#   (向こうの実測: v1 は60反復で累計1局、v2 は30反復で0局)。行動の log 確率に直接足すと 5反復で 0/100 → 102/102 になった。
#   損失 = c(it) · mean(−log π(BUY_LAND | 買える状態の行))。π は市場ヘッドの**最初の注文位置**の class 20。
#   対象は「自分の区画数が AUX_LAND_Q・所持金が AUX_LAND_MONEY 以上・手数が AUX_LAND_T0〜T1」の行だけ。
#   c(it) = AUX_LAND · max(0,(zero − n)/(zero − 1)) で AUX_LAND_ZERO 反復で厳密に 0 にし、外しても残るかを見る。
#   AUX_LAND=0(既定)なら対象の計算も損失項も一切通らず、これまでと完全に同じ。
# 09-21(ユーザー「グリーディでなく温度を上げて多様な手も試したい」): ロールアウトの温度。
#   ROLL_T → ROLL_T_END へ ROLL_T_ANNEAL 反復かけて線形に下げる。1.0 で従来どおり。
#   サンプリングと PER0(行動の対数確率)の両方に同じ温度を掛けるので、重要度比 ρ = π_θ/π_行動 は正しいまま。
ROLL_T = float(os.environ.get("ROLL_T", "1.0")); ROLL_T_END = float(os.environ.get("ROLL_T_END", "1.0"))
ROLL_T_ANNEAL = int(os.environ.get("ROLL_T_ANNEAL", "0"))
def roll_t(it):
    if ROLL_T_ANNEAL <= 0: return ROLL_T
    f = min(1.0, max(0.0, it / float(ROLL_T_ANNEAL)))
    return ROLL_T + (ROLL_T_END - ROLL_T) * f
AUX_LAND = float(os.environ.get("AUX_LAND", "0")); AUX_LAND_ZERO = int(os.environ.get("AUX_LAND_ZERO", "60"))
AUX_LAND_MONEY = float(os.environ.get("AUX_LAND_MONEY", "4000")); AUX_LAND_Q = int(os.environ.get("AUX_LAND_Q", "3"))
AUX_LAND_T0 = int(os.environ.get("AUX_LAND_T0", "120")); AUX_LAND_T1 = int(os.environ.get("AUX_LAND_T1", "480"))
AUX_ON = AUX_LAND > 0
# exp312: 買った区画を「使う」補助損失(denden さんの land4use の写し)。買わせるだけでは 4,000 を捨てるだけになる。
#   対象は (行, ユニット): 自分の区画数が4・そのユニットが生存・x と y が両方 5〜9(南東区画)・
#   そのユニットの植え付け(15〜19)が合法(遮蔽されていない = 足元が空きタイルで種を持つ)。
#   損失 = −mean(logsumexp(lsu[b,u,15:20]))。「どれでもいいから植えろ」であって、作物は指定しない。
#   条件は観測(ou の x,y、g[39] の区画数)と合法手から作れるので、局に列を足す必要は無い。
AUX_USE = float(os.environ.get("AUX_USE", "0")); AUX_USE_ZERO = int(os.environ.get("AUX_USE_ZERO", "60"))
AUX_USE_ON = AUX_USE > 0
def aux_use_c(it, start=0):
    n = it - start + 1
    return AUX_USE * max(0.0, (AUX_USE_ZERO - n) / max(1, AUX_USE_ZERO - 1)) if AUX_USE_ZERO > 1 else 0.0
# 09-22(チームメイトの助言「特定の棋譜に寄せたいなら、その棋譜との cross entropy を足す」): PPO の各ミニバッチに、棋譜
#   (bc_<AUX_BC_SPLIT>.json の train。既定は PFN の今の提出2つ = 3区画型の 285局)から AUX_BC_MB 局面を引き、BC の損失
#   (bc_teama.losses と同じ: ユニット + 0.5·数量 + 市場の列 + 0.5·数量。区画の購入の重みは AUX_BC_LANDW)を係数 AUX_BC で足す。
#   AUX_BC=0(既定)なら読み込みも損失項も一切通らない。ctl.json で途中から変えられる。
AUX_BC = float(os.environ.get("AUX_BC", "0")); AUX_BC_MB = int(os.environ.get("AUX_BC_MB", "256"))
AUX_BC_SPLIT = os.environ.get("AUX_BC_SPLIT", "p3now"); AUX_BC_CACHE = os.environ.get("AUX_BC_CACHE", "/workspace/kaggriculture/ext4p/mmpq2")
_BCT = {}
def bc_data():
    """棋譜を GPU に1回だけ載せる(20万局面で約 1.6GB)。"""
    if not _BCT:
        import bc_teama as BCM
        BCM.LANDW = float(os.environ.get("AUX_BC_LANDW", "1"))
        es = json.load(open("/content/local_logs/bc_%s.json" % AUX_BC_SPLIT))["train"]; pairs = [(AUX_BC_CACHE, e) for e in es]
        T, _ = BCM.load_dev_pairs(pairs)
        lens = [np.load("%s/%s.npz" % pe)["g"].shape[0] for pe in pairs]; n = sum(lens)
        T["step"] = torch.as_tensor(np.concatenate([np.arange(L) for L in lens]), device=DEV)
        T["w"] = torch.ones(n, device=DEV); T["wm"] = torch.ones(n, device=DEV); T["mm"] = torch.ones(n, dtype=torch.bool, device=DEV)
        _BCT.update(T=T, n=n, f=BCM.losses, ne=len(es))
    return _BCT
M_BUYLAND = 20   # 市場の行動 class。obs_feat.py の `if c == 20: market.append(["BUY_LAND"])` と対

def aux_c(it, start=0):
    """補助損失の係数。AUX_LAND_ZERO 反復で厳密に 0。"""
    n = it - start + 1
    return AUX_LAND * max(0.0, (AUX_LAND_ZERO - n) / max(1, AUX_LAND_ZERO - 1)) if AUX_LAND_ZERO > 1 else 0.0


_SEEDC = np.array([10, 20, 50, 100, 80], np.float64); _BASEP = np.array([25, 35, 60, 120, 250, 50, 160, 200, 100], np.float64)
_ANIMC = np.array([300, 400, 500], np.float64); _LANDC = np.array([0, 0, 1000, 3000, 7000], np.float64)   # 区画数 1〜4 までに払った土地代の合計


def pool_assets(pool, t=None):
    """(局, 2) 絶対座席ごとの資産。買い(種・家畜・土地)で下がらず、雇い賃・売り損・腐らせた分だけ下がる。BatchSPPool 専用"""
    e = pool.env; out = np.asarray(pool.money(), np.float64).copy(); n_ = out.shape[0]
    for s_ in (0, 1):
        e.set_view(s_)
        ol = np.asarray(e.obs_legal()); seeds = np.rint(ol[:, 27:32] * 20)
        items = np.asarray(e.shed_all(), np.float64) + np.asarray(e.unit_inv(S.NU), np.float64).sum(1)   # 納屋 + 手持ち(12品目、9〜11 は家畜)
        planted = np.asarray(e.mkt_aux(0))[:, :5, 1]                                                      # 作物ごとの植えた区画の数
        an = np.asarray(e.animals()).reshape(n_, -1); aid = np.where(an >= 9, an - 9, an - 1)
        placed = sum(((an > 0) & (aid == k)).sum(1) * _ANIMC[k] for k in range(3))
        nq = np.clip(np.rint(np.asarray(e.farm_pub(0))[:, 2] * 4).astype(int), 0, 4)
        out[:, s_] += seeds @ _SEEDC + planted @ _SEEDC + items[:, :9] @ _BASEP + items[:, 9:12] @ _ANIMC + placed + _LANDC[nq]
    e.set_view(0)
    if SHAPE_ASSET_TP > 0 and t is not None:
        m_ = np.asarray(pool.money(), np.float64); out = m_ + (1.0 - (t / 718.0) ** SHAPE_ASSET_TP) * (out - m_)
    return out


def pool_quads(pool):
    """(局, 2) 絶対座席ごとの解放した区画数(1〜4)。BatchSPPool 専用(pool_assets の nq と同じ読み方)"""
    e = pool.env; out = np.zeros((pool.n, 2), np.float32)
    for s_ in (0, 1):
        e.set_view(s_); out[:, s_] = np.clip(np.rint(np.asarray(e.farm_pub(0))[:, 2] * 4), 0, 4)
    e.set_view(0)
    return out


OWN_IT0 = os.environ.get("OWN_IT0", "")
# exp304(貯め込みの罠を外す。すべて既定で無効):
#   WIN_RAMP=N>0: 終端の報酬 = w·勝敗 + (1−w)·tanh((最終の資産 − 3000)/GROW_SCALE)、w = min(1, it/N)。貯め込みは 0、増やせば +
#   GATE_DRAW=1: 関門で引き分けも通す(勝ち+引き分けの率 ≥ GATE_WR)。GATE_SAMPLE=1: 関門を確率的に打つ
#   GUIDE_P>0: 局集め係の局のうち割合 GUIDE_P の学習側の席を、最初の GUIDE_N 手だけテープ(OPPDIR)で打たせる。
#     GUIDE_ANNEAL 反復かけて GUIDE_P → GUIDE_P_END、GUIDE_N → GUIDE_N_END に動かす(N を減らせば後ろからの学習)。
GUIDE_DIR = os.environ.get("GUIDE_DIR") or os.environ["OPPDIR"]   # exp313: 誘導に使うテープ(既定は OPPDIR と同じ。分ければ対局相手のテープを変えずに誘導だけ差し替えられる)
#     打たせた手は方策の手ではないので、その行の優位度を 0 にして方策の損失から外す(価値・勝敗・形勢の計算には使う)。
WIN_RAMP = int(os.environ.get("WIN_RAMP", "0")); GROW_SCALE = float(os.environ.get("GROW_SCALE", "3000"))
GATE_DRAW = os.environ.get("GATE_DRAW") == "1"; GATE_SAMPLE = os.environ.get("GATE_SAMPLE") == "1"
GUIDE_P = float(os.environ.get("GUIDE_P", "0")); GUIDE_P_END = float(os.environ.get("GUIDE_P_END", os.environ.get("GUIDE_P", "0")))
GUIDE_N = int(os.environ.get("GUIDE_N", "144")); GUIDE_N_END = int(os.environ.get("GUIDE_N_END", str(GUIDE_N))); GUIDE_ANNEAL = int(os.environ.get("GUIDE_ANNEAL", "0"))


def term_reward(z, grow, it):
    """z: 勝敗 (N,)、grow: 最終の資産 (N,) か None"""
    if WIN_RAMP <= 0: return z
    wz = min(1.0, it / WIN_RAMP)
    return wz * z + (1.0 - wz) * np.tanh((grow - 3000.0) / GROW_SCALE)   # exp302: 空なら従来どおり (it − start) で OWN を下げる。数を入れると (it − OWN_IT0)(学習係の再開で下げ直しにならない)


def shape_c(it):
    return max(0.0, 1.0 - (it - SHAPE_IT0) / SHAPE_ANNEAL) if SHAPE_ANNEAL > 0 else 1.0


def shape_phi(own, opp, c):
    """own, opp: (T, N) 各手の前の所持金(自分・相手)。返り値 Φ: (T, N)"""
    T_ = own.shape[0]; Phi = np.zeros(own.shape, np.float32)
    if c <= 0: return Phi
    a = np.log1p(np.maximum(own, 0)).astype(np.float32); b = np.log1p(np.maximum(opp, 0)).astype(np.float32)
    if SHAPE > 0 and SHAPE_W != 0:
        w = (np.arange(T_, dtype=np.float32) / max(T_ - 1, 1)) ** SHAPE_TP
        Phi += SHAPE_W * w[:, None] * np.tanh((a - b) / SHAPE)
    if SHAPE_OWN != 0: Phi += SHAPE_OWN * a / np.float32(np.log1p(1e5))
    return (c * Phi).astype(np.float32)
def quad_steps(QD):
    """QD: (T, N, 2) 各手の前の区画数 → (自分が2区画目を取った平均手番, 3区画目, 相手の3区画目)。取らなければ T"""
    T_ = QD.shape[0]
    f = lambda a, k: np.where((a >= k).any(0), (a >= k).argmax(0), T_).mean()
    return f(QD[..., 0], 2), f(QD[..., 0], 3), f(QD[..., 1], 3)


def quad_c(it, start=0):
    it0 = int(QUAD_IT0) if QUAD_IT0 else start
    return max(0.0, 1.0 - (it - it0) / QUAD_ANNEAL) if QUAD_ANNEAL > 0 else 1.0


def quad_phi(own, opp, c):
    """own, opp: (T, N) 各手の前の区画数(自分・相手)。返り値 Φ: (T, N)"""
    if c <= 0: return np.zeros(own.shape, np.float32)
    return (c * QUAD_A * ((own - 1.0) + QUAD_B * (own - opp))).astype(np.float32)


# exp220: 学習した批評家は 価値EV(対 勝敗) −0.37〜−0.40 で「定数を答える」より悪かった(o6/o7、80反復)。
#   1反復の独立な勝敗ラベルが192局ぶんしかなく、720手ぶんの相関した局面に当てはめて過学習している。
#   CRITIC=0 では批評家を使わず、V = その反復で同じ種類の相手と打った局の平均勝敗(基準線)にする。
CRITIC = os.environ.get("CRITIC", "1") == "1"
ANTI = os.environ.get("ANTI") == "1"   # exp242(PTCG 神託批評家): 批評家を V = (v(自分の観測) − v(相手の観測)) / 2 に。相手の観測は学習のときだけ使う特権情報(提出する方策は変わらない)
EMA = float(os.environ.get("EMA", 0))
EXTRA = [x for x in os.environ.get("LEAGUE_EXTRA", "").split(",") if x]   # exp252(PTCG 5・7位): 最初からリーグに入れる凍った強い相手(重みのパス)
# 09-22(学習分析): リーグの相手は GROUP=1 だと標本で打つので、PFN 型の BC がとても弱く出る(本番の貪欲より約0.07勝率が上がる)。
#   LEAGUE_GREEDY=1 でリーグの相手だけ貪欲にする(best と自己対戦はそのまま)。既定は従来どおり。
LEAGUE_GREEDY = os.environ.get("LEAGUE_GREEDY") == "1"
LEAGUE_EXTERNAL_ONLY = os.environ.get("LEAGUE_EXTERNAL_ONLY") == "1"   # 09-30: チームメイト(mmpqbcw)と同じ。既定オフ
PFSP = os.environ.get("PFSP") == "1"         # exp252(PTCG 7位): リーグの相手を 0.15 + (1 − 勝率)^2 の重みで選ぶ(苦手な相手ほど多く)
# 09-22: 段階表(ラダー)。LADDER=<名前\tパス\tElo の TSV>(tools/ladder_fit.py が総当たりの実測から作る)。
#   最新の世代だけが相手だと、強くなったかが分からず(新マスクの系統は第117世代に 0/384)、相手の強さも一段ずつ上がらない。
#   学習係が自分のレーティングをリーグの対局の結果から毎反復 Elo で更新して pub/state.json で配り、
#   対局係は各相手への予想勝率 p = 1/(1+10^((R相手 − R自分)/400)) から重みを出して相手を抽選する。
#   LADDER_W=var: p(1−p)(五分の相手ほど多く)/ hard: (1−p)^2(勝てない相手ほど多く)。どちらも +0.01 の下駄。
#   関門を通った自分の世代も、通ったときのレーティングで段階表に加わる(直近 8 世代)。LEAGUE_EXTRA とは併用しない。
LADDER = os.environ.get("LADDER", ""); LADDER_W = os.environ.get("LADDER_W", "var"); ELO_K = float(os.environ.get("ELO_K", "4"))
ELO0 = os.environ.get("ELO0", "")
def ladder_load(fn):
    d = {}
    for ln in open(fn):
        if not ln.strip() or ln.startswith("#"): continue
        f_ = ln.rstrip("\n").split("\t"); assert f_[1] not in d, "段階表に同じ重みが2行: %s" % f_[1]; d[f_[1]] = (f_[0], float(f_[2]), int(f_[3]) if len(f_) > 3 and f_[3] else -1)   # 4列目 = その相手の合法判定の版(負なら学習と同じ)
    return d
def ladder_weights(elo_d, r_me):
    ps = {x: 1.0 / (1.0 + 10 ** ((v[1] - r_me) / 400.0)) for x, v in elo_d.items()}
    w = {x: ((1 - p) ** 2 if LADDER_W == "hard" else p * (1 - p)) + 0.01 for x, p in ps.items()}
    return w, ps
SAFE_KL = float(os.environ.get("SAFE_KL", 0))
# 09-21(PTCG 9位): BC(基準方策)への KL の手綱 β を、BETA_ZERO 反復かけて 0 まで線形に下げる。
#   9位は「サイドの形づけと BC への KL を 8更新で 0 に減衰」。0 なら BETA は一定(従来どおり)。
BETA_ZERO = int(os.environ.get("BETA_ZERO", "0"))
def beta_at(it):
    if BETA_ZERO <= 0: return BETA
    return BETA * max(0.0, 1.0 - it / float(BETA_ZERO))
# 09-21(PTCG 3位): 近似KL だけでなく、批評家の損失が大きい更新も捨てる
SAFE_VL = float(os.environ.get("SAFE_VL", 0))
CRIT_HL = os.environ.get("CRIT_HL") == "1"   # exp256(PTCG 7位): 反対称の批評家の頭を「最終の差額の分布」(HL-Gauss 51ビン)に。ANTI=1 と一緒に。HLINIT = exp250 の vl5 a2_hl
HL_NB, HL_LO, HL_HI = 51, -60000.0, 60000.0   # exp252(PTCG 3位): ミニバッチの近似 KL(旧方策との)がこれを超える更新は捨てる。0 で従来どおり   # exp238(PTCG 4・9位): >0 で重みの指数移動平均(反復ごとに ema = EMA·ema + (1−EMA)·現行)を関門・ものさしの挑戦者にする。0 で従来どおり"
# exp223: GRPO(グループ相対)。同じシード・同じ座席・同じ相手の対局を GROUP 回ずつ打ち、その中の平均との差を優位度にする。
#   批評家が要らない(o6/o7 の批評家は 価値EV −0.4 で定数より悪かった)。報酬は所持金の差の tanh(REWARD=margin)か勝ち負け(win)。
GROUP = int(os.environ.get("GROUP", 1)); REWARD = os.environ.get("REWARD", "margin"); RSCALE = float(os.environ.get("RSCALE", 20000))
if GROUP > 1: CRITIC = False
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 0))   # >0 で形勢ポテンシャル(所持金の差の尺度)。0 で従来どおり終端のみ
OPPDIR = os.environ["OPPDIR"]
MKTNF = int(os.environ.get("MKTNF", "0") or 0)
FASTCOL = os.environ.get("FASTCOL", "0") == "1"   # exp280: 盤面の GPU 転送を1手1回にまとめる          # exp270: 市場の先読み特徴(0 で無効)
KEYS = ("tiles", "ou", "pu", "g", "lm") + (("mkt",) if MKTNF else ()) + ("ul", "uq", "ms", "mq")
OKEYS = ("otiles", "oou", "opu", "og")
EVSEEDS = [2_000_000 + 7919 * k for k in range(64)]      # 対 tape_ug のものさし(128局。64局では揺れが大きく舵取りを誤った — weval で確認)
# 09-21: 関門の局数を設定できるようにした(既定 32種×2席=64局で従来どおり)。
#   64局では勝率 0.50 と 0.55 を区別できない。実測: pf2/op1 は関門で何度も「昇格」したのに、
#   リーグ384局で測ると出発点と互角(0.479 / 0.495)だった。主系列の第112/114世代も同じ型。
#   GATEN=96 なら 192局になり、標準誤差が 0.0625 → 0.036 に下がる。
GATEN = int(os.environ.get("GATEN", "32")); GATE_BEST_MIN = float(os.environ.get("GATE_BEST_MIN", "0"))   # 09-22: 最良版そのものへの最低勝率(0 で従来どおり)
GATESEEDS = [5_000_000 + 7919 * k for k in range(GATEN)]    # 関門用(ものさしとは別)
# 09-21: 関門で過去の世代も相手にする体数(0 で従来どおり最良版だけ)。漂流を止めるため。
GATE_POOL = int(os.environ.get("GATE_POOL", "0"))
# 09-23(exp317): 関門に「外の流派への退行を止める」条件を足す。GATE_FIX に重みのパスを , 区切りで並べると、
#   その相手とも毎回戦い、**勝率が GATE_FIX_MIN を下回ったら昇格させない**。平均には入れない(平均に入れると
#   勝率 0.9 の相手が水増しになって関門が緩むため。g4a3_gen2 が 0.984 で本線の関門を緩めていたのと同じ罠)。
#   背景: 身内の1v1だけで世代を選ぶと、勾配の97%が自分の流派なので PFN 戦を削る世代を拾う。
#   実測 09-23: gen48→gen54 は身内 +0.047(z=+2.61)なのに pn_ft 相手は 0.902→0.867(z=-2.2)。
GATE_FIX = [x for x in os.environ.get("GATE_FIX", "").split(",") if x.strip()]
GATE_FIX_MIN = float(os.environ.get("GATE_FIX_MIN", "0"))
GATE_FIX_N = int(os.environ.get("GATE_FIX_N", "96"))
FIXSEEDS = [9_000_000 + 7919 * k for k in range(GATE_FIX_N)]   # 関門・ものさしとは別の種
MEMLOG = os.environ.get("MEMLOG") == "1"   # 09-21: どこで GPU の記憶が膨らむかを反復ごとに記録する(既定 off)


def _mem(where):
    if not MEMLOG: return ""
    import torch as _t
    return " | 記憶 %s 使用 %.1fGB ピーク %.1fGB 確保 %.1fGB" % (
        where, _t.cuda.memory_allocated() / 2**30, _t.cuda.max_memory_allocated() / 2**30,
        _t.cuda.memory_reserved() / 2**30)
# 09-21(ユーザー「バラバラのGPUでも学習できるか」): Local SGD。
#   クラスタ はノード丸ごとの確保が4日待ちで、密結合の DDP を広げられない。そこで
#   「1ノード4GPU の学習係を N 本立て、AVG_EVERY 反復ごとに共有ディスク経由で重みを平均する」。
#   通信は k 反復に1回・ファイル経由なのでノードが散らばっていても成立する。
#   AVGDIR に <タグ>_<反復>.pt を書き、AVG_N 本そろったら平均して読み直す。
#   **PPO の数式は変わる**(独立に進めた重みを平均する)のでリーグで判定すること。
AVGDIR = os.environ.get("AVGDIR", ""); AVG_EVERY = int(os.environ.get("AVG_EVERY", "0"))
AVG_N = int(os.environ.get("AVG_N", "0")); AVG_WAIT = int(os.environ.get("AVG_WAIT", "900"))


def h2h(a, b, seeds, greedy=True, draw_win=False, mig_a=True):
    """a 対 b を各シード両座席で。a から見た勝率と差。
       09-23: MASK_MIG のとき a(挑戦者 = 移行中のモデル)の席だけ新マスク＋罰。b は旧マスクのまま壊さない。"""
    mg = []
    for s in (0, 1):
        if MASK_MIG and mig_a and FASTPOOL_ON: sp_fast._MIG_SEAT[0] = s
        m = S.play(seeds, {s: S.PolicyOpp(a, greedy=greedy, cap=True), 1 - s: S.PolicyOpp(b, greedy=greedy, cap=True)})
        if MASK_MIG and mig_a and FASTPOOL_ON: sp_fast._MIG_SEAT[0] = None
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg)
    if draw_win: return float((mg >= 0).mean()), float(mg.mean())    # exp304: 引き分けも通す
    return float((mg > 0).mean() + 0.5 * (mg == 0).mean()), float(mg.mean())


YARD = os.environ.get("YARD", "")    # exp258: ものさしの相手。空なら tape_ug、重みのパスを渡すとその方策(例: v65)と 128局
_yard_net = [None]


def vs_tape(net, seeds=EVSEEDS):
    mg = []
    if YARD:
        if _yard_net[0] is None:
            cfg = json.load(open(YARD + ".cfg.json")) if os.path.exists(YARD + ".cfg.json") else {}
            y = M.Policy(**cfg).to(DEV); y.load_state_dict(torch.load(YARD, map_location=DEV)); y.eval(); _yard_net[0] = y
        opp = lambda: S.PolicyOpp(_yard_net[0], greedy=True, cap=True)
    else:
        opp = lambda: S.TapeRouterOpp(OPPDIR)
    for s in (0, 1):
        if not YARD and FASTPOOL_ON: sp_fast._PACE_SKIP[0] = 1 - s       # exp286: テープの席には安全装置をかけない
        if FASTPOOL_ON: sp_fast._LAND_SKIP[0] = 1 - s                     # exp290 修正: ものさしの相手には4区画目のルールをかけない
        if MASK_MIG and FASTPOOL_ON: sp_fast._MIG_SEAT[0] = s              # 09-23: ものさしも自分の席だけ新マスク＋罰
        m = S.play(seeds, {s: S.PolicyOpp(net, greedy=True, cap=True), 1 - s: opp()})
        if FASTPOOL_ON: sp_fast._PACE_SKIP[0] = None; sp_fast._LAND_SKIP[0] = None; sp_fast._MIG_SEAT[0] = None
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg)
    return float((mg > 0).mean() + 0.5 * (mg == 0).mean()), float(mg.mean())   # 引き分けは 0.5(同じ方策どうしだと大量に出る)


def main():
    tag, init = sys.argv[1], sys.argv[2]
    L = "/content/local_logs"; logf = open("%s/orbit_%s.log" % (L, tag), "a")
    def log(s):
        if RANK: return
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    cfg = json.load(open(init + ".cfg.json")) if os.path.exists(init + ".cfg.json") else {}
    if MKTNF:                                               # exp270: 市場の先読み特徴つきの方策(PolicyMkt)。pid は焼き込み済みの重みを使う
        import model_tf_mkt as MM
        cfg = {k: v for k, v in cfg.items() if k not in ("nf", "mkt", "ptr", "nid")}   # 構成ファイルの市場用の鍵は引数で渡すので外す
        def _ld(path):
            n = MM.PolicyMkt(nid=1, nf=MKTNF, **cfg).to(DEV)
            d = torch.load(path, map_location=DEV)
            if isinstance(d, dict) and "model" in d: d = d["model"]
            r = n.load_state_dict(d, strict=False)
            assert not [k for k in r.missing_keys if k != "pid.weight"], r.missing_keys
            return n
        mk = lambda: MM.PolicyMkt(nid=1, nf=MKTNF, **cfg).to(DEV)
        net = _ld(init); best = _ld(init); ref = _ld(init); ema = _ld(init) if EMA > 0 else None
        crit = M.load(CRITINIT, DEV)                        # 批評家は素の Policy のまま(市場特徴なし)
    else:
        mk = lambda: M.Policy(**cfg).to(DEV)                # 過去の版もすべて初期値と同じ構成で作る(ppo_sp の d2 構成の不具合を避ける)
        net = M.load(init, DEV); crit = M.load(CRITINIT, DEV); best = M.load(init, DEV); ref = M.load(init, DEV)
    if not MKTNF: ema = M.load(init, DEV) if EMA > 0 else None
    for p in list(best.parameters()) + list(ref.parameters()) + (list(ema.parameters()) if ema is not None else []): p.requires_grad_(False)
    if ema is not None: ema.eval()
    net.eval(); best.eval(); ref.eval()
    if CRIT_LAST:                                # exp298: compile より前に差し替える
        import crit_fast; crit_fast.install_last(crit); log("批評家の最後の層は大域トークンだけ計算する(CRIT_LAST=1)")
    if CRIT_EVERY > 1 or CRIT_FRAC < 1: log("批評家の更新: %dミニバッチに1回・行の %.2f(CRIT_EVERY/CRIT_FRAC)" % (CRIT_EVERY, CRIT_FRAC))
    if os.environ.get("COMPILE_UPD") == "1":     # exp280: 更新の計算を compile(値は同じ、最初の1回だけコンパイル時間がかかる)
        P.evaluate = torch.compile(P.evaluate, dynamic=False)
        crit.encode = torch.compile(crit.encode, dynamic=False)
        log("更新を torch.compile で実行する")

    hlh = None
    if CRIT_HL:
        dd = torch.load(os.environ["HLINIT"], map_location=DEV, weights_only=False); crit.load_state_dict(dd["crit"])
        dh = [m for m in crit.value.modules() if isinstance(m, torch.nn.Linear)][0].in_features
        hlh = torch.nn.Sequential(torch.nn.Linear(dh, dh), torch.nn.GELU(), torch.nn.Linear(dh, HL_NB)).to(DEV); hlh.load_state_dict(dd["head"])
        hl_c = torch.linspace(HL_LO, HL_HI, HL_NB, device=DEV); hl_sgn = torch.sign(hl_c); hl_w = (HL_HI - HL_LO) / (HL_NB - 1)
        hl_edges = torch.cat([hl_c - hl_w / 2, hl_c[-1:] + hl_w / 2]).unsqueeze(0)
        def hl_target(m):
            m = m.clamp(HL_LO, HL_HI).unsqueeze(1); sig = 0.75 * hl_w
            cdf = 0.5 * (1 + torch.erf((hl_edges - m) / (sig * 2 ** 0.5))); pr = cdf[:, 1:] - cdf[:, :-1]
            return pr / pr.sum(1, keepdim=True).clamp(min=1e-8)
        log("批評家: 反対称 + 差額の分布(HL-Gauss %dビン)、初期値 %s" % (HL_NB, os.environ["HLINIT"]))
    cparams = list(crit.parameters()) + (list(hlh.parameters()) if hlh is not None else [])
    _ovl_net = overlap_of(list(net.parameters()), "net"); _ovl_crit = overlap_of(cparams, "crit")
    if _ovl_net is not None: log("勾配の同期を逆伝播と重ねる(バケツ %d/%d 個、%.0fMB)" % (len(_ovl_net.buckets), len(_ovl_crit.buckets), BUCKET_MB))
    if WORLD > 1:
        _nb = sync_bench(sum(q.numel() for q in net.parameters()))
        if _nb: log("[NCCL の実測] %d GPU で %.1fMB の all_reduce が %.2fms(実効 %.0fGB/s)" % (WORLD, _nb[1], _nb[0], _nb[2]))
        if os.environ.get("COMPILE_UPD") != "1":
            log("!! COMPILE_UPD=1 が付いていない。1GPU の本線と 8GPU の走りは付けている(更新が 2〜3割 遅くなる)")
    # 09-23(MFU): ADAM_FUSED=1 で融合版の Adam(1本のカーネルで更新する)。更新の式は同じ(fp32 の丸め程度の差)。
    _akw = dict(fused=True) if os.environ.get("ADAM_FUSED") == "1" else {}
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5, **_akw); opt_c = torch.optim.Adam(cparams, lr=float(os.environ.get("CLR", 1e-4)), eps=1e-5, **_akw)   # exp228: CLR で批評家の学習率(既定 1e-4 は従来どおり)
    scaler = torch.amp.GradScaler("cuda"); scaler_c = torch.amp.GradScaler("cuda")
    split = json.load(open(os.environ.get("SPLITFILE", L + "/bc_d2.json"))); hold_e = split["hold"]
    ck = "%s/orbit_%s.pt" % (L, tag); start = 0; gen = 0; league = []; vw_until = VWARM; pfsp_wr = {}
    mig_it0_ck = None   # 09-23: マスク移行の日程の起点(途中保存にあれば引き継ぐ)
    if not os.path.exists(ck) and RESUME_FROM and os.path.exists(RESUME_FROM):
        c = torch.load(RESUME_FROM, map_location=DEV, weights_only=False)
        if _akw:                                        # 09-23: 上と同じ理由。読む前に旗を立てる
            for _g in c["opt"]["param_groups"]: _g["fused"] = True
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); best.load_state_dict(c["best"]); ref.load_state_dict(c["best"])
        start = c["it"] + 1; gen = c["gen"]; league = c["league"]; vw_until = start + 2
        log("%s の反復 %d から引き継ぎ(最良版は第%d世代、リーグ %d体)。評価役は小さいネットワークで作り直すので、最初の2反復は評価役だけ学習" % (RESUME_FROM, c["it"], gen, len(league)))
    elif os.path.exists(ck):
        c = torch.load(ck, map_location=DEV, weights_only=False)
        if _akw:   # 09-23: **読む前に**控え側の param_groups に旗を立てる。旗を後から立てると torch が
                   #   state["step"] を CPU に置いたままにして _fused_adam が「state_steps is on cpu」で落ちる
            for _sd in (c["opt"], c["opt_c"]):
                for _g in _sd["param_groups"]: _g["fused"] = True
        net.load_state_dict(c["net"]); crit.load_state_dict(c["crit"]); opt.load_state_dict(c["opt"]); opt_c.load_state_dict(c["opt_c"])
        if hlh is not None and "hlh" in c: hlh.load_state_dict(c["hlh"])
        best.load_state_dict(c["best"]); ref.load_state_dict(c["best"]); start = c["it"] + 1; gen = c["gen"]; league = c["league"]; vw_until = c.get("vw_until", VWARM); pfsp_wr = c.get("pfsp_wr", {})
        if ema is not None: ema.load_state_dict(c.get("ema", c["net"]))
        mig_it0_ck = c.get("mig_it0")
        log("再開 it%d(最良版は第%d世代、リーグ %d体)" % (start, gen, len(league)))
    # 09-22: 段階表。lad_d = {パス: (名前, Elo)} は固定の物差し、own_elo は関門を通った自分の世代(通ったときの値)
    lad_d = ladder_load(LADDER) if LADDER else {}; own_elo = {}; elo_me = None
    if LADDER:
        assert not EXTRA, "LADDER と LEAGUE_EXTRA は併用しない(段階表に入れる)"
        assert ROLE, "LADDER は非同期(ROLE=learner/actor)専用(同期の実行では自分のレーティングを更新しない)"
        if os.path.exists(ck) and ROLE != "actor":
            c_ = torch.load(ck, map_location="cpu", weights_only=False); own_elo = c_.get("own_elo", {}); elo_me = c_.get("elo_me"); del c_
        if elo_me is None:
            init_ = init
            elo_me = float(ELO0) if ELO0 else (lad_d[init_][1] if init_ in lad_d else float(np.median([v[1] for v in lad_d.values()])))
        miss_ = [v[0] for x, v in lad_d.items() if not os.path.exists(x)]
        log("段階表 %s: %d体(Elo %.0f〜%.0f)、自分の世代 %d体、自分の初期レーティング %.0f、抽選の重み %s、K=%g%s" % (
            os.path.basename(LADDER), len(lad_d), min(v[1] for v in lad_d.values()), max(v[1] for v in lad_d.values()), len(own_elo), elo_me, LADDER_W, ELO_K,
            (" / !! 重みが無い: %s" % ",".join(miss_)) if miss_ else ""))
    landr_t0 = int(LANDR_IT0) if LANDR_IT0 else None; landr_ok = 0; lc_last = 0.0   # 09-22: 区画報酬を下げ始めた反復
    ctl_mt = None; ctl_tok = None
    e2_t0 = None; e2_p0 = None; e2_ok = 0; E2F = None
    e4_t0 = None; e4_p0 = None; e4_ok = 0; E4F = None
    if E4 and os.path.exists(ck) and ROLE == "learner":
        c_ = torch.load(ck, map_location="cpu", weights_only=False); e4_t0 = c_.get("e4_t0"); e4_p0 = c_.get("e4_p0"); del c_
        if e4_t0 is not None: globals()["E4_P"] = e4_p0 * max(0.0, 1.0 - (start - e4_t0) / float(max(E4_ZERO, 1)))
    if E2 and os.path.exists(ck) and ROLE == "learner":
        c_ = torch.load(ck, map_location="cpu", weights_only=False); e2_t0 = c_.get("e2_t0"); e2_p0 = c_.get("e2_p0"); del c_
        if e2_t0 is not None:
            globals()["E2_P"] = e2_p0 * max(0.0, 1.0 - (start - e2_t0) / float(max(E2_ZERO, 1)))
            log("介入E2: 第%d反復から下げ始めている(割合 %.2f → 今 %.2f)" % (e2_t0, e2_p0, E2_P))
    if os.path.exists(ck) and ROLE == "learner":
        c_ = torch.load(ck, map_location="cpu", weights_only=False); ctl_tok = c_.get("ctl_tok"); del c_
    if LANDR_AUTO and landr_t0 is None and os.path.exists(ck) and ROLE != "actor":
        c_ = torch.load(ck, map_location="cpu", weights_only=False); landr_t0 = c_.get("landr_t0"); del c_
        if LANDR_T0_RESET and landr_t0 is not None:
            log("区画報酬: 途中保存では第%d反復から下げ始めていたが、LANDR_T0_RESET=1 なので取り消して全力に戻す" % landr_t0); landr_t0 = None
    if LANDR_AUTO or LANDR_IT0:
        log("区画報酬: LANDR %g、目標 2区画目 %.0f手・3区画目 %.0f手(%s)に %d反復続けて届いたら %d反復かけて 0 に下げる%s%s" % (
            LANDR, LANDR_T2, LANDR_T3, "中央値" if LANDR_MED else "平均", LANDR_HOLD, LANDR_ZERO,
            (" / 2区画目を %.0f手より1手早いごとに +%g" % (LANDR_P2, LANDR_B2)) if LANDR_B2 > 0 else "",
            (" / 第%d反復から下げ始めている" % landr_t0) if landr_t0 is not None else ""))
    if MASK_MIG and MIG_IT0 < 0: globals()["MIG_IT0"] = int(mig_it0_ck) if mig_it0_ck is not None else start
    if MASK_MIG: M.mig_warm(DEV); M.mig_set(mig_pen_at(start))   # 09-23: compile より前に罰の入れ物を作り、起動時のものさしも移行後の打ち方で測る
    if start == 0:
        w, m = vs_tape(net); log("it  -1 [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f (初期)" % (w, m))
    log("設定: NSELF %d NBEST %d NLEAGUE %d NTAPEUG %d / 関門 %d反復ごと・勝率 %.2f 以上で置き換え / LR %g BETA %g / GAMMA %g LAM %g / 形勢 SHAPE %g / 批評家 %s / GRPO GROUP %d REWARD %s" % (
        NSELF, NBEST, NLEAGUE, NTAPEUG, GATE, GATE_WR, LR, BETA, GAMMA, LAM, SHAPE, "あり" if CRITIC else "なし(相手の種類ごとの平均勝敗を基準線)", GROUP, REWARD))
    rng = random.Random(int(os.environ.get("RSEED", 11)) + 1000 * RANK)   # exp292: GPU ごとに別の局   # exp286: 分岐したランで対局の種を変える(既定は従来どおり 11)
    if MASK_MIG:   # 09-23: マスク移行
        assert FASTPOOL_ON, "MASK_MIG=1 は FASTPOOL=1(sp_fast.BatchSPPool)のときだけ対応している"
        assert MIG_PEN > 0 or MIG_ZERO <= 0, "MIG_PEN を較正して渡すこと(mig_calib.py。他のモデルの値を流用しない)"
        log("マスク移行: 新マスク版 %d%s / 旧マスク版 %d / 罰 初期 %.3f・%d反復保持・第%d反復で 0(第%d反復起点)。相手の席にはかけない" % (
            MIG_V, ("(mask2' 残す型 %s、エンジンには opt=%d)" % (",".join(sorted(sp_fast.MIG_KEEP)), sp_fast.MIG_VENG)) if MIG_V == 3 else "",
            sp_fast.MIG_BASE, MIG_PEN, MIG_HOLD, MIG_IT0 + MIG_ZERO, MIG_IT0))
    if ROLE:                                            # exp296: 非同期(対局係 actor / 学習係 learner)
        os.makedirs(ADIR + "/buf", exist_ok=True); os.makedirs(ADIR + "/pub", exist_ok=True)
        assert GROUP <= 1, "非同期版は GROUP に未対応"   # exp302: SHAPE は対応済み。09-21: CRIT_HL も対応(HL の頭を pub/ で配る)
    pub_ver = [-1, -1]
    def _publish(ver):                                  # 学習係: 最新の重みと状態を公開(書きかけを読まれないよう一時ファイル → rename)
        if RANK: return
        for nm_, obj_ in ([("net", net), ("crit", crit), ("best", best)] + ([("hlh", hlh)] if hlh is not None else [])):
            torch.save(obj_.state_dict(), ADIR + "/pub/%s.pt.tmp" % nm_); os.replace(ADIR + "/pub/%s.pt.tmp" % nm_, ADIR + "/pub/%s.pt" % nm_)
        json.dump(dict(version=ver, gen=gen, league=league, **({"elo": {**lad_d, **own_elo}, "elo_me": elo_me} if LADDER else {}), **({"e2p": E2_P} if E2 else {}), **({"e4p": E4_P} if E4 else {}), **({"lmix": LMIX_P} if LMIX else {}), **({"mig_it0": MIG_IT0} if MASK_MIG else {})),
                  open(ADIR + "/pub/state.json.tmp", "w")); os.replace(ADIR + "/pub/state.json.tmp", ADIR + "/pub/state.json")
    def _sync():                                        # 対局係: 公開された重みが新しければ読み込む
        while not os.path.exists(ADIR + "/pub/state.json"): time.sleep(10)
        for _try in range(20):
            try:
                st_ = json.load(open(ADIR + "/pub/state.json"))
                if st_["version"] != pub_ver[0]:
                    net.load_state_dict(torch.load(ADIR + "/pub/net.pt", map_location=DEV)); crit.load_state_dict(torch.load(ADIR + "/pub/crit.pt", map_location=DEV))
                    if hlh is not None: hlh.load_state_dict(torch.load(ADIR + "/pub/hlh.pt", map_location=DEV))   # 09-21: HL-Gauss の頭も配る
                    if st_["gen"] != pub_ver[1]:
                        best.load_state_dict(torch.load(ADIR + "/pub/best.pt", map_location=DEV)); ref.load_state_dict(best.state_dict())
                    pub_ver[0], pub_ver[1] = st_["version"], st_["gen"]
                return st_
            except Exception as e_:
                time.sleep(3)
        raise RuntimeError("公開された重みを読めない")
    if ROLE == "learner" and start == 0 or (ROLE == "learner" and not os.path.exists(ADIR + "/pub/state.json")): _publish(start)
    lmix_fixed = False
    for it in range(start, ITERS):
        if LMIX and ROLE == "learner" and not lmix_fixed:   # 09-22: 新マスクで打つ割合を段階的に上げる
            globals()["LMIX_P"] = LMIX_P0 + (LMIX_P1 - LMIX_P0) * min(1.0, it / float(max(LMIX_ANNEAL, 1)))
        if ROLE == "learner" and WORLD == 1:              # 09-22: 制御ファイル(走ったまま設定を変える)
            try:
                cf_ = ADIR + "/ctl.json"
                if os.path.exists(cf_) and os.path.getmtime(cf_) != ctl_mt:
                    ctl_mt = os.path.getmtime(cf_); cd_ = json.load(open(cf_)); ch_ = []
                    for k_, v_ in cd_.items():
                        if k_ in CTL_KEYS and globals().get(k_) != CTL_KEYS[k_](v_):
                            ch_.append("%s %s→%s" % (k_, globals().get(k_), CTL_KEYS[k_](v_))); globals()[k_] = CTL_KEYS[k_](v_)
                    if "LMIX_P" in cd_: lmix_fixed = True
                    if cd_.get("landr_reset") and cd_["landr_reset"] != ctl_tok:
                        ctl_tok = cd_["landr_reset"]; landr_t0 = None; landr_ok = 0; ch_.append("区画報酬の減衰を取り消して全力に戻す(%s)" % ctl_tok)
                    if ch_: log("it%4d [制御ファイル] %s" % (it, " / ".join(ch_)))
            except Exception as e_: log("制御ファイルを読めない: %s" % str(e_)[:80])
        if ROLE == "actor":
            st_ = _sync(); league = st_["league"]; gen = st_["gen"]; it = st_["version"]; start = 0
            if "elo" in st_: lad_all = st_["elo"]; elo_me = st_["elo_me"]   # 09-22: 段階表は学習係が配る(対局係に LADDER は要らない)
            else: lad_all = None
            if E4 and "e4p" in st_: globals()["E4_P"] = float(st_["e4p"])   # 09-22: 介入 E4 の割合も学習係が配る
            if LMIX and "lmix" in st_: globals()["LMIX_P"] = float(st_["lmix"])   # 09-22: 新マスクで打つ割合も学習係が配る
            if MASK_MIG and "mig_it0" in st_: globals()["MIG_IT0"] = int(st_["mig_it0"])   # 09-23: 罰の日程の起点は学習係が配る
            if E2 and "e2p" in st_: globals()["E2_P"] = float(st_["e2p"])   # 09-22: 介入 E2 の割合は学習係が配る(本体の局所変数にしないよう globals 経由)
            while len(glob.glob(ADIR + "/buf/*.pt")) >= BUFMAX: time.sleep(5)   # 学習係が追いつくまで待つ(バッファの上限)
        else: lad_all = {**lad_d, **own_elo} if LADDER else None
        t0 = time.time()
        _T_ = roll_t(it)       # 09-21: この反復のロールアウト温度(サンプリングと PER0 の両方に同じ値を使う)
        _BETA_ = beta_at(it)   # 09-21: この反復の BC への KL の手綱(BETA_ZERO>0 なら減衰)
        _MIGP_ = mig_pen_at(it)
        if MASK_MIG: M.mig_set(_MIGP_)   # 09-23: マスク移行の罰(採取・ものさし・関門はこの値。学習係は局に書かれた値で上書きする)
        if ROLE != "learner":
            games = []
            gid = 0
            lg_pool = [x for x in lad_all if os.path.exists(x)] if lad_all else (list(EXTRA) if LEAGUE_EXTERNAL_ONLY else EXTRA + league)   # (pool はゲームの束の名前なので別名に) 09-30: LEAGUE_EXTERNAL_ONLY=1 はチームメイトの設定(外部の相手だけ)
            for kind, n in (("self", NSELF), ("best", NBEST), ("league", NLEAGUE if lg_pool else 0), ("tape", NTAPEUG)):
                if GROUP > 1:
                    if kind == "self": continue                      # GRPO では自己対戦を使わない(両席とも自分で、組の比較が崩れる)
                    for _ in range(max(1, n // GROUP)):
                        seat, seed = rng.randrange(2), rng.randrange(10 ** 9)
                        for _k in range(GROUP): games.append(dict(kind=kind, seat=seat, seed=seed, gid=gid))
                        gid += 1
                else:
                    for _ in range(n): games.append(dict(kind=kind, seat=rng.randrange(2), seed=rng.randrange(10 ** 9), gid=-1))
            gN_ = 0; guide = None; gidx_ = []
            if GUIDE_P > 0:                                   # exp304: テープで序盤を打たせる局
                gfr_ = min(1.0, it / GUIDE_ANNEAL) if GUIDE_ANNEAL > 0 else 0.0
                gp_ = GUIDE_P + (GUIDE_P_END - GUIDE_P) * gfr_; gN_ = int(round(GUIDE_N + (GUIDE_N_END - GUIDE_N) * gfr_))
                for g in games: g["guide"] = gN_ > 0 and rng.random() < gp_
                gidx_ = [i for i, g in enumerate(games) if g["guide"]]
                if gidx_: guide = sp_fast.FastTapeRouterOpp(GUIDE_DIR)
            pool = S.SPPool([g["seed"] for g in games]); N = len(games)
            if getattr(pool, "pace_ok", None) is not None and sp_fast.PACE:   # exp286: どの局・席に安全装置をかけるか
                pp = PACE_P0 + (PACE_P1 - PACE_P0) * (min(1.0, (it - start) / PACE_ANNEAL) if PACE_ANNEAL > 0 else 1.0)
                for i, g in enumerate(games):
                    ls = g["seat"]
                    if g["kind"] == "self": pool.pace_ok[i, :] = [rng.random() < pp, rng.random() < pp]     # 両席とも学習中の自分
                    else:
                        pool.pace_ok[i, ls] = rng.random() < pp
                        pool.pace_ok[i, 1 - ls] = g["kind"] != "tape"      # 凍った自分の世代は出荷時と同じく装置つき、テープにはかけない
                pace_now = pp
            if getattr(pool, "land_ok", None) is not None and sp_fast.LAND:   # exp290
                lp = LAND_P0 + (LAND_P1 - LAND_P0) * (min(1.0, (it - start) / LAND_ANNEAL) if LAND_ANNEAL > 0 else 1.0)
                for i, g in enumerate(games):
                    ls = g["seat"]
                    if g["kind"] == "self": pool.land_ok[i, :] = [rng.random() < lp, rng.random() < lp]
                    else:
                        pool.land_ok[i, ls] = rng.random() < lp
                        pool.land_ok[i, 1 - ls] = LAND_OPP and g["kind"] != "tape"   # exp290 修正: 既定では相手(過去の世代・リーグ)にかけない
            if getattr(pool, "nl4_ok", None) is not None and getattr(sp_fast, "NOLAND4", False):   # exp318: 4区画目を買わせない
                np_ = NOLAND4_P0 + (NOLAND4_P1 - NOLAND4_P0) * (min(1.0, (it - start) / NOLAND4_ANNEAL) if NOLAND4_ANNEAL > 0 else 1.0)
                for i, g in enumerate(games):
                    ls = g["seat"]
                    if g["kind"] == "self": pool.nl4_ok[i, :] = [rng.random() < np_, rng.random() < np_]
                    else:
                        pool.nl4_ok[i, ls] = rng.random() < np_
                        pool.nl4_ok[i, 1 - ls] = NOLAND4_OPP and g["kind"] != "tape"   # 既定では相手にはかけない(4区画の相手と戦いながら3区画で仕上げる)
            if getattr(pool, "pfnw_ok", None) is not None and getattr(sp_fast, "PFNW", False):   # exp314
                fp_ = PFNW_P0 + (PFNW_P1 - PFNW_P0) * (min(1.0, (it - start) / PFNW_ANNEAL) if PFNW_ANNEAL > 0 else 1.0)
                for i, g in enumerate(games):
                    ls = g["seat"]
                    if g["kind"] == "self": pool.pfnw_ok[i, :] = [rng.random() < fp_, rng.random() < fp_]
                    else:
                        pool.pfnw_ok[i, ls] = rng.random() < fp_
                        pool.pfnw_ok[i, 1 - ls] = PFNW_OPP and g["kind"] != "tape"
            if getattr(pool, "open_ok", None) is not None and sp_fast.OPEN:   # exp305: 序盤のガードレール
                op_ = OPEN_P0 + (OPEN_P1 - OPEN_P0) * (min(1.0, (it - start) / OPEN_ANNEAL) if OPEN_ANNEAL > 0 else 1.0)
                for i, g in enumerate(games):
                    ls = g["seat"]
                    if g["kind"] == "self": pool.open_ok[i, :] = [rng.random() < op_, rng.random() < op_]
                    else:
                        pool.open_ok[i, ls] = rng.random() < op_
                        pool.open_ok[i, 1 - ls] = OPEN_OPP and g["kind"] != "tape"   # 既定では相手(過去の世代・リーグ・テープ)にはかけない
            lg_net = None; lg_path = None; lad_p = None
            if lg_pool:
                if lad_all:                                   # 09-22: 段階表。予想勝率から出した重みで抽選
                    wl_, pl_ = ladder_weights({x: lad_all[x] for x in lg_pool}, float(elo_me))
                    lg_path = rng.choices(lg_pool, weights=[wl_[x] for x in lg_pool])[0]; lad_p = pl_[lg_path]
                elif PFSP:
                    wts = [0.15 + (1 - pfsp_wr.get(x, 0.5)) ** 2 for x in lg_pool]; lg_path = rng.choices(lg_pool, weights=wts)[0]
                else: lg_path = rng.choice(lg_pool)
                if lad_p is not None: lg_net = M.load(lg_path, DEV).eval()   # 09-22: 段階表の相手は自分の cfg.json で作る(構造が違っても読める)
                else: lg_net = mk(); lg_net.load_state_dict(torch.load(lg_path, map_location=DEV)); lg_net.eval()
                if lad_p is not None and len(lad_all[lg_path]) > 2 and int(lad_all[lg_path][2]) >= 0:   # 09-22: 旧マスクで学習した相手は旧マスクで打たせる
                    pool.legal_opt = np.full((N, 2), -1, np.int32)
                    for i, g in enumerate(games):
                        if g["kind"] == "league": pool.legal_opt[i, 1 - g["seat"]] = int(lad_all[lg_path][2])
            tape = sp_fast.FastTapeRouterOpp(OPPDIR)      # TapeRouterOpp と719手すべて同じ手(check_tape で確認)
            allt = [(i, g["seat"]) for i, g in enumerate(games)] + [(i, 1 - g["seat"]) for i, g in enumerate(games) if g["kind"] == "self"]
            i0 = np.array([i for i, s in allt if s == 0], int); i1 = np.array([i for i, s in allt if s == 1], int)
            trajs = [(i, 0) for i in i0] + [(i, 1) for i in i1]; n0 = len(i0)
            e2i_ = np.array([i for i, _ in trajs], int); e2s_ = np.array([s_ for _, s_ in trajs], int); e2n_ = 0
            e2f_ = np.array([rng.random() < E2_P for _ in trajs], bool) if E2 else np.zeros(len(trajs), bool)
            e4f_ = np.array([rng.random() < E4_P for _ in trajs], bool) if E4 else np.zeros(len(trajs), bool); e4n_ = 0
            e5p_ = E5_P * (0.0 if (E5_ZERO > 0 and it - start >= E5_ZERO) else (1.0 - (it - start) / E5_ZERO if E5_ZERO > 0 else 1.0))
            e5f_ = np.array([rng.random() < e5p_ for _ in trajs], bool) if E5 else np.zeros(len(trajs), bool); e5n_ = 0
            e5tot_ = np.zeros(len(trajs), np.float64)      # exp320: 軌道ごとに出した家畜の注文の数(書き換え後の実物)
            e6f_ = np.array([rng.random() < E6_P for _ in trajs], bool) if E6 else np.zeros(len(trajs), bool); e6n_ = 0
            e6w_ = np.zeros(len(trajs), np.int32); e6h_ = np.zeros(len(trajs), np.int32)   # その日の通過件数(小麦・雇用)
            lmf_ = np.array([rng.random() < LMIX_P for _ in trajs], bool) if LMIX else np.zeros(len(trajs), bool)
            if LMIX and lmf_.any():                        # 09-22: 新マスクで打つ軌道(席ごと)
                if pool.legal_opt is None: pool.legal_opt = np.full((N, 2), -1, np.int32)
                for j_ in np.nonzero(lmf_)[0]: pool.legal_opt[trajs[j_][0], trajs[j_][1]] = LMIX_V
            if MASK_MIG:                                   # 09-23: マスク移行。学習側の軌道(自分の席)だけ新マスク＋罰の印
                if pool.legal_opt is None: pool.legal_opt = np.full((N, 2), -1, np.int32)
                pool.mig_ok[:] = False
                for i_, s_ in trajs: pool.mig_ok[i_, s_] = True; pool.legal_opt[i_, s_] = sp_fast.MIG_VENG
            gi0 = torch.as_tensor(i0, device=DEV); gi1 = torch.as_tensor(i1, device=DEV)      # 座席0の軌道を先に並べる(一括で取り出すため)
            buf = {k: [] for k in KEYS + ("v",) + (OKEYS if ANTI else ())}; tm = dict(env=0.0, samp=0.0, opp=0.0, step=0.0); mny = []; qd = []
            _PROF = os.environ.get("PROFILE") == "1" and it == max(start, 1)     # exp280: 実際の反復の中でどこが遅いかを測る
            _CPROF = os.environ.get("PROFILE") == "2" and it == max(start, 1)    # exp280: Python 関数ごと(torch 以外の CPU 処理を含む)
            if _CPROF:
                import cProfile, pstats, io
                _cp = cProfile.Profile(); _cp.enable(); _ct0 = time.time()
            if _PROF:
                from torch.profiler import profile, ProfilerActivity
                _pr = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); _pr.__enter__(); _pt0 = time.time()
            for t in range(719):
                if _CPROF and t == 60:
                    torch.cuda.synchronize(); _cp.disable()
                    log("[計測: Python 関数] 採取 60手 %.1f 秒" % (time.time() - _ct0))
                    _sio = io.StringIO(); pstats.Stats(_cp, stream=_sio).sort_stats("tottime").print_stats(30); log(_sio.getvalue())
                    _sio = io.StringIO(); pstats.Stats(_cp, stream=_sio).sort_stats("cumulative").print_stats(30); log(_sio.getvalue())
                    raise SystemExit(0)
                if _PROF and t == 60:
                    torch.cuda.synchronize(); _pr.__exit__(None, None, None)
                    log("[計測] 採取 60手 %.1f 秒" % (time.time() - _pt0))
                    log(_pr.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
                    log(_pr.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
                    _pr.export_chrome_trace("/content/local_logs/prof_%s_collect.json" % tag)
                    raise SystemExit(0)
                a_ = time.time(); st = {s: pool.state(s) for s in (0, 1)}; tm["env"] += time.time() - a_
                U = {s: np.zeros((N, S.NU, 3), np.int32) for s in (0, 1)}; O = {s: np.zeros((N, 10, 3), np.int32) for s in (0, 1)}
                a_ = time.time()
                amp = torch.autocast("cuda", dtype=torch.float16) if AMP_SAMPLE else contextlib.nullcontext()
                amp.__enter__()
                _NK = 5 + (1 if MKTNF else 0)
                if FASTCOL:                    # exp280: 盤面を1手に1回だけ GPU へ送り、以降は GPU 上で切り出す
                    _dt = lambda k: (torch.float16 if k == 0 else (torch.uint8 if k == 4 else torch.float32))
                    stg = {s_: [torch.as_tensor(st[s_][k], device=DEV, dtype=_dt(k)) for k in range(_NK)] for s_ in (0, 1)}
                    stL = tuple(torch.cat([stg[0][k][gi0], stg[1][k][gi1]]) for k in range(_NK))
                    ou_np = np.concatenate([st[0][1][i0], st[1][1][i1]])
                    if os.environ.get("FASTCOL_CHECK") == "1" and t < 3:      # exp280: GPU 上の切り出しが numpy 版と同じか確かめる
                        for k in range(_NK):
                            ref_ = np.concatenate([st[0][k][i0], st[1][k][i1]])
                            got_ = stL[k].cpu().numpy()
                            assert got_.shape == ref_.shape, (k, got_.shape, ref_.shape)
                            assert np.array_equal(got_.astype(ref_.dtype), ref_), ("stL の不一致", k, float(np.abs(got_.astype(np.float64) - ref_.astype(np.float64)).max()))
                        refO_ = [np.concatenate([st[1][k][i0], st[0][k][i1]]) for k in range(4)]
                        log("[検証] 手%d: GPU で切り出した盤面が numpy 版と完全一致(%d 種)" % (t, _NK))
                else:
                    stL = tuple(np.concatenate([st[0][k][i0], st[1][k][i1]]) for k in range(_NK))
                    ou_np = stL[1]
                _ch = int(os.environ.get("SAMPCHUNK", "256"))    # exp280: 学習側の推論をまとめる数
                _fastcrit = P.FASTINF and ANTI and CRITIC and not CRIT_HL    # exp280: 批評家は下で自分と相手をまとめて1回
                outs = [P.sample(net, tuple(x[j:j + _ch] for x in stL), crit=None if _fastcrit else (crit if CRITIC else None), temp=_T_) for j in range(0, len(trajs), _ch)]
                ul, uq, ms, mq, v = [np.concatenate([o[k] for o in outs]) for k in range(5)]
                if E4 and t >= E4_TMIN and e4f_.any():          # 09-22: 介入 E4(区画ごとの開始手番 E4_T2/T3/T0 を過ぎた軌道)
                    nq4_ = pool_quads(pool)[e2i_, e2s_]
                    ok4_ = np.zeros(len(nq4_), bool)
                    for k_, t_k in E4_TS.items():
                        if t_k >= 0 and t >= t_k: ok4_ |= (nq4_ == k_)
                    if E4_AFF:                                  # 買えるだけの所持金があるときだけ(観測 g の 32 列目 × 50000 = 所持金)
                        mon4_ = np.array([st[s_][3][i_, 32] for i_, s_ in trajs], np.float64) * 50000.0
                        pr4_ = np.array([E4_PRICE.get(int(k_), 1e18) for k_ in nq4_])
                        ok4_ &= mon4_ >= pr4_
                    sel4_ = np.nonzero(e4f_ & ok4_)[0]
                    for j_ in sel4_:
                        ms[j_], mq[j_], ch_ = e4_fix(ms[j_], mq[j_]); e4n_ += int(ch_)
                if E2 and E2_T0 <= t < E2_T1 and e2f_.any():   # 09-22: 介入 E2(区画が1つの軌道だけ)
                    nq_ = pool_quads(pool); sel_ = np.nonzero(e2f_ & (nq_[e2i_, e2s_] == 1))[0]
                    for j_ in sel_:
                        ms[j_], mq[j_], ch_ = e2_fix(ms[j_], mq[j_]); e2n_ += int(ch_)
                if E5 and E5_T0 <= t < E5_T1 and (t % max(1, E5_EVERY) == 0) and e5f_.any():   # exp320: 介入E5(家畜の買い増し)
                    sel5_ = np.nonzero(e5f_)[0]
                    for j_ in sel5_:
                        ms[j_], mq[j_], ch_ = e5_fix(ms[j_], mq[j_]); e5n_ += int(ch_)
                if E5:
                    try: e5tot_ += (np.asarray(ms) == E5_CLASS).sum(1)
                    except Exception: pass
                if E6 and e6f_.any():                          # exp322: 支出を絞る
                    if t % 24 == 0: e6w_[:] = 0; e6h_[:] = 0   # 日が変わったら数え直す
                    nq6_ = pool_quads(pool) if E6_LAND4 else None
                    for j_ in np.nonzero(e6f_)[0]:
                        nq_j = int(nq6_[trajs[j_][0], trajs[j_][1]]) if nq6_ is not None else 0
                        ms[j_], mq[j_], ch_, e6w_[j_], e6h_[j_] = e6_fix(ms[j_], mq[j_], t, int(e6w_[j_]), int(e6h_[j_]), nq_j)
                        e6n_ += int(ch_)
                UL, OL = P.to_codes(ul, uq, ms, mq, ou_np)
                U[0][i0], O[0][i0] = UL[:n0], OL[:n0]; U[1][i1], O[1][i1] = UL[n0:], OL[n0:]
                for k, a in zip(KEYS, list(stL) + [ul, uq, ms, mq]): buf[k].append(a)
                if ANTI and CRITIC:                            # 同じ局・同じ時刻の相手の席の観測で、反対称の価値
                    stO = tuple(torch.cat([stg[1][k][gi0], stg[0][k][gi1]]) for k in range(4)) if FASTCOL else tuple(np.concatenate([st[1][k][i0], st[0][k][i1]]) for k in range(4))
                    tq = lambda X, j: [torch.as_tensor(x[j:j + 256], device=DEV, dtype=torch.float16 if q == 0 else torch.float32) for q, x in enumerate(X)]
                    with torch.no_grad():
                        if CRIT_HL:
                            vv = []
                            for j in range(0, len(trajs), 256):
                                lg = (hlh(crit.encode(*tq(stL[:4], j))[0][:, -1]) + hlh(crit.encode(*tq(stO, j))[0][:, -1]).flip(-1)).float()
                                vv.append((lg.softmax(-1) * hl_sgn).sum(-1).cpu().numpy())
                            v = np.concatenate(vv)
                        elif _fastcrit:
                            vv = []
                            for j in range(0, len(trajs), _ch):
                                own_ = [torch.as_tensor(x[j:j + _ch], device=DEV, dtype=torch.float16 if q == 0 else torch.float32) for q, x in enumerate(stL[:4])]
                                opp_ = [torch.as_tensor(x[j:j + _ch], device=DEV, dtype=torch.float16 if q == 0 else torch.float32) for q, x in enumerate(stO)]
                                hh = crit.encode(*[torch.cat([own_[q], opp_[q]], 0) for q in range(4)])[0]
                                vr = crit.value(hh[:, -1]).squeeze(-1).float(); n_ = len(own_[0])
                                vv.append((0.5 * (vr[:n_] - vr[n_:])).cpu().numpy())
                            v = np.concatenate(vv)
                        else:
                            vo = np.concatenate([crit.value(crit.encode(*tq(stO, j))[0][:, -1]).squeeze(-1).float().cpu().numpy() for j in range(0, len(trajs), 256)])
                            v = 0.5 * (v - vo)
                    for k, a in zip(OKEYS, stO): buf[k].append(a)
                buf["v"].append(v); tm["samp"] += time.time() - a_; a_ = time.time()
                for kind, obj in (("best", S.PolicyOpp(best, greedy=GROUP > 1, cap=True)), ("league", S.PolicyOpp(lg_net, greedy=GROUP > 1 or LEAGUE_GREEDY, cap=True) if lg_net else None), ("tape", tape)):
                    ids = [i for i, g in enumerate(games) if g["kind"] == kind]
                    if not ids or obj is None: continue
                    for s in (0, 1):
                        sub = [i for i in ids if 1 - games[i]["seat"] == s]
                        if not sub: continue
                        if kind == "tape":
                            u, o = obj.act(tuple(x[sub] for x in st[s]), t, sub, pool, s)
                        elif FASTCOL:
                            gsub = torch.as_tensor(sub, device=DEV)
                            u, o = obj.act(tuple(x[gsub] for x in stg[s]), t, None, pool, s, ou_np=st[s][1][sub])
                        else:
                            u, o = obj.act(tuple(x[sub] for x in st[s]), t, None, pool, s)
                        U[s][sub], O[s][sub] = u, o
                amp.__exit__(None, None, None)
                if guide is not None and t < gN_:              # exp304: 学習側の席の手をテープで上書き(方策の出した手は捨てる)
                    for s in (0, 1):
                        sub = [i for i in gidx_ if games[i]["seat"] == s]
                        if sub: U[s][sub], O[s][sub] = guide.act(tuple(x[sub] for x in st[s]), t, sub, pool, s)
                tm["opp"] += time.time() - a_; a_ = time.time()
                if MNY_REC: mny.append(pool_assets(pool, t) if SHAPE_ASSET else pool.money().copy())   # exp302: 形勢ポテンシャル用。手を打つ前の所持金(= V と同じ局面 s_t)
                if QUAD_REC: qd.append(pool_quads(pool))          # exp305: 区画のポテンシャル用。手を打つ前の区画数
                pool.step(U[0], O[0], U[1], O[1]); tm["step"] += time.time() - a_
            fin = pool.money(); tr = time.time() - t0
            LIVE_ = None
            if SHOPW > 0:                       # exp321: 局ごとの家畜店の数(両席で共通 = 町の情報)
                try:
                    sh_ = np.asarray(pool.shops())
                    LIVE_ = np.array([sum(1 for v in r if int(v) in _LIVE_SHOP and int(v) >= 0) for r in sh_], np.float32)
                except Exception as e_: log("家畜店の数を取れない(%s)" % str(e_)[:50]); LIVE_ = None
            fina = pool_assets(pool) if WIN_RAMP > 0 else None     # exp304: 最終の資産(絶対座席)
            GFL = np.array([gN_ if games[i].get("guide") and s == games[i]["seat"] else 0 for i, s in trajs], np.int32)   # exp304: 軌道ごとの打たせた手数
            GROW = np.array([fina[i, s] for i, s in trajs], np.float64) if fina is not None else None
            # 報酬は最後の手の勝ち負けだけ(+1 / −1、引き分け 0)
            z = np.array([np.sign(fin[i, s] - fin[i, 1 - s]) for i, s in trajs])
            V = np.stack(buf["v"]); T_ = V.shape[0]; Nn = V.shape[1]
            if MNY_REC:                                    # exp302: 軌道ごとの (手, 軌道, [自分, 相手]) の所持金
                MNY = np.stack(mny); ii_ = np.array([i for i, _ in trajs], int); ss_ = np.array([s for _, s in trajs], int)
                MO = np.stack([MNY[:, ii_, ss_], MNY[:, ii_, 1 - ss_]], -1).astype(np.float32); del MNY
            if QUAD_REC:                                   # exp305: 軌道ごとの (手, 軌道, [自分, 相手]) の区画数
                QDD = np.stack(qd); ii2_ = np.array([i for i, _ in trajs], int); ss2_ = np.array([s for _, s in trajs], int)
                QD = np.stack([QDD[:, ii2_, ss2_], QDD[:, ii2_, 1 - ss2_]], -1).astype(np.float32); del QDD
            if not CRITIC:
                kinds = np.array([games[i]["kind"] + ("_g" if GFL[j] > 0 else "") for j, (i, _) in enumerate(trajs)]); bl = np.zeros(Nn, np.float32)
                zt_ = term_reward(z, GROW, it)                  # exp304: 既定では z のまま(誘導なしなら種類の名前も同じ)
                for k in set(kinds): bl[kinds == k] = zt_[kinds == k].mean()     # 相手の種類ごとの平均勝敗
                V = np.broadcast_to(bl.reshape(1, -1), V.shape).astype(np.float32).copy()
        else:                                           # exp296: 学習係は対局せず、対局係の局を読む
            t0 = time.time()
            if RANK == 0:
                while True:
                    fs_ = sorted(glob.glob(ADIR + "/buf/*.pt"), key=os.path.getmtime); ok_ = []
                    for f_ in fs_:
                        try: v_ = int(os.path.basename(f_).split("_v")[1].split("_")[0])
                        except Exception: continue
                        if v_ < it - MAXLAG: os.remove(f_); continue          # 古すぎる局は捨てる
                        ok_.append(f_)
                    if len(ok_) >= WORLD * KSHARD: break
                    time.sleep(3)
                pick_ = [ok_[:WORLD * KSHARD]]
            else: pick_ = [None]
            if WORLD > 1: tdist.broadcast_object_list(pick_, 0)
            mine_ = pick_[0][RANK::WORLD]
            # 09-21: ディスクが一杯のときに書かれた壊れた局は torch.load が落ちる。1個の破損でランごと
            #   死んでいたので、読めない局は捨てて次の反復に回す(全 GPU が同数を持つよう、足りない側に合わせる)。
            SH_ = []; bad_ = []
            for f_ in mine_:
                try: SH_.append(torch.load(f_, map_location="cpu", weights_only=False))
                except Exception as e_: bad_.append(f_); log("壊れた局を捨てる: %s (%s)" % (os.path.basename(f_), str(e_)[:60]))
            for f_ in bad_:
                try: os.remove(f_)
                except OSError: pass
            nkeep_ = [len(SH_)]
            if WORLD > 1:
                t_ = torch.tensor([len(SH_)], device=DEV); tdist.all_reduce(t_, op=tdist.ReduceOp.MIN); nkeep_ = [int(t_.item())]
            if nkeep_[0] == 0:
                log("読める局が無いので、この反復は飛ばす"); time.sleep(20); continue
            SH_ = SH_[:nkeep_[0]]
            if LADDER:                                     # 09-22: 段階表。リーグの局(誘導なし)の結果で自分のレーティングを Elo 更新
                lad_all_ = {**lad_d, **own_elo}; d_ = 0.0; nl_ = 0; tal_ = {}
                for s_ in SH_:
                    op_ = s_.get("opp")
                    if op_ is None or op_ not in lad_all_: continue
                    E_ = 1.0 / (1.0 + 10 ** ((lad_all_[op_][1] - elo_me) / 400.0)); t_ = tal_.setdefault(lad_all_[op_][0], [0.0, 0, E_])
                    for k_, f_ in zip(s_["kinds"], s_["fin"]):
                        if k_ != "league": continue
                        sc_ = 1.0 if f_[0] > f_[1] else (0.5 if f_[0] == f_[1] else 0.0); d_ += sc_ - E_; nl_ += 1; t_[0] += sc_; t_[1] += 1
                if WORLD > 1:
                    t_ = torch.tensor([d_, float(nl_)], device=DEV, dtype=torch.float64); tdist.all_reduce(t_); d_, nl_ = float(t_[0]), int(t_[1])
                k_eff_ = min(ELO_K, 347.4 / max(nl_, 1))           # 1反復で読む局が多いと一括更新が振動・発散するので、利得を 0.5 以下に抑える
                elo_prev_ = elo_me; elo_me = elo_me + k_eff_ * d_
                if RANK == 0 and nl_:
                    log("it%4d [段階表] 自分のレーティング %.0f → %.0f(%d局)| %s" % (it, elo_prev_, elo_me, nl_,
                        " / ".join("%s %.2f(%d、予想 %.2f)" % (k, v[0] / max(v[1], 1), v[1], v[2]) for k, v in sorted(tal_.items(), key=lambda kv: -kv[1][2]))))
            T_ = SH_[0]["T"]; Ns_ = [s_["N"] for s_ in SH_]; Nn = sum(Ns_)
            def _cat(key_, src_):
                xs_ = [s_[src_][key_].view(T_, s_["N"], *s_[src_][key_].shape[1:]) for s_ in SH_]
                return torch.cat(xs_, 1).reshape(T_ * Nn, *xs_[0].shape[2:]).to(DEV)
            D_ld = {k: _cat(k, "D") for k in SH_[0]["D"]}; DO_ld = {k: _cat(k, "DO") for k in SH_[0]["DO"]} if ANTI else None
            PER0_ld = torch.cat([s_["PER0"].view(T_, s_["N"], *s_["PER0"].shape[1:]) for s_ in SH_], 1).reshape(T_ * Nn, *SH_[0]["PER0"].shape[1:]).to(DEV).float()
            V = np.concatenate([s_["V"] for s_ in SH_], 1)
            kinds_ = sum([s_["kinds"] for s_ in SH_], []); fin = np.concatenate([s_["fin"] for s_ in SH_], 0)
            games = [dict(kind=k_) for k_ in kinds_]; trajs = [(j_, 0) for j_ in range(Nn)]
            z = np.sign(fin[:, 0] - fin[:, 1]); lg_path = None
            GFL = np.concatenate([s_["gfl"] if "gfl" in s_ else np.zeros(s_["N"], np.int32) for s_ in SH_])   # exp304
            LIVE_ = np.concatenate([s_["live"] for s_ in SH_]) if all("live" in s_ for s_ in SH_) else None   # exp321
            GROW = np.concatenate([s_["grow"] for s_ in SH_]) if WIN_RAMP > 0 else None
            if SHAPE_ON:                                   # exp302
                assert all("mny" in s_ for s_ in SH_), "SHAPE/SHAPE_OWN>0 には局集め係も SHAPE/SHAPE_OWN>0 で動かす(局に所持金の列が無い)"
            if MNY_REC and all("mny" in s_ for s_ in SH_): MO = np.concatenate([s_["mny"] for s_ in SH_], 1)
            if QUAD_ON:                                    # exp305
                assert all("quad" in s_ for s_ in SH_), "QUAD_A≠0 には局集め係も QUAD_A(か QUAD_REC=1)で動かす(局に区画数の列が無い)"
            if AUX_ON:                                     # exp308
                assert all("quad" in s_ and "mny" in s_ for s_ in SH_), "AUX_LAND>0 には局集め係も AUX_LAND>0 で動かす(局に区画数か所持金の列が無い)"
            E2F = np.concatenate([np.asarray(s_["e2f"], bool) if "e2f" in s_ else np.zeros(s_["N"], bool) for s_ in SH_]) if E2 else None   # 09-22
            E4F = np.concatenate([np.asarray(s_["e4f"], bool) if "e4f" in s_ else np.zeros(s_["N"], bool) for s_ in SH_]) if E4 else None   # 09-22
            QD = None   # 09-22: 対局係に QUAD_REC が渡っていないと局に "quad" が入らない。
                        #   学習係だけ有効にすると QD が未定義のままログ行で落ちていた(pfn4 が死んだ)。
            if QUAD_REC and all("quad" in s_ for s_ in SH_): QD = np.concatenate([s_["quad"] for s_ in SH_], 1)
            elif QUAD_REC: log("!! 局に区画の記録が無い。対局係にも QUAD_REC=1 を渡すこと(今回は区画のログを出さない)")
            lag_ = [it - s_["ver"] for s_ in SH_]
            if MASK_MIG:   # 09-23: **その局を集めたときの罰**で学習する(lag があっても採取と学習がずれない)
                if all("mig" in s_ for s_ in SH_):
                    mp_ = [float(s_["mig"]) for s_ in SH_]; M.mig_set(float(np.mean(mp_)))
                    if max(mp_) - min(mp_) > 1e-9: log("it%4d [マスク移行] 読んだ局の罰がばらついている %.3f〜%.3f(平均 %.3f で学習)" % (it, min(mp_), max(mp_), float(np.mean(mp_))))
                else:
                    log("!! 局に移行の罰が入っていない(対局係にも MASK_MIG=1 を渡すこと)。日程の値 %.3f を使う" % _MIGP_); M.mig_set(_MIGP_)
            tr = time.time() - t0; tm = dict(env=0.0, samp=0.0, opp=0.0, step=float(np.mean(lag_)))
            used_ = mine_
        Rw = np.zeros_like(V); Rw[-1] = term_reward(z, GROW, it)
        own_c = OWN_R0 * (max(0.0, 1.0 - (it - (int(OWN_IT0) if OWN_IT0 else start)) / OWN_ANNEAL) if OWN_ANNEAL > 0 else 1.0)
        if own_c > 0:   # exp294: 一から学ぶとき用。自分の最終の所持金(対数)を終端に足す(勝敗だけでは乱数どうしが 0 対 0 の引き分けで手がかりが無い)
            own = np.array([np.log1p(max(fin[i, s_], 0.0)) / np.log1p(1e5) for i, s_ in trajs], np.float32)
            Rw[-1] = Rw[-1] + own_c * own
        auxc_ = 0.0; AUXT = None
        if AUX_ON:   # exp308: 「買える状態」の行に旗を立てる(区画数・所持金・手数の3条件。収集済みの列だけで作れる)
            auxc_ = aux_c(it, start)
            if auxc_ > 0:
                tt_ = np.arange(QD.shape[0])[:, None]
                m_ = (QD[..., 0] == AUX_LAND_Q) & (MO[..., 0] >= AUX_LAND_MONEY) & (tt_ >= AUX_LAND_T0) & (tt_ <= AUX_LAND_T1)
                AUXT = torch.as_tensor(m_.reshape(-1), device=DEV)
        shc_ = 0.0; quc_ = 0.0; Phi_ = None
        if SHAPE_ON:
            # exp209: ポテンシャルに基づく報酬の付け替え(Ng+ 1999)。F_t = γΦ(s_{t+1}) − Φ(s_t)、終端では Φ=0 とする。
            #   γ=1 なので足し合わせると z − Φ(s_0) になり、定数ぶんずれるだけで**最適な方策は変わらない**。
            #   所持金の「差」をそのまま使うと序盤の投資を罰するので、尺度に依らない対数比を終盤ほど重く(SHAPE_TP)。
            #   exp302: 自分の所持金の項(SHAPE_OWN)と、係数を 0 に下げる SHAPE_ANNEAL を足し、非同期でも使えるように(shape_phi)。
            shc_ = shape_c(it); Phi_ = shape_phi(MO[..., 0], MO[..., 1], shc_)
        if QUAD_ON:                                        # exp305: 区画のポテンシャル(同じ付け替えに足す)
            quc_ = quad_c(it, start); qphi_ = quad_phi(QD[..., 0], QD[..., 1], quc_)
            Phi_ = qphi_ if Phi_ is None else Phi_ + qphi_
        if Phi_ is not None:
            Rw[:-1] += GAMMA * Phi_[1:] - Phi_[:-1]             # 途中の手: 形勢がどれだけ良くなったか
            Rw[-1] += -Phi_[-1]                                  # 終端では Φ=0(こうしないと最適方策が変わる)
        if E4 and E4_AUTO and ROLE == "learner" and QD is not None and E4F is not None and len(E4F) == QD.shape[1] and (~E4F).any():   # 09-22
            sh4_ = float(e4_reach(QD[..., 0][:, ~E4F]).mean())
            if e4_t0 is None:
                e4_ok = e4_ok + 1 if sh4_ >= E4_SHARE else 0
                if e4_ok >= E4_HOLD:
                    e4_t0 = it; e4_p0 = E4_P; log("it%4d [介入E4] 介入しなかった軌道で4区画目を買った割合 %.2f が %d反復続けて %.2f 以上。ここから %d反復かけて割合 %.2f → 0" % (it, sh4_, E4_HOLD, E4_SHARE, E4_ZERO, E4_P))
            if e4_t0 is not None: globals()["E4_P"] = e4_p0 * max(0.0, 1.0 - (it - e4_t0) / float(max(E4_ZERO, 1)))
        if E2 and E2_AUTO and ROLE == "learner" and QD is not None and E2F is not None and len(E2F) == QD.shape[1] and (~E2F).any():   # 09-22
            o_ = QD[..., 0][:, ~E2F]; T__ = o_.shape[0]; m2_ = float(np.median(np.where((o_ >= 2).any(0), (o_ >= 2).argmax(0), T__)))
            if e2_t0 is None:
                e2_ok = e2_ok + 1 if m2_ <= E2_T2 else 0
                if e2_ok >= E2_HOLD:
                    e2_t0 = it; e2_p0 = E2_P; log("it%4d [介入E2] 介入しなかった軌道の2区画目の中央値 %.0f手 が %d反復続けて %.0f手以下。ここから %d反復かけて割合 %.2f → 0" % (it, m2_, E2_HOLD, E2_T2, E2_ZERO, E2_P))
            if e2_t0 is not None: globals()["E2_P"] = e2_p0 * max(0.0, 1.0 - (it - e2_t0) / float(max(E2_ZERO, 1)))
        if LANDR > 0 and QD is not None:                       # exp314: 区画を早く買うほど大きい直接の報酬
            if LANDR_AUTO and landr_t0 is None and ROLE != "actor":   # 09-22: 目標に届いたかの判定(学習係だけ)
                if LANDR_MED:
                    o_ = QD[..., 0]; T__ = o_.shape[0]
                    if E2F is not None and len(E2F) == o_.shape[1] and (~E2F).any(): o_ = o_[:, ~E2F]   # 09-22: 介入しなかった軌道だけで判定
                    q2_ = float(np.median(np.where((o_ >= 2).any(0), (o_ >= 2).argmax(0), T__))); q3_ = float(np.median(np.where((o_ >= 3).any(0), (o_ >= 3).argmax(0), T__)))
                else: q2_, q3_, _ = quad_steps(QD)
                landr_ok = landr_ok + 1 if (q2_ <= LANDR_T2 and q3_ <= LANDR_T3) else 0
                t0_ = it if landr_ok >= LANDR_HOLD else -1
                if WORLD > 1:
                    t_ = torch.tensor([t0_], device=DEV); tdist.broadcast(t_, 0); t0_ = int(t_.item())
                if t0_ >= 0:
                    landr_t0 = t0_; log("it%4d [区画報酬] 目標に %d反復続けて届いた(直近 2区画目 %.0f手・3区画目 %.0f手)。ここから %d反復かけて 0 に下げる" % (it, LANDR_HOLD, q2_, q3_, LANDR_ZERO))
            if landr_t0 is not None: lc_ = LANDR * max(0.0, 1.0 - (it - landr_t0) / float(max(LANDR_ZERO, 1)))
            elif LANDR_AUTO: lc_ = LANDR
            else: lc_ = LANDR * (max(0.0, 1.0 - (it - start) / LANDR_ZERO) if LANDR_ZERO > 0 else 1.0)
            lc_last = lc_
            if lc_ > 0:
                own_ = QD[..., 0]                               # (T, N) 手を打つ前の自分の区画数
                gain_ = np.zeros_like(own_, np.float32)
                gain_[:-1] = np.clip(own_[1:] - own_[:-1], 0, None)   # 増えた手番に立つ
                tt_l = np.arange(own_.shape[0], dtype=np.float32).reshape(-1, 1)
                # 2区画目(買う前が1)と3区画目(買う前が2)だけ。区画ごとの基準で「早いほど大きい」
                is2_ = (np.rint(own_) == 1).astype(np.float32); is3_ = (np.rint(own_) == 2).astype(np.float32)
                w2_ = np.clip((LANDR_REF2 - tt_l) / float(LANDR_REF2), 0.0, 1.0)
                w3_ = np.clip((LANDR_REF3 - tt_l) / float(LANDR_REF3), 0.0, 1.0)
                if LANDR_B2 > 0:                                # 09-22: 2区画目を LANDR_P2 手より早く取るほど大きい(区画の記録の手番 = 増えた手 + 1)
                    w2_ = w2_ + LANDR_B2 * np.clip(LANDR_P2 - (tt_l + 1.0), 0.0, None)
                land_bonus = lc_ * gain_ * (is2_ * w2_ + LANDR_W3 * is3_ * w3_)
                Rw += land_bonus
                landr_sum = float(land_bonus.sum(0).mean())
            else: landr_sum = 0.0
        else: landr_sum = 0.0
        adv = np.zeros_like(Rw); last = np.zeros(Nn)
        for t in reversed(range(T_)):
            nv = V[t + 1] if t + 1 < T_ else np.zeros(Nn); delta = Rw[t] + GAMMA * nv - V[t]; last = delta + GAMMA * LAM * last; adv[t] = last
        if GROUP > 1:
            mg = np.array([fin[i, s_] - fin[i, 1 - s_] for i, s_ in trajs], np.float64)
            r = np.tanh(mg / RSCALE) if REWARD == "margin" else np.sign(mg)
            g_ = np.array([games[i]["gid"] for i, _ in trajs]); A_ = np.zeros(len(trajs))
            for k_ in set(g_.tolist()):
                ix = g_ == k_; sd = r[ix].std()
                A_[ix] = (r[ix] - r[ix].mean()) / (sd + 1e-6) if sd > 1e-9 else 0.0
            adv = np.broadcast_to(A_.reshape(1, -1), V.shape).astype(np.float32).copy()
            grpo_spread = float(np.mean([r[g_ == k_].std() for k_ in set(g_.tolist())]))
        if SHOPW > 0 and LIVE_ is not None and len(LIVE_) == adv.shape[1]:   # exp321: 家畜店の多い局を重く(自己対戦は等倍)
            w_ = np.clip(1.0 + SHOPW * (LIVE_ - float(LIVE_.mean())), 1.0 / SHOPW_MAX, SHOPW_MAX).astype(np.float32)
            if not SHOPW_SELF:
                self_ = np.array([games[i]["kind"] == "self" for i, _ in trajs], bool)
                w_ = np.where(self_, 1.0, w_).astype(np.float32)
            w_ = w_ / max(1e-6, float(w_.mean()))                                  # 平均1に正規化(学習率を変えないため)
            adv = adv * w_.reshape(1, -1)
        ret = adv + V
        byk = {}
        for j_k, ((i, s), zz) in enumerate(zip(trajs, z)):
            k_ = games[i]["kind"]; k_ = k_ + "_g" if GFL[j_k] > 0 and not k_.endswith("_g") else k_   # exp304: 誘導した軌道は別に数える
            byk.setdefault(k_, []).append(zz)
        if lg_path is not None and "league" in byk:        # PFSP の勝率(指数移動平均)
            pfsp_wr[lg_path] = 0.7 * pfsp_wr.get(lg_path, 0.5) + 0.3 * float((np.array(byk["league"]) > 0).mean())
        if ROLE != "learner":
            def flat(k, dt):
                b = buf[k]
                return (torch.cat(b).to(dt) if torch.is_tensor(b[0]) else torch.as_tensor(np.concatenate(b), device=DEV, dtype=dt))
            D = dict(tiles=flat("tiles", torch.float16), ou=flat("ou", torch.float32), pu=flat("pu", torch.float32), g=flat("g", torch.float32),
                     lm=flat("lm", torch.uint8), ul=flat("ul", torch.long), uq=flat("uq", torch.long), ms=flat("ms", torch.long), mq=flat("mq", torch.long))
            if MKTNF: D["mkt"] = flat("mkt", torch.float32)        # exp270: 市場の先読み特徴
            if ANTI:
                DO = dict(otiles=flat("otiles", torch.float16), oou=flat("oou", torch.float32), opu=flat("opu", torch.float32), og=flat("og", torch.float32))
            del buf
        else:
            D = D_ld; DO = DO_ld
        A = torch.as_tensor(adv.reshape(-1), device=DEV, dtype=torch.float32); RT = torch.as_tensor(ret.reshape(-1), device=DEV, dtype=torch.float32)
        if CRIT_HL:   # 各局面の行 = (手, 軌道)の順。目標は軌道ごとの最終の差額
            mg_t = np.array([fin[i, s_] - fin[i, 1 - s_] for i, s_ in trajs], np.float32)
            MGT = torch.as_tensor(np.broadcast_to(mg_t.reshape(1, -1), V.shape).reshape(-1).copy(), device=DEV)
        if WORLD > 1:                                     # exp292: 全 GPU の優位度で正規化
            st_ = _allsum(torch.stack([A.sum(), (A * A).sum(), torch.tensor(float(len(A)), device=A.device)]).double())
            mu_ = st_[0] / st_[2]; sd_ = ((st_[1] / st_[2] - mu_ * mu_) * st_[2] / (st_[2] - 1)).clamp(min=0).sqrt()
            A = ((A - mu_.float()) / (sd_.float() + 1e-8)); n = len(A)
        else:
            A = (A - A.mean()) / (A.std() + 1e-8); n = len(A)
        if GFL.any():                                     # exp304: テープに打たせた手は方策の損失から外す
            A[torch.as_tensor((np.arange(V.shape[0])[:, None] < GFL[None, :]).reshape(-1), device=A.device)] = 0.0
        MBL = MB // (WORLD * MICRO)                       # exp292: 1 GPU あたりのミニバッチ(全体では MB)。09-24: MICRO>1 は1枚で MICRO 回に分けて勾配を積む(WORLD×MICRO 枚ぶんと同じ更新)
        if ROLE == "learner": PER0 = PER0_ld
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                PER0 = torch.cat([P.evaluate(net, [D[k][i:i + MBL] for k in KEYS], temp=_T_)[4].float() for i in range(0, n, MBL)])   # 手を選んだときと同じ温度で測る
        if ROLE == "actor":                              # exp296: 局を書き出して、更新はしない
            nm_ = "%s/buf/a%s_v%d_%d.pt" % (ADIR, os.environ.get("SLURM_JOB_ID", "0"), it, int(time.time() * 1000))
            mig_sel_ = mig_av_ = 0.0
            if MASK_MIG:   # 09-23: 新マスクでだけ増えた手が、候補にどれだけ出て、実際にどれだけ選ばれたか
                _liv = D["ul"] >= 0
                _sel = D["lm"].gather(-1, D["ul"].clamp(min=0).unsqueeze(-1)).squeeze(-1)
                mig_sel_ = float(((((_sel & 2) != 0) & _liv).sum() / _liv.sum().clamp(min=1)).item())
                mig_av_ = float(((((D["lm"] & 2) != 0).any(-1) & _liv).sum() / _liv.sum().clamp(min=1)).item())
                _g_ = max(MIG_GUARD, 3.0 * _MIGP_) if M.MIG_MIX else MIG_GUARD   # 混合の方式では選択率は ε 前後になる
                if MIG_GUARD > 0 and mig_sel_ > _g_ and abs(_MIGP_ - MIG_PEN) < 1e-9:
                    log("!! マスク移行の罰が効いていない: 増えた手を選んだ割合 %.4f > %.4f(罰 %.3f のはず)。"
                        "FASTINF などが unit_logits を差し替えていないか、較正が甘くないかを見ること。ここで止める" % (mig_sel_, _g_, _MIGP_))
                    raise SystemExit(3)
                if MIG_Q2MAX > 0 and QD is not None:
                    _q2_ = quad_steps(QD)[0]
                    if _q2_ > MIG_Q2MAX:
                        log("!! マスク移行で経済が崩れている: 2区画目まで %.0f手(上限 %.0f、正常は150手前後)。ここで止める" % (_q2_, MIG_Q2MAX))
                        raise SystemExit(4)
            kinds_a = [games[i]["kind"] + ("_g" if GFL[j] > 0 else "") for j, (i, _) in enumerate(trajs)]; fin_a = np.array([[fin[i, s], fin[i, 1 - s]] for i, s in trajs], np.float64)
            torch.save(dict(D={k: v.detach().to("cpu") for k, v in D.items()}, DO={k: v.detach().to("cpu") for k, v in DO.items()} if ANTI else {},
                            PER0=PER0.to("cpu").float(), V=V.astype(np.float32), kinds=kinds_a, fin=fin_a, T=T_, N=Nn, ver=it, **({"mny": MO} if MNY_REC else {}), **({"quad": QD} if QUAD_REC else {}), **({"gfl": GFL} if GFL.any() else {}), **({"grow": GROW} if GROW is not None else {}), **({"e2f": e2f_} if E2 else {}), **({"e4f": e4f_} if E4 else {}), **({"live": np.array([LIVE_[i] for i, _ in trajs], np.float32)} if LIVE_ is not None else {}), **({"mig": _MIGP_} if MASK_MIG else {}), opp=lg_path), nm_ + ".tmp")   # 09-22: opp = リーグの相手(段階表のレーティング更新に使う)
            os.replace(nm_ + ".tmp", nm_)
            byk_ = " ".join("%s %.2f(%d)" % (k, (np.array(v) > 0).mean(), len(v)) for k, v in byk.items())
            e6msg_ = (" | 介入E6 落とした注文 %d手" % e6n_) if E6 else ""
            e5msg_ = ""
            if E5:                                   # exp320: 介入なしの軌道で家畜の注文が増えていれば方策が受け入れている
                try:
                    e5msg_ = " | 介入E5 書き換え %d手 家畜の注文 介入あり %.1f・**介入なし %.1f**" % (
                        e5n_, float(e5tot_[e5f_].mean()) if e5f_.any() else -1.0, float(e5tot_[~e5f_].mean()) if (~e5f_).any() else -1.0)
                except Exception as _e: e5msg_ = " | 介入E5 の集計に失敗(%s)" % str(_e)[:40]
            try:
                e5msg_ = " | 介入E5 書き換え %d手 家畜の注文 介入あり %.1f・**介入なし %.1f**" % (
                    e5n_, float(e5tot_[e5f_].mean()) if e5f_.any() else -1.0, float(e5tot_[~e5f_].mean()) if (~e5f_).any() else -1.0)
            except Exception as _e: e5msg_ = " | 介入E5 の集計に失敗(%s)" % str(_e)[:40]

            log("局 版%d %s | %d局面 収集 %.0fs → %s" % (it, byk_, n, tr, os.path.basename(nm_)) + (" | 自分の最終所持金 平均 %.0f 正の割合 %.2f" % (fin_a[:, 0].mean(), (fin_a[:, 0] > 0).mean()) if MNY_REC or OWN_R0 > 0 else "")
                + ((lambda q_: " | 区画 自分2 %.0f手 自分3 %.0f手 相手3 %.0f手" % q_)(quad_steps(QD)) if (QUAD_REC and QD is not None) else "")
                + ((lambda o_: " 4区画に届いた割合 %.2f(4区画目 中央 %.0f手)" % (float((o_ >= 4).any(0).mean()), float(np.median(np.where((o_ >= 4).any(0), (o_ >= 4).argmax(0), o_.shape[0])))))(QD[..., 0])
                   if (QUAD_REC and QD is not None) else "")   # 09-22: 介入 E4 を外した系統でも4区画目の購入が見えるように
                + (" 土地報酬 %.3f" % landr_sum if LANDR > 0 else "")
                + (" | 介入E2 割合 %.2f 軌道 %d 書き換え %d手" % (E2_P, int(e2f_.sum()), e2n_) if E2 else "")
                + ((lambda fo_: " | 新マスクの割合 %.2f 軌道 %d 最終所持金 新マスク %.0f・旧マスク %.0f" % (LMIX_P, int(lmf_.sum()),
                    float(fo_[lmf_].mean()) if lmf_.any() else -1, float(fo_[~lmf_].mean()) if (~lmf_).any() else -1))(np.array([fin[i, s_] for i, s_ in trajs]))
                   if LMIX else "")
                + ((" | マスク移行 罰 %.3f 新しい手が候補にある割合 %.3f 選んだ割合 %.4f" % (_MIGP_, mig_av_, mig_sel_)) if MASK_MIG else "")
                + ((lambda o_: " | 介入E4 割合 %.2f 軌道 %d 書き換え %d手 手%dまでに%d区画 介入あり %.2f・介入なし %.2f" % (E4_P, int(e4f_.sum()), e4n_, E4_BY, E4_QMAX,
                    float(e4_reach(o_[:, e4f_]).mean()) if e4f_.any() else 0.0, float(e4_reach(o_[:, ~e4f_]).mean()) if (~e4f_).any() else 0.0))(QD[..., 0])
                   if (E4 and QD is not None) else "")
                + ((lambda qa_, qb_: " 自分2 介入あり %.1f手・介入なし %.1f手(介入なしの128手以下 %.2f)" % (qa_, qb_[0], qb_[1]))(
                    quad_steps(QD[:, e2f_])[0], (lambda o_: (float(np.where((o_ >= 2).any(0), (o_ >= 2).argmax(0), o_.shape[0]).mean()), float((np.where((o_ >= 2).any(0), (o_ >= 2).argmax(0), o_.shape[0]) <= 128).mean())))(QD[:, ~e2f_, 0]))
                   if (E2 and QD is not None and e2f_.any() and (~e2f_).any()) else "")
                + (" | 段階表の相手 %s(R %.0f、自分 %.0f、予想勝率 %.2f)" % (lad_all[lg_path][0], lad_all[lg_path][1], float(elo_me), lad_p) if lad_p is not None else "")
                + (e5msg_ if E5 else "")
                + e6msg_
                + (" | 序盤ガードレール メロンの種 %.1f個" % float(np.mean([pool.open_mel[i, s] for i, s in trajs])) if getattr(sp_fast, "OPEN", False) and getattr(pool, "open_mel", None) is not None else ""))
            del D; DO = None; continue   # exp302: ANTI=0 だと DO が無く落ちていた
        if MEMLOG: log("it%4d%s" % (it, _mem("局を読んだ直後")))
        stats = []; vonly = CRITIC and it < vw_until; tu = time.time(); safe_skip = 0; aux_n = 0; aux_p = 0.0; aux_l = 0.0; aux_k = 0; auxu_n = 0; auxu_l = 0.0; auxu_k = 0; auxu_ = aux_use_c(it, start) if AUX_USE_ON else 0.0; auxb_l = 0.0; auxb_k = 0; auxb_u = 0.0; auxb_m = 0.0
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, MBL):
                idx = perm[i:i + MBL]; b = [D[k][idx] for k in KEYS]
                mi_ = (i // MBL) % MICRO; first_ = mi_ == 0; last_ = mi_ == MICRO - 1 or i + MBL >= n
                if CRITIC and (vonly or (i // MBL) % CRIT_EVERY == 0):     # exp298: CRIT_EVERY(既定 1 = 毎回)
                    idx_, b_ = idx, b
                    if CRIT_FRAC < 1: idx = idx[:max(1, int(len(idx) * CRIT_FRAC))]; b = [x[:len(idx)] for x in b]   # exp298: 批評家だけ行を間引く
                    with torch.autocast("cuda", dtype=torch.float16):
                        hc = None if (ANTI and P.FASTINF and not CRIT_HL) else crit.encode(b[0], b[1], b[2], b[3])[0]
                        if CRIT_HL:
                            ho, _ = crit.encode(*[DO[k][idx] for k in OKEYS]); lgc = (hlh(hc[:, -1]) + hlh(ho[:, -1]).flip(-1)).float()
                        elif ANTI and P.FASTINF:                   # exp280: 自分と相手の視点を1回の順伝播にまとめる(値は同じ)
                            ob = [DO[k][idx] for k in OKEYS]
                            hh, _ = crit.encode(*[torch.cat([b[q_], ob[q_]], 0) for q_ in range(4)])
                            vv = crit.value(hh[:, -1]).squeeze(-1); nb_ = len(idx)
                            vc = 0.5 * (vv[:nb_] - vv[nb_:])
                        else:
                            vc = crit.value(hc[:, -1]).squeeze(-1)
                            if ANTI:
                                ho, _ = crit.encode(*[DO[k][idx] for k in OKEYS]); vc = 0.5 * (vc - crit.value(ho[:, -1]).squeeze(-1))
                    vl = -(hl_target(MGT[idx]) * lgc.log_softmax(-1)).sum(-1).mean() if CRIT_HL else F.smooth_l1_loss(vc.float(), RT[idx])
                    if first_: opt_c.zero_grad()
                    scaler_c.scale(vl / MICRO if MICRO > 1 else vl).backward()
                    if last_:
                        if _ovl_crit is not None: _ovl_crit.wait()
                        else: _sync_grads(cparams, "crit")
                        scaler_c.unscale_(opt_c)
                        torch.nn.utils.clip_grad_norm_(cparams, 0.5); scaler_c.step(opt_c); scaler_c.update()
                    idx, b = idx_, b_
                if vonly: continue
                if MEMLOG and i == 0: log("it%4d%s" % (it, _mem("批評家の後・方策の前")))
                with torch.autocast("cuda", dtype=torch.float16):
                    lp, ent, v, dist, per, pmask = P.evaluate(net, b)
                if MEMLOG and i == 0: log("it%4d%s ミニバッチ %d行" % (it, _mem("方策の順伝播の後"), len(idx)))
                ent, per = ent.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
                rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
                pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    rd = P.evaluate(ref, b)[3]
                rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
                kl = P.kl_to_ref(dist, rd).mean()
                loss = pl + _BETA_ * kl - ENT * ent.mean()
                if AUX_USE_ON and auxu_ > 0:   # exp312: 4区画目を「使う」(南東区画に立つユニットに植えさせる)
                    ou_, g_ = b[1], b[3]
                    lsu_ = dist[0]
                    q4_ = (g_[:, 39] * 4).round() == 4
                    x_, y_ = ou_[:, :, 1], ou_[:, :, 2]
                    se_ = (x_ >= 5) & (x_ <= 9) & (y_ >= 5) & (y_ <= 9) & (ou_[:, :, 0] > 0)
                    pl_ = lsu_[:, :, 15:20].max(-1).values > -1e3
                    mu_ = q4_.unsqueeze(1) & se_ & pl_
                    if bool(mu_.any()):
                        lse_ = torch.logsumexp(lsu_[:, :, 15:20], -1)[mu_]
                        loss = loss + auxu_ * (-lse_.mean())
                        auxu_n += int(mu_.sum()); auxu_l += float(-lse_.mean()); auxu_k += 1
                if AUXT is not None:   # exp308: 買える状態の行で BUY_LAND の log 確率を上げる
                    m_ = AUXT[idx]
                    if bool(m_.any()):
                        lbl_ = dist[2][:, 0, M_BUYLAND][m_]
                        loss = loss + auxc_ * (-lbl_.mean())
                        aux_n += int(m_.sum()); aux_p += float(lbl_.exp().mean()) * int(m_.sum()); aux_l += float(-lbl_.mean())
                        aux_k += 1
                if AUX_BC > 0:   # 09-22: 棋譜との cross entropy
                    assert not MKTNF, "AUX_BC は素の Policy だけ(市場特徴つきは未対応)"
                    bd_ = bc_data(); bi_ = torch.randint(0, bd_["n"], (AUX_BC_MB,), device=DEV)
                    with torch.autocast("cuda", dtype=torch.float16):
                        lbc_, sbc_ = bd_["f"](net, bd_["T"], bi_, train=True)
                    loss = loss + AUX_BC * lbc_
                    auxb_l += float(lbc_); auxb_k += 1; auxb_u += float(sbc_["ua"] / sbc_["un"].clamp(min=1)); auxb_m += float(sbc_["mseq"] / sbc_["n"])
                if SAFE_KL > 0:
                    with torch.no_grad():
                        lr_ = (per - PER0[idx]).clamp(-20, 20); akl = (((lr_.exp() - 1) - lr_) * pmask).sum() / pmask.sum().clamp(min=1)
                        sk_ = (akl > SAFE_KL).float()
                        # 09-23: continue は _sync_grads も飛ばす。1台だけ飛ぶと以後の all_reduce が
                        #   ずれた組み合わせで噛み合い、黙って別のミニバッチの勾配を平均してしまう。
                        #   捨てるかどうかを全 GPU で揃える(WORLD==1 では float(akl) > SAFE_KL と同値)。
                        if WORLD > 1: tdist.all_reduce(sk_, op=tdist.ReduceOp.MAX)
                    if MICRO > 1:
                        if first_: mskip_ = False
                        if float(sk_) > 0: mskip_ = True   # 09-24: DDP と同じく、1片でも超えたらそのミニバッチ全体を捨てる
                    elif float(sk_) > 0: safe_skip += 1; continue
                if MICRO > 1:
                    if first_: opt.zero_grad()
                    if SAFE_KL > 0 and mskip_:
                        if last_: safe_skip += 1; opt.zero_grad()
                        continue
                    scaler.scale(loss / MICRO).backward()
                    if not last_:
                        stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean()))); continue
                    _sync_grads(list(net.parameters()), "net")
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
                else:
                    opt.zero_grad(); scaler.scale(loss).backward()
                    if _ovl_net is not None: _ovl_net.wait()
                    else: _sync_grads(list(net.parameters()), "net")
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
                stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean())))
        del D; tu = time.time() - tu
        if SAFE_KL > 0 and safe_skip: log("it%4d [安全装置] 近似KL > %.3f のミニバッチ %d 個を捨てた" % (it, SAFE_KL, safe_skip))
        if AUX_USE_ON and auxu_k: log("it%4d [使う補助損失] 係数 %.4f 対象 %d組 生の損失 %.2f (ミニバッチ %d個)" % (it, auxu_, auxu_n, auxu_l / auxu_k, auxu_k))
        if AUX_BC > 0 and auxb_k: log("it%4d [棋譜の補助損失] 係数 %.3f 棋譜 %d局(%s) BC損失 %.3f ユニット一致 %.3f 市場の列一致 %.3f (ミニバッチ %d個 × %d局面)" % (it, AUX_BC, _BCT["ne"], AUX_BC_SPLIT, auxb_l / auxb_k, auxb_u / auxb_k, auxb_m / auxb_k, auxb_k, AUX_BC_MB))
        if AUX_ON and aux_k: log("it%4d [補助損失] 係数 %.4f 対象 %d行 π(BUY_LAND) 平均 %.3e 生の損失 %.2f (ミニバッチ %d個)" % (it, auxc_, aux_n, aux_p / max(aux_n, 1), aux_l / aux_k, aux_k))
        if ANTI: del DO
        # 価値EV は ret = adv + V なので ret - V = adv、優位度が小さいほど 1 に近づく循環した指標だった。
        # 本当に見たいのは「勝敗 z を V がどれだけ当てているか」なので、z に対する説明率も出す。
        Z = np.broadcast_to(z.reshape(1, -1), V.shape)
        ev_ = 1 - np.var(Z.reshape(-1) - V.reshape(-1)) / (np.var(Z.reshape(-1)) + 1e-8)
        advstd_ = float(np.std(adv.reshape(-1))) if GROUP <= 1 else grpo_spread; s_ = np.mean(stats, 0) if stats else [0, 0, 0, 0]
        sync_drain()   # 09-23: 溜めた CUDA イベントをここで1回だけ読む(反復中は止めない)
        log("it%4d " % it + (("R %.0f " % elo_me) if LADDER else "") + " ".join("%s %.2f(%d)" % (k, (np.array(v) > 0).mean(), len(v)) for k, v in byk.items()) +
            " | KL %.3f ent %.2f 切 %.2f 価値EV(対 勝敗) %.2f %s %.3f %s| 局面 %d 収集 %.0fs(環境 %.0f 手選び %.0f 相手 %.0f 進める %.0f)更新 %.0fs 全体 %.0fs → %.2f百万局面/時" % (
            s_[1], s_[2], s_[3], ev_, "組内の報酬SD" if GROUP > 1 else "優位度SD", advstd_, "価値のみ " if vonly else "", n * WORLD, tr, tm["env"], tm["samp"], tm["opp"], tm["step"], tu, time.time() - t0, n * WORLD / (time.time() - t0) * 3600 / 1e6) + (" | 同期 %.2fs(all_reduce %d回 %.2fGB 実効 %.0fGB/s)" % (SYNC_T[0], SYNC_N[0], SYNC_N[1] * 4 / 1e9,
                SYNC_N[1] * 4 / 1e9 * 2.0 * (WORLD - 1) / WORLD / max(SYNC_T[0], 1e-9)) if WORLD > 1 else "")
            + (" | 最終所持金 自分 %.0f 相手 %.0f 自分>0 %.2f 形勢係数 %.2f 自分の所持金報酬 %.2f" % (np.mean([fin[i, s] for i, s in trajs]), np.mean([fin[i, 1 - s] for i, s in trajs]),
               np.mean([fin[i, s] > 0 for i, s in trajs]), shc_, own_c) if MNY_REC or OWN_R0 > 0 else "")
            + ((lambda q_: " | 区画 自分2 %.0f手 自分3 %.0f手 相手3 %.0f手 区画係数 %.2f" % (q_ + (quc_,)))(quad_steps(QD)) if (QUAD_REC and QD is not None) else "")
            + ((" | 介入E4 割合 %.2f%s 介入なしで目標に届いた割合 %.2f" % (E4_P, ("(第%d反復から減衰中)" % e4_t0) if e4_t0 is not None else ("(到達 %d/%d)" % (e4_ok, E4_HOLD) if E4_AUTO else ""),
                float(e4_reach(QD[..., 0][:, ~E4F]).mean()) if (E4F is not None and QD is not None and len(E4F) == QD.shape[1] and (~E4F).any()) else -1)) if E4 else "")
            + ((" | 介入E2 割合 %.2f%s" % (E2_P, ("(第%d反復から減衰中)" % e2_t0) if e2_t0 is not None else ("(到達 %d/%d)" % (e2_ok, E2_HOLD) if E2_AUTO else ""))) if E2 else "")
            + ((" 区画報酬(1軌道あたり %.3f)の係数 %.2f%s" % (landr_sum, lc_last, ("(第%d反復から減衰中)" % landr_t0) if landr_t0 is not None else ("(目標到達 %d/%d)" % (landr_ok, LANDR_HOLD) if LANDR_AUTO else ""))) if LANDR > 0 else "")
            + ((lambda ug_: " | 誘導なし 自分 %.0f >3000 %.2f 最大 %.0f 資産 %s 勝ち重み %.2f 誘導 %d軌道" % (
                np.mean([fin[i, s] for (i, s), u_ in zip(trajs, ug_) if u_]), np.mean([fin[i, s] > 3000 for (i, s), u_ in zip(trajs, ug_) if u_]),
                max([fin[i, s] for (i, s), u_ in zip(trajs, ug_) if u_]), "%.0f" % GROW[ug_].mean() if GROW is not None else "-",
                min(1.0, it / WIN_RAMP) if WIN_RAMP > 0 else 1.0, int((GFL > 0).sum())))(GFL == 0) if (WIN_RAMP > 0 or GFL.any()) and (GFL == 0).any() else ""))
        if MASK_MIG:
            log("it%4d [マスク移行] 学習の罰 %.3f → 関門・ものさしは %.3f(初期 %.3f・保持 %d反復・第%d反復で 0)" % (it, M.mig_get(), _MIGP_, MIG_PEN, MIG_HOLD, MIG_IT0 + MIG_ZERO))
            M.mig_set(_MIGP_)
        SYNC_T[0] = 0.0; SYNC_N[0] = SYNC_N[1] = 0
        if ema is not None and not vonly:               # 重みの移動平均(学習は現行 net のまま、挑戦者は平均)
            with torch.no_grad():
                for pe, pn in zip(ema.state_dict().values(), net.state_dict().values()):
                    if pe.dtype.is_floating_point: pe.mul_(EMA).add_(pn, alpha=1 - EMA)
                    else: pe.copy_(pn)
        chal = ema if ema is not None else net
        # 関門: 現行(EMA なら平均の重み)対 最良版(貪欲・種の上限 = 提出と同じ打ち方、64局)
        if it % GATE == GATE - 1 and not vonly:
            # 09-21: 関門を「直前の最良版」だけにすると、直前にだけ勝つ方向に漂流する。
            #   実測: land5 第112世代は第111世代に 0.609 で昇格したのに、第108世代には 384局 0.422(z=-3.10)。
            #   GATE_POOL=k で、過去の世代から k 体を選んで一緒に測り、**全部の平均**で判定する。
            #   ポケカ5位・7位のリーグ方式と同じ考え方。0 なら従来どおり最良版だけ。
            gopp = [("最良版", best)]
            if GATE_POOL > 0 and (league or gen > 0):
                # 09-22(巡回係の指摘): 昇格した最良版はリーグの末尾に入るので、そのままだと候補に最良版そのものが入り
                #   「最良版に2回」と同じになっていた。今の最良版のファイルを除き、起点(INIT)を候補に足す
                best_p_ = "%s/orbit_%s_gen%d.pt" % (L, tag, gen)
                cand_ = [x for x in league if os.path.exists(x) and x != best_p_]
                if gen > 0 and init and os.path.exists(init) and init not in cand_: cand_.append(init)
                gsel_ = rng.sample(cand_, min(GATE_POOL, len(cand_))) if cand_ else []   # 09-22: 名前を pick_ から変更(学習係の局リスト pick_ を上書きし、局の削除で落ちていた)
                for q_ in gsel_:
                    try:
                        old_ = M.load(q_, DEV).eval()
                        # 09-22(巡回係の指摘): 別の系統から分岐すると、最良版と同じ重みが別名(orbit_<元の系統>_genN)で残り、2回測っていた。重みが同一なら除く
                        if all(torch.equal(x_, y_) for x_, y_ in zip(old_.state_dict().values(), best.state_dict().values())):
                            del old_; continue
                        gopp.append((os.path.basename(q_).replace("orbit_%s_" % tag, "").replace(".pt", ""), old_))
                    except Exception as e_: log("関門の相手を読めない %s (%s)" % (q_, str(e_)[:50]))
            ws_ = []; ms_ = []; det_ = []
            nseed_ = max(8, len(GATESEEDS) // max(1, len(gopp)))
            for gi_, (nm_, op_) in enumerate(gopp):
                sd_ = GATESEEDS[gi_ * nseed_:(gi_ + 1) * nseed_] or GATESEEDS[:nseed_]
                if RANK == 0: w_, m_ = h2h(chal, op_, sd_, greedy=not GATE_SAMPLE, draw_win=GATE_DRAW)
                else: w_, m_ = 0.0, 0.0
                if WORLD > 1:
                    t_ = torch.tensor([w_, m_], device="cuda", dtype=torch.float64); tdist.broadcast(t_, 0); w_, m_ = float(t_[0]), float(t_[1])
                ws_.append(w_); ms_.append(m_); det_.append("%s %.3f" % (nm_, w_))
            w = sum(ws_) / len(ws_); m = sum(ms_) / len(ms_)
            nopp_ = len(gopp)   # 09-22: ログの「対 N体」は相手を消す前に数える
            if len(gopp) > 1: log("it%4d [関門の内訳] %s" % (it, " / ".join(det_)))
            if len(gopp) > 1:   # 09-21: 関門ごとに読んだ相手を GPU から外す(外さないと関門1回ごとに数十MB ずつ積み上がる)
                del gopp[1:]; del gsel_
                torch.cuda.empty_cache()
            wbest_ = ws_[0]   # 09-22: 最良版そのものへの勝率(GATE_BEST_MIN>0 なら、平均に加えてこれも満たすこと)
            fix_ok_ = True; fix_det_ = []                       # 09-23(exp317): 外の流派への退行を止める条件
            if GATE_FIX:
                for fp_ in GATE_FIX:
                    try:
                        fo_ = M.load(fp_, DEV).eval()
                        if RANK == 0: fw_, fm_ = h2h(chal, fo_, FIXSEEDS, greedy=not GATE_SAMPLE, draw_win=GATE_DRAW)
                        else: fw_, fm_ = 0.0, 0.0
                        if WORLD > 1:
                            t_ = torch.tensor([fw_, fm_], device="cuda", dtype=torch.float64); tdist.broadcast(t_, 0); fw_, fm_ = float(t_[0]), float(t_[1])
                        del fo_; torch.cuda.empty_cache()
                        nm_ = os.path.basename(os.path.dirname(fp_)) or os.path.basename(fp_)
                        fix_det_.append("%s %.3f(%d局)" % (nm_, fw_, len(FIXSEEDS)))
                        if fw_ < GATE_FIX_MIN: fix_ok_ = False
                    except Exception as e_: log("関門の固定相手を読めない %s (%s)" % (fp_, str(e_)[:50]))
                log("it%4d [関門の外側] %s → 下限 %.3f を%s" % (it, " / ".join(fix_det_), GATE_FIX_MIN, "満たす" if fix_ok_ else "**満たさない(昇格を止める)**"))
            if w >= GATE_WR and wbest_ >= GATE_BEST_MIN and fix_ok_:
                gen += 1; p = "%s/orbit_%s_gen%d.pt" % (L, tag, gen)
                if RANK == 0:
                    torch.save(chal.state_dict(), p + ".tmp"); os.replace(p + ".tmp", p)   # exp292: 書きかけを読まれないように
                    # 09-21: .pt.cfg.json を一緒に書く。無いと model_tf.load が d=128/3層にフォールバックして
                    #        次にこの世代から分岐するときに size mismatch で落ちる(CLAUDE.md 既出の落とし穴)。
                    try:
                        cfg_ = dict(d=net.d, layers=len(net.enc.layers), heads=net.enc.layers[0].self_attn.num_heads,
                                    ff=net.enc.layers[0].linear1.out_features, dec_layers=len(net.dec.layers))
                        json.dump(cfg_, open(p + ".cfg.json.tmp", "w")); os.replace(p + ".cfg.json.tmp", p + ".cfg.json")
                    except Exception as e_: log("cfg.json を書けなかった: %s" % str(e_)[:80])
                if WORLD > 1: tdist.barrier()                          # exp292: 全 GPU が新しい世代のファイルを読めるようになってから進む
                best.load_state_dict(chal.state_dict()); ref.load_state_dict(chal.state_dict()); league = (league + [p])[-8:]
                if LADDER:                                     # 09-22: 通った世代は、通ったときのレーティングで段階表に加わる(直近 8 世代だけ)
                    own_elo[p] = ("%s_g%d" % (tag, gen), float(elo_me)); own_elo = {x: v for x, v in own_elo.items() if x in league}
                log("it%4d [関門] 現行 対 %d体 勝率 %.3f 差 %+.0f → **置き換え(第%d世代)**" % (it, nopp_, w, m, gen))
            else:
                log("it%4d [関門] 現行 対 %d体 勝率 %.3f 差 %+.0f → 据え置き(第%d世代のまま)" % (it, nopp_, w, m, gen))
        # 09-21: Local SGD の重み平均(共有ディスク経由)
        if AVGDIR and AVG_EVERY > 0 and AVG_N > 1 and it > 0 and it % AVG_EVERY == 0 and not vonly:
            try:
                os.makedirs(AVGDIR, exist_ok=True)
                mine_ = "%s/%s_%d.pt" % (AVGDIR, tag, it)
                if RANK == 0:
                    torch.save(net.state_dict(), mine_ + ".tmp"); os.replace(mine_ + ".tmp", mine_)
                if WORLD > 1: tdist.barrier()
                t0_ = time.time(); want_ = "%s/*_%d.pt" % (AVGDIR, it)
                while len(glob.glob(want_)) < AVG_N and time.time() - t0_ < AVG_WAIT: time.sleep(5)
                fs_ = sorted(glob.glob(want_))
                if len(fs_) >= 2:
                    acc_ = None; nok_ = 0
                    for f_ in fs_:
                        try: sd_ = torch.load(f_, map_location=DEV)
                        except Exception: continue
                        nok_ += 1
                        if acc_ is None: acc_ = {k_: v_.float().clone() for k_, v_ in sd_.items()}
                        else:
                            for k_ in acc_: acc_[k_] += sd_[k_].float()
                    if nok_ >= 2:
                        cur_ = net.state_dict()
                        for k_ in acc_: cur_[k_].copy_((acc_[k_] / nok_).to(cur_[k_].dtype))
                        ref.load_state_dict(net.state_dict())
                        log("it%4d [重み平均] %d 本を平均して読み直した(%.0f秒待ち)" % (it, nok_, time.time() - t0_))
                else:
                    log("it%4d [重み平均] %d 本しか揃わず見送り(%.0f秒待ち)" % (it, len(fs_), time.time() - t0_))
                if RANK == 0:
                    for f_ in glob.glob("%s/*_%d.pt" % (AVGDIR, it - AVG_EVERY * 3)):
                        try: os.remove(f_)
                        except OSError: pass
            except Exception as e_: log("重み平均で例外(無視して続行): %s" % str(e_)[:100])
        if SAVE_EVERY and it % SAVE_EVERY == SAVE_EVERY - 1:
            torch.save(net.state_dict(), "%s/orbit_%s_it%d.pt" % (L, tag, it + 1))
        if RANK == 0: torch.save(dict(net=net.state_dict(), crit=crit.state_dict(), opt=opt.state_dict(), opt_c=opt_c.state_dict(), it=it,
                        best=best.state_dict(), gen=gen, league=league, vw_until=vw_until, pfsp_wr=pfsp_wr, **({"mig_it0": MIG_IT0} if MASK_MIG else {}), own_elo=own_elo, elo_me=elo_me, landr_t0=landr_t0, ctl_tok=ctl_tok, e2_t0=e2_t0, e2_p0=e2_p0, e4_t0=e4_t0, e4_p0=e4_p0, **({"ema": ema.state_dict()} if ema is not None else {}), **({"hlh": hlh.state_dict()} if hlh is not None else {})), ck + ".tmp"); os.replace(ck + ".tmp", ck)   # 09-22: 書き込み中の横取り(GraceTime=0)で控えが壊れないように
        if ROLE == "learner":                            # exp296: 使った局を消して、新しい重みを公開
            if WORLD > 1: tdist.barrier()
            if RANK == 0:
                for f_ in pick_[0]:
                    try: os.remove(f_)
                    except OSError: pass
            _publish(it + 1)
        if it % EVAL == EVAL - 1 and RANK == 0:
            w, m = vs_tape(chal); log("it%4d [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f(最良版は第%d世代)%s" % (it, w, m, gen, "(EMA の重み)" if ema is not None else ""))
            me, wr, mg = P.eval_policy(chal, hold_e, True, 2, cap=True); log("it%4d [58局 凍った相手 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))


if __name__ == "__main__":
    main()
