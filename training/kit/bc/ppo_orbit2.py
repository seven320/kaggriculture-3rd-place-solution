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
import copy, json, os, random, sys, time
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
OWN_R0 = float(os.environ.get("OWN_R0", "0")); OWN_ANNEAL = int(os.environ.get("OWN_ANNEAL", "0"))   # exp294: 自分の所持金の報酬(係数と、0 に下げる反復数)
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
SYNC_T = [0.0]
def _sync_grads(params):
    """拡大率をかけたままの勾配を全 GPU で平均する(unscale の前に呼ぶので、オーバーフローの判定が全 GPU で揃う)"""
    if WORLD == 1: return
    _t = time.time()
    gs = [q.grad for q in params if q.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in gs]); tdist.all_reduce(flat); flat /= WORLD
    o = 0
    for g in gs: n_ = g.numel(); g.copy_(flat[o:o + n_].view_as(g)); o += n_
    if os.environ.get("SYNCPROF") == "1": torch.cuda.synchronize(); SYNC_T[0] += time.time() - _t   # exp292: 勾配の同期にかかった時間
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
# exp220: 学習した批評家は 価値EV(対 勝敗) −0.37〜−0.40 で「定数を答える」より悪かった(o6/o7、80反復)。
#   1反復の独立な勝敗ラベルが192局ぶんしかなく、720手ぶんの相関した局面に当てはめて過学習している。
#   CRITIC=0 では批評家を使わず、V = その反復で同じ種類の相手と打った局の平均勝敗(基準線)にする。
CRITIC = os.environ.get("CRITIC", "1") == "1"
ANTI = os.environ.get("ANTI") == "1"   # exp242(PTCG 神託批評家): 批評家を V = (v(自分の観測) − v(相手の観測)) / 2 に。相手の観測は学習のときだけ使う特権情報(提出する方策は変わらない)
EMA = float(os.environ.get("EMA", 0))
EXTRA = [x for x in os.environ.get("LEAGUE_EXTRA", "").split(",") if x]   # exp252(PTCG 5・7位): 最初からリーグに入れる凍った強い相手(重みのパス)
PFSP = os.environ.get("PFSP") == "1"         # exp252(PTCG 7位): リーグの相手を 0.15 + (1 − 勝率)^2 の重みで選ぶ(苦手な相手ほど多く)
SAFE_KL = float(os.environ.get("SAFE_KL", 0))
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
GATESEEDS = [5_000_000 + 7919 * k for k in range(32)]    # 関門用(ものさしとは別)


def h2h(a, b, seeds, greedy=True):
    """a 対 b を各シード両座席で。a から見た勝率と差。"""
    mg = []
    for s in (0, 1):
        m = S.play(seeds, {s: S.PolicyOpp(a, greedy=greedy, cap=True), 1 - s: S.PolicyOpp(b, greedy=greedy, cap=True)})
        mg.append(m[:, s] - m[:, 1 - s])
    mg = np.concatenate(mg)
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
        m = S.play(seeds, {s: S.PolicyOpp(net, greedy=True, cap=True), 1 - s: opp()})
        if FASTPOOL_ON: sp_fast._PACE_SKIP[0] = None; sp_fast._LAND_SKIP[0] = None
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
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5); opt_c = torch.optim.Adam(cparams, lr=float(os.environ.get("CLR", 1e-4)), eps=1e-5)   # exp228: CLR で批評家の学習率(既定 1e-4 は従来どおり)
    scaler = torch.amp.GradScaler("cuda"); scaler_c = torch.amp.GradScaler("cuda")
    split = json.load(open(os.environ.get("SPLITFILE", L + "/bc_d2.json"))); hold_e = split["hold"]
    ck = "%s/orbit_%s.pt" % (L, tag); start = 0; gen = 0; league = []; vw_until = VWARM; pfsp_wr = {}
    if not os.path.exists(ck) and RESUME_FROM and os.path.exists(RESUME_FROM):
        c = torch.load(RESUME_FROM, map_location=DEV, weights_only=False)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); best.load_state_dict(c["best"]); ref.load_state_dict(c["best"])
        start = c["it"] + 1; gen = c["gen"]; league = c["league"]; vw_until = start + 2
        log("%s の反復 %d から引き継ぎ(最良版は第%d世代、リーグ %d体)。評価役は小さいネットワークで作り直すので、最初の2反復は評価役だけ学習" % (RESUME_FROM, c["it"], gen, len(league)))
    elif os.path.exists(ck):
        c = torch.load(ck, map_location=DEV, weights_only=False)
        net.load_state_dict(c["net"]); crit.load_state_dict(c["crit"]); opt.load_state_dict(c["opt"]); opt_c.load_state_dict(c["opt_c"])
        if hlh is not None and "hlh" in c: hlh.load_state_dict(c["hlh"])
        best.load_state_dict(c["best"]); ref.load_state_dict(c["best"]); start = c["it"] + 1; gen = c["gen"]; league = c["league"]; vw_until = c.get("vw_until", VWARM); pfsp_wr = c.get("pfsp_wr", {})
        if ema is not None: ema.load_state_dict(c.get("ema", c["net"]))
        log("再開 it%d(最良版は第%d世代、リーグ %d体)" % (start, gen, len(league)))
    if start == 0:
        w, m = vs_tape(net); log("it  -1 [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f (初期)" % (w, m))
    log("設定: NSELF %d NBEST %d NLEAGUE %d NTAPEUG %d / 関門 %d反復ごと・勝率 %.2f 以上で置き換え / LR %g BETA %g / GAMMA %g LAM %g / 形勢 SHAPE %g / 批評家 %s / GRPO GROUP %d REWARD %s" % (
        NSELF, NBEST, NLEAGUE, NTAPEUG, GATE, GATE_WR, LR, BETA, GAMMA, LAM, SHAPE, "あり" if CRITIC else "なし(相手の種類ごとの平均勝敗を基準線)", GROUP, REWARD))
    rng = random.Random(int(os.environ.get("RSEED", 11)) + 1000 * RANK)   # exp292: GPU ごとに別の局   # exp286: 分岐したランで対局の種を変える(既定は従来どおり 11)
    for it in range(start, ITERS):
        t0 = time.time()
        games = []
        gid = 0
        lg_pool = EXTRA + league                             # (pool はゲームの束の名前なので別名に)
        for kind, n in (("self", NSELF), ("best", NBEST), ("league", NLEAGUE if lg_pool else 0), ("tape", NTAPEUG)):
            if GROUP > 1:
                if kind == "self": continue                      # GRPO では自己対戦を使わない(両席とも自分で、組の比較が崩れる)
                for _ in range(max(1, n // GROUP)):
                    seat, seed = rng.randrange(2), rng.randrange(10 ** 9)
                    for _k in range(GROUP): games.append(dict(kind=kind, seat=seat, seed=seed, gid=gid))
                    gid += 1
            else:
                for _ in range(n): games.append(dict(kind=kind, seat=rng.randrange(2), seed=rng.randrange(10 ** 9), gid=-1))
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
        lg_net = None; lg_path = None
        if lg_pool:
            if PFSP:
                wts = [0.15 + (1 - pfsp_wr.get(x, 0.5)) ** 2 for x in lg_pool]; lg_path = rng.choices(lg_pool, weights=wts)[0]
            else: lg_path = rng.choice(lg_pool)
            lg_net = mk(); lg_net.load_state_dict(torch.load(lg_path, map_location=DEV)); lg_net.eval()
        tape = sp_fast.FastTapeRouterOpp(OPPDIR)      # TapeRouterOpp と719手すべて同じ手(check_tape で確認)
        allt = [(i, g["seat"]) for i, g in enumerate(games)] + [(i, 1 - g["seat"]) for i, g in enumerate(games) if g["kind"] == "self"]
        i0 = np.array([i for i, s in allt if s == 0], int); i1 = np.array([i for i, s in allt if s == 1], int)
        trajs = [(i, 0) for i in i0] + [(i, 1) for i in i1]; n0 = len(i0)
        gi0 = torch.as_tensor(i0, device=DEV); gi1 = torch.as_tensor(i1, device=DEV)      # 座席0の軌道を先に並べる(一括で取り出すため)
        buf = {k: [] for k in KEYS + ("v",) + (OKEYS if ANTI else ())}; tm = dict(env=0.0, samp=0.0, opp=0.0, step=0.0); mny = []
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
            outs = [P.sample(net, tuple(x[j:j + _ch] for x in stL), crit=None if _fastcrit else (crit if CRITIC else None)) for j in range(0, len(trajs), _ch)]
            ul, uq, ms, mq, v = [np.concatenate([o[k] for o in outs]) for k in range(5)]
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
            for kind, obj in (("best", S.PolicyOpp(best, greedy=GROUP > 1, cap=True)), ("league", S.PolicyOpp(lg_net, greedy=GROUP > 1, cap=True) if lg_net else None), ("tape", tape)):
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
            tm["opp"] += time.time() - a_; a_ = time.time()
            pool.step(U[0], O[0], U[1], O[1]); tm["step"] += time.time() - a_
            if SHAPE > 0: mny.append(pool.money().copy())   # 形勢ポテンシャル用(1手 0.1ms 程度)
        fin = pool.money(); tr = time.time() - t0
        # 報酬は最後の手の勝ち負けだけ(+1 / −1、引き分け 0)
        z = np.array([np.sign(fin[i, s] - fin[i, 1 - s]) for i, s in trajs])
        V = np.stack(buf["v"]); T_ = V.shape[0]; Nn = V.shape[1]
        if not CRITIC:
            kinds = np.array([games[i]["kind"] for i, _ in trajs]); bl = np.zeros(Nn, np.float32)
            for k in set(kinds): bl[kinds == k] = z[kinds == k].mean()     # 相手の種類ごとの平均勝敗
            V = np.broadcast_to(bl.reshape(1, -1), V.shape).astype(np.float32).copy()
        Rw = np.zeros_like(V); Rw[-1] = z
        own_c = OWN_R0 * (max(0.0, 1.0 - (it - start) / OWN_ANNEAL) if OWN_ANNEAL > 0 else 1.0)
        if own_c > 0:   # exp294: 一から学ぶとき用。自分の最終の所持金(対数)を終端に足す(勝敗だけでは乱数どうしが 0 対 0 の引き分けで手がかりが無い)
            own = np.array([np.log1p(max(fin[i, s_], 0.0)) / np.log1p(1e5) for i, s_ in trajs], np.float32)
            Rw[-1] = Rw[-1] + own_c * own
        if SHAPE > 0:
            # exp209: ポテンシャルに基づく報酬の付け替え(Ng+ 1999)。F_t = Φ(s_{t+1}) − Φ(s_t)、終端では Φ=0 とする。
            #   γ=1 なので足し合わせると z − Φ(s_0) になり、定数ぶんずれるだけで**最適な方策は変わらない**。
            #   720手の勝負で終端の ±1 だけでは信用割当が遠すぎるので、毎手「所持金の差」で形勢を渡す。
            MNY = np.stack(mny)                                 # (手, 局, 2) 各手の所持金(M はモデルの別名なので使わない)
            Phi = np.zeros((T_, Nn), np.float32)
            for j, (i, sseat) in enumerate(trajs):
                # 所持金の「差」をそのまま使うと序盤の投資(土地・家畜)を罰する。
                # 実際この競技は序盤に所持金で負けているほうが勝つ(実戦例: 第1日 $27 対 $144 → 最終 166k 対 35k)。
                # そこで尺度に依らない対数比を使い、さらに終盤ほど重くする(所持金が実力に一致するのは終盤だけ)。
                a = np.log1p(np.maximum(MNY[:, i, sseat], 0)); b = np.log1p(np.maximum(MNY[:, i, 1 - sseat], 0))
                w = (np.arange(T_, dtype=np.float32) / max(T_ - 1, 1)) ** 2   # 序盤ほぼ0、終盤で1
                Phi[:, j] = w * np.tanh((a - b) / SHAPE)
            Rw[:-1] += Phi[1:] - Phi[:-1]                        # 途中の手: 形勢がどれだけ良くなったか
            Rw[-1] += -Phi[-1]                                   # 終端では Φ=0(こうしないと最適方策が変わる)
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
        ret = adv + V
        byk = {}
        for (i, s), zz in zip(trajs, z): byk.setdefault(games[i]["kind"], []).append(zz)
        if lg_path is not None and "league" in byk:        # PFSP の勝率(指数移動平均)
            pfsp_wr[lg_path] = 0.7 * pfsp_wr.get(lg_path, 0.5) + 0.3 * float((np.array(byk["league"]) > 0).mean())
        def flat(k, dt):
            b = buf[k]
            return (torch.cat(b).to(dt) if torch.is_tensor(b[0]) else torch.as_tensor(np.concatenate(b), device=DEV, dtype=dt))
        D = dict(tiles=flat("tiles", torch.float16), ou=flat("ou", torch.float32), pu=flat("pu", torch.float32), g=flat("g", torch.float32),
                 lm=flat("lm", torch.uint8), ul=flat("ul", torch.long), uq=flat("uq", torch.long), ms=flat("ms", torch.long), mq=flat("mq", torch.long))
        if MKTNF: D["mkt"] = flat("mkt", torch.float32)        # exp270: 市場の先読み特徴
        if ANTI:
            DO = dict(otiles=flat("otiles", torch.float16), oou=flat("oou", torch.float32), opu=flat("opu", torch.float32), og=flat("og", torch.float32))
        del buf
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
        MBL = MB // WORLD                                 # exp292: 1 GPU あたりのミニバッチ(全体では MB)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            PER0 = torch.cat([P.evaluate(net, [D[k][i:i + MBL] for k in KEYS])[4].float() for i in range(0, n, MBL)])
        stats = []; vonly = CRITIC and it < vw_until; tu = time.time(); safe_skip = 0
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, MBL):
                idx = perm[i:i + MBL]; b = [D[k][idx] for k in KEYS]
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
                    opt_c.zero_grad(); scaler_c.scale(vl).backward(); _sync_grads(cparams); scaler_c.unscale_(opt_c)
                    torch.nn.utils.clip_grad_norm_(cparams, 0.5); scaler_c.step(opt_c); scaler_c.update()
                    idx, b = idx_, b_
                if vonly: continue
                with torch.autocast("cuda", dtype=torch.float16):
                    lp, ent, v, dist, per, pmask = P.evaluate(net, b)
                ent, per = ent.float(), per.float(); dist = tuple(x.float() if x.dtype.is_floating_point else x for x in dist)
                rr = torch.exp((per - PER0[idx]).clamp(-20, 20)); aa = A[idx].unsqueeze(1)
                pl = -((torch.min(rr * aa, rr.clamp(1 - CLIP, 1 + CLIP) * aa) * pmask).sum(1) / pmask.sum(1).clamp(min=1)).mean()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    rd = P.evaluate(ref, b)[3]
                rd = tuple(x.float() if x.dtype.is_floating_point else x for x in rd)
                kl = P.kl_to_ref(dist, rd).mean()
                loss = pl + BETA * kl - ENT * ent.mean()
                if SAFE_KL > 0:
                    with torch.no_grad():
                        lr_ = (per - PER0[idx]).clamp(-20, 20); akl = (((lr_.exp() - 1) - lr_) * pmask).sum() / pmask.sum().clamp(min=1)
                    if float(akl) > SAFE_KL: safe_skip += 1; continue
                opt.zero_grad(); scaler.scale(loss).backward(); _sync_grads(list(net.parameters())); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); scaler.step(opt); scaler.update()
                stats.append((float(pl), float(kl), float(ent.mean()), float(((rr - 1).abs() > CLIP)[pmask.bool()].float().mean())))
        del D; tu = time.time() - tu
        if SAFE_KL > 0 and safe_skip: log("it%4d [安全装置] 近似KL > %.3f のミニバッチ %d 個を捨てた" % (it, SAFE_KL, safe_skip))
        if ANTI: del DO
        # 価値EV は ret = adv + V なので ret - V = adv、優位度が小さいほど 1 に近づく循環した指標だった。
        # 本当に見たいのは「勝敗 z を V がどれだけ当てているか」なので、z に対する説明率も出す。
        Z = np.broadcast_to(z.reshape(1, -1), V.shape)
        ev_ = 1 - np.var(Z.reshape(-1) - V.reshape(-1)) / (np.var(Z.reshape(-1)) + 1e-8)
        advstd_ = float(np.std(adv.reshape(-1))) if GROUP <= 1 else grpo_spread; s_ = np.mean(stats, 0) if stats else [0, 0, 0, 0]
        log("it%4d " % it + " ".join("%s %.2f(%d)" % (k, (np.array(v) > 0).mean(), len(v)) for k, v in byk.items()) +
            " | KL %.3f ent %.2f 切 %.2f 価値EV(対 勝敗) %.2f %s %.3f %s| 局面 %d 収集 %.0fs(環境 %.0f 手選び %.0f 相手 %.0f 進める %.0f)更新 %.0fs 全体 %.0fs → %.2f百万局面/時" % (
            s_[1], s_[2], s_[3], ev_, "組内の報酬SD" if GROUP > 1 else "優位度SD", advstd_, "価値のみ " if vonly else "", n * WORLD, tr, tm["env"], tm["samp"], tm["opp"], tm["step"], tu, time.time() - t0, n * WORLD / (time.time() - t0) * 3600 / 1e6) + (" | 同期 %.0fs" % SYNC_T[0] if WORLD > 1 else ""))
        SYNC_T[0] = 0.0
        if ema is not None and not vonly:               # 重みの移動平均(学習は現行 net のまま、挑戦者は平均)
            with torch.no_grad():
                for pe, pn in zip(ema.state_dict().values(), net.state_dict().values()):
                    if pe.dtype.is_floating_point: pe.mul_(EMA).add_(pn, alpha=1 - EMA)
                    else: pe.copy_(pn)
        chal = ema if ema is not None else net
        # 関門: 現行(EMA なら平均の重み)対 最良版(貪欲・種の上限 = 提出と同じ打ち方、64局)
        if it % GATE == GATE - 1 and not vonly:
            if RANK == 0: w, m = h2h(chal, best, GATESEEDS)
            else: w, m = 0.0, 0.0
            if WORLD > 1:
                wm_ = torch.tensor([w, m], device="cuda", dtype=torch.float64); tdist.broadcast(wm_, 0); w, m = float(wm_[0]), float(wm_[1])
            if w >= GATE_WR:
                gen += 1; p = "%s/orbit_%s_gen%d.pt" % (L, tag, gen)
                if RANK == 0: torch.save(chal.state_dict(), p + ".tmp"); os.replace(p + ".tmp", p)   # exp292: 書きかけを読まれないように
                if WORLD > 1: tdist.barrier()                          # exp292: 全 GPU が新しい世代のファイルを読めるようになってから進む
                best.load_state_dict(chal.state_dict()); ref.load_state_dict(chal.state_dict()); league = (league + [p])[-8:]
                log("it%4d [関門] 現行 対 最良版 64局 勝率 %.3f 差 %+.0f → **置き換え(第%d世代)**" % (it, w, m, gen))
            else:
                log("it%4d [関門] 現行 対 最良版 64局 勝率 %.3f 差 %+.0f → 据え置き(第%d世代のまま)" % (it, w, m, gen))
        if SAVE_EVERY and it % SAVE_EVERY == SAVE_EVERY - 1:
            torch.save(net.state_dict(), "%s/orbit_%s_it%d.pt" % (L, tag, it + 1))
        if RANK == 0: torch.save(dict(net=net.state_dict(), crit=crit.state_dict(), opt=opt.state_dict(), opt_c=opt_c.state_dict(), it=it,
                        best=best.state_dict(), gen=gen, league=league, vw_until=vw_until, pfsp_wr=pfsp_wr, **({"ema": ema.state_dict()} if ema is not None else {}), **({"hlh": hlh.state_dict()} if hlh is not None else {})), ck)
        if it % EVAL == EVAL - 1 and RANK == 0:
            w, m = vs_tape(chal); log("it%4d [ものさし 対tape_ug 128局 貪欲] 勝率 %.3f 差 %+.0f(最良版は第%d世代)%s" % (it, w, m, gen, "(EMA の重み)" if ema is not None else ""))
            me, wr, mg = P.eval_policy(chal, hold_e, True, 2, cap=True); log("it%4d [58局 凍った相手 貪欲] 自分 %.0f 勝率 %.3f 差 %.0f" % (it, me, wr, mg))


if __name__ == "__main__":
    main()
