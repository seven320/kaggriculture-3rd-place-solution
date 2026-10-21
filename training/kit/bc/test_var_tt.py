"""2026-09-25(rule-variant curriculum §10 tomtown): kagsim_var の tt(店の抽選をピザ屋/ファーマーズマーケットへ偏らせる)の試験。

  (A) つまみなし・tt=0 を明示した局が元の kagsim とビット一致(test_var_equiv.run_pair と同じ比較 + set_variant(tt_permille=0))
  (B) 乱択の行動で tt の頻度: 日9(店3つ)・日12(店4つ)の時点でピザ+市場が 3以上の局の割合。tt なしの基準と比べる。
      tt の局でも、最初の店が開く前(手 0〜71)の観測は基準とビット一致(店の抽選以外は変えていない)
  (C) apo_var の var_draw(VAR_MIX=tt:1)→ sp_fast_var.BatchSPPool.set_variants → 方策(既定 pn2p gen45)に T=1 で打たせて同じ割合
      (C は torch が要る。PPODEV=cpu。TB 手で打ち切る)

  REF=<元の kagsim の .so の場所> python3 test_var_tt.py [A の局数] [B の局数] [C の局数]
"""
import os, sys, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FAST = os.path.join(os.path.dirname(HERE), "fast")
sys.path.insert(0, HERE)
import test_var_equiv as TE   # 元の kagsim(REF)と kagsim_var(VAR)を掴む
kagsim, kagsim_var = TE.kagsim, TE.kagsim_var
PIZZA, MARKET = 5, 2          # sim_var.hpp の ShopId(sorted(SHOPS) の順): FARMERS_MARKET=2, PIZZA_SHOP=5
TTP = int(os.environ.get("TTP", "700")); TTN = int(os.environ.get("TTN", "4"))


def pm_count(sh, k):
    """最初の k 店のうちピザ+市場の数"""
    s = sh[:, :k]
    return ((s == PIZZA) | (s == MARKET)).sum(1)


def check_identity(n_games, n_batches=2):
    ok = TE.check_equiv(n_games, n_batches)
    # tt_permille=0 を明示した set_variant(= apo_var で tt を選ばなかった局と同じ経路ではないが、つまみ 0 の Sim の作り直し)も一致
    w = TE.wide_blank(); seeds = [int(x) for x in np.random.default_rng(99).integers(0, 10 ** 9, n_games)]
    a = kagsim.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); a.keep_nonsell = False; a.reset()
    b = kagsim_var.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); b.keep_nonsell = False; b.reset()
    for i in range(n_games): b.set_variant(i, tt_permille=0, tt_nmax=4)
    rng = np.random.default_rng(5); bad = 0
    for t in range(719):
        if t % 24 == 0:
            for x, y in zip(TE.views(a, []), TE.views(b, [])): bad += x.tobytes() != y.tobytes()
        acts = []
        for s in (0, 1):
            a.set_view(s); acts.append(TE.rand_actions(rng, np.asarray(a.legal_mask(TE.NU)), np.asarray(a.money())[:, 0], t))
        a.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1]); b.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    a.set_view(0); b.set_view(0); same = np.array_equal(np.asarray(a.money()), np.asarray(b.money()))
    print("(A2) set_variant(tt_permille=0) を明示した %d局: 日ごとの観測の不一致 %d・最終所持金一致 %s" % (n_games, bad, same), flush=True)
    return ok and bad == 0 and same


def run_freq(n_games, ttp, ttn, T=300, seed0=31337):
    w = TE.wide_blank(); seeds = [int(x) for x in np.random.default_rng(seed0).integers(0, 10 ** 9, n_games)]
    a = kagsim_var.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); a.keep_nonsell = False; a.reset()   # 基準(つまみなし)
    b = kagsim_var.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); b.keep_nonsell = False; b.reset()
    for i in range(n_games): b.set_variant(i, tt_permille=ttp, tt_nmax=ttn)
    rng = np.random.default_rng(6); pre_bad = 0; sh9 = sh12 = None
    for t in range(T):
        if t < 72:   # 最初の店(日3)が開く前: tt の局も観測は基準と同じはず
            for x, y in zip(TE.views(a, []), TE.views(b, [])): pre_bad += x.tobytes() != y.tobytes()
        if t == 24 * 9: a.set_view(0); b.set_view(0); sh9 = (np.asarray(a.shops()).copy(), np.asarray(b.shops()).copy())
        if t == 24 * 12: sh12 = (np.asarray(a.shops()).copy(), np.asarray(b.shops()).copy())
        acts = []
        for s in (0, 1):
            a.set_view(s); acts.append(TE.rand_actions(rng, np.asarray(a.legal_mask(TE.NU)), np.asarray(a.money())[:, 0], t))
        a.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1]); b.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    b.set_view(0); shT = np.asarray(b.shops())
    r = {}
    for nm, (x, y) in (("日9", sh9), ("日12", sh12)):
        k = 3 if nm == "日9" else 4
        assert (x[:, :k] >= 0).all() and (y[:, :k] >= 0).all() and (x[:, k] < 0).all(), "日9 で店3つ・日12 で店4つのはず"
        r[nm] = (float((pm_count(x, k) >= 3).mean()), float((pm_count(y, k) >= 3).mean()), float(pm_count(x, k).mean()), float(pm_count(y, k).mean()))
    n_late = (shT[:, :12] >= 0).sum(1).min()
    late_pm = float(((shT[:, ttn:n_late] == PIZZA) | (shT[:, ttn:n_late] == MARKET)).mean()) if n_late > ttn else float("nan")
    print("(B) 乱択 %d局 tt=%d‰ 最初 %d店: 観測(手0〜71)の不一致 %d" % (n_games, ttp, ttn, pre_bad))
    for nm, (p0, p1, m0, m1) in r.items():
        print("    %s: ピザ+市場 ≥3 の局 基準 %.3f → tt %.3f(平均 %.2f → %.2f 店)" % (nm, p0, p1, m0, m1))
    if late_pm == late_pm: print("    %d店目以降(手%d まで)でピザ/市場の割合 %.3f(公式の一様 0.25)" % (ttn + 1, T, late_pm), flush=True)
    return pre_bad == 0, r


def run_policy(n_games, TB):
    os.environ.setdefault("PPODEV", "cpu"); os.environ.setdefault("FASTINF", "1")
    sys.path.append(FAST)
    import torch, ppo_tf as P, model_tf as M, fast_codes, sp_fast_var
    P.to_codes = fast_codes.to_codes_fast
    torch.set_num_threads(int(os.environ.get("NTH", "3")))
    W = os.environ.get("W", "/workspace/kaggriculture/local_logs/orbit_pn2p_gen45.pt")
    net = M.load(W, "cpu").eval()
    src = open(os.path.join(HERE, "apo_var.py")).read()     # apo_var の抽選部分だけを実行(test_var_play.py と同じ)
    s0 = src.index("# ==================================================================== rule-variant curriculum"); s1 = src.index("\ndef main():")
    os.environ["VAR_MIX"] = "tt:1"; os.environ["VAR_TT_P"] = str(TTP / 1000.0); os.environ["VAR_TT_N"] = str(TTN)
    ns = dict(os=os, np=np, torch=torch, _allsum=lambda t: t); exec(src[s0:s1], ns)
    rng = random.Random(11); seeds = [int(x) for x in np.random.default_rng(424242).integers(0, 10 ** 9, n_games)]
    # apo_var の対局係と同じ抽選(VAR_P=1, kind=self)
    specs = []; codes = []
    for i in range(n_games):
        c, sp, t0 = ns["var_draw"](rng); codes.append(c); specs.append(sp); assert t0 == 0
    pool = sp_fast_var.BatchSPPool(seeds); pool.set_variants(specs)
    base = sp_fast_var.BatchSPPool(seeds)
    vi = np.asarray(pool.env.variant_info())
    torch.manual_seed(0); t_ = time.time(); sh9 = sh12 = None; sh9b = sh12b = None
    for t in range(TB):
        if t == 24 * 9: sh9 = np.asarray(pool.shops()).copy(); sh9b = np.asarray(base.shops()).copy()
        if t == 24 * 12: sh12 = np.asarray(pool.shops()).copy(); sh12b = np.asarray(base.shops()).copy()
        for pl in (pool, base):
            acts = []
            for s in (0, 1):
                st = pl.state(s)
                with torch.no_grad(): ul, uq, ms, mq, _ = P.sample(net, st, greedy=False, cap=True)
                acts.append(P.to_codes(ul, uq, ms, mq, st[1]))
            pl.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
        if t % 60 == 0: print("   手%d (%.0fs)" % (t, time.time() - t_), flush=True)
    m = pool.money(); mb = base.money()
    print("(C) %s に打たせた %d局(var_draw → set_variants、変種の番号 %s、variant_info の tt 列 %s)" % (os.path.basename(W), n_games, sorted(set(codes)), sorted(set(vi[:, 7].tolist()))))
    print("    日9: ピザ+市場 ≥3 基準 %.3f → tt %.3f | 日12: 基準 %.3f → tt %.3f" % (
        (pm_count(sh9b, 3) >= 3).mean(), (pm_count(sh9, 3) >= 3).mean(), (pm_count(sh12b, 4) >= 3).mean(), (pm_count(sh12, 4) >= 3).mean()))
    print("    手%d の所持金 平均 基準 %.0f / tt %.0f(方策がまともに動いているか)" % (TB, mb.mean(), m.mean()), flush=True)
    return float((pm_count(sh12, 4) >= 3).mean())


if __name__ == "__main__":
    na = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ok = True
    if na > 0: ok &= check_identity(na)
    if nb > 0:
        pre_ok, r = run_freq(nb, TTP, TTN); ok &= pre_ok
        ok &= r["日12"][1] >= 0.70
    if nc > 0: ok &= run_policy(nc, int(os.environ.get("TB", "290"))) >= 0.70
    print("tomtown 試験:", "OK" if ok else "**NG**")
    sys.exit(0 if ok else 1)
