"""2026-09-24(rule-variant curriculum): kagsim_var が「つまみを触らなければ元の kagsim とビット一致」かを確かめる。

同じ種・同じ行動で元の kagsim.VecMarket と kagsim_var.VecMarket を並べて 719手進め、
**毎手・両座席の観測の全部**(obs_legal, farm_pub, tiles_base, tiles_dyn, unit_pos2, unit_inv, shed_all,
legal_mask の版 0/2, shops, money, mkt_aux, obs, unit_obs)をバイト単位で比べる。
行動は合法手からの乱択(ユニット)と、雇用・区画・種・家畜・小麦/肥料・売りを混ぜた乱択の注文(市場)。
numpy だけで動く(torch 不要)ので、ログインノードの python3.10 でも、実際にジョブが使う
kagsim.cpython-310 と比べられる。

  REF=<元の kagsim の .so がある場所> VAR=<kagsim_var の .so がある場所> python3 test_var_equiv.py [局数] [束の数]
後半で変種(開始区画・遅い開始・値段・摂動)が本当に効いているかの簡単な確認もする。
"""
import os, sys, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FAST = os.path.join(os.path.dirname(HERE), "fast")
REF = os.environ.get("REF", FAST); VAR = os.environ.get("VAR", FAST)
sys.path.insert(0, REF); import kagsim; sys.path.pop(0)
sys.path.insert(0, VAR); import kagsim_var; sys.path.pop(0)
sys.path.insert(0, FAST); import kagpack; sys.path.pop(0)
print("元:", kagsim.__file__, "\n写し:", kagsim_var.__file__, "\nPython", sys.version.split()[0], flush=True)
NU = 20


def wide_blank(n=720, nh=NU - 1):
    return kagpack.pack([{"farmer": ["PASS"], "hands": [["PASS"]] * nh, "market": []} for _ in range(n)])


# sim.hpp の Op / MOp / 候補(kagsim.cpp cand_of と同じ並び)
BASE = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6]   # PASS N S E W WATER HARVEST FERT DIG COOP PASTURE FEED COLLECT CARE DROP
OP_PLANT, OP_PICKUP, OP_PLACE = 8, 5, 7
M_HIRE, M_LAND, M_SEED, M_PROD, M_ANIMAL, M_SELL = 1, 2, 3, 4, 5, 6


def cand(c, n=1):
    if c < 15: return (BASE[c], 0, 1)
    if c < 20: return (OP_PLANT, c - 15, 1)
    if c < 32: return (OP_PICKUP, c - 20, n)
    return (OP_PLACE, c - 32, n)


def rand_actions(rng, lm, money, t):
    """lm (N,20,44) -> U (N,20,3)。ユニットは合法手から乱択(生産の手を優先、移動と待機は少なめ)、
    市場は所持金を見て乱択した注文列 (N,10,3)。所持金が 0 に張り付くと市場の枝を通らないので、支出を所持金の範囲に抑える。"""
    N = lm.shape[0]
    U = np.zeros((N, NU, 3), np.int32); U[:, :, 0] = 255
    O = np.zeros((N, 10, 3), np.int32)
    for i in range(N):
        for u in range(NU):
            c = np.flatnonzero(lm[i, u])
            if len(c) == 0: continue
            prod = c[c >= 5]
            if len(prod) and rng.random() < 0.75: c = prod
            elif len(c) > 1 and rng.random() < 0.8: c = c[c != 0]
            U[i, u] = cand(int(rng.choice(c)), int(rng.integers(1, 6)))
        k = 0; mo = float(money[i])
        for _ in range(int(rng.integers(0, 7))):
            r = rng.random()
            if r < 0.10 and mo > 300: O[i, k] = (M_HIRE, 0, 0); mo -= 50
            elif r < 0.14 and mo > 2500: O[i, k] = (M_LAND, 0, 0); mo -= 2000
            elif r < 0.36 and mo > 200: q = int(rng.integers(1, 6)); O[i, k] = (M_SEED, int(rng.integers(0, 5)), q); mo -= 60 * q
            elif r < 0.44 and mo > 1500: O[i, k] = (M_ANIMAL, int(rng.integers(9, 12)), 1); mo -= 500
            elif r < 0.50 and mo > 500: O[i, k] = (M_PROD, int(rng.choice([0, 8])), int(rng.integers(1, 10))); mo -= 300
            elif r < 0.54: O[i, k] = (254, 0, 0)                                   # 空だが枠を占める
            else: O[i, k] = (M_SELL, int(rng.integers(0, 9)), int(rng.integers(1, 40)))
            k += 1
        if k < 10 and rng.random() < 0.3: O[i, k] = (255, 0, 0)                      # 終端
    return U, O


def views(e, legal_opts):
    """元と写しで同じ呼び出しをして、全部の観測をバイト列に"""
    out = []
    for s in (0, 1):
        e.set_view(s)
        for f in (lambda: e.obs_legal(), lambda: e.farm_pub(0), lambda: e.farm_pub(1), lambda: e.tiles_base(0), lambda: e.tiles_base(1),
                  lambda: e.tiles_dyn(0), lambda: e.tiles_dyn(1), lambda: e.unit_pos2(0), lambda: e.unit_pos2(1), lambda: e.unit_inv(NU),
                  lambda: e.shed_all(), lambda: e.shops(), lambda: e.money(), lambda: e.mkt_aux(0), lambda: e.mkt_aux(1),
                  lambda: e.obs(), lambda: e.seeds_held(), lambda: e.legal_mask(NU)):
            out.append(np.ascontiguousarray(np.asarray(f())))
        for lo in legal_opts:
            out.append(np.ascontiguousarray(np.asarray(e.legal_mask(NU, lo))))
    return out


def run_pair(seeds, rng_seed, T=719, full_every=1, unit_obs_every=97):
    w = wide_blank()
    a = kagsim.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); a.keep_nonsell = False; a.reset()
    b = kagsim_var.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); b.keep_nonsell = False; b.reset()
    N = len(seeds); rng = np.random.default_rng(rng_seed)
    lo_mix = np.array([(0, 2, -1, 3)[k % 4] for k in range(N)], np.int32)
    bad = 0; ncmp = 0; h = hashlib.sha1()
    for t in range(T):
        if t % full_every == 0:
            va, vb = views(a, [lo_mix]), views(b, [lo_mix])
            for x, y in zip(va, vb):
                ncmp += 1
                if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes(): bad += 1
                h.update(x.tobytes())
            if t % unit_obs_every == 0:
                for s in (0, 1):
                    a.set_view(s); b.set_view(s); ncmp += 1
                    if np.asarray(a.unit_obs(NU)).tobytes() != np.asarray(b.unit_obs(NU)).tobytes(): bad += 1
        acts = []
        for s in (0, 1):
            a.set_view(s); lm = np.asarray(a.legal_mask(NU)); mo = np.asarray(a.money())[:, 0]
            acts.append(rand_actions(rng, lm, mo, t))
        a.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1]); b.step_both(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
    a.set_view(0); b.set_view(0)
    ma, mb = np.asarray(a.money()), np.asarray(b.money())
    return bad, ncmp, bool(np.array_equal(ma, mb)), ma, h.hexdigest()


def check_equiv(n_games=64, n_batches=4):
    t0 = time.time(); tot_bad = 0; tot_cmp = 0; allm = []
    for k in range(n_batches):
        seeds = [int(x) for x in np.random.default_rng(1000 + k).integers(0, 10 ** 9, n_games)]
        bad, ncmp, same, m, hx = run_pair(seeds, 77 + k)
        tot_bad += bad; tot_cmp += ncmp; allm.append(m)
        print("  束%d: %d局 x 719手 比較 %d 個 不一致 %d 最終所持金一致 %s 所持金 平均 %.0f 最大 %.0f 観測ハッシュ %s  (%.0fs)" % (
            k, n_games, ncmp, bad, same, m.mean(), m.max(), hx[:12], time.time() - t0), flush=True)
        if not same: tot_bad += 1
    allm = np.concatenate(allm)
    print("等価性: %d局 x 719手 x 両座席、観測 %d 個の比較で不一致 %d 件 → %s" % (n_games * n_batches, tot_cmp, tot_bad, "ビット一致" if tot_bad == 0 else "**不一致あり**"))
    print("  (乱択の行動でも経済が動いているか: 最終所持金 平均 %.0f・0 以下 %.2f・10,000 超 %.2f)" % (allm.mean(), (allm <= 0).mean(), (allm > 10000).mean()))
    return tot_bad == 0


def check_variants():
    """変種が本当に効くか(ビット一致の検査とは別に、つまみを入れたときの動き)"""
    w = wide_blank(); seeds = [11, 12, 13, 14, 15, 16]
    e = kagsim_var.VecMarket(w[0], w[1], w[0], w[1], seeds, 720, 0); e.keep_nonsell = False; e.reset()
    e.set_variant(1, start_money=8000, start_quads=4)
    e.set_variant(2, start_quads=3, start_step=360)
    e.set_variant(3, land_pct=50, hire_pct=50)
    e.set_variant(4, start_step=240, start_money=6000, start_quads=2)
    vi = np.asarray(e.variant_info()); fp = np.asarray(e.farm_pub(0)); ob = np.asarray(e.obs_legal()); sh = np.asarray(e.shops())
    ok = True
    def chk(c, msg):
        nonlocal ok
        print("  [%s] %s" % ("OK" if c else "NG", msg)); ok &= bool(c)
    chk(vi[0, 1:6].tolist() == [3000, 1, 0, 100, 100], "局0 は既定のまま %s" % vi[0].tolist())
    chk(abs(fp[1, 0] * 50000 - 8000) < 1e-3 and round(fp[1, 2] * 4) == 4, "局1: 開始資金 8000・4区画 (farm_pub %s)" % fp[1, :3].round(3).tolist())
    chk(round(ob[2, 34] * 720) == 360 and round(ob[2, 36] * 30) == 15 and (sh[2] >= 0).sum() == 5, "局2: 手360・日15 から、店は日3,6,9,12,15 の5つを先に開く (店 %s)" % sh[2][:6].tolist())
    kinds = np.asarray(e.tiles_base(0))[1, :, :, 0] * 5
    chk(int((np.rint(kinds) == 1).sum()) == 0, "局1: 施錠マスが無い")
    # 手を進める: 遅い開始の局は cur が追いつくまで止まっている
    U = np.full((len(seeds), NU, 3), 255, np.int32); O = np.zeros((len(seeds), 10, 3), np.int32)
    O[3, 0] = (M_LAND, 0, 0); O[3, 1] = (M_HIRE, 0, 0); O[3, 2] = (M_HIRE, 0, 0); O[3, 3] = (M_HIRE, 0, 0)
    O[0, 0] = (M_LAND, 0, 0); O[0, 1] = (M_HIRE, 0, 0); O[0, 2] = (M_HIRE, 0, 0); O[0, 3] = (M_HIRE, 0, 0)
    e.step_both(U, O, U, np.zeros_like(O)); e.set_view(0); m = np.asarray(e.money())
    chk(m[0, 0] == 3000 - 1000 - 1 - 1 - 2 and m[3, 0] == 3000 - 500 - 0 - 0 - 1, "区画・雇用の値段: 局0 %.0f(公式 1000+1+1+2)・局3 %.0f(半額 500+0+0+1)" % (m[0, 0], m[3, 0]))
    st = e.sim_steps(); chk(st[2] == 360 and st[4] == 240 and st[0] == 1, "遅い開始の局は止まっている: 手 %s" % st)
    O2 = np.zeros_like(O)
    for t in range(1, 241): e.step_both(U, O2, U, O2)
    st = e.sim_steps(); chk(st[4] == 241 and st[2] == 360 and st[0] == 241, "手241: 局4(手240開始)が1手進んだ、局2 はまだ: %s" % st)
    # 摂動
    e.add_money(5, 0, 1234.0); e.unlock_quad(5, 1); e.add_seeds(5, 0, 4, 3); e.add_shed(5, 0, 10, 2); p0 = np.asarray(e.obs_legal())[5, 9:18].copy()
    e.market_add(5, 3, -400); p1 = np.asarray(e.obs_legal())[5, 9:18]
    e.set_view(0); m = np.asarray(e.money()); fp1 = np.asarray(e.farm_pub(1)); sd = np.asarray(e.seeds_held()); sa = np.asarray(e.shed_all())
    chk(m[5, 0] == 3000 + 1234 and round(fp1[5, 2] * 4) == 2 and sd[5, 4] == 3 and sa[5, 10] == 2 and p1[3] > p0[3],
        "摂動: 所持金 %.0f・相手の区画 %d・メロンの種 %d・納屋の牛 %d・イチゴの値 %.0f → %.0f" % (m[5, 0], round(fp1[5, 2] * 4), sd[5, 4], sa[5, 10], p0[3] * 250, p1[3] * 250))
    for t in range(241, 719): e.step_both(U, O2, U, O2)
    e.set_view(0); m = np.asarray(e.money()); st = e.sim_steps()
    chk(all(x == 719 for x in st), "全局 手719 で終わる(遅い開始も同じ時刻に終わる): %s" % st)
    print("変種の動作確認:", "OK" if ok else "**NG**")
    return ok


if __name__ == "__main__":
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 64; nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ok1 = check_equiv(ng, nb)
    ok2 = check_variants()
    sys.exit(0 if (ok1 and ok2) else 1)
