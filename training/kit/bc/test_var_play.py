"""2026-09-24(rule-variant curriculum): 方策(既定 gen22)に打たせて
  (A) sp_fast.BatchSPPool(元の kagsim)と sp_fast_var.BatchSPPool(kagsim_var、つまみなし)が
      毎手・両座席の特徴5種(tiles/ou/pu/g/lm)と最終所持金でビット一致するか
      (両方に同じ行動を渡す。行動は元の側の観測から T=1 の標本で選ぶ = 学習の採取と同じ経済)
  (B) 変種の局(rs/ls/pr/rule)を同じ方策で打ち、方策がまともに動くか(所持金・区画・作付け・家畜)
を CPU で確かめる。ログインノード(Python 3.12 + torch)用。
  PPODEV=cpu REF=<元の kagsim の .so の場所> python3 test_var_play.py [局数] [重み]
"""
import os, sys, time, random
os.environ.setdefault("PPODEV", "cpu"); os.environ.setdefault("FASTINF", "1")   # FASTINF: 種の上限つきの選択をまとめて計算(結果は逐次版と同じ)。CPU で約2倍速い
HERE = os.path.dirname(os.path.abspath(__file__)); FAST = os.path.join(os.path.dirname(HERE), "fast")
REF = os.environ.get("REF", FAST)
sys.path.insert(0, REF); import kagsim; sys.path.pop(0)          # 元のエンジン(今の kagsim.cpp から作ったもの)を先に掴む
sys.path.insert(0, HERE); sys.path.append(FAST)
import numpy as np, torch
import ppo_tf as P, model_tf as M
import fast_codes; P.to_codes = fast_codes.to_codes_fast
import sp_fast, sp_fast_var
assert sp_fast.kagsim is kagsim and sp_fast_var.kagsim.__name__ == "kagsim_var"
torch.set_num_threads(int(os.environ.get("NTH", "3")))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
W = sys.argv[2] if len(sys.argv) > 2 else "/workspace/kaggriculture/local_logs/orbit_g5g4_gen22.pt"
net = M.load(W, "cpu").eval()
print("重み %s / 元 %s / 写し %s" % (os.path.basename(W), kagsim.__file__, sp_fast_var.kagsim.__file__), flush=True)


def act(st, greedy=False):
    with torch.no_grad():
        ul, uq, ms, mq, _ = P.sample(net, st, greedy=greedy, cap=True)
    return P.to_codes(ul, uq, ms, mq, st[1])


def check_same(seed0=4242001, T=719):
    seeds = [seed0 + 7919 * k for k in range(N)]
    a, b = sp_fast.BatchSPPool(seeds), sp_fast_var.BatchSPPool(seeds)
    torch.manual_seed(0); bad = 0; t0 = time.time()
    for t in range(T):
        acts = []
        for s in (0, 1):
            sa, sb = a.state(s), b.state(s)
            for k in range(5):
                if not (sa[k].dtype == sb[k].dtype and np.array_equal(sa[k], sb[k])): bad += 1
            acts.append(act(sa))
        a.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1]); b.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
        if t % 120 == 0: print("   手%d 不一致 %d (%.0fs)" % (t, bad, time.time() - t0), flush=True)
    ma, mb = a.money(), b.money()
    print("(A) 方策の対局 %d局 x 719手 x 両座席: 特徴5種の不一致 %d 件 / 最終所持金一致 %s / 所持金 平均 %.0f 最小 %.0f 最大 %.0f" % (
        N, bad, np.array_equal(ma, mb), ma.mean(), ma.min(), ma.max()), flush=True)
    return bad == 0 and np.array_equal(ma, mb)


def play_variants(seed0=5151001, T=719):
    """1つの pool に 通常/rs/ls/pr/rule を N 局ずつ混ぜて打つ(apo_var の対局係と同じ形)。両座席とも同じ方策(T=1 の標本)"""
    rng = random.Random(7)
    # apo_var の抽選関数をそのまま使う(import すると OPPDIR が要るので、定義部分だけ実行)
    src = open(os.path.join(HERE, "apo_var.py")).read()
    s0 = src.index("# ==================================================================== rule-variant curriculum"); s1 = src.index("\ndef main():")
    ns = dict(os=os, np=np, torch=torch, _allsum=lambda t: t); exec(src[s0:s1], ns)
    names = ["通常", "rs", "ls", "pr", "rule"]; G = len(names) * N
    seeds = [seed0 + 7919 * j for j in range(G)]
    pool = sp_fast_var.BatchSPPool(seeds); specs = [None] * G; t0s = [0] * G; prs = {}; grp = np.repeat(np.arange(len(names)), N)
    for i in range(G):
        k = names[grp[i]]
        if k == "通常": continue
        ns["VAR_MIX"].clear(); ns["VAR_MIX"][k] = 1.0
        c, sp, t0 = ns["var_draw"](rng); t0s[i] = t0
        if c == 3: prs[i] = sp
        else: specs[i] = sp
    pool.set_variants(specs)
    vi = np.asarray(pool.env.variant_info())
    buys = np.zeros(G); m_at = []; plant_se = np.zeros(G); anim = np.zeros(G); t_run = time.time()
    torch.manual_seed(1)
    for t in range(T):
        for i_, sp_ in prs.items():
            if t0s[i_] == t:
                m0 = pool.money()[i_].copy(); ns["var_apply_pr"](pool, i_, sp_); m_at.append((i_, t, m0.round().tolist(), pool.money()[i_].round().tolist()))
        acts = []
        for s in (0, 1):
            st = pool.state(s); U, O = act(st); acts.append((U, O))
            if s == 0:
                alive = np.array([t >= t0s[i] for i in range(G)])
                buys += (((O[:, :, 0] == 3) | (O[:, :, 0] == 5)).sum(1) * alive)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
        if t == min(600, T - 1):   # 手600 の盤面: 南東区画(x,y>=5)の作付け、家畜
            tb = np.asarray(pool.state(0)[0], np.float32)[:, 0]
            kind = np.rint(tb[..., 0] * 5).reshape(G, 10, 10)
            plant_se = (kind[:, 5:, 5:] == 5).sum((1, 2)); anim = (tb[..., 6] > 0.5).sum(1)
        if t % 120 == 0: print("   手%d (%.0fs)" % (t, time.time() - t_run), flush=True)
    pool.env.set_view(0); fq0 = np.rint(np.asarray(pool.env.farm_pub(0))[:, 2] * 4); fq1 = np.rint(np.asarray(pool.env.farm_pub(1))[:, 2] * 4)
    m = pool.money()
    for g_, name in enumerate(names):
        ix = grp == g_
        print("(B) %-4s %d局: 最終所持金 平均 %6.0f(最小 %6.0f 最大 %6.0f) 区画 %.1f/%.1f 手600 南東の作付け %.1f 家畜 %.1f 種・家畜の注文 %.0f回 | 仕様の例(資金,区画,開始手,区画%%,雇用%%) %s" % (
            name, N, m[ix].mean(), m[ix].min(), m[ix].max(), fq0[ix].mean(), fq1[ix].mean(), plant_se[ix].mean(), anim[ix].mean(), buys[ix].mean(),
            [vi[i, 1:6].tolist() for i in np.flatnonzero(ix)[:2]]), flush=True)
    print("(B) 摂動の例(局, 手, 前の所持金, 後の所持金): %s" % m_at[:4], flush=True)
    return m


if __name__ == "__main__":
    ok = True
    if os.environ.get("SKIP_A") != "1": ok = check_same()
    if os.environ.get("SKIP_B") != "1": play_variants(T=int(os.environ.get("TB", "719")))
    sys.exit(0 if ok else 1)
