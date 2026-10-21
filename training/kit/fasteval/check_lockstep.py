"""fast_league の恒等式の点検(FASTCOL_CHECK の評価版)。
  league.py と同じ打ち方(Rust arena batch2 + 各フォルダの featurize + greedy_batch + to_action + _pace/_land + sanitize)で局を進め、
  同時に kagsim(fast_league.Pool)を**同じ行動**で進めて、毎手・両エージェントについて
    (1) 特徴5種(tiles, ou, pu, g, 合法手の印)が featurize と同じか
    (2) fast_league.act が出すエンジン用コードが、league.py の辞書(sanitize 後)を符号化したものと同じか
    (3) 最終の所持金が arena と kagsim で同じか
  を数える。使い方: A=名前:フォルダ B=名前:フォルダ NSEED=8 SEED0=880001 python3 check_lockstep.py"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fast_league as FL
LG = FL.LG

def enc(a, ou):
    a = LG.sanitize(a); U = np.tile(FL._EMPTY_ROW, (FL.NU, 1)); us = [a["farmer"]] + a["hands"]
    for u in range(FL.NU):
        if ou[u, 0] > 0: U[u] = FL.unit_code_dict(us[u]) if u < len(us) else (FL.OPI["PASS"], 0, 1)
    return U, FL.mkt_codes(a["market"])

def main():
    na, fa = os.environ["A"].split(":", 1); nb, fb = os.environ["B"].split(":", 1)
    A, B = FL.Agent(na, fa), FL.Agent(nb, fb)
    NS = int(os.environ.get("NSEED", 8)); seeds = [int(os.environ.get("SEED0", 880001)) + 7919 * k for k in range(NS)]
    seats = np.array([k % 2 for k in range(NS)]); n = NS
    pool = FL.Pool(seeds); need = [0, 2]
    pr = subprocess.Popen([LG.ARENA, "batch2"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    pr.stdin.write(json.dumps({"games": [dict(seed=s) for s in seeds]}) + "\n"); pr.stdin.flush()
    bad = {k: 0 for k in ("tiles", "ou", "pu", "g", "lm", "U", "O", "Ufast", "Ofast")}; first = {}
    t = 0; nchk = 0
    while True:
        msg = json.loads(pr.stdout.readline())
        if msg.get("done"): res = msg["results"]; break
        o = (msg["obs0"], msg["obs1"])
        S = (pool.state(0, need), pool.state(1, need))
        act = ([None] * n, [None] * n); UU = [np.zeros((n, 20, 3), np.int32) for _ in range(2)]; OO = [np.zeros((n, 10, 3), np.int32) for _ in range(2)]
        for who, ag in ((0, A), (1, B)):
            sr = seats if who == 0 else 1 - seats
            obs = [o[sr[i]][i] for i in range(n)]
            F = [ag.bc.m.featurize(x) for x in obs]; st = tuple(np.stack([f[k] for f in F]) for k in range(5))
            mine = [np.stack([S[sr[i]][k][i] for i in range(n)]) for k in range(4)]
            mine.append(np.stack([FL.agent_lm(ag, {v: S[sr[i]][4][v][i:i + 1] for v in need})[0] for i in range(n)]))
            for k, nm in enumerate(("tiles", "ou", "pu", "g", "lm")):
                d = ~np.isclose(st[k].astype(np.float32), mine[k].astype(np.float32), atol=1e-5).reshape(n, -1).all(1)
                if d.any():
                    bad[nm] += int(d.sum()); first.setdefault(nm, (t, who, int(np.flatnonzero(d)[0])))
            nchk += n
            # league.py の行動(featurize の特徴で推論)
            acts = ag.bc.act(obs, list(range(n)))
            # fast_league の行動(kagsim の特徴で推論)
            Uf, Of = FL.act(ag, tuple(mine), np.arange(n), sr, t, None)
            for i in range(n):
                a = LG.sanitize(acts[i]); act[sr[i]][i] = a
                U, O = enc(a, st[1][i])
                UU[sr[i]][i] = U; OO[sr[i]][i] = O
                if not np.array_equal(U, Uf[i]): bad["Ufast"] += 1; first.setdefault("Ufast", (t, who, i, U.tolist(), Uf[i].tolist()))
                Of_ = Of[i].copy(); Of_[(Of_[:, 0] == 254)] = (FL.MOPI["SELL"], 0, 0)   # 254(空き枠を占める)= Rust の SELL WHEAT 0
                if not np.array_equal(O, Of_): bad["Ofast"] += 1; first.setdefault("Ofast", (t, who, i, O.tolist(), Of[i].tolist()))
        pr.stdin.write(json.dumps({"a0": act[0], "a1": act[1]}) + "\n"); pr.stdin.flush()
        pool.env.step_both(UU[0], OO[0], UU[1], OO[1]); t += 1
        if t % 100 == 0: print("手", t, bad, flush=True)
    pr.wait()
    m = pool.money()
    same = [abs(res[i][0] - m[i, 0]) < 1e-6 and abs(res[i][1] - m[i, 1]) < 1e-6 for i in range(n)]
    print("手数 arena %d / kagsim step_index %d" % (t, pool.env.step_index()))
    print("不一致(局×手×エージェント、全 %d):" % nchk, bad)
    print("最初の不一致:", {k: v for k, v in first.items()})
    for i in range(n):
        print("局 %d 種 %d  arena %s  kagsim %s %s" % (i, seeds[i], res[i], m[i].tolist(), "一致" if same[i] else "★不一致"))
    print("最終所持金の一致 %d/%d" % (sum(same), n))

if __name__ == "__main__":
    main()
