"""09-22: kagsim の legal_mask(opts) の検証。SODIR の kagsim を先に読み込み、貪欲の対局を打って
各手の合法手(両席)のハッシュと最終所持金を書く。OPTS=none/0/2/mix(席0 = 0、席1 = 環境)。"""
import os, sys, json, hashlib
SODIR = os.environ["SODIR"]; sys.path.insert(0, SODIR); import kagsim; sys.path.pop(0)
print("kagsim:", kagsim.__file__, flush=True)
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_lib as S
import sp_fast; S.SPPool = sp_fast.BatchSPPool
import fast_codes; P.to_codes = fast_codes.to_codes_fast
assert sp_fast.kagsim is kagsim
OPTS = os.environ.get("OPTS", "none"); L = "/content/local_logs"
a = M.load(L + "/orbit_apo5_gen79.pt", P.DEV).eval(); b = M.load(L + "/tf_l2bc_id22.pt", P.DEV).eval()
seeds = [9100000 + 7919 * j for j in range(8)]
pool = sp_fast.BatchSPPool(seeds)
if OPTS != "none":
    pool.legal_opt = np.full((pool.n, 2), -1, np.int32)
    if OPTS == "mix": pool.legal_opt[:, 0] = 0
    else: pool.legal_opt[:, :] = int(OPTS)
ag = {0: S.PolicyOpp(a, greedy=True, cap=True), 1: S.PolicyOpp(b, greedy=True, cap=True)}; h = hashlib.sha1(); idx = list(range(pool.n))
with torch.no_grad():
    for t in range(719):
        acts = {}
        for s in (0, 1):
            st = pool.state(s); h.update(np.ascontiguousarray(st[4]).tobytes()); acts[s] = ag[s].act(st, t, idx, pool, s)
        pool.step(acts[0][0], acts[0][1], acts[1][0], acts[1][1])
m = pool.money()
out = dict(opts=OPTS, legal_env=os.environ.get("LEGAL_OPT"), mask_sha=h.hexdigest(), money=[[float(x) for x in r] for r in m])
print(json.dumps(out), flush=True)
json.dump(out, open(os.environ["OUT"], "w"))
