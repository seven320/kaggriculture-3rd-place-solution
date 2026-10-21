"""exp280: autocast のまま torch.compile した encode(推論)と、学習の1歩(evaluate 順+逆)の速さ。値の一致も確認。"""
import os, sys, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FASTINF", "1")
import ppo_tf as P, model_tf as M, sp_fast
DEV = "cuda"
net = M.load("/content/local_logs/orbit_o19_gen9.pt", DEV).eval()
B = 960
pool = sp_fast.BatchSPPool([770001 + i for i in range(B)])
for t in range(20):
    st = pool.state(0)
    with torch.autocast("cuda", dtype=torch.float16): ul, uq, ms, mq, _ = P.sample(net, st[:5], greedy=False, cap=True)
    U, O = P.to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O, U, O)
st = pool.state(0)
tt = lambda a, dt=torch.float32: torch.as_tensor(a, device=DEV, dtype=dt)
X = (tt(st[0], torch.float16), tt(st[1]), tt(st[2]), tt(st[3]))
def bench(fn, R=20, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / R * 1000
def enc_amp(*x):
    with torch.autocast("cuda", dtype=torch.float16):
        return net.encode(*x)
with torch.no_grad():
    base, _ = enc_amp(*X)
    print("推論 encode B=%d: いま %.1f ms" % (B, bench(lambda: enc_amp(*X))), flush=True)
    for mode in ("default", "max-autotune-no-cudagraphs", "reduce-overhead"):
        try:
            f = torch.compile(enc_amp, mode=mode, dynamic=False)
            t0 = time.time(); f(*X); f(*X); torch.cuda.synchronize(); ct = time.time() - t0
            h, _ = f(*X)
            print("  compile %-28s コンパイル %4.0f 秒 | %.1f ms | 差 最大 %.2e" % (mode, ct, bench(lambda: f(*X)), float((h.float() - base.float()).abs().max())), flush=True)
        except Exception as e:
            print("  compile %s 失敗: %s" % (mode, str(e)[:200]), flush=True)
# 学習の1歩
train = copy.deepcopy(net).train()
opt = torch.optim.Adam(train.parameters(), lr=1e-6, fused=True)
MB = 512; idx = np.arange(MB)
ul_t = tt(np.where(st[1][..., 0] > 0, 0, -1)[idx], torch.long)
b = [X[0][idx], X[1][idx], X[2][idx], X[3][idx], tt(st[4], torch.uint8)[idx], ul_t, torch.zeros_like(ul_t),
     torch.full((MB, P.NS), P.M_STOP, device=DEV, dtype=torch.long), torch.zeros((MB, P.NS), device=DEV, dtype=torch.long)]
def loss_fn(*bb):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, ent, v, dist, per, pm = P.evaluate(train, list(bb))
        return -(per * pm).sum() / pm.sum() - 0.001 * ent.mean()
def step(fn):
    loss = fn(*b); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print("学習 MB=512 bf16 + fused Adam: %.1f ms" % bench(lambda: step(loss_fn), R=10), flush=True)
try:
    cl = torch.compile(loss_fn, dynamic=False)
    t0 = time.time(); step(cl); step(cl); torch.cuda.synchronize(); ct = time.time() - t0
    print("学習 MB=512 bf16 + compile: コンパイル %.0f 秒 | %.1f ms" % (ct, bench(lambda: step(cl), R=10)), flush=True)
except Exception as e:
    print("学習 compile 失敗:", str(e)[:300], flush=True)
