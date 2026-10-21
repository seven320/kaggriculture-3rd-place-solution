"""exp280: 読み取り部分(encode)と学習の1歩を、実装の選択肢ごとに測る。
推論: autocast fp16(いま)/ 半精度モデル(PyTorch の高速経路)/ torch.compile。
学習: evaluate の順伝播+逆伝播 を fp16+GradScaler(いま)/ bf16 / compile、ミニバッチ 512 と 1024。"""
import os, sys, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FASTINF", "1")
import ppo_tf as P, model_tf as M, sp_fast
DEV = "cuda"
print("torch", torch.__version__, flush=True)
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
def bench(fn, R=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(R): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / R * 1000
with torch.no_grad():
    ref, _ = net.encode(*X)
    def f_amp():
        with torch.autocast("cuda", dtype=torch.float16): return net.encode(*X)
    print("推論 encode B=%d: autocast fp16 %.1f ms" % (B, bench(f_amp)), flush=True)
    half = copy.deepcopy(net).half().eval()
    Xh = (X[0], X[1].half(), X[2].half(), X[3].half())
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            a, _ = net.encode(*X)
        b, _ = half.encode(*Xh)
        print("  半精度モデル: 差 最大 %.2e、%.1f ms" % (float((a.float() - b.float()).abs().max()), bench(lambda: half.encode(*Xh))), flush=True)
    except Exception as e:
        print("  半精度モデル 失敗:", str(e)[:200], flush=True)
    try:
        cenc = torch.compile(half.encode, dynamic=False)
        t0 = time.time(); cenc(*Xh); torch.cuda.synchronize(); ct = time.time() - t0
        print("  半精度 + compile: 最初のコンパイル %.0f 秒、%.1f ms" % (ct, bench(lambda: cenc(*Xh))), flush=True)
    except Exception as e:
        print("  compile 失敗:", str(e)[:300], flush=True)
    try:
        cenc2 = torch.compile(half.encode, mode="reduce-overhead", dynamic=False)
        t0 = time.time(); cenc2(*Xh); cenc2(*Xh); torch.cuda.synchronize(); ct = time.time() - t0
        print("  半精度 + compile(CUDA グラフ): コンパイル %.0f 秒、%.1f ms" % (ct, bench(lambda: cenc2(*Xh))), flush=True)
    except Exception as e:
        print("  CUDA グラフ 失敗:", str(e)[:300], flush=True)

# 学習の1歩(evaluate の順伝播+逆伝播)
train = copy.deepcopy(net).train()
opt = torch.optim.Adam(train.parameters(), lr=1e-6)
for MB in (512, 1024):
    idx = np.arange(MB) % B
    ul_t = tt(np.where(st[1][..., 0] > 0, 0, -1)[idx], torch.long)
    b = [X[0][idx], X[1][idx], X[2][idx], X[3][idx], tt(st[4], torch.uint8)[idx], ul_t, torch.zeros_like(ul_t),
         torch.full((MB, P.NS), P.M_STOP, device=DEV, dtype=torch.long), torch.zeros((MB, P.NS), device=DEV, dtype=torch.long)]
    for name, dt in (("fp16+scaler", torch.float16), ("bf16", torch.bfloat16)):
        scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))
        def step():
            with torch.autocast("cuda", dtype=dt):
                lp, ent, v, dist, per, pm = P.evaluate(train, b)
                loss = -(per * pm).sum() / pm.sum() - 0.001 * ent.mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        try:
            ms = bench(step, R=10, warm=3)
            print("学習 MB=%d %s: 1歩 %.1f ms → 1局面 %.3f ms、メモリ %.1f GB" % (MB, name, ms, ms / MB, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        except Exception as e:
            print("学習 MB=%d %s 失敗: %s" % (MB, name, str(e)[:200]), flush=True)
        torch.cuda.reset_peak_memory_stats()
