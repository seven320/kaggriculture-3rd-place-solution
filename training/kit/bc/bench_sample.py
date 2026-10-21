"""exp280: P.sample を FASTINF=0/1 で同じ局面に当てて、貪欲の行動が完全に一致するかと速さを比べる。
使い方: python3 bench_sample.py run <出力.npz>(FASTINF は環境変数)/ python3 bench_sample.py cmp"""
import os, sys, time
import numpy as np
if sys.argv[1] == "cmp":
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("0", "1"):
        subprocess.run([sys.executable, __file__, "run", "/tmp/bs_%s.npz" % f], env=dict(os.environ, FASTINF=f), check=True, cwd=here)
    a, b = np.load("/tmp/bs_0.npz"), np.load("/tmp/bs_1.npz")
    for k in ("ul", "uq", "ms", "mq"):
        print("貪欲の行動 %s: 一致率 %.5f" % (k, float((a[k] == b[k]).mean())))
    print("1手あたり(%d局まとめ): 元 %.1f ms → 高速 %.1f ms(%.2f倍)" % (int(a["B"]), float(a["ms_step"]), float(b["ms_step"]), float(a["ms_step"]) / float(b["ms_step"])))
    print("標本(貪欲でない)も1手あたり: 元 %.1f ms → 高速 %.1f ms" % (float(a["ms_samp"]), float(b["ms_samp"])))
    sys.exit()
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_fast
torch.manual_seed(0)
net = M.load("/content/local_logs/orbit_o19_gen9.pt", "cuda").eval()
B = int(os.environ.get("B", 960))
pool = sp_fast.BatchSPPool([990001 + i for i in range(B)])
acts = {k: [] for k in ("ul", "uq", "ms", "mq")}; tg = 0.0; ts = 0.0; n = 0
for t in range(60):
    st = pool.state(0)
    with torch.autocast("cuda", dtype=torch.float16):
        torch.cuda.synchronize(); t0 = time.time()
        ul, uq, ms, mq, _ = P.sample(net, st[:5], greedy=True, cap=True)
        torch.cuda.synchronize(); t1 = time.time()
        P.sample(net, st[:5], greedy=False, cap=True)
        torch.cuda.synchronize(); t2 = time.time()
    if t >= 5: tg += t1 - t0; ts += t2 - t1; n += 1
    for k, v in zip(acts, (ul, uq, ms, mq)): acts[k].append(v)
    U, O = P.to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O, U, O)
np.savez(sys.argv[2], B=B, ms_step=tg / n * 1000, ms_samp=ts / n * 1000, **{k: np.stack(v) for k, v in acts.items()})
