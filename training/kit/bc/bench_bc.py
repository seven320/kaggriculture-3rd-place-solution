"""exp254: BC の1歩の速さを測る(A100 を使い切る設定を探す)。idtop2 の学習局 300局を GPU に載せ、設定ごとに 200歩の時間。
設定: 精度(fp32 / tf32 / bf16 の autocast)、1歩の局面数 B、1歩ごとの同期(float())の有無、torch.compile。"""
import os, sys, time, json, glob
for d in ("/.singularity.d/libs", "/usr/local/cuda/compat/lib"):
    if glob.glob(d + "/libcuda.so*"): os.environ.setdefault("TRITON_LIBCUDA_PATH", d); break
print("TRITON_LIBCUDA_PATH", os.environ.get("TRITON_LIBCUDA_PATH"), glob.glob("/.singularity.d/libs/libcuda*"), flush=True)
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf_id as MI
import bc_tf_id as BT
from bc_tf import load_dev
DEV = "cuda"
S = json.load(open("/content/local_logs/bc_idtop2.json")); tr = S["train"][:300]; ids = S["ids"]
T, _ = load_dev(tr, "/content/local_ext4all")
lens = [int(np.load("/content/local_ext4all/%s.npz" % e)["g"].shape[0]) for e in tr]
T["pid"] = BT.pid_tensor(tr, lens, ids); N = len(T["g"])
print("局面 %d" % N, flush=True)
CFG = json.load(open("/content/local_logs/tf_d4xl.pt.cfg.json"))
def run(name, B, prec, sync, comp, steps=200):
    torch.backends.cuda.matmul.allow_tf32 = prec in ("tf32", "bf16"); torch.backends.cudnn.allow_tf32 = prec in ("tf32", "bf16")
    net = MI.PolicyID(nid=len(S["names"]), **CFG).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    f = torch.compile(BT.losses) if comp else BT.losses
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if prec == "bf16" else (lambda: torch.autocast("cuda", enabled=False))
    def step():
        idx = np.random.randint(0, N, B)
        with ctx(): l, a, m = f(net, T, idx, True, sync)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        return l
    for _ in range(20): step()                       # 暖機(compile を含む)
    torch.cuda.synchronize(); t = time.time(); acc = []
    for _ in range(steps):
        l = step()
        if sync: float(l)
        else: acc.append(l.detach())
    torch.cuda.synchronize(); dt = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 2 ** 30; torch.cuda.reset_peak_memory_stats()
    print("%-28s B=%4d  1歩 %6.1f ms  %8.0f 局面/秒  最大メモリ %.1fGB" % (name, B, dt / steps * 1000, B * steps / dt, mem), flush=True)
    del net, opt; torch.cuda.empty_cache()
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
if MODE == "compile":
    for args in [("bf16", 256, "bf16", False, False), ("bf16 compile", 256, "bf16", False, True), ("bf16 compile", 512, "bf16", False, True)]:
        try: run(*args)
        except Exception as ex: print(args[0], args[1], "失敗", str(ex)[:300], flush=True)
    sys.exit(0)
for args in [("fp32 同期あり(今)", 64, "fp32", True, False), ("tf32", 64, "tf32", True, False), ("bf16", 64, "bf16", True, False),
             ("bf16 同期なし", 64, "bf16", False, False), ("bf16", 256, "bf16", False, False), ("bf16", 512, "bf16", False, False),
             ("bf16", 1024, "bf16", False, False), ("fp32", 512, "fp32", False, False), ("bf16 compile", 512, "bf16", False, True)]:
    try: run(*args)
    except Exception as ex: print(args[0], args[1], "失敗", str(ex)[:200], flush=True)
