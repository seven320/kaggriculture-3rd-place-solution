#!/usr/bin/env python3
# 09-23: 勾配の all_reduce が本当はどれだけかかるかを測る(数秒)。
#   apo.py の _sync_grads とまったく同じ形(平らな器にコピー → all_reduce → 割る → 書き戻す)を
#   再現し、さらに SYNCPROF=1 が入れている torch.cuda.synchronize() の値段も別に出す。
#
# 起動は torchrun(複数 GPU)。環境変数は launcher と同じ:
#   RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
# 設定:
#   SIZES="10.9e6,40e6"  測る要素数(fp32)。既定は「今の net(約42MB)」と「40M(160MB)」
#   ITERS=30             各サイズの計測回数(前に 10 回の空回し)
#   CALLS=674            1反復ぶんに直すときの呼び出し回数(345120 行 ÷ MB 512)
import os, time, json, torch, torch.distributed as tdist

RANK = int(os.environ.get("RANK", "0"))
LRANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
ITERS = int(os.environ.get("ITERS", "30"))
WARM = int(os.environ.get("WARM", "10"))
CALLS = int(os.environ.get("CALLS", "674"))
SIZES = [int(float(s)) for s in os.environ.get("SIZES", "10.9e6,40e6").split(",")]

torch.cuda.set_device(LRANK)
if WORLD > 1:
    import datetime
    tdist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


if RANK == 0:
    p = torch.cuda.get_device_properties(LRANK)
    log("== bench_allreduce ==")
    log("torch %s / NCCL %s / GPU %s x%d rank" % (torch.__version__, ".".join(map(str, torch.cuda.nccl.version())), p.name, WORLD))
    log("host(rank0) %s  IB_HCA=%s  SOCKET_IFNAME=%s  IB_DISABLE=%s" % (
        os.uname().nodename, os.environ.get("NCCL_IB_HCA", "(auto)"),
        os.environ.get("NCCL_SOCKET_IFNAME", "(auto)"), os.environ.get("NCCL_IB_DISABLE", "(auto)")))
    # どのノードに何 rank 居るかを見せる(1ノードなのか2ノードなのかを取り違えないため)
hosts = [None] * WORLD
if WORLD > 1:
    tdist.all_gather_object(hosts, "%s:gpu%d" % (os.uname().nodename, LRANK))
    log("rank の置き場所: %s" % json.dumps(hosts, ensure_ascii=False))


def timed(fn, iters, warm):
    """fn を warm 回空回しして iters 回測る。戻りは 1 回あたりの秒。"""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    if WORLD > 1:
        tdist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


log("")
log("%12s %9s | %10s %10s %10s | %10s %10s" % (
    "要素数", "MB", "純all_red", "algbw", "busbw", "_sync風", "+同期毎回"))
log("%12s %9s | %10s %10s %10s | %10s %10s" % (
    "", "", "ms", "GB/s", "GB/s", "ms", "ms"))
rows = []
for n in SIZES:
    mb = n * 4 / 1e6
    buf = torch.empty(n, device="cuda", dtype=torch.float32).normal_()
    # (a) 素の all_reduce だけ
    def pure():
        tdist.all_reduce(buf)
    t_pure = timed(pure, ITERS, WARM) if WORLD > 1 else float("nan")

    # (b) apo.py の _sync_grads と同じ形: 勾配 -> 平らな器 -> all_reduce -> /WORLD -> 勾配
    #     勾配の本数は実機に近づけて 300 本に割る(層ごとに小さなコピーが並ぶのを再現する)
    nparts = 300
    step = max(1, n // nparts)
    grads, views, o = [], [], 0
    while o < n:
        m = min(step, n - o)
        grads.append(torch.empty(m, device="cuda", dtype=torch.float32).normal_())
        views.append(buf[o:o + m])
        o += m
    def sync_like():
        torch._foreach_copy_(views, grads)
        tdist.all_reduce(buf)
        buf.div_(WORLD)
        torch._foreach_copy_(grads, views)
    t_sync = timed(sync_like, ITERS, WARM) if WORLD > 1 else float("nan")

    # (c) SYNCPROF=1 と同じく、毎回 torch.cuda.synchronize() を足したとき
    def sync_prof():
        sync_like()
        torch.cuda.synchronize()
    t_prof = timed(sync_prof, ITERS, WARM) if WORLD > 1 else float("nan")

    algbw = n * 4 / 1e9 / t_pure if WORLD > 1 else float("nan")
    busbw = algbw * 2 * (WORLD - 1) / WORLD if WORLD > 1 else float("nan")
    log("%12d %9.1f | %10.3f %10.1f %10.1f | %10.3f %10.3f" % (
        n, mb, t_pure * 1e3, algbw, busbw, t_sync * 1e3, t_prof * 1e3))
    rows.append(dict(n=n, mb=mb, pure_ms=t_pure * 1e3, algbw=algbw, busbw=busbw,
                     sync_ms=t_sync * 1e3, prof_ms=t_prof * 1e3))
    del grads, views, buf
    torch.cuda.empty_cache()

log("")
log("1反復(%d 回呼ぶ)に直すと:" % CALLS)
log("%12s | %12s %12s %12s" % ("MB", "純all_red", "_sync風", "+同期毎回"))
for r in rows:
    log("%12.1f | %10.2f s %10.2f s %10.2f s" % (
        r["mb"], r["pure_ms"] * CALLS / 1e3, r["sync_ms"] * CALLS / 1e3, r["prof_ms"] * CALLS / 1e3))
log("")
log("※ ログの「同期 61s / 80s」と比べる。純 all_reduce ぶんが 1s 前後なら、")
log("   あの数字は通信ではなく、SYNCPROF=1 の torch.cuda.synchronize() が")
log("   逆伝播の待ち時間を同期の勘定に付け替えていただけ、ということになる。")

if WORLD > 1:
    tdist.barrier()
    tdist.destroy_process_group()
