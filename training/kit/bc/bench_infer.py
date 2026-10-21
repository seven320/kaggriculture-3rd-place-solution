"""exp280: PPO の手選び(P.sample)のどこが遅いかを区間ごとに測る。GPU で実行。
区間: 読み取り(encode)/ ユニットの頭と種の上限ループ / 市場デコーダ(最大11枠の逐次)/ 合計。バッチの大きさ別。"""
import os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_fast
DEV = "cuda"
net = M.load("/content/local_logs/orbit_o19_gen9.pt", DEV).eval()
out = {}
for B in (64, 256, 480, 960):
    pool = sp_fast.BatchSPPool([880001 + i for i in range(B)])
    # 少し進めて現実的な局面にする
    for t in range(30):
        st = pool.state(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ul, uq, ms, mq, _ = P.sample(net, st[:5], greedy=False, cap=True)
        U, O = P.to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O, U, O)
    st = pool.state(0)
    tt = lambda a, dt=torch.float32: torch.as_tensor(a, device=DEV, dtype=dt)
    def sync(): torch.cuda.synchronize()
    T = {"encode": 0.0, "unit": 0.0, "market": 0.0, "total": 0.0}
    R = 20
    for r in range(R):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            sync(); t0 = time.time()
            h, pad = net.encode(tt(st[0], torch.float16), tt(st[1]), tt(st[2]), tt(st[3])); sync(); t1 = time.time()
            lg, q = net.unit_logits(h, tt(st[1]), tt(st[4], torch.uint8))
            seeds = torch.round(tt(st[3][:, 27:32]) * 20).long(); used = torch.zeros_like(seeds); live = tt(st[1][..., 0] > 0, torch.bool)
            cols = []
            for u in range(lg.shape[1]):
                l = lg[:, u].clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
                c = torch.distributions.Categorical(logits=l.float()).sample(); pl = (c >= 15) & (c < 20) & live[:, u]
                used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
            ul = torch.stack(cols, 1); uq = net.unit_qty(q, ul).argmax(-1); sync(); t2 = time.time()
            prev = torch.full((len(st[3]), 1), P.NM, device=DEV, dtype=torch.long); done = torch.zeros(len(st[3]), dtype=torch.bool, device=DEV)
            for j in range(P.NS):
                mo, y = net.market(h, pad, prev); lj = mo[:, -1].float()
                cj = torch.distributions.Categorical(logits=lj).sample(); cj = torch.where(done, torch.full_like(cj, P.M_STOP), cj)
                qj = net.market_qty(y[:, -1:], cj.view(-1, 1))[:, 0]
                done = done | (cj == P.M_STOP); prev = torch.cat([prev, cj.clamp(max=P.NM).view(-1, 1)], 1)
            sync(); t3 = time.time()
        if r >= 3:
            T["encode"] += t1 - t0; T["unit"] += t2 - t1; T["market"] += t3 - t2; T["total"] += t3 - t0
    n = R - 3
    out[B] = {k: v / n * 1000 for k, v in T.items()}
    print("B=%4d  1手あたり ms: 読み取り %.1f / ユニット %.1f / 市場 %.1f / 合計 %.1f  → 1局面あたり %.3f ms" % (
        B, out[B]["encode"], out[B]["unit"], out[B]["market"], out[B]["total"], out[B]["total"] / B), flush=True)
json.dump(out, open("/content/lab/bench_infer.json", "w"))
