"""exp280: fastinfer が元と同じ値を出すか(ユニットの頭・市場の逐次キャッシュ)と、どれだけ速いか。"""
import os, sys, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_tf as P, model_tf as M, sp_fast, fastinfer as FI
DEV = "cuda"
net = M.load("/content/local_logs/orbit_o19_gen9.pt", DEV).eval()
orig_unit = M.Policy.unit_logits
tt = lambda a, dt=torch.float32: torch.as_tensor(a, device=DEV, dtype=dt)
for B in (256, 960):
    pool = sp_fast.BatchSPPool([880001 + i for i in range(B)])
    for t in range(40):
        st = pool.state(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ul, uq, ms, mq, _ = P.sample(net, st[:5], greedy=False, cap=True)
        U, O = P.to_codes(ul, uq, ms, mq, st[1]); pool.step(U, O, U, O)
    st = pool.state(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        h, pad = net.encode(tt(st[0], torch.float16), tt(st[1]), tt(st[2]), tt(st[3]))
        a, _ = orig_unit(net, h, tt(st[1]), tt(st[4], torch.uint8)); b, _ = FI.unit_logits_fast(net, h, tt(st[1]), tt(st[4], torch.uint8))
        leg = tt(st[4], torch.uint8).bool()
        print("B=%d ユニットの頭: 合法手の差 最大 %.2e、argmax 一致 %.4f" % (B, float((a.float() - b.float())[leg].abs().max()), float((a.argmax(-1) == b.argmax(-1)).float().mean())))
        # 市場: 元の逐次(毎回全部)と、キャッシュ版に同じ系列を入れてロジットを比べる
        prev = torch.full((B, 1), P.NM, device=DEV, dtype=torch.long); mc = FI.MarketCache(net, h, pad)
        tok = prev[:, 0]; mx = 0.0; agree = []
        for j in range(P.NS):
            mo, _ = net.market(h, pad, prev); la = mo[:, -1].float()
            lb, _ = mc.step(tok); lb = lb.float()
            mx = max(mx, float((la - lb).abs().max())); agree.append(float((la.argmax(-1) == lb.argmax(-1)).float().mean()))
            tok = la.argmax(-1).clamp(max=P.NM); prev = torch.cat([prev, tok.view(-1, 1)], 1)
        print("B=%d 市場デコーダ: ロジットの差 最大 %.2e、各枠の argmax 一致の最小 %.4f" % (B, mx, min(agree)))
    def timeit(fn, R=15):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(R): fn()
        torch.cuda.synchronize(); return (time.time() - t0) / R * 1000
    def unit_old():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): orig_unit(net, h, tt(st[1]), tt(st[4], torch.uint8))
    def unit_new():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): FI.unit_logits_fast(net, h, tt(st[1]), tt(st[4], torch.uint8))
    def mkt_old():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prev = torch.full((B, 1), P.NM, device=DEV, dtype=torch.long)
            for j in range(P.NS):
                mo, _ = net.market(h, pad, prev); prev = torch.cat([prev, mo[:, -1].argmax(-1).clamp(max=P.NM).view(-1, 1)], 1)
    def mkt_new():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            mc = FI.MarketCache(net, h, pad); tok = torch.full((B,), P.NM, device=DEV, dtype=torch.long)
            for j in range(P.NS):
                lg, _ = mc.step(tok); tok = lg.argmax(-1).clamp(max=P.NM)
    def enc():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): net.encode(tt(st[0], torch.float16), tt(st[1]), tt(st[2]), tt(st[3]))
    print("B=%d ms/手: 読み取り %.1f | ユニット 元 %.1f → 新 %.1f | 市場 元 %.1f → 新 %.1f" % (B, timeit(enc), timeit(unit_old), timeit(unit_new), timeit(mkt_old), timeit(mkt_new)), flush=True)
