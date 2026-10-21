"""P9 の相手50体(チームの Rust エンジン)に、提出形のエージェント(フォルダ)を GPU でまとめて打たせて評価する。
   各フォルダの main.py の修理層・在庫の目安をそのまま使い、重みだけ GPU に載せて全局まとめて貪欲に推論する。
   勝率・差に加えて「何を育てたか」(マス×日)と区画・お手伝いの人数を記録する(勝ち方の指紋)。
   使い方: AGENTS="名前:フォルダ,..." SEEDS="730001 ..." python3 rust_eval2.py 出力.json"""
import os, sys, json, time, types, subprocess, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_tf as M
DEV = "cuda"; NM, NS, M_STOP = 23, 11, 22
ROOT = "/content/p9/p9_opponents/p9_opponents"; ARENA = "/content/lab/rust_arena/target/release/arena"
IDS = [l.strip() for l in open("/content/lab/rust_arena/ids.txt") if l.strip()]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "730001 730002 730003 730004").split()]
tt = lambda a, dt=torch.float32: torch.as_tensor(a, device=DEV, dtype=dt)
@torch.no_grad()
def greedy_batch(net, st, mkt=None):
    tiles, ou, pu, g, lm = st; B = len(g)
    with torch.autocast("cuda", dtype=torch.float16):
        h, pad = net.encode(tt(tiles, torch.float16), tt(ou), tt(pu), tt(g), **({"mkt": tt(mkt)} if mkt is not None else {}))
        lg, q = net.unit_logits(h, tt(ou), tt(lm, torch.uint8))
        live = tt(ou[..., 0] > 0, torch.bool)
        seeds = torch.round(tt(g[:, 27:32]) * 20).long(); used = torch.zeros_like(seeds); cols = []
        for u in range(lg.shape[1]):
            l = lg[:, u].float().clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
            c = l.argmax(-1); pl = (c >= 15) & (c < 20) & live[:, u]
            used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
        ul = torch.stack(cols, 1); ul = torch.where(live, ul, torch.full_like(ul, -1))
        uq = net.unit_qty(q, ul).argmax(-1); uq = torch.where(ul >= 20, uq, torch.zeros_like(uq))
        prev = torch.full((B, 1), NM, device=DEV, dtype=torch.long); done = torch.zeros(B, dtype=torch.bool, device=DEV)
        ms = torch.full((B, NS), M_STOP, device=DEV, dtype=torch.long); mq = torch.zeros((B, NS), device=DEV, dtype=torch.long)
        for j in range(NS):
            mo, y = net.market(h, pad, prev); lj = mo[:, -1].float()
            if j == NS - 1: lj = torch.full_like(lj, torch.finfo(lj.dtype).min); lj[:, M_STOP] = 0
            cj = lj.argmax(-1); cj = torch.where(done, torch.full_like(cj, M_STOP), cj); ms[:, j] = cj
            qc = net.market_qty(y[:, -1:], cj.view(-1, 1))[:, 0].argmax(-1)
            mq[:, j] = torch.where((cj < 19) & ~done, qc, torch.zeros_like(qc))
            done = done | (cj == M_STOP); prev = torch.cat([prev, cj.clamp(max=NM).view(-1, 1)], 1)
            if bool(done.all()): break
    return ul.cpu().numpy(), uq.cpu().numpy(), ms.cpu().numpy(), mq.cpu().numpy()
def greedy_batch_ens(nets, st, mkt=None):
    """exp309: 複数の方策の**対数確率を各決定で平均**してから argmax(logit アンサンブル)。
    nets が1体なら greedy_batch と同じ手を返す(log_softmax は単調なので argmax は変わらない)。
    合法手の遮蔽はどの方策でも同じなので、平均しても遮蔽は残る。"""
    if len(nets) == 1: return greedy_batch(nets[0], st, mkt=mkt)
    tiles, ou, pu, g, lm = st; B = len(g); K = len(nets)
    LS = lambda x: torch.log_softmax(x.float(), -1)
    with torch.autocast("cuda", dtype=torch.float16):
        hs = []; lgs = []; qs = []
        for net in nets:
            h, pad = net.encode(tt(tiles, torch.float16), tt(ou), tt(pu), tt(g), **({"mkt": tt(mkt)} if mkt is not None else {}))
            lg_, q_ = net.unit_logits(h, tt(ou), tt(lm, torch.uint8))
            hs.append((net, h, pad)); lgs.append(LS(lg_)); qs.append(q_)
        lg = sum(lgs) / K
        live = tt(ou[..., 0] > 0, torch.bool)
        seeds = torch.round(tt(g[:, 27:32]) * 20).long(); used = torch.zeros_like(seeds); cols = []
        for u in range(lg.shape[1]):
            l = lg[:, u].float().clone(); l[:, 15:20] = l[:, 15:20].masked_fill(used >= seeds, -1e4)
            c = l.argmax(-1); pl = (c >= 15) & (c < 20) & live[:, u]
            used.scatter_add_(1, (c - 15).clamp(0, 4).view(-1, 1), pl.long().view(-1, 1)); cols.append(c)
        ul = torch.stack(cols, 1); ul = torch.where(live, ul, torch.full_like(ul, -1))
        uq = (sum(LS(net.unit_qty(q_, ul)) for (net, _, _), q_ in zip(hs, qs)) / K).argmax(-1)
        uq = torch.where(ul >= 20, uq, torch.zeros_like(uq))
        prev = torch.full((B, 1), NM, device=DEV, dtype=torch.long); done = torch.zeros(B, dtype=torch.bool, device=DEV)
        ms = torch.full((B, NS), M_STOP, device=DEV, dtype=torch.long); mq = torch.zeros((B, NS), device=DEV, dtype=torch.long)
        for j in range(NS):
            outs = [net.market(h, pad, prev) for net, h, pad in hs]
            lj = sum(LS(o[0][:, -1]) for o in outs) / K
            if j == NS - 1: lj = torch.full_like(lj, torch.finfo(lj.dtype).min); lj[:, M_STOP] = 0
            cj = lj.argmax(-1); cj = torch.where(done, torch.full_like(cj, M_STOP), cj); ms[:, j] = cj
            qc = (sum(LS(net.market_qty(o[1][:, -1:], cj.view(-1, 1))[:, 0]) for (net, _, _), o in zip(hs, outs)) / K).argmax(-1)
            mq[:, j] = torch.where((cj < 19) & ~done, qc, torch.zeros_like(qc))
            done = done | (cj == M_STOP); prev = torch.cat([prev, cj.clamp(max=NM).view(-1, 1)], 1)
            if bool(done.all()): break
    return ul.cpu().numpy(), uq.cpu().numpy(), ms.cpu().numpy(), mq.cpu().numpy()


def run(folder):
    tp = os.path.join(folder, "main.py"); sys.path.insert(0, folder)
    mod = types.ModuleType("ag"); exec(compile(open(tp).read(), tp, "exec"), mod.__dict__); sys.path.pop(0)
    featurize, to_action, pace = mod.featurize, mod.to_action, getattr(mod, "_pace", None)
    net = M.load(os.path.join(folder, "policy.pt"), DEV) if os.path.exists(os.path.join(folder, "policy.pt.cfg.json")) else None
    if net is None:
        cfg = json.load(open(os.path.join(folder, "cfg.json"))); net = M.Policy(**cfg).to(DEV); net.load_state_dict(torch.load(os.path.join(folder, "policy.pt"), map_location=DEV))
    net.eval()
    games = [dict(opp=o, seed=s, seat=0) for o in IDS for s in SEEDS]
    pr = subprocess.Popen([ARENA, "batch", ROOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    pr.stdin.write(json.dumps({"games": games}) + "\n"); pr.stdin.flush()
    grow = [collections.Counter() for _ in games]; hands = [0] * len(games); quad = [0] * len(games)
    while True:
        msg = json.loads(pr.stdout.readline())
        if msg.get("done"): res = msg["results"]; break
        obs = msg["obs"]; idx = [i for i, o in enumerate(obs) if o is not None]
        F = [featurize(obs[i]) for i in idx]; st = tuple(np.stack([f[k] for f in F]) for k in range(5))
        ul, uq, ms, mq = greedy_batch(net, st); acts = [None] * len(obs)
        for j, i in enumerate(idx):
            a = to_action(ul[j], uq[j], ms[j], mq[j], int((st[1][j][:, 0] > 0).sum()))
            acts[i] = pace(a, obs[i]) if pace is not None else a
            o = obs[i]
            if o["hour"] == 12:   # 1日1回、何を育てているかを数える
                fa = o["farms"][0]
                for row in fa["tiles"]:
                    for x in row:
                        if isinstance(x, dict):
                            if x.get("kind") == "PLANT": grow[i][x["crop"]] += 1
                            elif x.get("animal"): grow[i][x["animal"]] += 1
                hands[i] += len(fa["hands"]); quad[i] = len(fa["unlocked_quadrants"])
        pr.stdin.write(json.dumps({"actions": acts}) + "\n"); pr.stdin.flush()
    pr.wait()
    me = np.array([r[0] for r in res]); op = np.array([r[1] for r in res])
    G = collections.Counter()
    for c in grow: G.update(c)
    return dict(n=len(res), win=float((me > op).mean()), margin=float((me - op).mean()), me=float(np.median(me)), op=float(np.median(op)),
                grow={k: v / len(res) for k, v in G.items()}, hands=float(np.mean(hands) / 30), quad=float(np.mean(quad)),
                games=[dict(g, me=float(m), op=float(o_)) for g, m, o_ in zip(games, me, op)])
if __name__ == "__main__":
    out = sys.argv[1]; R = json.load(open(out)) if os.path.exists(out) else {}
    for x in os.environ["AGENTS"].split(","):
        n, f = x.split(":", 1)
        if n in R: continue
        t = time.time(); r = run(f); R[n] = r; json.dump(R, open(out, "w"))
        g = r["grow"]; top = sorted(g.items(), key=lambda kv: -kv[1])[:6]
        print("%-14s %d局 勝率 %.3f 差 %+6.0f 自分 %6.0f 相手 %6.0f | お手伝い %.1f人 区画 %.1f | 育てた(マス×日): %s (%.0fs)" % (
            n, r["n"], r["win"], r["margin"], r["me"], r["op"], r["hands"], r["quad"], " ".join("%s %.0f" % (k[:5], v) for k, v in top), time.time() - t), flush=True)
