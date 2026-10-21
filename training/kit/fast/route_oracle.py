"""Is there headroom in choosing the route per game?

The market residual keeps coming back small: in a self-mirror, knowing per game
which deviations help is worth +199 coins over the fixed rule. The route is where
the big numbers were -- switching it was worth +32,570. So: if the best route
differs by game, a policy that picks one has real headroom; if one route is best
almost everywhere, route selection is closed and so is that whole direction.

Measured against a fixed strong opponent, per seed, with every candidate route
playing the same seeds.
"""
import os, sys, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagsim, kagpack
D = "/content/drive/MyDrive/kaggriculture/data/"

def main():
    rr = json.load(open("/content/drive/MyDrive/kaggriculture/runs/rr47.json"))
    top = [k for k, v in rr if not k.startswith("boat29")][:int(os.environ.get("TOPK", 10))]
    files = {os.path.basename(f).rsplit("_", 1)[0][:20]: f for f in glob.glob(D + "nemesis/*.tape.json")}
    T = {k: kagpack.load(files[k]) for k in top if k in files}
    OPP = T[top[4]]                        # a fixed strong opponent, not one of the two best
    seeds = list(range(6600000, 6600000 + int(os.environ.get("NSEED", 300))))
    res = {}
    for k, t in T.items():
        r = kagsim.run_tapes(t[0], t[1], OPP[0], OPP[1], seeds, 720, 2)
        res[k] = (r[:, 0] > r[:, 1]).astype(float) + 0.5*(r[:, 0] == r[:, 1])
        print(f"  {k:<22} 勝率 {res[k].mean():.4f}", flush=True)
    M = np.stack([res[k] for k in res])     # [routes, games]
    best_fixed = M.mean(1).max()
    oracle = M.max(0).mean()
    print(f"\n最良の単一経路: {best_fixed:.4f}")
    print(f"局ごとに最良を選べた場合(オラクル): {oracle:.4f}")
    print(f"経路選択の伸びしろ: {oracle-best_fixed:+.4f}")
    # how often is the best fixed route NOT the per-game best?
    bi = int(M.mean(1).argmax())
    notbest = float((M.max(0) > M[bi] + 1e-9).mean())
    print(f"最良固定経路が局ごと最良でない割合: {notbest:.3f}")

if __name__ == "__main__":
    main()
