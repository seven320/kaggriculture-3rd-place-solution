"""league.play と fast_league.play を同じ種・席で打ち、局ごとの所持金と時間を比べる。
  使い方: A=名前:フォルダ B=名前:フォルダ NSEED=192 SEED0=880001 python3 compare.py 出力.json [fast_only]"""
import os, sys, json, time, atexit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fast_league as FL
LG = FL.LG

def main():
    out = sys.argv[1]; fast_only = len(sys.argv) > 2 and sys.argv[2] == "fast_only"
    na, fa = os.environ["A"].split(":", 1); nb, fb = os.environ["B"].split(":", 1)
    A, B = FL.Agent(na, fa), FL.Agent(nb, fb)          # A.bc / B.bc は league.BC そのもの(fire の書き出しは外してある)
    NS = int(os.environ.get("NSEED", 192)); seeds = [int(os.environ.get("SEED0", 880001)) + 7919 * k for k in range(NS)]; seats = [k % 2 for k in range(NS)]
    R = {"seeds": seeds, "seats": seats}
    t = time.time(); rf = FL.play(A, B, seeds, seats); R["fast_sec"] = time.time() - t; R["fast"] = rf; R["fast_shops"] = FL.play.shops
    R["fast_parts"] = FL.play.time
    print("fast   %d局 %.1f 秒 (%.2f 局/秒) %s" % (NS, R["fast_sec"], NS / R["fast_sec"], {k: round(v, 1) for k, v in FL.play.time.items()}) + " GPU 確保 %.1fGB" % (FL.torch.cuda.max_memory_reserved() / 2**30), flush=True)
    json.dump(R, open(out, "w"))
    if fast_only: return
    t = time.time(); rl = LG.play(A.bc, B.bc, seeds, seats); R["league_sec"] = time.time() - t; R["league"] = rl; R["league_shops"] = LG.play.shops
    print("league %d局 %.1f 秒 (%.2f 局/秒)" % (NS, R["league_sec"], NS / R["league_sec"]), flush=True)
    json.dump(R, open(out, "w"))
    rf = np.array(rf); rl = np.array(rl)
    same = np.all(np.abs(rf - rl) < 1e-6, 1); wf = rf[:, 0] > rf[:, 1]; wl = rl[:, 0] > rl[:, 1]
    print("局ごとの所持金 完全一致 %d/%d、勝敗一致 %d/%d、店の一致 %d/%d" % (same.sum(), NS, (wf == wl).sum(), NS,
          sum(a == b for a, b in zip(R["fast_shops"], R["league_shops"])), NS))
    for i in range(min(NS, 12)): print("  局 %3d 席 %d  fast %s  league %s" % (i, seats[i], rf[i].tolist(), rl[i].tolist()))
    print("%s の勝率 fast %.3f / league %.3f、平均差 fast %+.0f / league %+.0f、速さ %.1f 倍" % (
        na, wf.mean(), wl.mean(), (rf[:, 0] - rf[:, 1]).mean(), (rl[:, 0] - rl[:, 1]).mean(), R["league_sec"] / R["fast_sec"]))

if __name__ == "__main__":
    main()
