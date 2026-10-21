"""leagueq.py と同じ待ち行列の書式で、依頼を fast_league.py で処理する常駐ワーカー(案)。
  依頼 = JSON {"agents": "名前:bc:パス,...", "nseed": 192, "seed0": 123, "out": "/content/lab/xxx.json"}(leagueq.py と同じ)
  bc 型だけの依頼は fast_league.py、py 型や市場16特徴(cfg の mkt)を含む依頼は従来の league.py に回す。
  待ち行列は既定で lab/fasteval/q_test(試験用)。本番の lab/leagueq を読ませるときだけ LQF_Q=/content/lab/leagueq を明示する。
  使い方(コンテナの中): LQF_Q=/content/lab/fasteval/q_test DURATION=60 python3 leagueq_fast.py"""
import os, sys, json, time, glob, subprocess, socket, threading
HERE = os.path.dirname(os.path.abspath(__file__))
BC_DIR = os.environ.get("KAGG_BC", "/content/drive/MyDrive/kaggriculture/bc")
Q = os.environ.get("LQF_Q", os.path.join(HERE, "q_test"))
WID = "%s_%s_fast" % (socket.gethostname(), os.environ.get("SLURM_JOB_ID", str(os.getpid())))
DUR = float(os.environ.get("DURATION", 1380)) * 60; t_end = time.time() + DUR
ONESHOT = os.environ.get("ONESHOT") == "1"          # 1 = 待ち行列が空になったら終わる(試験用)
sys.path.insert(0, HERE)
def hooks_supported(folder):          # fast_league.hooks_supported と同じ(torch を読まずに済むよう、ここでは文字列から取り出して実行)
    import re, types
    src = open(os.path.join(HERE, "fast_league.py")).read()
    i = src.index("_OBS_OK ="); j = src.index("class Agent:")
    m = types.ModuleType("hk"); m.os = os; exec(src[i:j], m.__dict__); return m.hooks_supported(folder)
if __name__ == "__main__":
    for d in ("todo", "run", "done"): os.makedirs(os.path.join(Q, d), exist_ok=True)
    print("ワーカー", WID, "待ち行列", Q, flush=True)


def fast_ok(agents):
    """bc 型だけで、どのフォルダも cfg に mkt が無ければ fast_league で打てる"""
    for x in agents.split(","):
        n, t, f = x.split(":", 2)
        if t != "bc": return False
        try:
            if json.load(open(os.path.join(f, "cfg.json"))).get("mkt"): return False
            if not hooks_supported(f)[0]: return False
        except OSError: return False
    return True


def _requeue_stale():
    for f in glob.glob(Q + "/run/*.json.*"):
        try:
            if time.time() - os.path.getmtime(f) > 900: os.rename(f, os.path.join(Q, "todo", os.path.basename(f).split(".json.")[0] + ".json"))
        except OSError: pass


PAR = int(os.environ.get("LQ_PAR", "1"))
def _slot(k):
    while time.time() < t_end:
        _requeue_stale(); todo = []
        for f_ in glob.glob(Q + "/todo/*.json"):
            try: todo.append((os.path.getmtime(f_), f_))
            except OSError: pass
        todo = [f_ for _, f_ in sorted(todo)]
        if not todo:
            if ONESHOT: return
            time.sleep(15); continue
        src = todo[0]; mine = os.path.join(Q, "run", os.path.basename(src) + "." + WID + ("_%d" % k if PAR > 1 else ""))
        try: os.rename(src, mine); os.utime(mine)
        except OSError: continue
        try: r = json.load(open(mine))
        except OSError: continue
        t0 = time.time(); fast = fast_ok(r["agents"])
        env = dict(os.environ, AGENTS=r["agents"], NSEED=str(r.get("nseed", 192)), SEED0=str(r.get("seed0", 880001)))
        lg = open(os.path.join(Q, "done", os.path.basename(src)[:-5] + ".log"), "w")
        cmd = [sys.executable, os.path.join(HERE, "fast_league.py") if fast else "league.py", r["out"]]
        pr = subprocess.Popen(cmd, env=env, cwd=BC_DIR, stdout=lg, stderr=subprocess.STDOUT)
        while pr.poll() is None:
            time.sleep(10)
            if time.time() - os.path.getmtime(mine) > 300:
                try: os.utime(mine)
                except OSError: pass
        rc = pr.returncode; lg.close()
        try: os.rename(mine, os.path.join(Q, "done", os.path.basename(src)))
        except OSError: pass
        print(time.strftime("%H:%M"), "済", k, os.path.basename(src), "fast" if fast else "league.py", "rc", rc, "%.0f秒" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    ths = [threading.Thread(target=_slot, args=(k,)) for k in range(PAR)]
    for th in ths: th.start(); time.sleep(2)
    for th in ths: th.join()
