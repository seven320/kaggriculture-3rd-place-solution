"""exp292: リーグの常駐ワーカー。GPU を握ったまま lab/leagueq/todo/ の依頼を順に処理する。
Slurm の待ち行列とコンテナのコピー(1本あたり1〜2分)を省き、評価の出し入れを速くする。
依頼 = JSON {"agents": "名前:bc:パス,...", "nseed": 192, "seed0": 123, "out": "/content/lab/xxx.json"}
複数のワーカーが同じ待ち行列を読む(rename で取り合うので二重には走らない)。"""
import os, sys, json, time, glob, subprocess, socket
Q = "/content/lab/leagueq"; WID = "%s_%s" % (socket.gethostname(), os.environ.get("SLURM_JOB_ID", str(os.getpid())))
DUR = float(os.environ.get("DURATION", 1380)) * 60; t_end = time.time() + DUR
for d in ("todo", "run", "done"): os.makedirs(os.path.join(Q, d), exist_ok=True)
print("ワーカー", WID, flush=True)
def _requeue_stale():
    """ワーカーが止められて取り残された依頼(40分以上動きなし)を待ち行列に戻す"""
    for f in glob.glob(Q + "/run/*.json.*"):
        try:
            if time.time() - os.path.getmtime(f) > 900: os.rename(f, os.path.join(Q, "todo", os.path.basename(f).split(".json.")[0] + ".json"))
        except OSError: pass
PAR = int(os.environ.get("LQ_PAR", "1"))   # 09-19: 1ワーカーで同時に回す依頼の数(league.py は CPU 律速で GPU が空くため)。各局は独立なので結果は同じ
import threading
def _slot(k):
    while time.time() < t_end:
        _requeue_stale()
        # 09-21 修正: 他のワーカーが同じファイルを先に取ると getmtime が FileNotFoundError を投げ、
        #   スレッドごと死んで並列スロットが1つずつ減っていた。取れなかったファイルは黙って捨てる。
        todo = []
        for f_ in glob.glob(Q + "/todo/*.json"):
            try: todo.append((os.path.getmtime(f_), f_))
            except OSError: pass
        todo = [f_ for _, f_ in sorted(todo)]
        if not todo: time.sleep(15); continue
        src = todo[0]; mine = os.path.join(Q, "run", os.path.basename(src) + "." + WID + ("_%d" % k if PAR > 1 else ""))
        try: os.rename(src, mine); os.utime(mine)                # 取った時刻を刻む(取り残し判定に使う)
        except OSError: continue                                  # 他のワーカーが先に取った
        try: r = json.load(open(mine))
        except OSError: continue
        t0 = time.time()
        env = dict(os.environ, AGENTS=r["agents"], NSEED=str(r.get("nseed", 192)), SEED0=str(r.get("seed0", 880001)))
        lg = open(os.path.join(Q, "done", os.path.basename(src)[:-5] + ".log"), "w")
        pr = subprocess.Popen([sys.executable, "league.py", r["out"]], env=env, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=lg, stderr=subprocess.STDOUT)
        while pr.poll() is None:                                  # 長い依頼が取り残し扱いされないよう、5分ごとに時刻を刻む
            time.sleep(30)
            if time.time() - os.path.getmtime(mine) > 300:
                try: os.utime(mine)
                except OSError: pass
        rc = pr.returncode; lg.close()
        try: os.rename(mine, os.path.join(Q, "done", os.path.basename(src)))
        except OSError: pass
        print(time.strftime("%H:%M"), "済", k, os.path.basename(src), "rc", rc, "%.0f秒" % (time.time() - t0), flush=True)
ths = [threading.Thread(target=_slot, args=(k,)) for k in range(PAR)]
for k, th in enumerate(ths): th.start(); time.sleep(2)
for th in ths: th.join()
