"""exp280: 放置用の自動判定。新しくできた PPO の世代を見つけてエージェント化し、v67 とバッチ化リーグで192局。
結果は /content/lab/autoeval/<run>_gen<k>.json と summary.json。GPU ジョブの中で DURATION 分まで繰り返す。"""
import os, sys, json, glob, time, math, shutil, subprocess
import torch
L = "/content/local_logs"; A = "/content/lab/agents"; OUT = os.environ.get("AUTOOUT", "/content/lab/autoeval")
BASE = os.environ.get("BASE", "v67"); BASEDIR = os.environ.get("BASEDIR", A + "/o19g9_sm3")   # exp282: 現行最良を差し替えられるように
os.makedirs(OUT, exist_ok=True)
RUNS = os.environ.get("RUNS", "o37,mk16,mk16v67,o39rec").split(",")
MKT_RUNS = set(os.environ.get("MKT_RUNS", "mk16,mk16v67").split(","))
NSEED = int(os.environ.get("NSEED", 192)); DURATION = float(os.environ.get("DURATION", 1380)) * 60
t_end = time.time() + DURATION


def build(run, k):
    name = "%s_g%d%s" % (run, k, "_nr" if os.environ.get("NOREP") == "1" else ""); d = os.path.join(A, name)
    if not os.path.exists(os.path.join(d, "policy.pt")):
        base = "idtop3lk2_sm3" if run in MKT_RUNS else "o19g9_sm3"
        if os.path.exists(d): shutil.rmtree(d)
        shutil.copytree(os.path.join(A, base), d, ignore=shutil.ignore_patterns("__pycache__"))
        sd = torch.load("%s/orbit_%s_gen%d.pt" % (L, run, k), map_location="cpu")
        torch.save({kk: v for kk, v in sd.items() if kk != "pid.weight"}, os.path.join(d, "policy.pt"))
        if os.environ.get("NOREP") == "1":          # exp286: 在庫修理層(_pace)を外した版で判定する
            mp = os.path.join(d, "main.py"); src = open(mp).read()
            src = src.replace("def _pace(a, obs):\n    try:", "def _pace(a, obs):\n    return a   # 装置なし\n    try:").replace("        return _pace(_act(o), o)", "        return _act(o)   # 装置なし")
            assert "装置なし" in src; open(mp, "w").write(src)
    return name, d


def summarize():
    rows = []
    for f in sorted(glob.glob(OUT + "/*_gen*.json")):
        R = json.load(open(f)); key = [k for k in R if k.startswith(BASE + "|")]
        if not key: continue
        v = R[key[0]]; n = len(v); w = (sum(1 for a, b in v if b > a) + 0.5 * sum(1 for a, b in v if a == b)) / n     # 候補の勝率(引き分けは 0.5。exp286 で修正: 以前は引き分けを候補の勝ちに数えていた)
        se = math.sqrt(w * (1 - w) / n) if 0 < w < 1 else 0
        rows.append(dict(cand=os.path.basename(f)[:-5], games=n, win=round(w, 3), se=round(se, 3), z=round((w - .5) / se, 2) if se else None,
                         margin=round(-sum(a - b for a, b in v) / n)))
    rows.sort(key=lambda r: -(r["z"] or -9))
    json.dump(rows, open(OUT + "/summary.json", "w"), ensure_ascii=False, indent=1)
    return rows


while time.time() < t_end:
    todo = []
    for run in RUNS:
        for p in sorted(glob.glob("%s/orbit_%s_gen*.pt" % (L, run))):
            k = int(p.rsplit("gen", 1)[1][:-3])
            if not os.path.exists("%s/%s_gen%d.json" % (OUT, run, k)): todo.append((run, k))
    if not todo:
        time.sleep(300); continue
    todo.sort(key=lambda x: -x[1])                    # exp286: 新しい世代から先に測る
    run, k = todo[0]
    name, d = build(run, k)
    env = dict(os.environ, AGENTS="%s:bc:%s,%s:bc:%s" % (BASE, BASEDIR, name, d), NSEED=str(NSEED), SEED0=str(700001 + 1000 * k))
    tmp = "%s/%s_gen%d.json.part" % (OUT, run, k)
    r = subprocess.run([sys.executable, "league.py", tmp], env=env, cwd="/content/drive/MyDrive/kaggriculture/bc")
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, "%s/%s_gen%d.json" % (OUT, run, k))
        rows = summarize(); print(time.strftime("%H:%M"), "判定済み", run, k, "| 上位:", rows[:3], flush=True)
    else:
        print("失敗", run, k, r.returncode, flush=True); time.sleep(60)
