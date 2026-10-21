import os,sys,json,statistics,subprocess
B="/content/drive/MyDrive/kaggriculture"
def h2h(a,b,seeds):
    code=f'''
import json
from kaggle_environments import make
out=[]
for sd in {seeds!r}:
    for sw in (0,1):
        e=make("kaggriculture",configuration={{"episodeSteps":720,"seed":sd}})
        e.run(["{b}","{a}"] if sw else ["{a}","{b}"])
        f=e.steps[-1][0].observation.farms; i=1 if sw else 0
        bad=sum(1 for s in e.steps if s[i].status not in ("ACTIVE","DONE","INACTIVE"))
        out.append([float(f[i]["money"]),float(f[1-i]["money"]),bad])
print("R"+json.dumps(out))
'''
    r=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=40000)
    for ln in r.stdout.splitlines():
        if ln.startswith("R["): return json.loads(ln[1:])
    print(r.stderr[-300:]); return None
def rep(tag,r):
    if not r: print(f"  {tag}: 失敗",flush=True); return
    w=sum(1 for x in r if x[0]>x[1]); l=sum(1 for x in r if x[0]<x[1]); t=len(r)-w-l
    d=[x[0]-x[1] for x in r]
    print(f"  {tag:<22s} {w}勝{l}敗{t}分 勝率{(w+0.5*t)/len(r):.3f} 異常{sum(x[2] for x in r)} コイン差{statistics.mean(d):+,.0f}",flush=True)
S=[int(x) for x in os.environ.get("SEEDS","5101,5203,5307,5409,5511,5613").split(",")]
for v in (os.environ.get("VAR","v15"),):
    p=f"{B}/snap/{v}.py"
    if os.path.exists(p): rep(f"{v} vs "+os.environ.get("OPP","v13"), h2h(p,B+"/snap/"+os.environ.get("OPP","v13")+".py",S))
