"""公式エンジン(kaggle_environments)で N 局まわす。これが唯一の正しい測定器。

kagsim のプールは 09-05 断面のテープ12体で、実エンジンの相手とは市場が別物になる。
実際、生成器は kagsim で 73.9k、実エンジンでは同じ設定で動物が0頭になり 22k〜73k と割れた。
"""
import os, sys, json, time, numpy as np
from kaggle_environments import make
B = "/content/drive/MyDrive/kaggriculture"
A = os.environ.get("A", B + "/snap/v48.py")
OPP = os.environ.get("OPP", B + "/snap/v50.py")
N = int(os.environ.get("N", 12))
res = []
t0 = time.time()
for i in range(N):
    env = make("kaggriculture", configuration={"seed": 700000 + i})
    r = env.run([A, OPP])
    a, b = r[-1][0]["reward"], r[-1][1]["reward"]
    st = [x["status"] for x in r[-1]]
    res.append((a, b))
    print("  %2d  自分 %9s  相手 %9s  %s  (%.0fs)" % (i, format(int(a or 0), ","), format(int(b or 0), ","),
          "" if st == ["DONE", "DONE"] else st, time.time() - t0), flush=True)
A_ = np.array([x[0] or 0 for x in res], float); B_ = np.array([x[1] or 0 for x in res], float)
print("\n%s 対 %s  %d局" % (os.path.basename(A), os.path.basename(OPP), N))
print("  自分 平均 %s (中央 %s)  相手 平均 %s  勝ち %d/%d  margin 平均 %s" % (
    format(int(A_.mean()), ","), format(int(np.median(A_)), ","), format(int(B_.mean()), ","),
    int((A_ > B_).sum()), N, format(int((A_ - B_).mean()), ",")))
