"""09-22: 提出形のフォルダ(main.py ほか6ファイル)を公式の kaggle_environments で最後まで打たせる。
  使い方: python3 official_check.py <A の main.py> <B の main.py>(A が席0)。コンテナ(Python 3.10)の中で動かす。"""
import sys, time
sys.path.append("/workspace/kaggriculture/py310")   # 公式エンジン本体。部品(rpds など)は利用者用の py_env(Python 3.10)を先に使う
from kaggle_environments import make
A, B = sys.argv[1], sys.argv[2]
qn = lambda f: f.get("unlocked_quadrants") if isinstance(f.get("unlocked_quadrants"), int) else len(f.get("unlocked_quadrants") or [])
for seed in (424242, 515151):
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run([A, B])
    last = env.steps[-1]; farms = last[0].observation.farms
    f4 = [next((t for t, s in enumerate(env.steps) if qn(s[0].observation.farms[k]) >= 4), None) for k in (0, 1)]
    print("種 %d: 手数 %d 状態 %s 所持金 %s 区画 %s 4区画に届いた手 %s %.0f秒" % (seed, len(env.steps) - 1, [a.status for a in last],
          [f["money"] for f in farms], [qn(f) for f in farms], f4, time.time() - t0), flush=True)
print("== 済 ==", flush=True)
