"""exp191: 学習した BC の重みを、公式環境で走る提出形のエージェントに詰める。
   common/bc/agent_main_tf.py は `M.Policy()`(既定の大きさ)で読むので、**構成つきの重み**が読めない。
   ここでは `M.load` を使う版の main.py を書き出す(common/ は変えていない)。
   使い方: python3 pack_bc_agent.py <tag> <出力dir>
"""
import json, os, shutil, sys
BC = os.path.dirname(os.path.abspath(__file__))
tag, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in ("model_tf.py", "obs_feat.py"):
    shutil.copy(os.path.join(BC, f), os.path.join(out, f))
w = "/content/local_logs/tf_%s.pt" % tag
shutil.copy(w, os.path.join(out, "policy.pt"))
cfg = w + ".cfg.json"
json.dump(json.load(open(cfg)) if os.path.exists(cfg) else {}, open(os.path.join(out, "policy.pt.cfg.json"), "w"))
src = open(os.path.join(BC, "agent_main_tf.py")).read()
old = '_NET = M.Policy(); _NET.load_state_dict(torch.load(os.path.join(_D, "policy.pt"), map_location="cpu")); _NET.eval()'
new = '_NET = M.load(os.path.join(_D, "policy.pt"), "cpu")'
assert src.count(old) == 1, "agent_main_tf.py の読み込み行が見つからない"
open(os.path.join(out, "main.py"), "w").write(src.replace(old, new))
print("詰めた %s (tag=%s, 構成 %s)" % (out, tag, open(os.path.join(out, "policy.pt.cfg.json")).read()))
