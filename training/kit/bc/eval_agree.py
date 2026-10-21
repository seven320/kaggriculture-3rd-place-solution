"""exp302: 重み1つを、ある分割の未見(または学習)の局で測るだけの道具(学習はしない)。
bc_teama.py の評価と同じ値: 損失・ユニット一致・市場の注文列一致・BUY_LAND 再現率(教師の直前の注文を与えた teacher forcing)・
4区画目の再現率・誤って BUY_LAND を出した枠の数。古い棋譜と新しい提出の棋譜で、同じ重みの一致がどれだけ違うかを比べるのに使う。
使い方(容器内): CKPT=/content/local_logs/tf_x.pt SPLIT=mmq CACHE=/content/local_team [PART=hold|train|all] [MAX=局数] python3 eval_agree.py
"""
import json, os, sys, contextlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "/content/drive/MyDrive/kaggriculture/fast")
import model_tf as M
import bc_teama as B
from bc_tf import load_dev
DEV = "cuda"

ck = os.environ["CKPT"]; cache = os.environ["CACHE"]; part = os.environ.get("PART", "hold")
S = json.load(open("/content/local_logs/bc_%s.json" % os.environ["SPLIT"]))
es = S["hold"] if part == "hold" else (S["train"] if part == "train" else S["train"] + S["hold"])
if os.environ.get("MAX"): es = es[:int(os.environ["MAX"])]
T, _ = load_dev(es, cache)
lens = [np.load("%s/%s.npz" % (cache, e))["g"].shape[0] for e in es]
T["step"] = torch.as_tensor(np.concatenate([np.arange(L) for L in lens]), device=DEV)
CFG = json.load(open(ck + ".cfg.json")) if os.path.exists(ck + ".cfg.json") else {}
net = M.Policy(**CFG).to(DEV); net.load_state_dict(torch.load(ck, map_location=DEV)); net.eval()
N = len(T["g"]); acc = {}; tl = 0.0
with torch.no_grad():
    for k in range(0, N, 256):
        ix = np.arange(k, min(k + 256, N))
        with torch.autocast("cuda", dtype=torch.bfloat16): l, st = B.losses(net, T, ix, train=False)
        tl += float(l) * len(ix)
        for kk, v in st.items(): acc[kk] = acc.get(kk, 0.0) + float(v)
print("%s | %s の %s %d 局(%d手): 損失 %.3f ユニット %.3f 市場列 %.3f | 土地 再現率 %.3f(p %.3f, %d枠) 4区画目 %.3f(%d枠) 誤発 %.2f/局" % (
    os.path.basename(ck), os.environ["SPLIT"], part, len(es), N, tl / N, acc["ua"] / acc["un"], acc["mseq"] / acc["n"],
    acc["land_hit"] / max(acc["land_n"], 1), acc["land_p"] / max(acc["land_n"], 1), acc["land_n"],
    acc["l4_hit"] / max(acc["l4_n"], 1), acc["l4_n"], acc["land_fp"] / len(es)), flush=True)
