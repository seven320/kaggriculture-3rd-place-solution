"""重み1つからリーグ用のエージェント一式を作る。土台は v67 の一式(在庫修理層つき)で、方策と構成だけ差し替える。
使い方: python3 tools/mkagent.py <重み.pt> <エージェント名> [土台のエージェント]"""
import json, os, shutil, sys
import torch
W = "/workspace/kaggriculture"


def main(ckpt, name, base="lab/agents/o19g9_sm3"):
    dst = os.path.join(W, "lab/agents", name)
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(os.path.join(W, base), dst)
    shutil.copy(ckpt, dst + "/policy.pt")
    cfg = ckpt + ".cfg.json"
    if os.path.exists(cfg): shutil.copy(cfg, dst + "/cfg.json")
    sd = torch.load(dst + "/policy.pt", map_location="cpu")
    if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
    # 09-21(exp313): 合法判定の版を obs_feat.py に焼き込む。Kaggle 上には環境変数が無いので、
    #   ここで既定値そのものを書き換える。**学習に使った値と必ず同じにすること**(違うと提出だけ手が狭まる)。
    lo = os.environ.get("LEGAL_OPT", "0")
    f_ = dst + "/obs_feat.py"
    if lo != "0":
        # 土台のエージェントは古い obs_feat.py を持っていることがある。明示されたときだけ正典で上書きする
        # (既定 LEGAL_OPT=0 の組み立ては土台のまま = 従来とまったく同じ)。
        shutil.copy(os.path.join(W, "repo/common/bc/obs_feat.py"), f_)
        print("  obs_feat.py を正典(repo/common/bc)で置き換えた")
    t_ = open(f_).read()
    if 'LEGAL_OPT = int(os.environ.get("LEGAL_OPT", "0"))' in t_:
        open(f_, "w").write(t_.replace('LEGAL_OPT = int(os.environ.get("LEGAL_OPT", "0"))',
                                       'LEGAL_OPT = int(os.environ.get("LEGAL_OPT", "%s"))' % lo, 1))
        print("  合法判定の版 LEGAL_OPT=%s を焼き込んだ" % lo)
    elif lo != "0":
        print("  !! obs_feat.py に LEGAL_OPT の行が無い。提出前に確認すること")
    left = [k for k in sd if "pid" in k]
    d = [v.shape[1] for k, v in sd.items() if k.endswith("m_out.weight")]
    print("%s: 次元 %s、焼き込み前の pid が残っていないか %s" % (name, d, left or "残っていない"))
    return dst


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or []))
