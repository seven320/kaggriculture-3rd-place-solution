# 09-26: apo_nmbrel.py の写し。走らせる本体を apo_proposed_e7s.py にしただけ(4区画型の牛の状態だけ強制を新マスクで)。
"""09-24 nmb(newmask-distill-plan.md 9節): apo_proposed_e7s.py(E7 の状態だけ版)を変えずに、model_tf.mig_apply だけを rel 版(nmb_lib.rel_apply)に
差し替えて走らせる入口。training/run_ppo.sh の env ファイルで SCRIPT=apo_e7s_nmbrel.py として使う(torchrun が引数 TAG INIT で呼ぶ)。
罰の値・日程・安全装置(MIG_PEN / MIG_HOLD / MIG_ZERO / MIG_GUARD / MIG_Q2MAX)は apo.py の MASK_MIG の仕組みをそのまま使う。
下駄を一定にするには MIG_HOLD を十分大きく(例 1000000)する。"""
import os, sys, runpy
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
if os.environ.get("MASK_MIG") != "1": raise SystemExit("apo_e7s_nmbrel.py は MASK_MIG=1 で起動すること")
import model_tf as M, nmb_lib
nmb_lib.install(M)
print("[nmb] model_tf.mig_apply を rel 版に差し替えた(増えた手 = min(logit, 旧マスクの最大) − MIG_PEN)。MIG_PEN=%s MIG_HOLD=%s" % (
    os.environ.get("MIG_PEN"), os.environ.get("MIG_HOLD")), flush=True)
sys.argv[0] = os.path.join(D, "apo_proposed_e7s.py")
runpy.run_path(os.path.join(D, "apo_proposed_e7s.py"), run_name="__main__")
