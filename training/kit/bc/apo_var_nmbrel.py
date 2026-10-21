"""09-25 pntt(rule-variant-curriculum.md §10): apo_nmbrel.py と同じ入口を apo_var.py に向けたもの。
apo_var.py を runpy で走らせる前に model_tf.mig_apply を rel 版(nmb_lib.rel_apply)に差し替える(新マスクの rel 下駄)。
VAR と MASK_MIG の併用は apo_var.py では既定で止めているので、ここで VAR_ALLOW_MIG=1 を立てる。
training/run_ppo.sh の env ファイルで SCRIPT=apo_var_nmbrel.py として使う(torchrun が引数 TAG INIT で呼ぶ)。"""
import os, sys, runpy
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
if os.environ.get("MASK_MIG") != "1": raise SystemExit("apo_var_nmbrel.py は MASK_MIG=1 で起動すること")
os.environ["VAR_ALLOW_MIG"] = "1"
import model_tf as M, nmb_lib
nmb_lib.install(M)
print("[nmb] model_tf.mig_apply を rel 版に差し替えた(増えた手 = min(logit, 旧マスクの最大) − MIG_PEN)。MIG_PEN=%s MIG_HOLD=%s | apo_var: VAR=%s VAR_MIX=%s VAR_P=%s VAR_TT_P=%s VAR_TT_N=%s" % (
    os.environ.get("MIG_PEN"), os.environ.get("MIG_HOLD"), os.environ.get("VAR"), os.environ.get("VAR_MIX"), os.environ.get("VAR_P"),
    os.environ.get("VAR_TT_P"), os.environ.get("VAR_TT_N")), flush=True)
sys.argv[0] = os.path.join(D, "apo_var.py")
runpy.run_path(os.path.join(D, "apo_var.py"), run_name="__main__")
