"""切り替え型エージェントを作る。python3 mk_switch.py <名前> <A> <B> <D_SW> <K> <M>
A・B は lab/agents/ の単体エージェント。雛形は pr_rs01(policy.pt 以外をコピー)。policy.pt は a.* = A、b.* = B を fp16 で1つに。"""
import sys, os, re, shutil, torch
AG = "/workspace/kaggriculture/lab/agents"
name, A, B, D, K, M = sys.argv[1:7]
out = "%s/%s" % (AG, name); assert not os.path.exists(out), out
os.makedirs(out)
for f in ["main.py", "model_tf.py", "obs_feat.py", "cfg.json", "stock_target.json"]:
    shutil.copy("%s/pr_rs01/%s" % (AG, f), out)
src = open("%s/pr_rs01/model_ptr_infer.py" % AG).read()
src = re.sub(r"^D_SW = \d+", "D_SW = %d" % int(D), src, flags=re.M)
src = re.sub(r"^K = \d+", "K = %d" % int(K), src, flags=re.M)
src = re.sub(r"^M = \d+", "M = %d" % int(M), src, flags=re.M)
open("%s/model_ptr_infer.py" % out, "w").write(src)
sd = {}
for p, n in (("a", A), ("b", B)):
    s = torch.load("%s/%s/policy.pt" % (AG, n), map_location="cpu", weights_only=False)
    for k, v in s.items(): sd["%s.%s" % (p, k)] = v.half() if v.is_floating_point() else v
torch.save(sd, "%s/policy.pt" % out)
print(name, A, B, D, K, M, len(sd))
