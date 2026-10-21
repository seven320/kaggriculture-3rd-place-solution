"""Make a submission-format agent folder that plays a checkpoint with no postprocess.

The folder takes main.py, model_tf.py, model_ptr_infer.py, obs_feat.py and stock_target.json from a
template agent of the kit (default pntom45g28, whose main.py has no postprocess), policy.pt from the
checkpoint, and cfg.json = the checkpoint's .pt.cfg.json plus the new-mask inference keys
("ptr", "nmb_pen", "nmb_mode"), the same keys the kit's tools/mkagent_nmb.py writes. The penalty and
mode must match training (apo_nmbrel.py with MIG_PEN=6 -> --nmb-pen 6 --nmb-mode rel).

fast_league.py can play such a folder; it cannot play the shimishige packages, whose postprocess
reads obs["farms"] and obs["market"].

A checkpoint written by PPO (orbit_<TAG>.pt) holds optimizer state as well; pass a promoted
generation (orbit_<TAG>_gen<N>.pt) or a BC weight (tf_<TAG>.pt), which are plain state_dicts.

Usage: make_agent.py CHECKPOINT.pt NAME [--template pntom45g28] [--nmb-pen 6] [--nmb-mode rel] [--root ROOT]
"""
import argparse, json, os, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("name")
    ap.add_argument("--template", default="pntom45g28")
    ap.add_argument("--nmb-pen", type=float, default=6.0)
    ap.add_argument("--nmb-mode", default="rel", choices=("rel", "abs"))
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("TEAM_BC_PPO_ROOT", REPO / "training/work")))
    a = ap.parse_args()
    agents = a.root.resolve() / "content/lab/agents"
    out = agents / a.name
    out.mkdir(parents=True)
    for f in ("main.py", "model_tf.py", "model_ptr_infer.py", "obs_feat.py", "stock_target.json"):
        shutil.copyfile(agents / a.template / f, out / f)
    cfg = json.loads(Path(str(a.checkpoint) + ".cfg.json").read_text())
    cfg.update(ptr=True, nmb_pen=a.nmb_pen, nmb_mode=a.nmb_mode)
    (out / "cfg.json").write_text(json.dumps(cfg))
    (out / "policy.pt").symlink_to(a.checkpoint.resolve())
    print(out)
