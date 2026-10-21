# Evaluation code for the post-processing A/B (denden)

This folder holds the code that ran the same-game A/B of the post-processing stages on the final policy gen70
(write-up §9.1; results in `../docs/results/gen70_postprocess_ab.ja.md`). It is the import closure of
`experiments/gen70_postproc_20261017/run_gen70.py` and `report_gen70.py`, traced by importing them and recording every loaded
module, plus the modules that the harness imports lazily on the game path.

All code is in the repository. To run it you also need the weights of all policies below and a CUDA GPU; `setup_inputs.sh` lays
out the inputs (list below). Only gen70 is published (Kaggle Dataset); the opponents' weights are not, so the A/B cannot be re-run
from the public material alone. The recorded results are in `../docs/results/`.

## Layout

| Path | Content |
|---|---|
| `experiments/gen70_postproc_20261017/` | `run_gen70.py` (arms A, C, B, D, E), `report_gen70.py`, `protocol.json` (fixed before the run) |
| `experiments/gen41_postproc_20261001/` | `run_gen41.py` (opponents, waves, seeds, the stage chain), `report.py`, `report_c.py` (statistics) |
| `experiments/<other dated folders>/` | the stage implementations used by the chain: waste replacement (`waste_replace2.py`), market stages (`run_m.py`, `market_pp.py`), the day-29 planner (`planner_ed.py`, `run_p_ed.py`), D23S (`postprocess_d23s.py`, `postprocess_ev.py`, `run_ev.py`, `run_d23s.py`), loaders (`run.py`, `run_c.py`, `run_g35.py`, `run_g5new.py`), `postprocess_w9t.py` |
| `harness/scripts/` | the batched game runner on the Rust engine: `kagg_pfsp3/evaluate_land.py` (`play_land_multi`), `kagg_ppo/native.py` (FFI to `libkagg_ppo_native.so`), `kagg_ppo_eval_native/`, and the modules they import |
| `harness/src/kaggriculture_agent/rust_simulation.py` | client for the Rust `simulation-server` |
| `harness/nmb_eval.py` | wrapper that lets the harness play submissions with the new-mask bonus (`ptr`, `nmb_*` in cfg) |

The folder names keep the dated names of the private experiment folders, because the code, the protocols and the result tables
refer to them. The harness packages under `harness/scripts/` were renamed for publication (`kagg_*`, and the Rust crate
`kagg-ppo-native` / `libkagg_ppo_native.so`); the private originals carried a teammate's handle in their names.

The experiment files were copied with these changes only. Absolute paths became `$KAGG_SCRATCH` or paths relative to the repository
root, `ROOT` now points to the repository root, the path to the official `kaggriculture.py` is found from the installed package, and
`code_sha256` keys are made machine-independent (`relkey` / `resolve_key` in `run.py`, `_scratch_key` in `run_gen70.py`). The game
path was not changed.

## Requirements

- Python 3.11, `torch` (CUDA: the harness builds the network on `cuda`), `numpy`, `kaggle-environments==1.32.7`.
- The Rust build in `../engine/rust` (`cargo build --release`). `harness/scripts/kagg_ppo/native.py` loads
  `../engine/rust/target/release/libkagg_ppo_native.so` (override: `KAGG_NATIVE_LIB`).

| Variable | Meaning |
|---|---|
| `KAGG_SCRATCH` | root for the inputs and outputs that are not in the repository (default `./scratch`; use an absolute path) |
| `KAGG_BC_KIT_DIR` | masspeaks' BC kit folder with `original/code/{model_tf.py,obs_feat.py}` (default: `harness/kagg_bc_kit`, included) |
| `KAGG_NATIVE_LIB` | optional path to `libkagg_ppo_native.so` |
| `NVIDIA_SMI` | optional path to `nvidia-smi` (recorded in the manifests) |

## Inputs

| Input | In this repository | Notes |
|---|---|---|
| masspeaks' BC kit code used by the harness (`model_tf.py`, `obs_feat.py`, 09-14 version) | `harness/kagg_bc_kit/original/code/` | masspeaks |
| `harness/scripts/kagg_bc_kit/versioned_features.py` | in place | contains functions taken from the kit's `obs_feat.py` |
| Opponents: team submissions sub120_v108 (for the `_pace` check), sub121_v109, sub122_v4x20, sub123s_gen21, sub124 | `inputs/team_subs_20260930/`, `inputs/team_subs_20261001/sub124/` | code and `receipt.json` (file list with sha256); weights not included |
| Opponents: MMPQ-BCw gen32–35, and gen41 (its code files are the learner's code) | `inputs/shimishige_teamabcw_*/` | shimishige; weights not included |
| Final submissions sub125, sub126 | `../submission/` | `setup_inputs.sh` copies them into place |
| Weights (all of the above, and gen70) | not in git | only gen70 is published, in the [Kaggle Dataset](https://www.kaggle.com/datasets/denden12/kaggriculture-dsm-3rd-place-weights). `setup_inputs.sh` checks each sha256 |
| Game outputs of the original run (`games/`, logs, hashes) | not published | |

```bash
KAGG_SCRATCH=/abs/path/to/scratch KAGG_WEIGHTS=/abs/path/to/weights bash eval/setup_inputs.sh
```

A few comment lines that named a private computing environment were edited in the copied submissions; their `receipt.json` records
both the current and the submitted sha256.

`protocol.json` records the `code_sha256` of the files as they were run, and `learner.file_sha256` the files of the learner as
submitted. The copies here differ only in path constants and in the edited comment lines above, but their hashes differ, so
`run_gen70.py part ...` stops at the hash check. To re-run, build the Rust engine first (`cargo build --release` in
`../engine/rust`; `run_gen70.py` loads `libkagg_ppo_native.so` at import), run `setup_inputs.sh`, then write the current hashes
into `protocol.json`. The original values are kept under `*_as_run` keys; `git diff` shows the change and `git checkout` restores
the file.

```bash
env ED_START=708 ED_ALPHA=0.01 python - <<'EOF'
import hashlib, importlib.util, json, os, sys
p = 'eval/experiments/gen70_postproc_20261017/'
spec = importlib.util.spec_from_file_location('run_gen70', p + 'run_gen70.py'); m = importlib.util.module_from_spec(spec)
sys.argv = ['run_gen70.py']; spec.loader.exec_module(m)
d = json.load(open(p + 'protocol.json'))
d['code_sha256_as_run'] = d['code_sha256']; d['code_sha256'] = m.sources()
src = os.path.join(os.environ['KAGG_SCRATCH'], 'gen70_postproc_20261017/gen70_src')
lf = d['learner']['file_sha256']; d['learner']['file_sha256_as_run'] = dict(lf)
for f in lf:
    lf[f] = hashlib.sha256(open(os.path.join(src, f), 'rb').read()).hexdigest()
json.dump(d, open(p + 'protocol.json', 'w'), ensure_ascii=False, indent=1)
EOF
```

## How the A/B was run

Run from the repository root, with the inputs above in place (original commands, paths adapted):

```bash
H=eval/experiments/gen70_postproc_20261017
export KAGG_SCRATCH=/abs/path/to/scratch
ENV="ED_START=708 ED_ALPHA=0.01 EV_V=0 PLAN_PROCS=2"

# 1) staging (once, before the parallel runs)
env $ENV python $H/run_gen70.py stage

# 2) smoke: arms A and D, 32 games each (sub124 x 16 seeds x both players), plus a step-by-step comparison of 4 games
#    against the submissions' agent()
for a in A D; do env $ENV python $H/run_gen70.py smoke $a & done; wait
python $H/report_gen70.py smoke          # -> summary_smoke.json (integrity and parity mismatches must be 0)

# 3) main run: one arm at a time, 11 processes per arm (job j runs in part (j-1) % 11)
for a in A C B D E; do
  for k in $(seq 0 10); do env $ENV python $H/run_gen70.py part $a $k 11 & done; wait
done

# 4) report
python $H/report_gen70.py games          # -> summary_games.json
```

Conditions (from `protocol.json`): 7 opponents (sub121_v109, sub122_v4x20, sub123s_gen21, mmpqbcw_gen32, mmpqbcw_gen34,
mmpqbcw_gen35, sub124), seeds 174300000 to 174300127, both players, 1,792 games per arm, the same games in every arm. The original
run took about 20 to 24 minutes per arm on one GPU.

## Not included

- The W8 stages imported by `run_c.py` only when `w8` is set (`cap_overlap`, `replant`); the chain here runs with `w8=''`.
- Other modules of the private harness that the A/B does not load (training, PPO, dashboards, posting scripts).
- `run_gen41_c.py` (arm C of the gen41 A/B; `run_gen70.py` does not import it).
