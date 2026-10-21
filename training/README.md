# Training

The training side of the final policy: masspeaks' BC/PPO kit, and the scripts we used to run it locally.

| Path | Content | Author (Kaggle handle) |
|---|---|---|
| `kit/bc/` | BC (replay extraction, teacher-ID BC, Team-A fine-tuning) and PPO (synchronous and asynchronous learners/actors, league, gate, interventions) | masspeaks |
| `kit/fast/` | C++ game engine `kagsim` (pybind11; `build.sh`; the engine core is based on nikital7's public notebook, see NOTICE), fast evaluation and gate scripts | masspeaks |
| `kit/fasteval/` | league evaluation on the fast engine | masspeaks |
| `kit/overlap/` | env files of the PPO lines (`pntom45_base.env` and others) | masspeaks |
| `kit/tools/` | agent packaging (`mkagent.py`, `mkagent_nmb.py`) and analysis helpers | masspeaks |
| `env.sh`, `run_ppo.sh`, `envs/`, `smoke/`, `tools/` | local runner: lays out the kit (`tools/materialize.py`), extracts teacher data from replays (`tools/extract_teacher.py`), packs agents (`tools/make_agent.py`), runs one PPO line (`run_ppo.sh`), smoke tests | denden |

The gen70 weights of the final submissions come from shimishige's run (gen21–70) on the kit's `pntom45` line, starting from the
MMPQ-BCw policies; `_land` in the submissions is also shimishige's.

## Notes

- The kit hard-coded the paths of the machines it ran on. In this repository they are written as `/workspace/...` (cluster) and
  `/content/...` (Google Colab). `tools/materialize.py` copies the kit into a working tree and rewrites those prefixes.
- The gen70 run: the env file of the original run (`pntom45_mmpqbcw_sub112_a100x4_r1`) is not included. What we know: it ran the
  kit's PPO code on the `pntom45` line (`kit/overlap/pntom45_base.env`) with `INIT` = the MMPQ-BCw policy (the BC of write-up §4
  from `sub112`) and `AUX_BC=0`; the run name says 4 A100s; the league held strong past models; a generation was promoted at a
  win rate of 0.60 or higher; the melon intervention (below) was on (write-up §4, §6.2, §6.5).
- `envs/teamabcw.env` is our reconstruction of such a line, not the original env. Its critic, league and `AUX_BC=0.02` are our
  own choices; set `AUX_BC=0` to match the original run.
- Weights: the BC starting point `sub112/policy.pt` and the final `gen70/policy.pt` are in the [Kaggle Dataset](https://www.kaggle.com/datasets/denden12/kaggriculture-dsm-3rd-place-weights). Rerun the
  BC of write-up §4 from `sub112` to get MMPQ-BCw, then PPO to get gen70. sha256 of `sub112/policy.pt`:
  `0c8d40ee0fd958df2e181a69ab2616047ec04345a737790c1fc2970ff6a5f09c` (the weights of the team's submission sub112; its `cfg.json`
  is next to it in the Dataset). gen70: see `../submission/README.md`. Both load on CPU with
  `torch.load(path, map_location="cpu", weights_only=True)`.
- Not included: other weights (critics, intermediate generations), teacher data and caches
  extracted from replays, the Rust arena of opponent agents used by `kit/fast/league.py`, and the job scripts of the
  cluster we trained on (Slurm submission, piggybacking actors on running jobs, relaunching after preemption).
- `run_ppo.sh` runs one line on one GPU (`sync`, or one `learner` plus any number of `actor` processes sharing `ASYNC_DIR`).
  During the competition the same scripts ran as many single-GPU jobs: actors and learners only share files under
  `ASYNC_DIR` (`buf/` for games, `pub/` for weights), and the learners of one line were joined into one DDP group with
  `torchrun`. Any scheduler that starts these processes with the same environment variables reproduces that setup.
- BC demonstrations come from public leaderboard episodes of top teams (Kaggle episode replays). `tools/extract_teacher.py` rebuilds
  the training data from downloaded replays.
- The training-side counterpart of MELON16 (the melon top-up intervention, write-up §6.5) is intervention E7 in
  `kit/bc/apo_proposed_e7s.py`, run through the entry point `kit/bc/apo_e7s_nmbrel.py` (`SCRIPT=apo_e7s_nmbrel.py`, `MASK_MIG=1`).
  The actor rewrites the sampled market orders before they reach the environment and records the rewritten action as the
  policy's own. The switches are environment variables:
  `E7=1 E7_MODE=melon` turns it on; `E7_T0`/`E7_T1` set the step window, `E7_CAP` the melon count (seeds + plants) below which
  it adds seeds, `E7_QTY` the seeds per step, `E7_QMIN` the minimum number of plots, and `E7_P`/`E7_HOLD`/`E7_ZERO` the
  probability schedule. Our run v4xy used `E7_T0=408 E7_T1=409` (day 17, hour 0), `E7_CAP=9`, `E7_P=1.0`, `E7_HOLD=100000`
  (never annealed) and `E7_QMIN=2`; its full configuration is `kit/overlap/v4xy_base.env`. The E7 settings of the gen70 run
  are not recorded here.

## Building the C++ engine

```bash
pip install pybind11
cd kit/fast && bash build.sh      # builds kagsim for the running Python
```
