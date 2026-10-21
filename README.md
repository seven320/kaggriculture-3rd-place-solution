# Kaggriculture: 3rd place solution (team DSM)

Code for team DSM's 3rd-place solution to the Kaggle competition "Kaggriculture".

- Write-up: `<link to the Kaggle write-up, to be added>`
- Team members whose work is referenced here (Kaggle handles): denden, masspeaks, shimishige

The final agent is a neural policy trained with the team's BC/PPO pipeline (final weights "gen70"), followed by rule-based post-processing stages inside
`agent()`. This repository contains the final submissions (without weights), the post-processing stages, the BC/PPO training kit,
the Rust game engine used for local evaluation, and the code of the post-processing A/B reported in the write-up (§9.1). The
weights are published separately in the Kaggle Dataset [kaggriculture-dsm-3rd-place-weights](https://www.kaggle.com/datasets/denden12/kaggriculture-dsm-3rd-place-weights): `gen70/` (final
submission) and `sub112/` (the starting checkpoint of the BC runs that produced MMPQ-BCw, the start of the gen70 PPO run).

## What can be reproduced from the published material

| Result | With this repository + the Kaggle Dataset |
|---|---|
| The final submissions sub125 / sub126 (`submission/README.md`) | yes: code here, gen70 weights in the Dataset (sha256 checked) |
| Rust engine build and tests (`engine/rust`) | yes, CPU only |
| C++ engine build (`training/kit/fast`) | yes, needs `pybind11` |
| Equivalence test of the post-processing (`postprocess/`) | no: needs the weights of the team's submission sub121; the recorded result is `postprocess/validation.json` |
| Post-processing A/B of write-up §9.1 (`eval/`) | no: needs the weights of the 7 opponents (team submissions and generations of a teammate's run); the recorded result is `docs/results/` |
| Training gen70 from sub112 (`training/`) | the code and the starting weights are here; the env file of the original run, the critics and the league weights are not (`training/README.md`). Reinforcement learning does not give the same weights again |

## Layout

| Path | Content | Status |
|---|---|---|
| `postprocess/` | the post-processing stages used in the final submissions (byte-identical copies) and their equivalence test | included |
| `engine/rust/` | Rust re-implementation of the game rules (`kaggriculture-engine`), the Python agent bridge and simulation server, the trace parser, and the batched FFI library used by the evaluation harness | included |
| `eval/` | the A/B evaluation code (stage chain, harness, protocol, statistics), opponent code, `setup_inputs.sh` | included; needs the weights and a CUDA GPU |
| `docs/results/` | report of the gen70 post-processing A/B (Japanese, with an English summary) | included |
| `submission/` | final submissions sub125 and sub126 (all files except the weights) | included |
| `training/` | masspeaks' BC/PPO kit (BC, PPO, C++ engine `kagsim`) and our local runner | included (not the env file of the gen70 run itself; see `training/README.md`) |
| `LICENSE`, `LICENSES/`, `NOTICE` | licences | |

## Post-processing

Order in the final submissions, after the network and the policy's own `_postprocess` (`_land(_pace(.))`):

1. `postprocess_c1p_w9p.py`: W3, W5, W6 (replace wasted unit actions with the next-best action by network score), W9 and W9' (sell
   what the end-of-day deposit would discard; use empty order slots), WT (sell leftovers at the last step).
2. `postprocess_d23s_p1.py`: D23S_p1, deposit at hour 23 on days 0 to 28 and sell items priced at or above base.
3. `collect_plan_d29.py`: day-29 collection plan from step 708 (alpha = 0.01).
4. MELON16 (sub125 and sub126), then melon replanting (sub126 only): `submission/sub126/melon16.py`, `melon_replant.py`.

Details are in `postprocess/README.md`.

### Running the equivalence test (CPU)

Requires Python 3.11, `torch`, `numpy` and `kaggle-environments==1.32.7`. It needs the replays of 8 public leaderboard episodes and
the team's submission sub121 with the stage attached (code in this repository; its weights are not published); see `postprocess/README.md`.

```bash
python postprocess/test_postprocess_c1p_w9p.py --replays /path/to/replays --example /path/to/example_sub121 --jobs 8
```

The recorded output is `postprocess/validation.json` (50,704 calls, 0 mismatches, passed).

## Rust engine

```bash
cd engine/rust
cargo build --release     # CPU only; builds simulation-server, policy-planning-server, replay_fixture,
                          # bench_fixture and libkagg_ppo_native.so into target/release/
cargo test --release      # 31 tests, no external data
```

The engine was checked against the official Python environment (kaggle-environments 1.32.7) transition by transition during the
competition; the unit tests here include values exported from the official interpreter. Rust edition 2024 (tested with cargo 1.97).

## Post-processing A/B (write-up §9.1)

Same-game comparison on gen70: 7 team opponents, seeds 174300000 to 174300127, both players, 1,792 games per arm, identical games
across arms, Rust engine.

| Arm | Content | Win rate (draw = 0.5) |
|---|---|---:|
| A | network + `_postprocess` only | 74.22% |
| C | A + W3/W5/W6/W9'/WT + day-29 collection plan | 79.21% |
| B | C + D23S_p1 | 79.69% |
| D | B + MELON16 (= sub125) | 87.83% |
| E | D + melon replanting (= sub126) | 86.16% |

D − A = +13.62 pp [+10.88, +16.41] (95% bootstrap over seeds). The full report, including per-opponent tables and integrity
checks, is `docs/results/gen70_postprocess_ab.ja.md`.

How it was run (from the repository root; the paths of the original commands were adapted):

```bash
H=eval/experiments/gen70_postproc_20261017
export KAGG_SCRATCH=/abs/path/to/scratch   # after eval/setup_inputs.sh
ENV="ED_START=708 ED_ALPHA=0.01 EV_V=0 PLAN_PROCS=2"
env $ENV python $H/run_gen70.py stage                                   # staging, once
for a in A D; do env $ENV python $H/run_gen70.py smoke $a & done; wait  # smoke + parity against the submissions' agent()
python $H/report_gen70.py smoke
for a in A C B D E; do                                                  # main run, one arm at a time, 11 parts each
  for k in $(seq 0 10); do env $ENV python $H/run_gen70.py part $a $k 11 & done; wait
done
python $H/report_gen70.py games
```

To re-run it you need the weights of the opponents (not published; only gen70 is in the Kaggle Dataset) and a CUDA GPU;
`eval/setup_inputs.sh` lays out the inputs. The recorded
`code_sha256` in `protocol.json` must be regenerated first, because path constants were changed for publication (see
`eval/README.md`).

## Licence

- The repository's own code and documents: [CC BY 4.0](LICENSE) (the competition's winner licence).
- Parts derived from kaggle-environments 1.32.7 (Apache License 2.0) remain under Apache-2.0: the rule definitions copied into
  `postprocess/postprocess_c1p_w9p.py` and `eval/experiments/share_w9t_20260929/postprocess_w9t.py`, and the Rust port of the game
  rules in `engine/rust/kaggriculture-engine`. See [NOTICE](NOTICE) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- The C++ engine of the training kit (`training/kit/fast/pyrandom.hpp`, `validate.cpp`, `sim.hpp`, `sim_var.hpp`, `export_trace.py`)
  is based on nikital7's public Kaggle notebook
  ["4000x environment speedup | Kaggriculture"](https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture)
  and remains under Apache-2.0 (see NOTICE).
