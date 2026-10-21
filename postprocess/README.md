# Post-processing stages (denden)

These are rule-based stages applied to the network's action inside `agent()`. They are not trained. The three modules here are
**byte-identical** to the copies in the final submissions sub125 and sub126:

| File | sha256 (same as in the final submissions) | Role |
|---|---|---|
| `postprocess_c1p_w9p.py` | `ff95f1117b07027724c2974cb32c195ef5435b05830b20f76928c3c417a94eb9` | W3 + W5 + W6 + W9 + W9' + WT: replace wasted unit actions, sell overflow and leftovers |
| `postprocess_d23s_p1.py` | `438d043a83d9f462b127575235d0d75d243da617a274585861452670d9acf74b` | D23S_p1: at hour 23 of days 0 to 28, deposit at the shed and sell items priced at or above base |
| `collect_plan_d29.py` | `563977472ac0ac0a6e59827f5adb7aeb7ab159e140f324da6830d4bb0f055505` | Day-29 collection plan from step 708 (alpha = 0.01) |

All three use only the Python standard library. `postprocess_d23s_p1.py` imports `postprocess_c1p_w9p.py` from the same folder.
Each stage returns the action it was given if it raises an exception (and `postprocess_c1p_w9p` also does so when it exceeds 1 second
for one observation).

The comments in these files are in Japanese and mention the private experiment folders as `artifacts/<name>/`. The same folders are
in this repository as `eval/experiments/<name>/`. The files were left unedited so that their hashes match the submitted files.

## What each stage does

`postprocess_c1p_w9p.py` (`KitHook` wraps the network's `unit_logits` to remember the scores of the forward pass):

| Code | Target | Change |
|---|---|---|
| W3 | CARE whose first production day it would affect falls after the end of the game | the unit's next-best action by network score (moves included) |
| W5 | WATER that cannot lead to a harvest or a yield increase late in the game | same |
| W6 | WATER on an ongoing crop on the last day, FEED on the last day, FERTILIZE with no yield opportunity in the extended window | same |
| W9 | Hour 23: products that the end-of-day automatic deposit would discard because the shed (capacity 100) is full | extra SELL orders, up to the 10 order slots |
| W9' | Hour 23, when the market orders contain empty orders `[]` | treats `[]` as free slots: recomputes W9 without them and writes the missing SELL orders into the `[]` slots, front first |
| WT | Last step (718): items left in the shed | extra SELL orders |

W9' exists because an empty order `[]` does nothing under the official rules (`_parse_order`) but still counts as one of the 10 order
slots (`q[:max_orders]` in `_process_market`).

`postprocess_d23s_p1.py`: on days 0 to 28 at hour 23, a unit standing on a shed access tile whose action is a move, PASS, DROP or
PICKUP is changed to DROP. For STRAWBERRY, MILK, MELON, CARROT, TOMATO and EGG whose price / base price is at least 1.0, the deposited
quantity is sold in the same step. WHEAT, FERTILIZER and WOOL are deposited but not sold. Day 29 is untouched.

`collect_plan_d29.py`: from step 708 to 718 on day 29, replaces the units' actions with a multi-unit collection route (harvest,
water-then-harvest, deposit) and sells in advance what would overflow the shed. One instance per game.

## Order in the final submissions

```python
# after _NET is built
import postprocess_c1p_w9p as _C1P
_C1P_HOOK = _C1P.KitHook(_NET)
import collect_plan_d29 as _CP
_CP_PLAN = _CP.CollectPlanner(start_step=708, alpha=0.01)
import postprocess_d23s_p1 as _D23S

# end of agent(): network -> _postprocess -> W3..WT -> D23S_p1 -> day-29 plan
return _CP_PLAN(converted, _D23S.apply(converted, config, _C1P_HOOK.apply(converted, config, _postprocess(_act(converted), converted))))
```

sub125 and sub126 add two more stages after the day-29 plan (MELON16, and in sub126 also a melon replanting stage). Those files
come with the submissions (see `../submission/README.md`).

## Equivalence test

`test_postprocess_c1p_w9p.py` checks that `postprocess_c1p_w9p` gives exactly the same output as the implementation used in the
experiments (`eval/experiments/sub121_c1c2_20260930/run_c.py` path + `run_m.MTop.m5`, built on
`eval/experiments/pass_noop_replace_v104_20260929/waste_replace2.py` and `eval/experiments/sub121_market_pp_20260930/market_pp.py`).
It also checks that the copied rule definitions are identical to the installed `kaggle_environments` 1.32.7.

It replays every step of 8 leaderboard episodes (both players) and calls the stage on six kinds of inputs: the network's actions,
the recorded actions, random actions, random scores, actions padded with `[]`, and synthetic shed-overflow states. Pass condition:
same output, same events, no mutation of the inputs, every code (W3, W5, W6, W9, W9', WT) fires at least once.

Requirements: Python 3.11, `torch`, `numpy`, `kaggle-environments==1.32.7`, CPU only.

Inputs that are not in this repository:

- `--replays DIR` (or `KAGG_REPLAYS_DIR`): the episode files `<episode>.json.gz` for episodes 114643260, 114563237, 114638818,
  114551580, 114558870, 114641776, 114624074, 114539949, from Kaggle's public episode data (for example the replay JSON of each
  episode, compressed with `gzip`).
- `--example DIR` (or `KAGG_EXAMPLE_SUB121`): the team's Kit-type submission sub121 with this stage attached. It provides the network
  scores. Build it from the sub121 code and `example_sub121/main.py` (sub121's `main.py` plus 3 lines that attach
  `postprocess_c1p_w9p`; its sha256 is `example_main_sha256` in `validation.json`):

  ```bash
  mkdir example_sub121
  cp eval/inputs/team_subs_20260930/sub121_v109/source/* example_sub121/
  cp postprocess/example_sub121/main.py postprocess/postprocess_c1p_w9p.py example_sub121/
  cp /path/to/sub121/policy.pt example_sub121/   # sha256 in eval/inputs/team_subs_20260930/sub121_v109/receipt.json
  ```

  The sub121 weights are not published, so this test cannot be re-run from the public material alone; `validation.json` is the
  recorded result.

```bash
python postprocess/test_postprocess_c1p_w9p.py --replays /path/to/replays --example /path/to/example_sub121 --jobs 8
```

`validation.json` is the output of this command run on 2026-10-18 from this repository layout (CPU, 8 processes, 109 s): 50,704 calls,
50,704 equal, 0 mismatches, 0 input mutations, `passed: true`. The totals are identical to the run made when the stage was first
shared (2026-09-30).
