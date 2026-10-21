#!/bin/bash
# PPO smoke on one GPU: gen41 as INIT, 64 games per iteration, 3 iterations
# (VWARM=2 makes iterations 0 and 1 value-only; iteration 2 is the first policy update with the AUX_BC loss).
# Needs the split bc_mmpq_smoke.json and its cache (made by smoke/bc.sh).
# Pass: no Traceback and three "it N self ..." lines in $ROOT/smoke/ppo.log.
#
# MICRO stays at the env's 4 (a minibatch of 2048 is updated in 4 parts). With MICRO=1 the update reserved
# 34.6 GB on a 24 GB GPU and one value-only iteration took 743 s (2026-10-03, RTX 4090).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
mkdir -p "$ROOT/smoke"
rm -rf "$LOGS"/orbit_smoke_ppo* "$ROOT/content/async/smoke_ppo"   # start from INIT every time, not from the last smoke
"$TB_HERE/run_ppo.sh" "$TB_HERE/envs/teamabcw.env" sync \
  TAG=smoke_ppo ASYNC_DIR=$ROOT/content/async/smoke_ppo \
  AUX_BC_SPLIT=mmpq_smoke AUX_BC_CACHE=$ROOT/cluster/ext4p/mmpq_smoke \
  NSELF=16 NBEST=16 NLEAGUE=24 NTAPEUG=8 KSHARD=1 ITERS=3 > "$ROOT/smoke/ppo.log" 2>&1
echo "exit=$?"
grep -E "^it +[0-9]+ self|Traceback|Error" "$ROOT/smoke/ppo.log" | cut -c1-400
