#!/bin/bash
# Evaluation smoke.
#   1. fast_league.py (GPU, the kit's C++ engine): gen41 vs gen35, weights only, 8 games.
#   2. official environment (CPU): the gen41 package vs the gen35 package, 1 game of 720 steps.
# Pass: a "g41 対 g35 8局" line, and statuses ["DONE", "DONE"] with 720 steps.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
mkdir -p "$ROOT/smoke"
for g in 41 35; do
  [ -d "$AGENTS_DIR/mmpqbcw_gen${g}_net" ] || "$PY" "$TB_HERE/tools/make_agent.py" "$LOGS/orbit_mmpqbcw_gen$g.pt" mmpqbcw_gen${g}_net
done
rm -f "$ROOT/smoke/eval_g41_g35.json"   # fast_league.py skips pairs already in the output file
(cd "$KAGG/bc" && AGENTS="g41:bc:$AGENTS_DIR/mmpqbcw_gen41_net,g35:bc:$AGENTS_DIR/mmpqbcw_gen35_net" NSEED=8 SEED0=880001 \
  "$PY" "$ROOT/content/lab/fasteval/fast_league.py" "$ROOT/smoke/eval_g41_g35.json" 2>&1 | tail -1)
"$PY" "$TB_HERE/smoke/official_game.py" "$AGENTS_DIR/mmpqbcw_gen41_paceland_liq" "$AGENTS_DIR/mmpqbcw_gen35_paceland_liq" \
  --seed 1 --out "$ROOT/smoke/official_g41_g35.json" 2>/dev/null | tail -1
