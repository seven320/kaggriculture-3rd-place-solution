#!/bin/bash
# BC smoke on one GPU: extract M & M & P & Q seats from one official daily archive, then fine-tune gen41
# on them for 300 steps with the new mask (NM=rel, the same rule as apo_nmbrel.py and the agents).
#   bc.sh OFFICIAL_DAILY.zip
# Pass: no Traceback, and the hold-out loss of "ep 0" below the "初期" line in $ROOT/smoke/bc.log.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
mkdir -p "$ROOT/smoke" "$ROOT/cluster/ext4p/mmpq_smoke"
"$PY" "$TB_HERE/tools/extract_teacher.py" --zips "$1" --out "$ROOT/cluster/ext4p/mmpq_smoke" \
  --records "$ROOT/smoke/extract_records.json" --team "M & M & P & Q" --split mmpq_smoke --limit 1500
cd "$KAGG/bc"
TAG=smoke_mmpq_bc SPLIT=mmpq_smoke CACHE=$ROOT/cluster/ext4p/mmpq_smoke INIT=$LOGS/orbit_mmpqbcw_gen41.pt \
  NM=rel NMB_PEN=6 LANDW=20 EPOCHS=1 MAXSTEPS=300 LRMAX=2e-5 "$PY" bc_teama_nm.py > "$ROOT/smoke/bc.log" 2>&1
grep -E "初期|ep +[0-9]|未見の損失|Traceback" "$ROOT/smoke/bc.log" | cut -c1-200
