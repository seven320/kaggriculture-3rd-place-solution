#!/bin/bash
# Lay out the inputs of the post-processing A/B under $KAGG_SCRATCH.
#   KAGG_SCRATCH=/abs/path/to/scratch KAGG_WEIGHTS=/abs/path/to/weights bash eval/setup_inputs.sh
# KAGG_WEIGHTS holds one policy.pt per policy, in folders named as below. Only gen70 is published (Kaggle Dataset
# denden12/kaggriculture-dsm-3rd-place-weights); the other policies' weights are not.
#   sub120_v108 sub121_v109 sub122 sub123s sub124 mmpqbcw_gen32 mmpqbcw_gen33 mmpqbcw_gen34 mmpqbcw_gen35 mmpqbcw_gen41 gen70
# Every copied policy.pt is checked against the sha256 recorded in the receipts / submission README.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PUB=$(dirname "$HERE")
: "${KAGG_SCRATCH:?set KAGG_SCRATCH}"; : "${KAGG_WEIGHTS:?set KAGG_WEIGHTS}"
mkdir -p "$KAGG_SCRATCH"
cp -r "$HERE/inputs/." "$KAGG_SCRATCH/"

put() {  # put <weights folder name> <destination folder> <expected sha256>
  local src="$KAGG_WEIGHTS/$1/policy.pt" dst="$2/policy.pt"
  [ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
  cp "$src" "$dst"
  echo "$3  $dst" | sha256sum -c --quiet
}
want() {  # sha256 of policy.pt from a receipt.json
  python3 -c "import json,sys; print(next(f['sha256'] for f in json.load(open(sys.argv[1]))['files'] if f['path']=='policy.pt'))" "$1"
}

T0=$KAGG_SCRATCH/team_subs_20260930
for n in sub120_v108 sub121_v109 sub122 sub123s; do put "$n" "$T0/$n/source" "$(want "$T0/$n/receipt.json")"; done
T1=$KAGG_SCRATCH/team_subs_20261001
put sub124 "$T1/sub124/source" "$(want "$T1/sub124/receipt.json")"

GEN70=76ec119f8513c8b728fbb0501b2839a0fe4556fdba96bbf6b8bd7e93b007f23c
mkdir -p "$T1/sub126/source" "$T1/sub125/source"
cp "$PUB/submission/sub126/"* "$T1/sub126/source/"
put gen70 "$T1/sub126/source" "$GEN70"
cp "$T1/sub126/source/"* "$T1/sub125/source/"
cp "$PUB/submission/sub125/main.py" "$T1/sub125/source/main.py"
rm "$T1/sub125/source/melon_replant.py"

MM=$KAGG_SCRATCH/shimishige_teamabcw_gen32_gen34_20260930/extracted
declare -A GSHA=(
  [32]=aa652f9295e8d6ee920abf0eb814f30b0b7501b04d44fd7d6175764c12cabea4
  [33]=18f9b5b399328dc1fe613437be7bc50905fdf19c1da3e0361afa0641de0afde1
  [34]=cbbf9e9ceaf1d955a135cff35d1601ecae79cc80f157ad3f5897f165917ffa35
  [35]=deea2ca5c8c6610af6440db785dbf087019567e8546246126e1e19f253b51f8f
)
for g in 32 33 34 35; do put "mmpqbcw_gen$g" "$MM/teamabcw_gen${g}_paceland_liq" "${GSHA[$g]}"; done
G41=$KAGG_SCRATCH/shimishige_teamabcw_20261001/teamabcw_gen41_paceland_liq
put mmpqbcw_gen41 "$G41" d184128261351d3a222f284e1db91bee7726d59ade5b3760db7800c34fb0dbcc

# learner of the gen70 A/B: gen41's code files with the gen70 weights (run_gen70.py stage checks the sha)
SRC=$KAGG_SCRATCH/gen70_postproc_20261017/gen70_src; mkdir -p "$SRC"
for f in main.py model_ptr_infer.py model_tf.py obs_feat.py cfg.json stock_target.json; do cp "$G41/$f" "$SRC/"; done
put gen70 "$SRC" "$GEN70"
echo "inputs ready under $KAGG_SCRATCH"
