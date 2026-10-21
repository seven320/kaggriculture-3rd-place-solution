#!/bin/bash
# Run the kit's PPO for one line on one GPU.
#   run_ppo.sh ENV_FILE [sync|learner|actor] [KEY=VALUE ...]
# ENV_FILE is sourced on top of the kit's pntom45_base.env; trailing KEY=VALUE pairs override both.
#   sync    : one process alternates games and updates
#   learner / actor : asynchronous, joined through ASYNC_DIR (start one learner and one or more actors)
# The same TAG resumes from $LOGS/orbit_<TAG>.pt (optimizer state included). Delete $ASYNC_DIR/buf after stopping.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ENV_FILE=$(realpath "$1"); MODE=${2:-sync}; shift; shift || true
set -a
source "$KIT_ENVS/pntom45_base.env"
source "$ENV_FILE"
for kv in "$@"; do eval "$kv"; done
set +a
mkdir -p "$ASYNC_DIR"
cd "$KAGG/bc"
case $MODE in
  sync)    exec "$PY" "$SCRIPT" "$TAG" "$INIT" ;;
  learner) ROLE=learner exec "$PY" "$SCRIPT" "$TAG" "$INIT" ;;
  actor)   ROLE=actor RSEED=${RSEED:-$RANDOM} exec "$PY" "$SCRIPT" "$TAG" "$INIT" ;;
  *) echo "mode must be sync, learner or actor" >&2; exit 2 ;;
esac
