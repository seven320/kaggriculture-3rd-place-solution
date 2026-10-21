# source this. Paths of the runnable tree built by tools/materialize.py.
TB_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath -m "${TEAM_BC_PPO_ROOT:-$TB_HERE/work}")
export TEAM_BC_PPO_ROOT=$ROOT ROOT
export LOGS=$ROOT/content/local_logs            # weights, splits, logs (the kit's /content/local_logs)
export KAGG=$ROOT/content/drive/MyDrive/kaggriculture   # bc/ and fast/
export AGENTS_DIR=$ROOT/content/lab/agents
export KIT_ENVS=$ROOT/content/repo/cluster/overlap  # the kit's env file per PPO line
export PY=${PY:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
