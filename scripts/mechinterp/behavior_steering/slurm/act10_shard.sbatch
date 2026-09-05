#!/usr/bin/env bash
#SBATCH --job-name=tilt_shard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%A_%a.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# -----------------------------------------------------------------------------
# One shard of the plan-tilt evaluation. Each array task takes its own GPU and
# a disjoint slice of the maps, so wall-clock falls roughly linearly in the
# number of shards. Shards write separate json; act10_merge.py combines them.
#
# Env vars: PHASE (fit|test|grid), NSHARD, MAPS, ROLLS, WORKERS, LAM_TUNNEL,
#           LAM_BRIDGE, TAG
#
#   sbatch --array=0-7 scripts/mechinterp/behavior_steering/slurm/act10_shard.sbatch
# -----------------------------------------------------------------------------
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
PHASE="${PHASE:-test}"
NSHARD="${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
MAPS="${MAPS:-24}"
ROLLS="${ROLLS:-6}"
WORKERS="${WORKERS:-4}"
TAG="${TAG:-main}"

echo "[job] $(date) host=$(hostname) phase=$PHASE shard=$SHARD/$NSHARD tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:$REPO/r2dreamer_model:${PYTHONPATH:-}"

srun python scripts/mechinterp/behavior_steering/act10_success.py \
  --phase "$PHASE" --shard "$SHARD" --nshard "$NSHARD" \
  --maps "$MAPS" --rolls "$ROLLS" --workers "$WORKERS" --tag "$TAG" \
  ${LAM_TUNNEL:+--lam-tunnel "$LAM_TUNNEL"} \
  ${LAM_BRIDGE:+--lam-bridge "$LAM_BRIDGE"}
echo "[job] $(date) shard $SHARD done"
