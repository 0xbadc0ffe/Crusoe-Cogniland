#!/usr/bin/env bash
#SBATCH --job-name=mem_dreamer
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%A_%a.out
# -----------------------------------------------------------------------------
# Train the 3 MemoryEnv R2-Dreamer models as a job array (one model per task):
#   array index 0 -> memory_2cue (entangled: green_up / blue_down)
#               1 -> memory_3cue (partial:   + green_down)
#               2 -> memory_4cue (factorized: all four cues)
# Each: size25M, 10M env steps, 16 parallel envs, progress-shaped MemoryEnv.
#
# Submit:  sbatch scripts/memory_env/slurm/train_memory.sbatch
# Cluster-specific knobs (edit above or override on the CLI), e.g.:
#   sbatch --partition=gpu --account=myacct --time=36:00:00 \
#          scripts/memory_env/slurm/train_memory.sbatch
#
# Env vars (optional):
#   REPO        path to the repo on the cluster   (default: $SLURM_SUBMIT_DIR)
#   CONDA_ENV   conda env name                    (default: r2dreamer)
#   STEPS       env steps per model               (default: 10e6)
#   MODEL       r2dreamer size preset             (default: size25M)
#   SEED        training seed                      (default: 0)
# -----------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
STEPS="${STEPS:-10e6}"
MODEL="${MODEL:-size25M}"
SEED="${SEED:-0}"

CUES=(2cue 3cue 4cue)
CUE="${CUES[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "[job] $(date) host=$(hostname) repo=$REPO env=$CONDA_ENV model=$MODEL steps=$STEPS cue=memory_$CUE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# activate conda (works whether conda is a function or only `conda run` exists)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

srun python r2dreamer_model/train.py \
  env=memory env.task=memory_${CUE} \
  model=${MODEL} env.steps=${STEPS} \
  device=cuda:0 seed=${SEED} \
  logdir=r2dreamer_model/runs/memory_${CUE}

echo "[job] $(date) memory_${CUE} finished -> r2dreamer_model/runs/memory_${CUE}/latest.pt"
