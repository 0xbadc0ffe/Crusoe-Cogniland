#!/usr/bin/env bash
#SBATCH --job-name=mem_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
# -----------------------------------------------------------------------------
# Evaluate the 3 trained models on the SAME held-out 4-cue test set and render
# the average-reward-per-cue plot. Run AFTER train_memory.sbatch finishes — the
# submit.sh helper wires the dependency automatically. To run standalone:
#   sbatch scripts/memory_env/slurm/eval_memory.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

srun python scripts/memory_env/eval_r2dreamer.py \
  --ckpt-2cue external/r2dreamer/runs/memory_2cue/latest.pt \
  --ckpt-3cue external/r2dreamer/runs/memory_3cue/latest.pt \
  --ckpt-4cue external/r2dreamer/runs/memory_4cue/latest.pt \
  --device cuda:0

echo "[eval] done -> outputs/report/memoryenv_reward_per_cue.png"
