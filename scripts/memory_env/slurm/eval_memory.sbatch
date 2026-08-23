#!/usr/bin/env bash
#SBATCH --job-name=mem_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
# The cluster TaskProlog (/etc/slurm/sync/prolog_sync.sh) runs `set -u` and
# dereferences $SRC_DIR before its own empty-guard, so an unset SRC_DIR makes
# the prolog fail with "TaskProlog failed status=1" (it never reaches this
# script). job_memory.slurm exports it; the eval must too. Without this the
# job dies in <1s on ANY node regardless of --exclude.
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
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
  --ckpt-2cue r2dreamer_model/runs/memory_2cue/latest.pt \
  --ckpt-3cue r2dreamer_model/runs/memory_3cue/latest.pt \
  --ckpt-4cue r2dreamer_model/runs/memory_4cue/latest.pt \
  --device cuda:0

echo "[eval] done -> outputs/report/memoryenv_reward_per_cue.png"
