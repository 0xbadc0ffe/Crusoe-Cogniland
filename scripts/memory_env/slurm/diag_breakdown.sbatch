#!/usr/bin/env bash
#SBATCH --job-name=mem_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r2dreamer
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
SUFFIX="${SUFFIX:-.2m}"   # run-dir suffix: e.g. ".2m", "_gate", "_rew2"
srun python scripts/memory_env/diag_breakdown.py \
  --ckpt-2cue "r2dreamer_model/runs/memory_2cue${SUFFIX}/latest.pt" \
  --ckpt-3cue "r2dreamer_model/runs/memory_3cue${SUFFIX}/latest.pt" \
  --ckpt-4cue "r2dreamer_model/runs/memory_4cue${SUFFIX}/latest.pt" \
  --device cuda:0 --n 96
echo "[diag] done"
