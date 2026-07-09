#!/usr/bin/env bash
#SBATCH --job-name=memjax_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Per-cue branch/door breakdown for the 3 JAX symbolic models (tag via $TAG).
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crusoe
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
TAG=${TAG:-embed}
for cue in 2cue 3cue 4cue; do
  echo "########## ${cue} (tag=${TAG}) ##########"
  python scripts/memory_env/diag_jax.py --run-dir "outputs/dreamer_runs/memjax_${cue}_${TAG}" --n 96 \
    || echo "diag failed for ${cue}"
done
echo "[memjax_diag] done"
