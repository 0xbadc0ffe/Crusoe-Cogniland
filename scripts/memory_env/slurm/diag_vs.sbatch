#!/usr/bin/env bash
#SBATCH --job-name=memvs_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Head-to-head per-cue breakdown: PPO+GRU vs DreamerV3 for tag=$TAG.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crusoe
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
TAG=${TAG:-vs}
for cue in 2cue 3cue 4cue; do
  echo "########## PPO ${cue} (tag=${TAG}) ##########"
  python scripts/memory_env/diag_ppo.py --run-dir "outputs/ppo_runs/ppo_${cue}_${TAG}" --n 96 \
    || echo "diag_ppo failed for ${cue}"
  echo "########## DREAMER ${cue} (tag=${TAG}) ##########"
  python scripts/memory_env/diag_jax.py --run-dir "outputs/dreamer_runs/memjax_${cue}_${TAG}" --n 96 \
    || echo "diag_jax failed for ${cue}"
done
echo "[memvs_diag] done"
