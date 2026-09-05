#!/usr/bin/env bash
#SBATCH --job-name=ppo_steer
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# PPO+GRU steering: CPU only, so no --gres and it never queues behind the GPU
# jobs. Three arms on held-out balanced maps, plus the corr2 belief readback.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
MAPS="${MAPS:-36}"; ROLLS="${ROLLS:-6}"; WORKERS="${WORKERS:-44}"
echo "[job] $(date) host=$(hostname) maps=$MAPS rolls=$ROLLS"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
srun python scripts/mechinterp/behavior_steering/act11_ppo.py \
  --maps "$MAPS" --rolls "$ROLLS" --workers "$WORKERS"
echo "[job] $(date) done"
