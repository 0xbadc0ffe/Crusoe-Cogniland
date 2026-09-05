#!/usr/bin/env bash
#SBATCH --job-name=pgs_ppo
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
# Policy Gradient Steering on the PPO+GRU agent (CPU only): fit the direction
# on train-pool maps, then run the dose grid on held-out test maps.
# Env vars: OBJECTIVE (mine|build), ALPHAS, and the pgs_steer.py knobs.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
OBJECTIVE="${OBJECTIVE:-mine}"
ALPHAS="${ALPHAS:-0.05,0.1,0.2,0.4,0.8}"
EXTRA="${EXTRA:-}"
echo "[job] $(date) host=$(hostname) agent=ppo objective=$OBJECTIVE extra=$EXTRA"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
for st in ${STAGES:-fit eval}; do
  srun python scripts/mechinterp/behavior_steering/pgs_steer.py \
    --agent ppo --objective "$OBJECTIVE" --stage "$st" --device cpu \
    --alphas "$ALPHAS" $EXTRA
done
echo "[job] $(date) done"
