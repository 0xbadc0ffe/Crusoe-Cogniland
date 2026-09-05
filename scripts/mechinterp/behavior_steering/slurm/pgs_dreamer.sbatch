#!/usr/bin/env bash
#SBATCH --job-name=pgs_dreamer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
# Policy Gradient Steering on Dreamer (RSSM deter site): fit then dose grid.
# Env vars: OBJECTIVE (mine|build), ALPHAS, and the pgs_steer.py knobs.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
OBJECTIVE="${OBJECTIVE:-mine}"
ALPHAS="${ALPHAS:-0.05,0.1,0.2,0.4,0.8}"
EXTRA="${EXTRA:-}"
echo "[job] $(date) host=$(hostname) agent=dreamer objective=$OBJECTIVE extra=$EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/r2dreamer_model:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
for st in ${STAGES:-fit eval}; do
  srun python scripts/mechinterp/behavior_steering/pgs_steer.py \
    --agent dreamer --objective "$OBJECTIVE" --stage "$st" --device cuda \
    --alphas "$ALPHAS" $EXTRA
done
echo "[job] $(date) done"
