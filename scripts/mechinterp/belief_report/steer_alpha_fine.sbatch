#!/usr/bin/env bash
#SBATCH --job-name=steer_alpha
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Dose-response of the belief write h <- h + alpha v at corr2 entry, PPO, alpha 0..2 step 0.1.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_WARNINGS=0
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:${PYTHONPATH:-}"
ALPHAS=$(python -c "print(','.join(f'{i/10:.1f}' for i in range(0, 21)))")
echo "[job] $(date) host=$(hostname) alphas=$ALPHAS"
MODE="${MODE:-add}"; TAG="${TAG:-fine}"; EXTRA="${EXTRA:---control}"
echo "[job] mode=$MODE tag=$TAG"
srun python scripts/mechinterp/belief_report/steer_alpha.py --agent ppo --n 100 --workers 44 --site corr2 --alphas "$ALPHAS" --mode "$MODE" --tag "$TAG" $EXTRA
echo "[job] $(date) done"
