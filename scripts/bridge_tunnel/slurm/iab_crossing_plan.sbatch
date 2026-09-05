#!/usr/bin/env bash
#SBATCH --job-name=cross_plan
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_WARNINGS=0
export PYTHONPATH="$REPO/src:$REPO/scripts/figures:${PYTHONPATH:-}"
echo "[job] $(date) host=$(hostname)"
srun python scripts/figures/paper/iab_crossing_plan.py --workers 44 ${ARGS:-}
echo "[job] $(date) done"
