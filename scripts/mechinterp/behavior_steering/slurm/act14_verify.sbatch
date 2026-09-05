#!/usr/bin/env bash
#SBATCH --job-name=ppo_verify
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# act14 verify only: re-run the per-arm winner under RULE on the 36 held-out
# act-11 maps (sweep + analyze must have run already).
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
VROLLS="${VROLLS:-10}"; WORKERS="${WORKERS:-44}"; RULE="${RULE:-guard90}"
echo "[job] $(date) host=$(hostname) vrolls=$VROLLS rule=$RULE"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_WARNINGS=0
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
srun python scripts/mechinterp/behavior_steering/act14_sweep.py --stage verify \
  --maps 36 --rolls "$VROLLS" --workers "$WORKERS" --rule "$RULE"
echo "[job] $(date) done"
