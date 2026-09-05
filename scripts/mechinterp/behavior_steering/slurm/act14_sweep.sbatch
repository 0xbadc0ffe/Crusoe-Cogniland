#!/usr/bin/env bash
#SBATCH --job-name=ppo_sweep
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# act14: clamp-strength sweep (theta x gate window x arm) on TRAIN balanced maps,
# then verify the per-arm winner on the 36 held-out act-11 maps.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
MAPS="${MAPS:-36}"; ROLLS="${ROLLS:-6}"; VROLLS="${VROLLS:-10}"; WORKERS="${WORKERS:-44}"
RULE="${RULE:-score}"
echo "[job] $(date) host=$(hostname) maps=$MAPS rolls=$ROLLS vrolls=$VROLLS rule=$RULE"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_WARNINGS=0
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
srun python scripts/mechinterp/behavior_steering/act14_sweep.py --stage sweep \
  --maps "$MAPS" --rolls "$ROLLS" --workers "$WORKERS"
srun python scripts/mechinterp/behavior_steering/act14_sweep.py --stage analyze
echo "[job] $(date) verify rule=$RULE"
srun python scripts/mechinterp/behavior_steering/act14_sweep.py --stage verify \
  --maps 36 --rolls "$VROLLS" --workers "$WORKERS" --rule "$RULE"
echo "[job] $(date) done"
