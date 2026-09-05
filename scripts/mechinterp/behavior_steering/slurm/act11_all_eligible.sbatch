#!/usr/bin/env bash
#SBATCH --job-name=ppo_steer_all
#SBATCH --array=0-2
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%A_%a.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# The act-11 gated clamp (act-5 operating points) on EVERY eligible held-out map
# of one category: screen all 400 maps of the category with two unsteered
# rollouts, keep those where both tools were used, run the three arms.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CATS=(balanced lakes rocky); CAT=${CATS[$SLURM_ARRAY_TASK_ID]}
echo "[job] $(date) host=$(hostname) category=$CAT"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_WARNINGS=0
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
srun python scripts/mechinterp/behavior_steering/act11_ppo.py \
  --category "$CAT" --screen 400 --all-eligible --rolls 6 --workers 44 --tag "all_$CAT"
echo "[job] $(date) done"
