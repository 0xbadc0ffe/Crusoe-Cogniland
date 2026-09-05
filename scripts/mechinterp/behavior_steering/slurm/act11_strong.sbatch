#!/usr/bin/env bash
#SBATCH --job-name=ppo_steer_strong
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# act11 re-run with a STRONGER clamp than the frozen act-5 operating points:
# P(suppressed tool) < THETA at every clamp-active step, ten rollouts per map,
# on the SAME 36 held-out balanced maps as the original act11 grid.
# Two variants back to back: never-released, and released after WINDOW stuck steps.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
MAPS="${MAPS:-36}"; ROLLS="${ROLLS:-10}"; WORKERS="${WORKERS:-44}"
THETA="${THETA:-1e-4}"; ITERS="${ITERS:-80}"; WINDOW="${WINDOW:-20}"
PREV="${PREV:-outputs/behavior_steering/act11/summary.json}"
echo "[job] $(date) host=$(hostname) maps=$MAPS rolls=$ROLLS theta=$THETA iters=$ITERS window=$WINDOW"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
echo "[job] variant A: ungated"
srun python scripts/mechinterp/behavior_steering/act11_ppo.py \
  --maps "$MAPS" --rolls "$ROLLS" --workers "$WORKERS" --reuse-maps "$PREV" \
  --theta "$THETA" --iters "$ITERS" --ungated --tag strong_ungated
echo "[job] $(date) variant B: gated window=$WINDOW"
srun python scripts/mechinterp/behavior_steering/act11_ppo.py \
  --maps "$MAPS" --rolls "$ROLLS" --workers "$WORKERS" --reuse-maps "$PREV" \
  --theta "$THETA" --iters "$ITERS" --window "$WINDOW" --tag strong_gated_w${WINDOW}
echo "[job] $(date) done"
