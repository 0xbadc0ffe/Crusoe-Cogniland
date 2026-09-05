#!/usr/bin/env bash
#SBATCH --job-name=pg_dist
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
#SBATCH --error=scripts/mechinterp/behavior_steering/slurm/logs/%x_%j.out
# act2 per-step gradient steering, door-outcome distributions: no-mine on
# rocky, no-build on lakes, both (separately) on balanced; K rollouts per map
# with paired seeds. final_models PPO, CPU.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DOSES="${DOSES:--0.1,-0.25,-0.5,-0.75}"
NMAPS="${NMAPS:-30}"
ROLLS="${ROLLS:-10}"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
export PYTHONPATH="$REPO/src:$REPO/scripts/mechinterp:$REPO/scripts/mechinterp/belief_report:$REPO/scripts/figures:$REPO/scripts/mechinterp/behavior_steering:${PYTHONPATH:-}"
srun python scripts/mechinterp/behavior_steering/act2_ppo.py \
  --stage dist --doses="$DOSES" --n-maps "$NMAPS" --rolls "$ROLLS"
echo "[job] $(date) done"
