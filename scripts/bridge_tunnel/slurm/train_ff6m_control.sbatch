#!/usr/bin/env bash
#SBATCH --job-name=ff6m_ctrl
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%A_%a.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Feed-forward (memoryless) control for the IAB paper: the belief-free PPO
# recipe of outputs/ppo_noaux/noaux_ent15_s* (ent 0.15 annealed, belief_coef 0,
# 6M steps) with --no-recurrence, so the ONLY difference from the released
# agent is the GRU. One seed per array task.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SEED="${SLURM_ARRAY_TASK_ID}"
RUN="ff6m_control_s${SEED}"
echo "[job] $(date) host=$(hostname) run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
srun python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
  --config configs/bridge_tunnel/btc_ppo_forkwall_plain_solved.yaml \
  --maps-path data/bridge_tunnel/forkwall6k/train.pkl \
  --total-timesteps 6000000 --seed "$SEED" \
  --ent-coef 0.15 --anneal-ent --belief-coef 0 --no-recurrence \
  --wandb-mode offline --wandb-project bridge_tunnel \
  --run-name "$RUN" --checkpoint-dir "outputs/ppo_noaux/$RUN"
echo "[job] $(date) done"
