#!/usr/bin/env bash
#SBATCH --job-name=fw_fixed_ppo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# -----------------------------------------------------------------------------
# PyTorch PPO+GRU (belief head) on fork_wall BT-rules, trained on the SAME FIXED
# 6k-map dataset (train split) as the DreamerV3 model.
# Env vars: REPO, CONDA_ENV (crusoe), STEPS (default 8e6), SEED, MAPS, WANDB_MODE.
# -----------------------------------------------------------------------------
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-crusoe}"
STEPS="${STEPS:-8000000}"
SEED="${SEED:-0}"
MAPS="${MAPS:-$REPO/data/bridge_tunnel/forkwall6k/train.pkl}"
CONFIG="${CONFIG:-configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml}"
RUN_NAME="${RUN_NAME:-forkwall_fixed_ppo_seed${SEED}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[job] $(date) host=$(hostname) steps=$STEPS maps=$MAPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
set -a; [ -f .env ] && source .env; set +a          # WANDB_API_KEY

srun python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
  --config "$CONFIG" \
  --maps-path "$MAPS" \
  --total-timesteps "$STEPS" --seed "$SEED" \
  --wandb-mode "${WANDB_MODE:-offline}" --wandb-project bridge_tunnel \
  --run-name "$RUN_NAME" \
  --checkpoint-dir outputs/ppo_checkpoints $EXTRA_ARGS
echo "[job] $(date) ppo done"
