#!/usr/bin/env bash
#SBATCH --job-name=fw_fixed_dreamer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# -----------------------------------------------------------------------------
# DreamerV3 (25M, rep_loss=dreamer -> proper reconstruction world model) on the
# fork_wall BT-rules task, trained on the FIXED 6k-map dataset (train split).
# rep_loss=dreamer (not the r2dreamer default) so the world model is generative
# and the imagined rollouts decode natively -> realistic dreams.
# Resubmitting the same logdir warm-starts from latest.pt (train.py).
#
# Env vars: REPO, CONDA_ENV, STEPS (default 8e6), SEED, ACT_ENTROPY (0.01),
#   MAPS (default data/bridge_tunnel/forkwall6k/train.pkl).
# -----------------------------------------------------------------------------
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
STEPS="${STEPS:-8e6}"
SEED="${SEED:-0}"
ACT_ENTROPY="${ACT_ENTROPY:-0.01}"
MODEL="${MODEL:-size25M}"
BATCH_LENGTH="${BATCH_LENGTH:-64}"
IMAG_HORIZON="${IMAG_HORIZON:-15}"
TRAIN_RATIO="${TRAIN_RATIO:-64}"
MAPS="${MAPS:-$REPO/data/bridge_tunnel/forkwall6k/train.pkl}"
RUN_NAME="${RUN_NAME:-forkwall_fixed_dreamer}"

echo "[job] $(date) host=$(hostname) steps=$STEPS maps=$MAPS run=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export BT_MAPS="$MAPS"          # fixed-map dataset for the wrapper

srun python external/r2dreamer/train.py \
  env=bridge_tunnel_forkwall env.task=bridgetunnel_forkwall \
  model="$MODEL" model.rep_loss=dreamer env.steps="$STEPS" \
  model.act_entropy="$ACT_ENTROPY" model.imag_horizon="$IMAG_HORIZON" \
  env.train_ratio="$TRAIN_RATIO" batch_length="$BATCH_LENGTH" \
  device=cuda:0 seed="$SEED" \
  logdir="external/r2dreamer/runs/${RUN_NAME}"
echo "[job] $(date) ${RUN_NAME} done -> external/r2dreamer/runs/${RUN_NAME}/latest.pt"
