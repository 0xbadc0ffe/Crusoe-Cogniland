#!/usr/bin/env bash
#SBATCH --job-name=bt_forkwall_dreamer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# -----------------------------------------------------------------------------
# R2-Dreamer on bridge_tunnel fork_wall, BT-rules (commit disabled). Never
# trained with any Dreamer variant before -- PPO+GRU is the only released
# agent (100% success, configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml).
#
# Vector obs (one-hot minimap + scalars), no CNN -- mirrors the pure-JAX
# Dreamer baseline's proven encoding for this env. Raised model.act_entropy
# (0.01 vs the 3e-4 paper default): the pure-JAX Dreamer needed exactly this
# to learn the analogous category-dependent skill on the btc variant.
#
# Submit:  sbatch scripts/bridge_tunnel/slurm/train_bridge_tunnel_forkwall.sbatch
# Re-submitting with the SAME logdir warm-starts from its `latest.pt` (see
# train.py) instead of random init -- safe to resubmit with a larger STEPS if
# one 24h job isn't enough.
#
# Env vars (optional):
#   REPO        path to the repo on the cluster   (default: $SLURM_SUBMIT_DIR)
#   CONDA_ENV   conda env name                    (default: r2dreamer)
#   STEPS       env steps                          (default: 8e6)
#   MODEL       r2dreamer size preset             (default: size25M)
#   SEED        training seed                      (default: 0)
#   TAG         logdir suffix for a fresh arm      (default: '' -> forkwall_nocommit)
#   ACT_ENTROPY actor entropy coefficient          (default: 0.01)
# -----------------------------------------------------------------------------
set -eo pipefail   # NOT -u (crusoe MKL activate reads unset vars)

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
STEPS="${STEPS:-8e6}"
MODEL="${MODEL:-size25M}"
SEED="${SEED:-0}"
TAG="${TAG:-}"
ACT_ENTROPY="${ACT_ENTROPY:-0.01}"
RUN_NAME="forkwall_nocommit${TAG:+_$TAG}"

echo "[job] $(date) host=$(hostname) repo=$REPO env=$CONDA_ENV model=$MODEL steps=$STEPS run=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

srun python r2dreamer_model/train.py \
  env=bridge_tunnel_forkwall env.task=bridgetunnel_forkwall \
  model="${MODEL}" env.steps="${STEPS}" \
  model.act_entropy="${ACT_ENTROPY}" \
  device=cuda:0 seed="${SEED}" \
  logdir="r2dreamer_model/runs/${RUN_NAME}"

echo "[job] $(date) ${RUN_NAME} finished -> r2dreamer_model/runs/${RUN_NAME}/latest.pt"
