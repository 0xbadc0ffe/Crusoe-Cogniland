#!/usr/bin/env bash
#SBATCH --job-name=mem_forkgate_dreamer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# -----------------------------------------------------------------------------
# R2-Dreamer on MemoryEnv 4cue with the NEW marker-door mechanic (explicit
# OPEN action + purple/yellow marker doors at the branch fork -- replaces the
# old passive-position branch detection). ~15 prior r2dreamer runs on the OLD
# mechanic all plateaued at chance-level door selection ("learns direction,
# never solves the door"). A PPO+GRU sanity run on the NEW mechanic (same
# machine, 2026-08-19) went 0.50 -> 0.98 success over 25M steps after a long
# plateau, confirming the new mechanic is solvable -- this is the actual
# Dreamer attempt at it.
#
# Submit:  sbatch scripts/memory_env/slurm/train_memory_forkgate.sbatch
# Re-submitting with the SAME logdir warm-starts from its `latest.pt` (see
# r2dreamer_model/train.py) -- safe to resubmit with a larger STEPS if one
# 24h job doesn't reach the breakthrough (the PPO run's breakthrough came late,
# ~68% through its budget).
#
# Env vars (optional): REPO, CONDA_ENV, STEPS (default 7e6), MODEL (size25M),
#   SEED (0), TAG (logdir suffix).
# -----------------------------------------------------------------------------
set -eo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
STEPS="${STEPS:-7e6}"
MODEL="${MODEL:-size25M}"
SEED="${SEED:-0}"
TAG="${TAG:-}"
RUN_NAME="memory_4cue_forkgate${TAG:+_$TAG}"

echo "[job] $(date) host=$(hostname) repo=$REPO env=$CONDA_ENV model=$MODEL steps=$STEPS run=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

srun python r2dreamer_model/train.py \
  env=memory env.task=memory_4cue \
  model="${MODEL}" env.steps="${STEPS}" \
  device=cuda:0 seed="${SEED}" \
  logdir="r2dreamer_model/runs/${RUN_NAME}"

echo "[job] $(date) ${RUN_NAME} finished -> r2dreamer_model/runs/${RUN_NAME}/latest.pt"
