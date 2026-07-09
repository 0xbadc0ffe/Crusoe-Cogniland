#!/usr/bin/env bash
#SBATCH --job-name=memppo_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%A_%a.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Recurrent PPO (PPO+GRU) on the SYMBOLIC MemoryEnv — same env/obs as the Dreamer
# runs, for a head-to-head. Array of 3: one model per cue set (2cue/3cue/4cue).
#   sbatch --array=0-2 [TAG=... STEPS=...] scripts/memory_env/slurm/ppo_memory.sbatch
set -eo pipefail   # NOT -u (crusoe MKL activate reads unset vars)
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crusoe
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
set -a; [ -f .env ] && source .env; set +a          # WANDB_API_KEY
CUES=(2cue 3cue 4cue)
CUE=${CUES[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=${TAG:-ppo}
STEPS=${STEPS:-20000000}
python -c "import jax; print('JAX devices:', jax.devices())"
srun python scripts/memory_env/train_ppo_memory.py \
  --cue "$CUE" --total-timesteps "$STEPS" --seed "${SEED:-0}" \
  --wandb-mode "${WANDB_MODE:-online}" --tag "$TAG" \
  --set ${EXTRA_SET:-num_envs=256}
echo "[memppo] done cue=$CUE tag=$TAG"
