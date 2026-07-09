#!/usr/bin/env bash
#SBATCH --job-name=memjax_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%A_%a.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Pure-JAX DreamerV3 on the SYMBOLIC MemoryEnv with a learned tile-embedding
# encoder. Array of 3: one model per cue set (2cue/3cue/4cue).
#   sbatch --array=0-2 [TAG=... STEPS=...] scripts/memory_env/slurm/dreamerv3_memory.sbatch
set -eo pipefail   # NOT -u (crusoe MKL activate reads unset vars)
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crusoe
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
set -a; [ -f .env ] && source .env; set +a          # WANDB_API_KEY
CUES=(2cue 3cue 4cue)
CUE=${CUES[${SLURM_ARRAY_TASK_ID:-0}]}
TAG=${TAG:-embed}
STEPS=${STEPS:-3000000}
python -c "import jax; print('JAX devices:', jax.devices())"
srun python scripts/memory_env/dreamerv3_memory.py \
  --size "${SIZE:-25M}" --cue "$CUE" --num-envs 32 --train-ratio 64 \
  --wandb-mode "${WANDB_MODE:-online}" --wandb-project memory_env_jax \
  --run-name "memjax_${CUE}_${TAG}" --run-dir outputs/dreamer_runs \
  --set total_env_steps="$STEPS" ${EXTRA_SET:-}
echo "[memjax] done cue=$CUE tag=$TAG"
