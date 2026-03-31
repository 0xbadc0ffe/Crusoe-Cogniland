#!/bin/bash
#SBATCH --job-name=cogniland_continue_160m
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=logs/continue_160m_%A_%a.log
#SBATCH --error=logs/continue_160m_%A_%a.err
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
#SBATCH --array=0-5

# ── Continue training from 1m_artifacts checkpoints ────────────────────────────
#
#   Each array task resumes one of the 6 saved checkpoints and trains for a
#   total budget of 160 M env moves (no curriculum).
#
#   Array  Checkpoint run
#   -----  --------------------------
#     0    1m_artifacts/9347uduq/ckpt_best.pt
#     1    1m_artifacts/c6mgazd4/ckpt_best.pt
#     2    1m_artifacts/iugy3zlz/ckpt_best.pt
#     3    1m_artifacts/le5r3dpn/ckpt_best.pt
#     4    1m_artifacts/um5ur4wo/ckpt_best.pt
#     5    1m_artifacts/v8qxpuj6/ckpt_best.pt
#
# ───────────────────────────────────────────────────────────────────────────────

RUN_IDS=(9347uduq c6mgazd4 iugy3zlz le5r3dpn um5ur4wo v8qxpuj6)

IDX=$SLURM_ARRAY_TASK_ID
RUN_ID=${RUN_IDS[$IDX]}

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/crusoe"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

CKPT_PATH="$PROJECT_DIR/1m_artifacts/${RUN_ID}/ckpt_best.pt"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ── Debug info ─────────────────────────────────────────────────────────────────
echo "Job ID:        $SLURM_JOB_ID (array task $IDX)"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Run ID:        $RUN_ID"
echo "Checkpoint:    $CKPT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── GPU sanity check ───────────────────────────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" || {
    echo "ERROR: GPU not available on $(hostname) — requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 1
}

# ── WandB ──────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

export WANDB_TAGS="continue_160m,no_curriculum,$RUN_ID"
export WANDB_RUN_NAME="cont_${RUN_ID}"
export WANDB_GROUP="continue_160m_$(date +%Y%m%d)"
# Guarantee a unique artifacts sub-dir even if WandB is offline ("local" fallback)
export WANDB_RUN_ID="cont_${RUN_ID}"

# ── Training ───────────────────────────────────────────────────────────────────
python train.py \
    models=ppo_1m \
    resume="$CKPT_PATH" \
    models.training.total_env_moves=160_000_000 \
    models.training.dataset.curriculum_switch_steps=0 \
    models.training.dataset.curriculum_switch_steps_2=0 \
    logging.wandb.mode=online

echo "Done: $(date)"
