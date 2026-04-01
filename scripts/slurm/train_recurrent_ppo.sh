#!/bin/bash
#SBATCH --job-name=train_rppo
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/crusoe"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ── Debug info ───────────────────────────────────────────────────────────────
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── WandB ────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

# ── Training ─────────────────────────────────────────────────────────────────
python train.py \
    models=recurrent_ppo \
    models.training.total_env_moves=300_000_000 \
    logging.wandb.mode=online

echo "Done: $(date)"
