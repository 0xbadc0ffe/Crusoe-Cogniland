#!/bin/bash
#SBATCH --job-name=cogniland_ppo
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                       # Enough for env workers + data loading
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00     
#SBATCH --output=logs/cogniland_%j.log
#SBATCH --error=logs/cogniland_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/cluster/software/anaconda3/bin:$PATH"
source activate /cluster/raid/home/fwang/.conda/envs/crusoe

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

# ── Debug info ────────────────────────────────────────────────────────────────
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Date:     $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── WandB (optional: set to disabled for a dry run) ──────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a   # load .env (exports WANDB_API_KEY etc.)
# export WANDB_MODE=disabled                  # uncomment to disable wandb

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \
    models=ppo \
    logging.wandb.mode=online \
    env.seed=$SLURM_JOB_ID        # use job ID as seed for reproducible but varied runs

echo "Done: $(date)"
