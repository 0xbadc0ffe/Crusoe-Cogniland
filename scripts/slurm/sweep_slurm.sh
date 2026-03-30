#!/bin/bash
#SBATCH --job-name=cogniland_sweep
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%A_%a.log
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
#SBATCH --array=0-4

# ── Sweep grid ─────────────────────────────────────────────────────────────────
#
#   1-D sweep over lambda_p (cost-to-go progress weight).
#   All other hyperparameters held at default.yaml values.
#
#   Axis            Values                              Baseline
#   --------------  ----------------------------------  --------
#   lambda_p        0.02 | 0.05 | 0.08* | 0.15 | 0.30    0.08
#
#   (* = baseline value)
#   Total: 5 jobs  (array indices 0–4)
#
# ───────────────────────────────────────────────────────────────────────────────

LP_VALUES=(0.02 0.05 0.08 0.15 0.30)

IDX=$SLURM_ARRAY_TASK_ID

LP=${LP_VALUES[$IDX]}

NAME="lp${LP}"
OVERRIDE="env.reward.lambda_p=${LP}"

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/crusoe"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ── Debug info ─────────────────────────────────────────────────────────────────
echo "Job ID:        $SLURM_JOB_ID (array task $IDX)"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Config:        $NAME"
echo "Overrides:     $OVERRIDE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── GPU sanity check ──────────────────────────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" || {
    echo "ERROR: GPU not available on $(hostname) — requeueing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 1
}

# ── WandB ──────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

export WANDB_TAGS="sweep,lambda_p_sweep,$NAME"
export WANDB_RUN_NAME="$NAME"
export WANDB_GROUP="lambda_p_sweep_$(date +%Y%m%d)"

# ── Training ───────────────────────────────────────────────────────────────────
python train.py \
    models=ppo \
    logging.wandb.mode=online \
    $OVERRIDE

echo "Done: $(date)"
