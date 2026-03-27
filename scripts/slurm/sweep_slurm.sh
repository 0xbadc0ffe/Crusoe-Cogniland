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
#SBATCH --array=0-26
#SBATCH --exclude=node01,node02,node03

# ── Sweep grid ─────────────────────────────────────────────────────────────────
#
#   Full 3x3x3 grid over progress signal, risk penalty, and death penalty.
#   All other hyperparameters held at default.yaml values.
#
#   Axis            Values                      Baseline
#   --------------  --------------------------  --------
#   lambda_p        0.05* | 0.20  | 0.50         0.05
#   lambda_rho      0.05  | 0.10* | 0.30         0.10
#   lambda_d        0.00  | 0.50  | 1.00*        1.00
#
#   (* = baseline value)
#   Total: 3 x 3 x 3 = 27 jobs  (array indices 0-26)
#
#   Index mapping (row-major: lambda_p outermost, lambda_d innermost):
#     lp_idx   = IDX / 9
#     lrho_idx = (IDX / 3) % 3
#     ld_idx   = IDX % 3
#
# ───────────────────────────────────────────────────────────────────────────────

LP_VALUES=(0.05 0.20 0.50)
LRHO_VALUES=(0.05 0.10 0.30)
LD_VALUES=(0.00 0.50 1.00)

IDX=$SLURM_ARRAY_TASK_ID

LP_IDX=$((IDX / 9))
LRHO_IDX=$(( (IDX / 3) % 3 ))
LD_IDX=$((IDX % 3))

LP=${LP_VALUES[$LP_IDX]}
LRHO=${LRHO_VALUES[$LRHO_IDX]}
LD=${LD_VALUES[$LD_IDX]}

NAME="lp${LP}_lrho${LRHO}_ld${LD}"
OVERRIDE="env.reward.lambda_p=${LP} env.reward.lambda_rho=${LRHO} env.reward.lambda_d=${LD}"

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/crusoe"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

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
    echo "ERROR: GPU not available on $(hostname) -- aborting job $SLURM_JOB_ID"
    exit 1
}

# ── WandB ──────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

export WANDB_TAGS="sweep,reward_sweep,$NAME"
export WANDB_RUN_NAME="$NAME"
export WANDB_GROUP="reward_sweep_$(date +%Y%m%d)"

# ── Training ───────────────────────────────────────────────────────────────────
python train.py \
    models=ppo \
    logging.wandb.mode=online \
    $OVERRIDE

echo "Done: $(date)"
