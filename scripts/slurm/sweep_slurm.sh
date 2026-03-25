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
#SBATCH --array=0-11

# ── Sweep grid ─────────────────────────────────────────────────────────────────
#
#   Each job is one hand-picked config exploring a different axis.
#   Baseline (job 0) uses default.yaml values as-is.
#
#   ID  Description                 Override
#   --  --------------------------  ----------------------------------------
#    0  baseline                    (none — default.yaml)
#    1  higher progress weight      lambda_p=0.08
#    2  lower progress weight       lambda_p=0.02
#    3  higher risk penalty         lambda_rho=0.7
#    4  lower risk penalty          lambda_rho=0.2
#    5  higher reach bonus          reach_bonus=200
#    6  lower reach bonus           reach_bonus=100
#    7  higher time bonus           lambda_t=80
#    8  higher death penalty        lambda_d=1.5
#    9  higher raft cost            beta_raft=20
#   10  higher LR                   learning_rate=1e-3
#   11  lower LR                    learning_rate=2e-4
#
#   Total: 12 jobs
#
# ───────────────────────────────────────────────────────────────────────────────

# Overrides per job (empty string = baseline)
OVERRIDES=(
    ""                                                     #  0: baseline
    "env.reward.lambda_p=0.08"                             #  1: higher progress
    "env.reward.lambda_p=0.02"                             #  2: lower progress
    "env.reward.lambda_rho=0.7"                            #  3: higher risk penalty
    "env.reward.lambda_rho=0.2"                            #  4: lower risk penalty
    "env.reward.reach_bonus=200"                           #  5: higher reach bonus
    "env.reward.reach_bonus=100"                           #  6: lower reach bonus
    "env.reward.lambda_t=80"                               #  7: higher time bonus
    "env.reward.lambda_d=1.5"                              #  8: higher death penalty
    "env.reward.beta_raft=20"                              #  9: higher raft cost
    "models.training.learning_rate=1e-3"                   # 10: higher LR
    "models.training.learning_rate=2e-4"                   # 11: lower LR
)

NAMES=(
    "baseline"
    "lp_0.08"
    "lp_0.02"
    "lrho_0.7"
    "lrho_0.2"
    "reach_200"
    "reach_100"
    "lt_80"
    "ld_1.5"
    "raft_20"
    "lr_1e-3"
    "lr_2e-4"
)

IDX=$SLURM_ARRAY_TASK_ID
OVERRIDE=${OVERRIDES[$IDX]}
NAME=${NAMES[$IDX]}

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
echo "Config:        $NAME  |  $OVERRIDE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── WandB ──────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

export WANDB_TAGS="sweep,new_reward_sweep,$NAME"
export WANDB_RUN_NAME="$NAME"
export WANDB_GROUP="sweep_$(date +%Y%m%d)"

# ── Training ───────────────────────────────────────────────────────────────────
python train.py \
    models=ppo \
    logging.wandb.mode=online \
    $OVERRIDE

echo "Done: $(date)"
