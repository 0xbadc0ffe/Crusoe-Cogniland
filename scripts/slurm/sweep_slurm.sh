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
#SBATCH --array=0-35

# ── Sweep grid ─────────────────────────────────────────────────────────────────
#
#   Axis           Values
#   -----------    -----------------------------------------
#   time_penalty   0.01  0.05  0.10  0.20          (4 vals)
#   lambda_p       0.2   0.5   1.0                 (3 vals)
#   difficulty     default  diff_half  diff_fifth   (3 vals)
#
#   Index layout:  TASK_ID = diff_idx*12 + tp_idx*3 + lp_idx
#   Total:         4 × 3 × 3 = 36 jobs
#
# ───────────────────────────────────────────────────────────────────────────────

TIME_PENALTIES=(0.01 0.05 0.10 0.20)
LAMBDA_PS=(0.2 0.5 1.0)
DIFFICULTIES=(default diff_half diff_fifth)

# Decode TASK_ID → indices
diff_idx=$(( SLURM_ARRAY_TASK_ID / 12 ))
tp_idx=$(( (SLURM_ARRAY_TASK_ID % 12) / 3 ))
lp_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))

TIME_P=${TIME_PENALTIES[$tp_idx]}
LAMBDA_P=${LAMBDA_PS[$lp_idx]}
DIFF=${DIFFICULTIES[$diff_idx]}

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/crusoe"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"

PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
cd "$PROJECT_DIR"

# ── Debug info ─────────────────────────────────────────────────────────────────
echo "Job ID:        $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "time_penalty:  $TIME_P  |  lambda_p: $LAMBDA_P  |  difficulty: $DIFF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── WandB ──────────────────────────────────────────────────────────────────────
set -a; source "$PROJECT_DIR/.env"; set +a

export WANDB_TAGS="sweep,tp_${TIME_P},lp_${LAMBDA_P},diff_${DIFF}"
export WANDB_RUN_NAME="tp${TIME_P}_lp${LAMBDA_P}_${DIFF}"
export WANDB_GROUP="sweep_$(date +%Y%m%d)"

# ── Training ───────────────────────────────────────────────────────────────────
python train.py \
    env=$DIFF \
    models=ppo \
    logging.wandb.mode=online \
    env.seed=$SLURM_ARRAY_TASK_ID \
    env.time_penalty=$TIME_P \
    env.lambda_p=$LAMBDA_P

echo "Done: $(date)"
