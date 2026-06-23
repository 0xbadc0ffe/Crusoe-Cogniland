#!/bin/bash
#SBATCH --job-name=memoryenv_sweep
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=logs/memoryenv_%A_%a.log
#SBATCH --error=logs/memoryenv_%A_%a.err
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
#
# Recycled from job_sweep.slurm for the MemoryEnv R2-Dreamer trainings.
# A wandb grid sweep (configs/sweeps/memory_r2dreamer.yaml) dispatches the 3 cue
# sets (2cue/3cue/4cue) as a 3-task SLURM array; each agent runs ONE training
# via scripts/memory_env/sweep_train.py -> external/r2dreamer/train.py.
#
# Env vars (set by launch_sweep.sh):
#   SWEEP_ID        - (required) entity/project/sweep_id
#   RUNS_PER_AGENT  - (optional) runs per agent (default: 1)

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not set"
    exit 1
fi
RUNS_PER_AGENT=${RUNS_PER_AGENT:-1}

# ── EDIT THESE PATHS FOR YOUR CLUSTER ───────────────────────────────
PROJECT_DIR="/cluster/raid/home/fwang/Crusoe-Cogniland"
# Dedicated r2dreamer env (NOT crusoe — r2dreamer has its own torch/hydra deps).
# Create it once with scripts/memory_env/slurm/setup_env.sh.
CONDA_ENV="/cluster/raid/home/fwang/.conda/envs/r2dreamer"
# ─────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
mkdir -p logs external/r2dreamer/runs

# Activate the conda env by prepending PATH (the agent + wrapper need python,
# wandb, torch). PROJECT_DIR is exported so sweep_train.py can set PYTHONPATH=src
# and find external/r2dreamer.
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export PROJECT_DIR

# W&B auth (WANDB_API_KEY etc. from .env)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

echo "============================================================"
echo "MemoryEnv R2-Dreamer Sweep Agent — SLURM Array Task"
echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Sweep ID:        $SWEEP_ID"
echo "Runs per agent:  $RUNS_PER_AGENT"
echo "PROJECT_DIR:     $PROJECT_DIR"
echo "CONDA_ENV:       $CONDA_ENV"
echo "============================================================"
nvidia-smi
echo "============================================================"

wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID"

echo "Agent $SLURM_ARRAY_TASK_ID completed"
