#!/bin/bash
#SBATCH --job-name=cogniland_ppo_rnn
#SBATCH -D ./
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fiwang@ethz.ch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=logs/ppo_rnn_sweep_%A_%a.log
#SBATCH --error=logs/ppo_rnn_sweep_%A_%a.err
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang

# Environment variables (passed via sbatch or launch_sweep.sh):
#   SWEEP_ID        - (required) entity/project/sweep_id
#   RUNS_PER_AGENT  - (optional) number of runs this agent handles (default: 1)

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not set"
    exit 1
fi

RUNS_PER_AGENT=${RUNS_PER_AGENT:-1}

# Load Anaconda
export PATH="/cluster/software/anaconda3/bin:$PATH"

# Activate environment
source activate /cluster/raid/home/fwang/.conda/envs/crusoe

# W&B auth — falls back to a file in $HOME if ~/.netrc isn't accessible
# from the compute nodes. Create once with:
#   echo "<YOUR_WANDB_API_KEY>" > ~/.wandb_key && chmod 600 ~/.wandb_key
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
fi

echo "============================================================"
echo "Sweep Agent — SLURM Array Task"
echo "============================================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Array Task ID:   $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Sweep ID:        $SWEEP_ID"
echo "Runs per agent:  $RUNS_PER_AGENT"
echo "SRC_DIR:         ${SRC_DIR:-<unset>}"
echo "============================================================"
echo "Allocated GPUs:"
nvidia-smi
echo "============================================================"

# Run the W&B agent
# --count N means this agent will pick up and run N jobs from the sweep queue
wandb agent --count "$RUNS_PER_AGENT" "$SWEEP_ID"

echo "Agent $SLURM_ARRAY_TASK_ID completed"
