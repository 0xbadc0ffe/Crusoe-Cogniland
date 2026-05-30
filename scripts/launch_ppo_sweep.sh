#!/usr/bin/env bash
# Launch N parallel W&B sweep agents on the local GPU.
# Usage: scripts/launch_ppo_sweep.sh <entity/project/sweep_id> [N=12]
# Each agent pulls trials from the sweep controller and runs them sequentially;
# OMP/MKL threads are pinned to 1 so the ~N numpy envs don't oversubscribe the CPU.
set -uo pipefail
SWEEP="${1:?usage: launch_ppo_sweep.sh <entity/project/sweep_id> [N]}"
N="${2:-12}"
WANDB=/home/filippo/miniconda3/envs/crusoe/bin/wandb
cd "$(dirname "$0")/.."
mkdir -p sweep_logs
for i in $(seq 1 "$N"); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    nohup "$WANDB" agent "$SWEEP" > "sweep_logs/agent_${i}.log" 2>&1 &
  echo "agent $i -> pid $!"
  sleep 3
done
echo "launched $N agents for sweep $SWEEP (logs in sweep_logs/)"
