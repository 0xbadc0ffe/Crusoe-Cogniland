#!/usr/bin/env bash
# Launch N parallel W&B sweep agents for the bridge_tunnel PPO sweep on the local
# 4090. Each agent pulls --count trials from the controller and runs them
# sequentially; OMP/MKL threads are pinned so the per-run numpy env loops
# (32 envs each) don't oversubscribe the 32-core CPU.
# Usage: scripts/launch_bridge_tunnel_sweep.sh <entity/project/sweep_id> [N=9] [COUNT=1]
# Default = 9 runs, all in parallel (keep total sweep size ~9; drop N for
# memory-heavy maps like the 21-view natural ones if the 4090 runs low on VRAM).
set -uo pipefail
SWEEP="${1:?usage: launch_bridge_tunnel_sweep.sh <entity/project/sweep_id> [N] [COUNT]}"
N="${2:-9}"
COUNT="${3:-1}"
WANDB=/home/filippo/miniconda3/envs/crusoe/bin/wandb
cd "$(dirname "$0")/.."
mkdir -p sweep_logs
for i in $(seq 1 "$N"); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    nohup "$WANDB" agent --count "$COUNT" "$SWEEP" > "sweep_logs/bridge_tunnel_agent_${i}.log" 2>&1 &
  echo "agent $i -> pid $!"
  sleep 4
done
echo "launched $N agents (count=$COUNT each → up to $((N*COUNT)) runs) for $SWEEP"
echo "logs: sweep_logs/bridge_tunnel_agent_*.log"
