#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=logs/slurm_%x_%j.log
#SBATCH --error=logs/slurm_%x_%j.log
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang

# usage: sbatch --job-name=<arm> scripts/slurm_storm2_sweep.sh \
#          configs/agents/sweep/<arm>.yaml [configs/envs/<env>.yaml]
set -eo pipefail
AGENT_CONFIG="$1"
ENV_CONFIG="${2:-configs/envs/bridge_tunnel_storm2_run6.yaml}"
cd /cluster/raid/home/fwang/Crusoe-Cogniland/STORM_model
source .venv/bin/activate
echo "[job] $(date) host=$(hostname) agent-config=$AGENT_CONFIG env-config=$ENV_CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
srun python -m scripts.train \
  --env-config "$ENV_CONFIG" \
  --agent-config "$AGENT_CONFIG" \
  --offline --device 0
echo "[job] $(date) done"
