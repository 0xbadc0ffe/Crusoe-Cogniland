#!/usr/bin/env bash
#SBATCH --job-name=eval_ff6m
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Held-out evaluation (1200 test maps, sampled policy) of the five 6M-step
# feed-forward control seeds, in parallel on CPU.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_WARNINGS=0
mkdir -p outputs/bridge_tunnel_forkwall/eval_noaux_seeds
echo "[job] $(date) host=$(hostname)"
for s in 1 2 3 4 5; do
  r=ff6m_control_s$s
  python scripts/bridge_tunnel/eval_forkwall_ppo.py --checkpoint outputs/ppo_noaux/$r/final.pt \
    --maps data/bridge_tunnel/forkwall6k/test.pkl --n 1200 --device cpu \
    --out outputs/bridge_tunnel_forkwall/eval_noaux_seeds/$r.json \
    > outputs/bridge_tunnel_forkwall/eval_noaux_seeds/$r.log 2>&1 &
done
wait
echo "[job] $(date) done"
