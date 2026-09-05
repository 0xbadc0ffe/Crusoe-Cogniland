#!/usr/bin/env bash
#SBATCH --job-name=eval_seeds
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Held-out (1200 test maps, sampled policy) evaluation of every belief-free PPO
# seed and every feed-forward control seed, 10 evaluations in parallel on CPU.
set -eo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-crusoe}"
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 KMP_WARNINGS=0
mkdir -p outputs/bridge_tunnel_forkwall/eval_noaux_seeds scripts/bridge_tunnel/slurm/logs
echo "[job] $(date) host=$(hostname)"
for r in noaux_ent15_s1 noaux_ent15_s2 noaux_ent15_s3 noaux_ent15_s4 noaux_ent15_s5 \
         ff_control_s1 ff_control_s2 ff_control_s3 ff_control_s4 ff_control_s5; do
  python scripts/bridge_tunnel/eval_forkwall_ppo.py --checkpoint outputs/ppo_noaux/$r/final.pt \
    --maps data/bridge_tunnel/forkwall6k/test.pkl --n 1200 --device cpu \
    --out outputs/bridge_tunnel_forkwall/eval_noaux_seeds/$r.json \
    > outputs/bridge_tunnel_forkwall/eval_noaux_seeds/$r.log 2>&1 &
done
wait
echo "[job] $(date) done"
