#!/usr/bin/env bash
#SBATCH --job-name=fw_eval_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --error=scripts/bridge_tunnel/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
set -eo pipefail
source /cluster/software/anaconda3/etc/profile.d/conda.sh
conda activate r2dreamer
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export BT_MAPS="$PWD/data/bridge_tunnel/forkwall6k/test.pkl"
declare -A SZ=( [fw_sw_12M_bl64_h15]=size12M [fw_sw_12M_bl128_h15]=size12M [fw_sw_25M_bl64_h15]=size25M [fw_sw_25M_bl128_h15]=size25M )
for n in fw_sw_12M_bl128_h15 fw_sw_12M_bl64_h15 fw_sw_25M_bl128_h15 fw_sw_25M_bl64_h15; do
  ck=external/r2dreamer/runs/$n/latest.pt
  [ -f "$ck" ] || { echo "=== $n: NO CHECKPOINT ==="; continue; }
  echo ""; echo "################## HELD-OUT TEST EVAL: $n (${SZ[$n]}) ##################"
  python scripts/bridge_tunnel/eval_forkwall_fixed.py \
    --checkpoint "$ck" --maps data/bridge_tunnel/forkwall6k/test.pkl \
    --n 150 --model-size "${SZ[$n]}" --device cuda:0 \
    --out outputs/bridge_tunnel_forkwall/eval_${n}.json || echo "  (eval failed for $n)"
done
echo ""; echo "[eval_sweep] done $(date)"
