#!/usr/bin/env bash
# Targeted retrain of ppo_grass20 and ppo_grass25 with **clipped PBRS shaping**:
# backward steps (Δctg < 0) pay only the flat SLACK, no −SHAPING penalty.
# Removes the asymmetric "always move toward top-right" bias.
#
# Same physics as grass_slip_hardland_mixtrain (hard-land slip 0.75 when any
# skill committed) and same training distribution (simplex + composed mixed
# per reset). 10M steps, only sweep values 20 and 25.
#
# Usage: scripts/crafter/launch_grass_slip_clipshape.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/filippo/miniconda3/envs/crusoe/bin/python
EXP=grass_slip_clipshape           # new experiment folder
GENERATOR="simplex,composed"
TOTAL_TIMESTEPS=10000000
mkdir -p "sweep_logs/${EXP}"

PCTS=(20 25)
for pct in "${PCTS[@]}"; do
  val=$(python3 -c "print(f'{$pct/100:.2f}')")
  name=$(printf "ppo_grass%02d" "$pct")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    nohup "$PY" scripts/crafter/train_ppo_gru.py \
      --config configs/efficient.yaml \
      --generator "$GENERATOR" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --grass-slip-noskill "$val" \
      --clip-neg-shaping \
      --run-name "$name" \
      --checkpoint-dir "checkpoints/${EXP}" \
      > "sweep_logs/${EXP}/${name}.log" 2>&1 &
  echo "launched $name (grass_slip_noskill=$val, clip_neg_shaping=ON) -> pid $!"
  sleep 3
done
echo "launched ${#PCTS[@]} clipshape runs -> checkpoints/${EXP}/  (logs in sweep_logs/${EXP}/)"
