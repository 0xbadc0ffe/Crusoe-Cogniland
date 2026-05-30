#!/usr/bin/env bash
# Grass-slip sweep: train one efficient.yaml PPO-GRU agent per value of the
# bare-handed (no-skill) grass slip probability, swept 0..30 %.
#
# Base config: configs/efficient.yaml. The ONLY thing that varies across runs
# is --grass-slip-noskill (the prob grass slips while no skill is committed);
# the 30 % weight-tax on grass *with* a skill is unchanged.
#
# All 7 runs launch in parallel on GPU 0 (small models; 32-core box). Threads
# are pinned to 1 so the 7x16 numpy envs don't oversubscribe the CPU.
#
# Usage: scripts/launch_grass_slip_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/filippo/miniconda3/envs/crusoe/bin/python
EXP=grass_slip_hardland_mixtrain     # experiment folder name
                                     # 2026-05-28: hard-land weight tax +
                                     # **augmented training**: training maps
                                     # sampled per reset from {simplex,
                                     # composed}; ``components`` is held out
                                     # as the test set. Budget bumped to 10M.
GENERATOR="simplex,composed"
TOTAL_TIMESTEPS=10000000
mkdir -p "sweep_logs/${EXP}"

# Training maps: the legacy "simplex" generator (the env/trainer default now,
# but pinned explicitly here so the experiment is self-documenting). The
# structured composed/components maps are held out for the trajectory-grid eval.
PCTS=(0 5 10 15 20 25 30)
for pct in "${PCTS[@]}"; do
  val=$(python3 -c "print(f'{$pct/100:.2f}')")
  name=$(printf "ppo_grass%02d" "$pct")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    nohup "$PY" scripts/train_ppo_gru.py \
      --config configs/efficient.yaml \
      --generator "$GENERATOR" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --grass-slip-noskill "$val" \
      --run-name "$name" \
      --checkpoint-dir "checkpoints/${EXP}" \
      > "sweep_logs/${EXP}/${name}.log" 2>&1 &
  echo "launched $name  (grass_slip_noskill=$val)  -> pid $!"
  sleep 3
done
echo "launched ${#PCTS[@]} grass-slip runs -> checkpoints/${EXP}/ (logs in sweep_logs/${EXP}/)"
