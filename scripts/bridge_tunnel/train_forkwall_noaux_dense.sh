#!/usr/bin/env bash
# fork_wall (no-commit, NO auxiliary belief loss) with LOG-SPACED checkpointing,
# for learning-dynamics analysis in the style of Huang, Singh & Rajan (RLC 2024):
# fixed/slow-point structure and stimulus-integration timescales as they evolve
# over training.
#
# The standard runs save every 300 of 1464 iterations, so the earliest snapshot
# is iteration 300 — by which point the agent is already largely trained. That
# leaves the window the paper actually studies (the first few dozen gradient
# steps) completely unsampled. --save-log-spaced 40 puts ~20 checkpoints inside
# the first 100 iterations, plus iter0 (the untrained init).
#
# Written to distinct run names so the existing seed{1..5} checkpoints, the
# fitted belief probe, and everything already published off them stay intact.
#
#   bash scripts/bridge_tunnel/train_forkwall_noaux_dense.sh
set -uo pipefail

CFG=configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml
SEEDS="${SEEDS:-1 2 3 4 5}"
NCKPT="${NCKPT:-40}"
LOGDIR=outputs/logs/forkwall_noaux_dense
PY="${PY:-/home/filippo/miniconda3/envs/crusoe/bin/python}"
mkdir -p "$LOGDIR"

for s in $SEEDS; do
  name="ppo_gru_forkwall_noaux_dense_seed${s}"
  if [ -f "outputs/ppo_checkpoints/${name}/final.pt" ]; then
    echo "[skip] ${name} already complete"
    continue
  fi
  echo "[start] ${name}  $(date '+%F %T')"
  "$PY" scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
    --config "$CFG" \
    --belief-coef 0 \
    --seed "$s" \
    --run-name "$name" \
    --save-log-spaced "$NCKPT" \
    --wandb-mode offline \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[done ] ${name}  exit=$?  $(date '+%F %T')"
done
echo "ALL DENSE SEEDS COMPLETE $(date '+%F %T')"
