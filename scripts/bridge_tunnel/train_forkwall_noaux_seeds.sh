#!/usr/bin/env bash
# fork_wall (no-commit) PPO+GRU trained WITHOUT the auxiliary belief loss,
# across 5 seeds — the control the forkwall chapter's §3 comparison needs.
#
# Identical config to the released aux agent except --belief-coef 0, so the
# only difference is representation supervision. 6M env steps per seed.
# The chapter reports a fixed-door basin (some seeds collapse to a constant
# door), which is exactly why we sweep seeds rather than trusting one run.
#
#   bash scripts/bridge_tunnel/train_forkwall_noaux_seeds.sh
set -uo pipefail

CFG=configs/bridge_tunnel/btc_ppo_forkwall_nocommit.yaml
SEEDS="${SEEDS:-1 2 3 4 5}"
LOGDIR=outputs/logs/forkwall_noaux
# the repo's deps live in the `crusoe` conda env, not the base interpreter
PY="${PY:-/home/filippo/miniconda3/envs/crusoe/bin/python}"
mkdir -p "$LOGDIR"

for s in $SEEDS; do
  name="ppo_gru_forkwall_noaux_seed${s}"
  if [ -f "outputs/ppo_checkpoints/${name}/final.pt" ]; then
    echo "[skip] ${name} already has final.pt"
    continue
  fi
  echo "[start] ${name}  $(date '+%F %T')"
  "$PY" scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
    --config "$CFG" \
    --belief-coef 0 \
    --seed "$s" \
    --run-name "$name" \
    --wandb-mode offline \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[done ] ${name}  exit=$?  $(date '+%F %T')"
done
echo "ALL SEEDS COMPLETE $(date '+%F %T')"
