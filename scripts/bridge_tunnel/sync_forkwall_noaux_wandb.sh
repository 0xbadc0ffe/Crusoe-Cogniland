#!/usr/bin/env bash
# Sync the offline fork_wall no-aux runs to W&B (crusoe/bridge_tunnel).
# Safe to re-run: wandb skips runs that are already synced.
#
#   bash scripts/bridge_tunnel/sync_forkwall_noaux_wandb.sh
set -uo pipefail

WB="${WB:-/home/filippo/miniconda3/envs/crusoe/bin/wandb}"
PY="${PY:-/home/filippo/miniconda3/envs/crusoe/bin/python}"
ENTITY="${ENTITY:-crusoe}"
PROJECT="${PROJECT:-bridge_tunnel}"
PATTERN="${PATTERN:-forkwall_noaux}"

for d in wandb/offline-run-*; do
  [ -d "$d" ] || continue
  name=$("$PY" - "$d" <<'EOF' 2>/dev/null
import sys, json
from pathlib import Path
sys.path.insert(0, "scripts/figures")
from forkwall_noaux_training_curves import read_offline_history
cfg, _ = read_offline_history(Path(sys.argv[1]))
print(cfg.get("run_name", ""))
EOF
)
  case "$name" in
    *"$PATTERN"*) ;;
    *) continue ;;
  esac
  echo "[sync] $name  ($d)"
  "$WB" sync --project "$PROJECT" --entity "$ENTITY" "$d" 2>&1 | tail -2
done
echo "SYNC COMPLETE $(date '+%F %T')"
