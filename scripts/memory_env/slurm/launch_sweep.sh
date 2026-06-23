#!/usr/bin/env bash
# Recycled from the cluster's launch_sweep.sh, pointed at job_memory.slurm.
# Creates the W&B sweep from a grid YAML, auto-detects the number of combos, and
# submits a SLURM array of wandb agents (one training per combo).
#
#   bash scripts/memory_env/slurm/launch_sweep.sh configs/sweeps/memory_r2dreamer.yaml
#
# Options: -n/--num-agents  -r/--runs-per  -N/--nodes  -x/--exclude  -t/--time
#          -m/--mem  --sweep-id <id>  --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SLURM="$SCRIPT_DIR/job_memory.slurm"
LOGS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/logs"
mkdir -p "$LOGS_DIR"

usage() { sed -n '2,12p' "$0"; }

NUM_AGENTS=""; RUNS_PER_AGENT=1; NODES=""; EXCLUDE=""; TIME="24:00:00"; MEM="64G"
DRY_RUN=false; SWEEP_ID=""; SWEEP_CONFIG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sweep-id)      SWEEP_ID="$2";       shift 2 ;;
        -n|--num-agents) NUM_AGENTS="$2";     shift 2 ;;
        -r|--runs-per)   RUNS_PER_AGENT="$2"; shift 2 ;;
        -N|--nodes)      NODES="$2";          shift 2 ;;
        -x|--exclude)    EXCLUDE="$2";        shift 2 ;;
        -t|--time)       TIME="$2";           shift 2 ;;
        -m|--mem)        MEM="$2";            shift 2 ;;
        --dry-run)       DRY_RUN=true;        shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "Unknown option: $1"; usage; exit 1 ;;
        *)               SWEEP_CONFIG="$1";   shift ;;
    esac
done

if [ -z "$SWEEP_ID" ]; then
    [ -z "$SWEEP_CONFIG" ] && { echo "Error: provide a sweep YAML or --sweep-id"; usage; exit 1; }
    echo "Creating W&B sweep from $SWEEP_CONFIG ..."
    SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" 2>&1 | grep -oP '(?<=wandb agent )\S+')
    echo "Created sweep: $SWEEP_ID"
fi

# Auto-detect grid size (combos) if -n not given.
if [ -z "$NUM_AGENTS" ] && [ -n "$SWEEP_CONFIG" ]; then
    GRID_SIZE=$(python - "$SWEEP_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
if cfg.get("method") != "grid": sys.exit(0)
total = 1
for v in (cfg.get("parameters") or {}).values():
    if isinstance(v, dict) and "values" in v: total *= len(v["values"])
print(total)
PY
)
    [ -n "$GRID_SIZE" ] && { NUM_AGENTS="$GRID_SIZE"; echo "Auto-detected grid: $NUM_AGENTS combos -> -n $NUM_AGENTS"; }
fi
[ -z "$NUM_AGENTS" ] && { echo "Error: -n required (non-grid or reused --sweep-id)"; exit 1; }

echo ""; echo "Sweep ID: $SWEEP_ID | agents: $NUM_AGENTS | runs/agent: $RUNS_PER_AGENT"; echo ""

SBATCH_CMD="sbatch --array=0-$((NUM_AGENTS - 1)) --time=$TIME --mem=$MEM"
[ -n "$NODES" ]   && SBATCH_CMD="$SBATCH_CMD --nodelist=$NODES"
[ -n "$EXCLUDE" ] && SBATCH_CMD="$SBATCH_CMD --exclude=$EXCLUDE"
SBATCH_CMD="$SBATCH_CMD $JOB_SLURM"

echo "Command: SWEEP_ID=$SWEEP_ID RUNS_PER_AGENT=$RUNS_PER_AGENT $SBATCH_CMD"
[ "$DRY_RUN" = true ] && { echo "[DRY RUN] Not submitted"; exit 0; }

export SWEEP_ID RUNS_PER_AGENT
$SBATCH_CMD
echo ""; echo "Submitted. Monitor: squeue -u \$USER | logs: tail -f logs/memoryenv_*.log"
echo "After training: sbatch scripts/memory_env/slurm/eval_memory.sbatch  (writes the per-cue reward plot)"
