#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$(dirname "$SCRIPT_DIR")/logs"
mkdir -p "$LOGS_DIR"

usage() {
    cat << 'EOF'
Usage:
  ./scripts/launch_sweep.sh <sweep_config.yaml> [options]
  ./scripts/launch_sweep.sh --sweep-id <existing_id> [options]

Options:
  -n, --num-agents NUM    Number of SLURM array tasks
                          (default: auto = number of grid combos for grid
                          sweeps; required for random/bayes)
  -r, --runs-per NUM      Runs per agent (default: 1)
  -N, --nodes LIST        SLURM --nodelist
  -x, --exclude LIST      SLURM --exclude
  -t, --time TIME         Max time (default: 24:00:00)
  -m, --mem MEM           Memory (default: 32G)
  --dry-run               Print command without submitting

Examples:
  # Grid sweep: -n auto-detected from parameters × values (one combo per task)
  ./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_reward.yaml

  # Override: force 8 array tasks (e.g. to serialize 2 combos per task)
  ./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_reward.yaml -n 8 -r 2

  # Random/bayes sweep: -n is required
  ./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_hpsearch.yaml -n 20

  # Reuse existing sweep
  ./scripts/launch_sweep.sh --sweep-id entity/project/abc123 -n 10
EOF
}

NUM_AGENTS=""
RUNS_PER_AGENT=1
NODES=""
EXCLUDE=""
TIME="24:00:00"
MEM="32G"
DRY_RUN=false
SWEEP_ID=""
SWEEP_CONFIG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --sweep-id)     SWEEP_ID="$2";       shift 2 ;;
        -n|--num-agents) NUM_AGENTS="$2";    shift 2 ;;
        -r|--runs-per)  RUNS_PER_AGENT="$2"; shift 2 ;;
        -N|--nodes)     NODES="$2";          shift 2 ;;
        -x|--exclude)   EXCLUDE="$2";        shift 2 ;;
        -t|--time)      TIME="$2";           shift 2 ;;
        -m|--mem)       MEM="$2";            shift 2 ;;
        --dry-run)      DRY_RUN=true;        shift ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "Unknown option: $1"; usage; exit 1 ;;
        *)              SWEEP_CONFIG="$1";   shift ;;
    esac
done

# Create sweep if no ID given
if [ -z "$SWEEP_ID" ]; then
    if [ -z "$SWEEP_CONFIG" ]; then
        echo "Error: provide a sweep YAML or --sweep-id"
        usage; exit 1
    fi
    echo "Creating W&B sweep from $SWEEP_CONFIG ..."
    SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" 2>&1 | grep -oP '(?<=wandb agent )\S+')
    echo "Created sweep: $SWEEP_ID"
fi

# Auto-detect number of agents for grid sweeps if -n not provided
if [ -z "$NUM_AGENTS" ]; then
    if [ -n "$SWEEP_CONFIG" ]; then
        GRID_SIZE=$(python - "$SWEEP_CONFIG" <<'PY'
import sys, yaml
try:
    cfg = yaml.safe_load(open(sys.argv[1]))
except Exception as e:
    sys.exit(f"yaml parse error: {e}")
if cfg.get("method") != "grid":
    sys.exit(0)  # print nothing, caller falls through
total = 1
for v in (cfg.get("parameters") or {}).values():
    if isinstance(v, dict) and "values" in v:
        total *= len(v["values"])
print(total)
PY
)
        if [ -n "$GRID_SIZE" ]; then
            NUM_AGENTS="$GRID_SIZE"
            echo "Auto-detected grid size: $NUM_AGENTS combos -> -n $NUM_AGENTS"
        fi
    fi
    if [ -z "$NUM_AGENTS" ]; then
        echo "Error: -n required (non-grid sweep or reused --sweep-id)"
        exit 1
    fi
fi

echo ""
echo "Sweep ID:        $SWEEP_ID"
echo "Num agents:      $NUM_AGENTS"
echo "Runs per agent:  $RUNS_PER_AGENT"
echo "Total runs:      $((NUM_AGENTS * RUNS_PER_AGENT))"
echo ""

# Build sbatch command
SBATCH_CMD="sbatch --array=0-$((NUM_AGENTS - 1)) --time=$TIME --mem=$MEM"
[ -n "$NODES" ]   && SBATCH_CMD="$SBATCH_CMD --nodelist=$NODES"
[ -n "$EXCLUDE" ] && SBATCH_CMD="$SBATCH_CMD --exclude=$EXCLUDE"
SBATCH_CMD="$SBATCH_CMD $SCRIPT_DIR/job_sweep.slurm"

echo "Command: SWEEP_ID=$SWEEP_ID RUNS_PER_AGENT=$RUNS_PER_AGENT $SBATCH_CMD"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Not submitted"
    exit 0
fi

export SWEEP_ID RUNS_PER_AGENT
$SBATCH_CMD

echo ""
echo "Submitted! Monitor with: squeue -u \$USER"
echo "Logs:    tail -f logs/sweep_*.log"
echo "Cancel:  scancel <job_id>"
