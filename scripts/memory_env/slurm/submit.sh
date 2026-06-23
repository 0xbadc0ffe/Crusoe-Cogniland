#!/usr/bin/env bash
# Submit the 3 MemoryEnv trainings (job array) + the eval/plot job chained to run
# after all three finish (SLURM dependency). Run from the repo root on the
# cluster login node:
#
#   bash scripts/memory_env/slurm/submit.sh
#
# Pass extra sbatch flags through SBATCH_FLAGS, e.g.:
#   SBATCH_FLAGS="--partition=gpu --account=myacct" bash scripts/memory_env/slurm/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
SB="scripts/memory_env/slurm"
FLAGS="${SBATCH_FLAGS:-}"

# 1) training array (0-2)
ARRAY_ID=$(sbatch ${FLAGS} --parsable "$SB/train_memory.sbatch")
echo "submitted training array: job $ARRAY_ID  (memory_2cue / 3cue / 4cue)"

# 2) eval after the WHOLE array succeeds
EVAL_ID=$(sbatch ${FLAGS} --parsable --dependency=afterok:${ARRAY_ID} "$SB/eval_memory.sbatch")
echo "submitted eval: job $EVAL_ID  (runs after $ARRAY_ID, writes the per-cue reward plot)"
echo
echo "watch:  squeue -j ${ARRAY_ID},${EVAL_ID}"
echo "logs :  tail -f $SB/logs/mem_dreamer_${ARRAY_ID}_*.out"
