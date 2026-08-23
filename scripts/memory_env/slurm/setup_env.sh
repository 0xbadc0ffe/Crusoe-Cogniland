#!/usr/bin/env bash
# One-time cluster setup: create the `r2dreamer` conda env used by the MemoryEnv
# trainings. Run this ONCE on the cluster (login node or an interactive GPU
# node) before submitting the sbatch jobs.
#
#   bash scripts/memory_env/slurm/setup_env.sh
#
# Assumes `conda` (or mamba) is on PATH. Override the env name with CONDA_ENV.
set -euo pipefail
CONDA_ENV="${CONDA_ENV:-r2dreamer}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"

echo "[setup] repo   = $REPO"
echo "[setup] env    = $CONDA_ENV"

# create env (python 3.11, as r2dreamer is tested on it)
conda create -y -n "$CONDA_ENV" python=3.11
# install r2dreamer (core deps: torch, torchrl, hydra, gymnasium, ...) + minigrid
# + wandb (the SLURM sweep agent needs it) + matplotlib (eval plot).
conda run -n "$CONDA_ENV" pip install -e "$REPO/r2dreamer_model"
conda run -n "$CONDA_ENV" pip install minigrid matplotlib wandb

# sanity: MemoryEnv must import + produce the (56,56,3) obs (cogniland.memory_env
# only needs numpy+gymnasium+minigrid; no heavy cogniland deps required).
PYTHONPATH="$REPO/src" conda run -n "$CONDA_ENV" python - <<'PY'
import cogniland.memory_env as M
e = M.make_memory_env(M.MemoryEnvConfig())
o, info = e.reset(seed=0)
print("MemoryEnv OK -> obs", o.shape, o.dtype, "| oracle solvable + shaping on")
PY
echo "[setup] done. Now: sbatch scripts/memory_env/slurm/train_memory.sbatch"
