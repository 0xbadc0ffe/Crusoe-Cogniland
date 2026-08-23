#!/usr/bin/env bash
# Launch the 3 R2-Dreamer MemoryEnv trainings (2cue / 3cue / 4cue), size25M,
# 30M env steps each, on the 4090. Run from the repo root.
#
#   bash scripts/memory_env/launch_r2dreamer.sh
#
# Notes
# -----
# * Uses the dedicated `r2dreamer` conda env (NOT `crusoe`) so it doesn't
#   touch the deps of the other running jobs.
# * PYTHONPATH=src makes `cogniland.memory_env` importable without installing
#   the rest of cogniland.
# * `task` lives under the `env` config group, so it is overridden as
#   `env.task=...` (top-level `task=...` is NOT a valid key for this repo).
# * `model.compile=True` (the default) is fine on CUDA; only the CPU smoke
#   test needed `model.compile=False`.
# * All three models are later evaluated on the SAME held-out 4-cue test set:
#     PYTHONPATH=src conda run -n r2dreamer python scripts/memory_env/eval_r2dreamer.py \
#         --ckpt-2cue r2dreamer_model/runs/memory_2cue/latest.pt \
#         --ckpt-3cue r2dreamer_model/runs/memory_3cue/latest.pt \
#         --ckpt-4cue r2dreamer_model/runs/memory_4cue/latest.pt --device cuda:0
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

DEVICE="${DEVICE:-cuda:0}"

PYTHONPATH=src conda run -n r2dreamer python r2dreamer_model/train.py \
  env=memory env.task=memory_2cue model=size25M env.steps=10e6 \
  device="${DEVICE}" seed=0 logdir=r2dreamer_model/runs/memory_2cue

PYTHONPATH=src conda run -n r2dreamer python r2dreamer_model/train.py \
  env=memory env.task=memory_3cue model=size25M env.steps=10e6 \
  device="${DEVICE}" seed=0 logdir=r2dreamer_model/runs/memory_3cue

PYTHONPATH=src conda run -n r2dreamer python r2dreamer_model/train.py \
  env=memory env.task=memory_4cue model=size25M env.steps=10e6 \
  device="${DEVICE}" seed=0 logdir=r2dreamer_model/runs/memory_4cue
