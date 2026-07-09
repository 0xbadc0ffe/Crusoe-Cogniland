#!/usr/bin/env bash
#SBATCH --job-name=mem_jax_bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=node01,node02,node03
#SBATCH --output=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --error=scripts/memory_env/slurm/logs/%x_%j.out
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/home/fwang
# Benchmark the pure-JAX DreamerV3 on MemoryEnv-JAX (25M, num_envs=16,
# train_ratio=64 — matching the PyTorch tr64 baseline) and read per-chunk fps.
set -eo pipefail   # NOT -u: crusoe's MKL conda-activate script reads unset vars
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crusoe
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
python -c "import jax; print('JAX devices:', jax.devices())"
srun python scripts/memory_env/dreamerv3_memory.py \
  --size 25M --cue 4cue --num-envs 16 --train-ratio 64 --wandb-mode disabled \
  --set total_env_steps=200000 ckpt_interval_updates=3000
echo "[bench] done"
