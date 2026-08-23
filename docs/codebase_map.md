# Codebase map — what is where

Navigation guide after the three-pipeline consolidation. The repo holds ONE
memory task (`bridge_tunnel` fork_wall) solved by three agents on identical
env/reward/data, plus the mech-interp tooling built on top.

## Top-level layout

```
final_models/           ★ the three checkpoints + ENVIRONMENT.md / ARCHITECTURES.md
                          + per-agent README with the exact reproduce command
src/cogniland/
  bridge_tunnel/        the env (PyTorch/numpy + bit-identical pure-JAX port)
  memory_env/           MiniGrid MemoryEnv fork (T-maze; secondary task)
  assets/sprites/       rendering sprites
purejaxwm/              in-tree DreamerV3 algorithm library (JAX)
r2dreamer_model/        DreamerV3 training pipeline (runs/ holds fw_sw_* sweep)
STORM_model/            STORM training pipeline (agent `storm2`; own README)
scripts/
  bridge_tunnel/        PPO train/eval/viz + slurm/ launchers + dataset builder
  memory_env/           memory_env training + analysis
  mechinterp/           activation datasets, probing, steering kits
  figures/              figure generation
configs/bridge_tunnel/  PPO/experiment configs + REGISTRY.md (released agents)
released_models/        earlier released agents (git-LFS)
data/bridge_tunnel/     forkwall6k/{train,test}.pkl fixed dataset + val maps
activation_datasets/    mech-interp bundles                     [gitignored]
outputs/ artifacts/     generated interp reports & figures      [gitignored]
tests/                  env contract + JAX↔PyTorch parity + purejaxwm units
paper/ docs/            write-ups
```

## Where do I start

| I want to… | where |
|---|---|
| understand the task + shared-env proof | `final_models/ENVIRONMENT.md` |
| the three architectures compared | `final_models/ARCHITECTURES.md` |
| reproduce PPO / Dreamer / STORM | `final_models/{ppo,dreamer,storm}/README.md` |
| regenerate the dataset | `scripts/bridge_tunnel/make_forkwall_dataset.py` |
| evaluate STORM (TRUE metric) | `STORM_model/scripts/true_eval_w.py --sampled --env-context 128` |
| evaluate PPO | `scripts/bridge_tunnel/eval_bridge_tunnel_forkwall.py` |
| change env rules | `src/cogniland/bridge_tunnel/env.py` (+ keep `jax/` parity; run tests) |
| build an activation dataset | `scripts/mechinterp/build_activation_dataset.py` |
| released agents registry | `configs/bridge_tunnel/REGISTRY.md` |

See `CLAUDE.md` for the task definition, evaluation convention (TRUE door
metric, not `return>0`), and the per-agent training lore (constant-door basin,
batch_length ≥ dependency span, entropy schedules).
