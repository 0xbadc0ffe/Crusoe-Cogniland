# Architecture Guide

How the codebase is organised, what each folder does, and how the pieces connect.

## Overview

```
scripts/train.py  ──Hydra──>  configs/  ──builds──>  cogniland/training/trainer.py
                                                          │
                                    ┌─────────────────────┤
                                    ▼                     ▼
                          cogniland/env/          cogniland/models/
                          (environment)           (agents)
                                    │                     │
                                    ▼                     │
                          cogniland/training/rollout.py <──┘
                                    │
                                    ▼
                          cogniland/logging/
                          (WandB + metrics)
```

A training run starts in `scripts/train.py`, which uses Hydra to load YAML configs and calls `trainer.train(cfg)`. The trainer builds an environment, a model, and a logger, then runs the PPO loop: collect rollouts, compute GAE, update the model, periodically evaluate and log.

---

## Folders

### `configs/`

Hydra YAML configuration, split into groups:

| Group | Files | What it controls |
|-------|-------|-----------------|
| `env/` | `default.yaml`, `easy.yaml`, `hard.yaml` | Island generation, agent params, minimap, reward coefficients, episode limits |
| `model/` | `ppo.yaml`, `compass.yaml` | Model architecture (hidden dim, CNN params, action dim) |
| `training/` | `default.yaml` | PPO hyperparams (LR, gamma, clip, epochs, batch sizes), eval/checkpoint intervals |
| `logging/` | `default.yaml` | WandB project/entity/mode, trajectory image settings |

`config.yaml` is the top-level file that sets defaults for each group and the global `device` setting.

Any value can be overridden from CLI: `python scripts/train.py training.learning_rate=1e-4 env=hard`.

### `cogniland/env/` — Environment

The core simulation. Generates an island, manages batched agent state, and steps the environment forward.

| File | Purpose | Depends on |
|------|---------|-----------|
| `constants.py` | All game constants: terrain levels, actions, visibility ranges. Also pre-computed tensor versions (`TERRAIN_THRESHOLDS`, `ACTION_DELTAS`) for vectorised lookups. | nothing |
| `types.py` | `EnvState` (NamedTuple), `StepResult` (NamedTuple), `EnvConfig` (frozen dataclass). These are the data structures everything else passes around. | `torch` |
| `core.py` | **Pure functions** that implement one environment step. `env_step()` is the main entry — it calls `apply_movement`, `compute_terrain_levels`, `apply_terrain_effects`, etc. No classes, no `self`, no mutation. | `constants`, `types`, `reward` |
| `reward.py` | `compute_reward()` — a single pure function. Takes state + terminal flags + distances, returns per-env reward tensor. All coefficients come from `EnvConfig`. | `types` |
| `islands.py` | `Islands` class — thin wrapper that owns the `world_map` tensor (generated once at init) and delegates to `core.env_step()`. Provides `reset()`, `step()`, `reset_done()`. Also contains `generate_island()` which runs the SimplexNoise terrain generator. | `core`, `types`, `constants`, `lib/simplexnoise` |
| `wrappers.py` | `BatchedIslandEnv` — training-oriented wrapper that auto-resets done environments and provides observations as `{"scalars": [B,6], "minimap": [B,1,H,W]}`. `IslandNavEnv` — single-instance Gymnasium wrapper for compatibility with standard RL libraries. | `islands`, `types`, `constants` |

**Data flow for one step:**

```
BatchedIslandEnv.step(action)
  └─> Islands.step(state, action, target_pos)
        └─> core.env_step(state, action, world_map, target_pos, config)
              ├─> apply_movement(state, action, map_size)
              ├─> compute_terrain_levels(world_map, positions)
              ├─> update_terrain_clock(state, old_terrain)
              ├─> apply_movement_costs(state, old_terrain, action, config)
              ├─> apply_terrain_effects(state, action, config)
              └─> reward.compute_reward(state, alive, reached, dist, prev_dist, config)
              => StepResult(state, reward, done, info)
```

### `cogniland/models/` — Agents

| File | Purpose |
|------|---------|
| `ppo.py` | `ActorCritic` — CNN processes the minimap, MLP processes scalar observations, features are concatenated into a shared trunk, then split into actor (action logits) and critic (value) heads. Orthogonal init (CleanRL pattern). |
| `compass_agent.py` | `CompassAgent` — greedy Manhattan-distance follower. Reads the compass observation and always picks the action that minimises Manhattan distance to the target. Same interface as `ActorCritic` (`get_action_and_value`, `get_value`) so the trainer doesn't need to special-case it. |

Both models take `obs = {"scalars": [B,6], "minimap": [B,1,H,W]}` and return `(action, log_prob, entropy, value)`.

### `cogniland/training/` — PPO Training Loop

| File | Purpose | Depends on |
|------|---------|-----------|
| `rollout.py` | `RolloutBuffer` (dataclass storing one rollout), `collect_rollout()` (runs N steps in the batched env, stores transitions), `compute_gae()` (GAE advantage estimation). | `env/wrappers` (via the env passed in) |
| `trainer.py` | `train(cfg)` — the main loop. Builds env + model + optimizer + logger from config. Outer loop: collect rollout → compute GAE → PPO update (clipped objective + value loss + entropy bonus) → periodic eval → periodic checkpoint. Also contains `_run_eval()` which runs evaluation episodes and renders trajectory images for WandB. | everything |

**Training loop structure:**

```
train(cfg)
  ├─ build env, model, optimizer, logger
  └─ for update in 1..num_updates:
       ├─ LR annealing
       ├─ collect_rollout(env, model, obs, rollout_steps)
       ├─ compute_gae(buffer, next_value)
       ├─ ppo_update(model, optimizer, data, advantages, returns)
       ├─ logger.log(train_metrics)
       ├─ if eval_interval: _run_eval() → logger.log(eval_metrics + trajectory images)
       └─ if checkpoint_interval: save_checkpoint()
```

### `cogniland/logging/` — Experiment Tracking

| File | Purpose |
|------|---------|
| `wandb_logger.py` | `WandBLogger` — wraps `wandb.init/log/finish`. When `mode="disabled"`, everything is a no-op. Supports `log()` for scalars and `log_image()` for trajectory figures. |
| `metrics.py` | `compute_behavioral_metrics()` — computes terrain distribution, resource management, HP score, and directness ratio from evaluation data. Used for policy identification (conservative vs greedy). |

### `cogniland/utils/` — Utilities

| File | Purpose |
|------|---------|
| `checkpoint.py` | `save_checkpoint()` / `load_checkpoint()` — saves model weights, optimizer state, and full RNG state (torch + numpy) for exact resume. |
| `reproducibility.py` | `set_reproducibility(seed)` — pins torch, numpy, and Python random seeds + CuDNN deterministic mode. |

### `scripts/` — Entry Points

| File | What it does |
|------|-------------|
| `train.py` | `@hydra.main` entry point. Loads config from `configs/`, calls `trainer.train(cfg)`. |
| `evaluate.py` | Loads a checkpoint, runs eval episodes, prints results. Also uses Hydra for config. |
| `demo.py` | Launches the interactive PyGame demo (delegates to `game_demo.py`). |

### `game_demo.py`

Interactive PyGame visualisation. Uses `cogniland.env.Islands` directly — creates an env, renders the world map and minimap as surfaces, handles keyboard input. Not used during training; purely for human play-testing.

### `lib/simplexnoise/`

Bundled third-party Simplex/Perlin noise library. Used only by `islands.generate_island()` to create the heightmap. Uses Python's `random` module internally for permutation tables (which is why we seed all RNGs before island generation).

### `outputs/` (git-ignored, auto-generated)

Created by Hydra at runtime. Each training or eval run gets a timestamped subfolder (e.g. `outputs/2026-02-22/16-43-15/`) containing:
- `.hydra/config.yaml` — the fully resolved config snapshot for that run
- `.hydra/overrides.yaml` — which CLI overrides were used
- `train.log` — Hydra's log output

This is useful for reproducing a run: look at the config snapshot to see exactly what params were used. The folder is git-ignored and safe to delete.

### `wandb/` (git-ignored, auto-generated)

Local WandB run cache. Contains offline copies of logged data, synced to the WandB server. Git-ignored and safe to delete.

### `checkpoints/` (git-ignored, auto-generated)

Model checkpoints saved during training (e.g. `ckpt_50.pt`). Contains model weights, optimizer state, and RNG state for exact resume.

### `tests/`

| File | What it tests |
|------|--------------|
| `test_env_step.py` | Reset produces valid state, step returns correct types, stay doesn't move, bounds clamping, vectorised terrain levels, batch consistency. |
| `test_reward.py` | Reach bonus is positive, death penalty is negative, closer = higher reward, low HP reduces reward. |
| `test_roundtrip.py` | Two runs with same seed produce identical trajectories (determinism). Trajectory positions and HP stay within bounds. Reference data for future JAX migration. |

---

## Key Design Decisions

**Immutable state**: `EnvState` is a `NamedTuple`. Updates create new tuples via `state._replace(field=new_value)`. No in-place mutation, no `copy.deepcopy`.

**Pure functions**: All step logic lives in `core.py` as standalone functions. This makes the code testable, readable, and directly portable to JAX (`torch.where` → `jnp.where`).

**Batched everything**: Every tensor has a batch dimension at index 0. The environment processes N agents simultaneously in a single call.

**Config as code boundary**: `EnvConfig` is a frozen dataclass. All magic numbers (reward coefficients, terrain costs, etc.) flow from the Hydra YAML through `EnvConfig` into the pure functions. Nothing is hardcoded in the step logic.
