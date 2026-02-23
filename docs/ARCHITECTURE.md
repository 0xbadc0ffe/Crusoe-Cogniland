# Architecture Guide

How the codebase is organised, what each module does, and how the pieces connect.

## Overview

```
train.py  ──Hydra──>  configs/config.yaml  ──builds──>  cogniland/models/
                                                                      │
                                                    ┌─────────────────┤
                                                    ▼                 ▼
                                          cogniland/env/     cogniland/logging.py
                                          (environment)      (WandB + metrics)
```

A training run starts in `train.py`, which uses Hydra to load config and calls `build_model(cfg).train(cfg)`. Each model is self-contained: it builds its own environment, optimizer, and logger, then runs its training loop.

---

## Modules

### `configs/`

Hydra YAML configuration. Training and logging params are inlined in `config.yaml`; only env and models have separate config groups:

| Group | Files | What it controls |
|-------|-------|-----------------|
| `env/` | `default.yaml`, `hard.yaml` | Island generation, agent params, minimap, reward coefficients, episode limits |
| `models/` | `ppo.yaml`, `compass.yaml` | Model architecture (hidden dim, CNN params, action dim) |
| *inlined* | `config.yaml` → `training:` | PPO hyperparams (LR, gamma, clip, epochs, batch sizes), eval/checkpoint intervals |
| *inlined* | `config.yaml` → `logging:` | WandB project/entity/mode, trajectory image settings |

Any value can be overridden from CLI: `python train.py models.training.learning_rate=1e-4 env=hard`.

### `cogniland/env/` — Environment

The core simulation. Generates an island, manages batched agent state, and steps the environment forward.

| File | Purpose | Depends on |
|------|---------|-----------|
| `constants.py` | All game constants: terrain levels, actions, visibility ranges. Also pre-computed tensor versions (`TERRAIN_THRESHOLDS`, `ACTION_DELTAS`) for vectorised lookups. | nothing |
| `types.py` | `EnvState` (NamedTuple), `StepResult` (NamedTuple), `EnvConfig` (frozen dataclass with `.from_hydra()` classmethod). | `torch` |
| `core.py` | **Pure functions** that implement one environment step. No classes, no `self`, no mutation. | `constants`, `types`, `reward` |
| `reward.py` | `compute_reward()` — a single pure function. All coefficients come from `EnvConfig`. | `types` |
| `islands.py` | `Islands` class — wraps `world_map` + delegates to `core.env_step()`. Also contains `generate_island()` using SimplexNoise. | `core`, `types`, `constants`, `simplexnoise` |
| `wrappers.py` | `BatchedIslandEnv` — auto-resets done envs, provides obs as `{"scalars": [B,6], "minimap": [B,1,H,W]}`. `IslandNavEnv` — Gymnasium wrapper. | `islands`, `types`, `constants` |

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

### `cogniland/models/` — Self-Contained Agents

Each model file defines its **architecture + full training loop**. To add a new model, create a file here with a class that has `.train(cfg)`.

| File | Purpose |
|------|---------|
| `__init__.py` | `build_model(cfg)` factory — returns a PPOAgent or CompassModel based on config. |
| `ppo.py` | `PPOAgent` — full PPO agent: `ActorCritic` (CNN + MLP), `RolloutBuffer`, GAE computation, PPO clipped update, training loop with periodic eval + checkpointing. |
| `compass.py` | `CompassModel` — greedy Manhattan-distance follower. Deterministic baseline. `.train()` runs evaluation only. |

Both models expose` .get_action_and_value(obs)` and `.train(cfg)`.

**PPO training loop structure:**

```
PPOAgent.train(cfg)
  ├─ build env, optimizer, logger
  └─ for update in 1..num_updates:
       ├─ LR annealing
       ├─ _collect_rollout(env, model, obs, rollout_steps)
       ├─ _compute_gae(buffer, next_value)
       ├─ _ppo_update(optimizer, data, advantages, returns)
       ├─ logger.log(train_metrics)
       ├─ if eval_interval: _run_eval() → trajectory images + behavioral profile
       └─ if checkpoint_interval: save_checkpoint()
```

### `cogniland/logging.py` — Experiment Tracking

`WandBLogger` — wraps `wandb.init/log/finish`. When `mode="disabled"`, everything is a no-op. Supports scalar logging and trajectory image panels.

`compute_behavioral_metrics()` — computes terrain distribution, resource management, HP score, and directness ratio from evaluation data.

### `cogniland/utils.py` — Utilities

| Function | Purpose |
|----------|---------|
| `save_checkpoint()` / `load_checkpoint()` | Model weights, optimizer state, and full RNG state for exact resume. |
| `set_reproducibility(seed)` | Pins torch, numpy, and Python random seeds + CuDNN deterministic mode. |

### `cogniland/simplexnoise/` — Noise Library

Bundled Simplex/Perlin noise library. Used only by `islands.generate_island()` to create the heightmap. Uses Python's `random` module internally.

### Entry Points

| File | What it does |
|------|-------------|
| `train.py` | `@hydra.main` entry point. Calls `build_model(cfg).train(cfg)`. |
| `demo.py` | Interactive PyGame demo — full game visualization with keyboard controls. |

---

## Key Design Decisions

**Self-contained models**: Each model defines architecture + training loop + eval in a single file. To swap models, change `models=ppo` to `models=compass`. To add a new model, add one file + one config.

**Immutable state**: `EnvState` is a `NamedTuple`. Updates create new tuples via `state._replace(field=new_value)`. No in-place mutation.

**Pure functions**: All step logic lives in `core.py` as standalone functions. Directly portable to JAX.

**Batched everything**: Every tensor has a batch dimension at index 0. N agents processed simultaneously.

**Config as code boundary**: All magic numbers flow from Hydra YAML → `EnvConfig` → pure functions. Nothing is hardcoded.
