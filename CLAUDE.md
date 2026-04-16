# Cogniland — Architecture & Developer Guide

## Project Overview

Cogniland is a multi-task RL framework where agents learn to navigate procedurally generated 128x128 maps. The agent starts at a random spawn point and must reach a target position while managing **HP** (health points) and **wood** (gathered from forests). Different terrain types impose HP drains, and the agent can **forage** (berries heal HP, forests yield wood) and **craft tools** (raft, rope, shoes) that reduce terrain costs. Maps are pre-generated in pools of 256 (train) / 16 (val/test) across 4 biomes.

The framework supports three agents: **PPO-RNN** (JAX/Flax), **DreamerV3**, and **STORM**. New agents plug in via a `@register_agent` decorator — all training orchestration, evaluation, and logging are agent-agnostic.

---

## Installation & Setup

```bash
# Conda environment
conda env create -f environment.yml
conda activate crusoe

# Editable install
pip install -e .

# Generate map datasets (required before training)
python scripts/generate_dataset.py
```

---

## How to Run

```bash
# PPO-RNN training (5M frames, default config)
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml

# Smoke test — offline W&B, fast
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  --offline trainer.num_train_frames=10000

# DreamerV3 / STORM
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/dreamerv3.yaml

python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/storm.yaml

# Override hyperparameters inline (OmegaConf dotlist)
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  agent.lr=1e-4 trainer.num_train_frames=1000000

# K-seed sweep on SLURM cluster
./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_seeds.yaml -n 10 -r 1

# Local parallel sweep (non-SLURM)
wandb sweep configs/sweeps/ppo_rnn_seeds.yaml
python scripts/run_sweep.py <SWEEP_ID> --num-agents 5 --count 1 --gpus 0 1
```

---

## Experiment Pipeline

```
scripts/train.py (OmegaConf)
  └─ load_agent(config)                      # registry auto-discovers agents
       └─ make_ppo_rnn(config, obs_space, act_space)  → Agent dataclass

Trainer(config, agent).run()
  │
  ├─ MultiTaskEnvWrapper(CognilandEnv)        # 32 parallel training envs
  ├─ MultiTaskEnvWrapper(CognilandEnv)        # 4 eval envs
  ├─ RunLogger(config)                       # W&B init + artifact upload
  ├─ TaskSampler(num_tasks, num_envs)        # task assignment per segment
  │
  ├─ while total_trained < num_train_frames:
  │    ├─ task_ids = task_sampler.sample()
  │    ├─ agent.train(state, env, rng, seg, task_ids=task_ids)
  │    │    ├─ collect rollout (128 steps × 32 envs = 4096 transitions)
  │    │    ├─ compute GAE(γ=0.99, λ=0.95)
  │    │    └─ PPO update (4 epochs × 4 minibatches)
  │    ├─ _log_training_metrics()            # train/* scalars
  │    │
  │    └─ [periodic] _run_evaluation()
  │         ├─ for task_id in range(num_tasks):
  │         │    ├─ agent.evaluate(state, eval_env, rng, task_ids=fixed(task_id))
  │         │    └─ log eval/task_{i}/* scalars
  │         └─ log eval/aggregate/* scalars + console table
  │
  └─ Training done
```

### Key numbers (default config)

| Parameter | Value |
|-----------|-------|
| Parallel envs (train) | 32 |
| Parallel envs (eval) | 4 |
| Steps per rollout | 128 |
| Total train frames | 5 000 000 |
| Eval interval | 500 000 frames |
| Eval frames per task | 20 000 |
| Max episode length | 1 000 steps |
| Map size | 128 x 128 |
| Num tasks | 1 (task 0 only, expandable to 7) |

---

## File Map

### Entry points

| File | Purpose |
|------|---------|
| `scripts/train.py` | OmegaConf entry point. Parses `--env-config` + `--agent-config`, calls `load_agent(config)` then `Trainer(config, agent).run()`. |
| `demo.py` | Playable pygame game (human mode). Ground truth for game mechanics. |
| `configs/` | OmegaConf YAML config hierarchy. |

### `src/cogniland/envs/` — Environment

| File | Purpose |
|------|---------|
| `env.py` | `CognilandEnv` — batched numpy env running B parallel games. 8 actions (4 cardinal, forage, 3 craft). Loads pre-generated maps from `.pt` files. Computes RGB minimap with occlusion. Auto-resets done envs. |
| `tile_effects.py` | `TileEffects` dataclass + `drain_for()` — terrain HP drain table, tool modifiers, foraging params. Single source of truth. |
| `tasks.py` | `compute_task_reward()` — task 0: sparse reach bonus + step penalty + distance shaping. Tasks 1-6 are stubs. |
| `multitask_wrapper.py` | `MultiTaskEnvWrapper` — wraps `CognilandEnv`, applies task-specific rewards, provides task embeddings. |
| `task_sampler.py` | `TaskSampler` — round-robin or random task assignment across parallel envs. |
| `registry.py` | `make_env(env_id, config, train)` — factory creating `CognilandEnv` wrapped with `MultiTaskEnvWrapper`. |
| `gym_adapter.py` | `GymAdapter` — state-based API adapter for DreamerV3/STORM compatibility. |
| `simplexnoise/` | Simplex noise for map generation. |

### `src/cogniland/agents/` — Agents

| File | Purpose |
|------|---------|
| `agent.py` | `Agent` dataclass — functional container for `init`, `train`, `evaluate`, `select_action`. NOT a PyTree. |
| `state.py` | `AgentState`, `RuntimeState`, `PolicyParams`, `OptState` — unified state for all agents. |
| `registry.py` | `@register_agent` decorator, `AgentRegistry`, `load_agent()` — auto-discovers agents on import. |
| `ppo_rnn.py` | PPO-RNN agent factory (JAX/Flax). CNN+LSTM actor-critic, 8 actions. |
| `dreamer.py` | DreamerV3 agent factory (JAX). RSSM world model + imagination-based policy. |
| `storm.py` | STORM agent factory (JAX). Transformer-based world model (TSSM). |
| `utils.py` | `sg()`, `count_parameters()`, `RatioTracker`. |
| `commons/` | Shared NN building blocks: distributions, networks (CNN/MLP/RNN), normalizers, optimizers, replay buffers, preprocessing. |
| `policy/` | Actor-critic heads: MLP policy, imagination rollouts. |
| `world_models/` | World model implementations: `dreamerv3/` (RSSM), `storm/` (TSSM with attention). |

### `src/cogniland/trainer/` — Training orchestration

| File | Purpose |
|------|---------|
| `trainer.py` | `Trainer` — main loop, periodic eval across all tasks, metric logging. Agent-agnostic. |
| `run_logger.py` | `RunLogger` — W&B init (sweep-aware), metric registration, config artifact upload. |
| `checkpoint.py` | `CheckpointCallback` — orbax save/load, keep_last rotation, best tracking, W&B artifact upload. |
| `utils.py` | `RNGManager` — deterministic JAX key splitting with checkpoint/restore. |

### `src/cogniland/config/` — Configuration

| File | Purpose |
|------|---------|
| `env.py` | `setup_environment()` — XLA env vars, must be called before `import jax`. |
| `utils.py` | `load_config()`, `configure_sweep_config()` — OmegaConf merge (env + agent YAMLs). |
| `jax_config.py` | `COMPUTE_DTYPE = jnp.float32`. |

### `src/cogniland/shared/` and `src/cogniland/metrics/`

| File | Purpose |
|------|---------|
| `shared/logger.py` | `setup_logger()` — consistent Python logging. |
| `metrics/tracker.py` | `MetricsTracker` — rolling stats for train (aggregate) and eval (per-task). |

### Scripts & Configs

| File | Purpose |
|------|---------|
| `scripts/train.py` | CLI entry point. |
| `scripts/run_sweep.py` | Local parallel W&B agent launcher (non-SLURM). |
| `scripts/launch_sweep.sh` | SLURM job array submitter. |
| `scripts/job_sweep.slurm` | SLURM job script. |
| `scripts/generate_maps.py` | Map generation pipeline (4 biomes, simplex noise). |
| `scripts/generate_dataset.py` | Builds train/val/test `.pt` datasets. |
| `scripts/tune_tile_effects.py` | HP drain parameter tuning simulation. |
| `configs/env/cogniland.yaml` | Env config: map paths, parallel envs, reward coefficients. |
| `configs/agent/ppo_rnn.yaml` | PPO-RNN hyperparameters. |
| `configs/agent/dreamerv3.yaml` | DreamerV3 hyperparameters. |
| `configs/agent/storm.yaml` | STORM hyperparameters. |
| `configs/sweeps/` | W&B sweep YAMLs (seed benchmarks, HP search). |

---

## Environment Mechanics

### Map & terrain

128x128 maps with 9 terrain types, pre-generated across 4 biomes (balanced, archipelago, grassland, highland). Maps include **berry tiles** (heal HP on forage) and a **deadly 1-pixel border** (instant death).

| Index | Name | HP drain | Visibility | With raft | With rope |
|-------|------|----------|------------|-----------|-----------|
| 0 | ocean | 16 | 16 | 8 | — |
| 1 | deep_water | 10 | 12 | 3 | — |
| 2 | water | 6 | 10 | 1 | — |
| 3 | beach | 1 | 7 | — | — |
| 4 | sandy | 1 | 7 | — | — |
| 5 | grassland | 1 | 7 | — | — |
| 6 | forest | 3 | 5 | — | — |
| 7 | rocky | 6 | 10 | — | 1 |
| 8 | mountains | 12 | 22 | — | 3 |

**Shoes**: After 10 consecutive grassland steps, drain drops to 0.5.

### Actions (8 total)

| Index | Action | Effect |
|-------|--------|--------|
| 0-3 | up/down/left/right | Move, apply terrain HP drain |
| 4 | forage | On berry: +10 HP (no drain). On forest: +10 wood (costs drain). Elsewhere: no-op. |
| 5 | craft_raft | Costs 100 wood. Reduces water/ocean drain. One tool only. |
| 6 | craft_rope | Costs 100 wood. Reduces rocky/mountain drain. One tool only. |
| 7 | craft_shoes | Costs 100 wood. Reduces grassland drain after 10 consecutive steps. One tool only. |

### Observation dict

```
obs["minimap"]:  float32 [B, 3, 45, 45]   (2*22+1 = 45)
    3 RGB channels of the map, centered on agent
    Occlusion applied via heightmap raycasting — unseen cells are black
    Target marker drawn if within visibility radius and not occluded

obs["scalars"]:  float32 [B, 6]
    compass_x, compass_y    — unit vector toward target
    terrain_idx / 8         — normalized terrain index
    hp / 100                — normalized HP
    wood / 100              — normalized wood
    tool_id / 3             — normalized tool (0=none, 1=raft, 2=rope, 3=shoes)

obs["task_embedding"]:  float32 [B, 7]
    Orthogonal task embedding vector (from MultiTaskEnvWrapper)
```

### Task 0 reward (reach target)

```
r_step = -step_penalty                                    # per-step cost (0.01)
r_reach = +reach_bonus                                    # on reaching target (100.0)
r_shape = distance_shaping_coef * (1 - d_final/d_init)   # at episode end (0.1)
```

Tasks 1-6 are stubs (return 0) — to be defined for multi-task experiments.

---

## Neural Network Architecture (PPO-RNN)

```
Minimap [B, 3, 45, 45]
  → Conv2d(3→16, 3×3) → ReLU → MaxPool2d(2)
  → Conv2d(16→32, 3×3) → ReLU → AdaptiveMaxPool2d(4×4)
  → Flatten → [B, 512]

Scalars [B, 6]
  → Dense(6→64) → ReLU → [B, 64]

Task embedding [B, 7]
  → concatenated directly

Concat → [B, 583]
  → Dense(583→256) → ReLU
  → Dense(256→256) → ReLU

LSTM → [B, 256]

Actor head:  Dense(256→8) → Categorical (init std=0.01)
Critic head: Dense(256→1) → scalar (init std=1.0)
```

Orthogonal weight initialization throughout.

---

## W&B Metrics

### Training (step metric: `train_steps`)

| Key | Description |
|-----|-------------|
| `train/reward` | Episode return |
| `train/success` | 1 if return > 0 |
| `train/length` | Episode length |
| `train/moving_avg_reward` | Rolling mean return |
| `train/moving_avg_success_rate` | Rolling mean success |
| `train/fps` | Training frames/sec |
| `train/<agent_key>` | Agent-specific scalars (policy_loss, value_loss, entropy, etc.) |

### Evaluation (step metric: `train_frames`)

| Key | Description |
|-----|-------------|
| `eval/task_{i}/avg_reward` | Mean reward on task i |
| `eval/task_{i}/avg_success` | Mean success rate on task i |
| `eval/task_{i}/avg_length` | Mean episode length on task i |
| `eval/aggregate/avg_reward` | Mean reward across all tasks |
| `eval/aggregate/avg_success` | Mean success across all tasks |

---

## How to Add a New Agent

1. Create `src/cogniland/agents/my_agent.py`.
2. Decorate the factory with `@register_agent("my_agent")`.
3. The factory receives `(config, obs_space, act_space)` and returns an `Agent(init=..., train=..., evaluate=..., select_action=..., state_from_checkpoint=...)`.
4. `train()` must accept `task_ids` kwarg and return `(new_state, metrics_dict)` where `metrics_dict` contains `"episode_info"` with `returned_episode_returns`, `returned_episode_lengths`, `returned_episode`.
5. Create `configs/agent/my_agent.yaml` with `agent.name: my_agent`.
6. Run: `python scripts/train.py --env-config configs/env/cogniland.yaml --agent-config configs/agent/my_agent.yaml`

No changes needed to Trainer, RunLogger, MetricsTracker, or any infrastructure.

---

## Design Invariants

- **Trainer never imports anything agent-specific.** No `if config.agent.name == "ppo"`.
- **Agent never imports anything trainer-specific.** No `import wandb`.
- **Task identity flows through `task_ids` kwarg**, not env switching or global state.
- **`AgentState` is immutable** and threaded through the trainer.
- **Resolved config uploaded as a W&B artifact every run.**
- **Only `train_state` is checkpointed** (params + optimizer), never runtime.
