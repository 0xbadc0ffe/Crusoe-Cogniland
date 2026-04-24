# Cogniland — Architecture & Developer Guide

## Project Overview

Cogniland is a multi-task RL framework where agents learn to navigate procedurally generated 128x128 maps. The agent starts at a random spawn point and must reach a target position while managing **HP** (health points) and **wood** (gathered from forests). Different terrain types impose HP drains, and the agent can **forage** (berries heal HP, forests yield wood) and **craft tools** (raft, rope, shoes) that reduce terrain costs. Maps are pre-generated in pools of 256 (train) / 16 (val/test) across 4 biomes.

The agent's observation at every step is a dict of three arrays:

1. `minimap`: `int8 [B, 45, 45]` — per-cell **tile-class id**. All salient entities live in this single channel and each class id gets its own row in the learned `nn.Embed(14, 8)` table (priority on collision: `TARGET_YES > TARGET_NO > BERRY > DEADLY > terrain > UNSEEN`). The agent is centred on the patch; unseen cells are occluded via heightmap raycasting or lie outside the visibility disk. **RGB is not fed to the agent** — it's loaded from the map `.pt` files purely for trajectory viz.
2. `scalars`: `float32 [B, 6]` — `compass_x, compass_y` (unit vector toward target midpoint), `tile_class/9` (0..8 base terrain, 9 berry), `hp/100`, `wood/100`, `tool/3`.
3. `task_embedding`: `float32 [B, 7]` — one-hot task id (from `MultiTaskEnvWrapper`).

The framework supports three agents: **PPO-RNN** (JAX/Flax, default; ~330k params — CNN trunk ≈25k, post-concat Dense ≈156k, LSTM ≈132k, heads ≈1k), **DreamerV3**, and **STORM**. New agents plug in via a `@register_agent` decorator — all training orchestration, evaluation, and logging are agent-agnostic.

### Training-dynamics notes (April 2026)

The reward defaults were re-tuned after a deep-dive on the "success climbs to 30% at 400k
then collapses" failure mode. Keep these in mind before tweaking:

- `reward.death_penalty=0` — a non-zero sparse death penalty creates a cliff in the
  value function that traps PPO in a "die quickly" local optimum (ma_r ≈ -5.8, 0% success)
  regardless of entropy / lr / clip_grad tuning. Confirmed across 8 parallel sweeps.
- `reward.shaping_coef=0.3` (up from 0.1) — stronger PBRS gives a per-step gradient
  that survives GAE discount and minibatch advantage normalisation.
- `agent.entropy_coef=3e-2` (up from 1.5e-3) — lower coefs let entropy collapse before
  the agent has stumbled on enough reach-target trajectories to learn from.
- `tasks: [0]` in `configs/env/cogniland.yaml` — multi-task round-robin across all
  7 tasks dilutes the task-0 signal 7× with stub tasks 1-6, swamping the shared
  trunk gradient. `tasks` is a list of task ids; widen it (e.g. `[0, 4]`) to
  train on a subset of tasks without forcing all 7 into the mix.
- `env.biome_filter=[balanced]` restricts training to the 64 balanced-biome maps for fast
  iteration. `null` uses all 256 maps.

The current PPO-RNN baseline at 3M frames converges to ma_r ≈ +13, success ≈ 20-24% on
balanced-biome task 0. Further gains (berry-detour skill for long episodes) are pending.

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
  ├─ TaskSampler(task_ids, num_envs)         # task assignment per segment
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
  │         ├─ for task_id in config.tasks:
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

128x128 maps with 9 base terrain types (+ berry overlay = 10 tile classes), pre-generated across 4 biomes (balanced, archipelago, grassland, highland). Maps also include a **deadly 1-pixel border** (instant death).

| Index | Name | HP drain | Visibility | With raft | With rope |
|-------|------|----------|------------|-----------|-----------|
| 0 | ocean | 8 | 22 | 4 | — |
| 1 | deep_water | 5 | 18 | 2 | — |
| 2 | water | 3 | 14 | 1 | — |
| 3 | beach | 1 | 12 | — | — |
| 4 | sandy | 1 | 12 | — | — |
| 5 | grassland | 1 | 12 | — | — |
| 6 | forest | 2 | 10 | — | — |
| 7 | rocky | 6 | 18 | — | 1 |
| 8 | mountains | 8 | 22 | — | 2 |
| 9 | **berry** (overlay) | **0** | — | — | — |

Drain values are defined in `src/cogniland/envs/tile_effects.py` — edit that
single source of truth rather than the table above if you retune.

**Shoes**: After 10 consecutive grassland steps, grassland drain drops to 0.5. **Berry**: an overlay on forest/beach — stepping onto it is free (0 drain) and `forage` on a berry tile heals +10 HP.

### Actions (8 total)

| Index | Action | Effect |
|-------|--------|--------|
| 0-3 | up/down/left/right | Move, apply terrain HP drain |
| 4 | forage | On berry: +10 HP (no drain). On forest: +10 wood (costs drain). Elsewhere: no-op. |
| 5 | craft_raft | Costs 100 wood. Reduces water/ocean drain. One tool only. |
| 6 | craft_rope | Costs 100 wood. Reduces rocky/mountain drain. One tool only. |
| 7 | craft_shoes | Costs 100 wood. Reduces grassland drain after 10 consecutive steps. One tool only. |

### Observation dict (returned every step and on reset)

```
obs["minimap"]:  int8 [B, 45, 45]         (patch radius 22 → diameter 45)
    Single-channel tile-class id for every visible cell. Agents embed this
    via nn.Embed(14, embed_dim). Computed on GPU via _compute_tile_idx_jax
    when occlude=True (default). Overlay priority on collision:
    TARGET_YES > TARGET_NO > BERRY > DEADLY > terrain > UNSEEN.
       0  TILE_UNSEEN       (occluded by heightmap or outside visibility disk / OOB)
       1  ocean
       2  deep_water
       3  water
       4  beach
       5  sandy
       6  grassland
       7  forest
       8  rocky
       9  mountains
      10  TILE_BERRY        (visible berry tile)
      11  TILE_TARGET_YES   (visible YES target)
      12  TILE_TARGET_NO    (visible NO decoy)
      13  TILE_DEADLY       (1-px deadly border)

obs["scalars"]:  float32 [B, 6]
    [0] compass_x           — dc/|d|, column direction to YES/NO midpoint
    [1] compass_y           — dr/|d|, row direction to YES/NO midpoint
    [2] tile_class / 9      — current cell; 0..8 terrain, 9 = berry overlay
    [3] hp / hp_max         — normalized HP (hp_max=100)
    [4] wood / wood_max     — normalized wood (wood_max=100)
    [5] tool_id / 3         — 0=none, 1=raft, 2=rope, 3=shoes

obs["task_embedding"]:  float32 [B, 7]
    One-hot of task_id (eye(7)[task_ids]), injected by MultiTaskEnvWrapper.
```

RGB is **not** part of the agent obs. The map dataset `.pt` still carries an
`rgb` key, but the env keeps it only for trajectory visualisation — see the
`# env obs is tile-idx` comment in `CognilandEnv.__init__` and `_get_obs`.

### Task 0 reward (reach target)

```
r_reach = +reach_bonus                                    # sparse, on reaching target (150.0)
r_shape = shaping_coef * (ctg_prev - ctg_curr)            # target PBRS (1.0)
```

The PBRS potential is the **Euclidean distance** from the agent's cell to
the YES/NO midpoint — no graph, no Dijkstra, just a sqrt evaluated at
every step. The info keys `ctg_prev` / `ctg_curr` / `ctg_spawn` (kept for
naming compatibility) all carry this distance. Along a successful
trajectory the shaping telescopes to `shaping_coef · ctg_spawn` — bounded.

`step_penalty`, `hp_coef`, and `death_penalty` default to 0 (the original
HP-aware PBRS is kept in the code for ablations but disabled in the shipped
config).

Tasks 1-6 are stubs (return 0) — to be defined for multi-task experiments.

---

## Neural Network Architecture (PPO-RNN)

The minimap is consumed as tile-class indices (int8 in `{0..13}`) and embedded
via a learned lookup table — no RGB. Berries and targets live in this single
channel as class ids 10, 11, 12; the `nn.Embed` table learns a distinct 8-dim
vector per class. Two CoordConv channels (normalised row/col in `[-1, 1]`)
are appended so the CNN can reason about direction to each pixel in the
egocentric patch without re-learning translation from scratch.

The CNN keeps a 7×7 spatial output all the way to the flatten step, so
fine-grained positions of targets and berries aren't averaged away before
the MLP.

```
Minimap      [B, 45, 45] (int8)    → nn.Embed(14, 8)     → [B, 45, 45, 8]
CoordConv (rr, cc) in [-1, 1]                              2 channels
  → concat → [B, 45, 45, 10]
  → Conv(10→24, 3×3 VALID) → ReLU → MaxPool(2,2)   # 45 → 43 → 21
  → Conv(24→32, 3×3 VALID) → ReLU → MaxPool(2,2)   # 21 → 19 →  9
  → Conv(32→48, 3×3 VALID) → ReLU                  #  9 →  7
  → Conv(48→24, 1×1)        → ReLU                 # channel bottleneck, spatial 7×7
  → Flatten → [B, 7·7·24 = 1176]

Scalars [B, 6]
  → Dense(6→32) → ReLU

Task embedding [B, 7]
  → concatenated directly

Concat [B, 1176 + 32 + 7 = 1215]
  → Dense(→128) → ReLU
  → Dense(→128) → ReLU

LSTM (OptimizedLSTMCell, features=lstm_size=128) → [B, 128]
  (skipped if agent.use_rnn=false; carry is threaded through unchanged.)

Actor head:  Dense(128→8) → Categorical (init std=0.01)
Critic head: Dense(128→1) → scalar          (init std=1.0)
```

Orthogonal weight initialisation throughout (embedding init: normal, std=0.5).
Defaults from `configs/agent/ppo_rnn.yaml`: `embed_dim=8`, `hidden_size=128`,
`lstm_size=128`, `num_tile_classes=14` (kept in sync with `NUM_TILE_CLASSES`).

---

## W&B Metrics

Raw per-episode values only — no rolling averages are maintained in code.
Use W&B's UI smoothing to visualise trends.

### Training (step metric: `train_steps` / `train_episode`)

| Key | Description |
|-----|-------------|
| `train/reward` | Episode return (one entry per finished episode) |
| `train/success` | 1 if task success criterion met, else 0 |
| `train/length` | Episode length in steps |
| `train/fps` | Training frames/sec |
| `train/frame`, `train/episode` | Global counters |
| `train/task_{t}/reward` | Same as `train/reward` but scoped to episodes run on task `t` |
| `train/task_{t}/success`, `train/task_{t}/length` | Per-task raw scalars |
| `train/biome_{b}/reward` | Per-biome raw episode return |
| `train/biome_{b}/success`, `train/biome_{b}/length` | Per-biome raw scalars |
| `train/<agent_key>` | Agent-specific scalars (policy_loss, value_loss, entropy, …) |

### Evaluation (step metric: `train_frames`)

Aggregated over every finished episode in one eval set.

| Key | Description |
|-----|-------------|
| `eval/task_{i}/reward` | Mean reward on task `i` |
| `eval/task_{i}/success` | Mean success on task `i` |
| `eval/task_{i}/length` | Mean episode length on task `i` |
| `eval/task_{i}/episodes` | Number of finished eval episodes |
| `eval/aggregate/reward` | Mean reward across all configured tasks |
| `eval/aggregate/success` | Mean success across all configured tasks |
| `eval/aggregate/length` | Mean length across all configured tasks |

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
