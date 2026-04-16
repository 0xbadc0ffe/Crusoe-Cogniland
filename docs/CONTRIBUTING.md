# Contributing to Cogniland

## Architecture overview

The codebase is split into four layers with strict dependency boundaries:

```
Config  -->  Trainer  -->  Agent  -->  Environment
```

Each layer imports only from the layer to its right. The Trainer never knows which agent it's running. The Agent never knows it's being logged to W&B. This means you can add a new agent, a new task, or a new sweep config without touching the other layers.

## Environment layer

**Location**: `src/cogniland/envs/`

The environment is a batched numpy game running B parallel episodes. No JAX, no PyTorch — just numpy arrays. This keeps the env simple and makes it easy to test.

### Game state per env

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `pos_r`, `pos_c` | int | [0, 127] | Agent position on 128x128 grid |
| `hp` | float | [0, 100] | Health points, dies at 0 |
| `wood` | int | [0, 100] | Wood gathered from forests |
| `tool` | int | 0-3 | 0=none, 1=raft, 2=rope, 3=shoes |
| `consec_grass` | int | >= 0 | Consecutive grassland steps (for shoes bonus) |
| `steps` | int | >= 0 | Step counter, episode ends at `max_steps` |

### Actions

| Index | Name | What happens |
|-------|------|-------------|
| 0 | up | Move (-1, 0), pay terrain HP drain |
| 1 | down | Move (+1, 0), pay terrain HP drain |
| 2 | left | Move (0, -1), pay terrain HP drain |
| 3 | right | Move (0, +1), pay terrain HP drain |
| 4 | forage | Stay in place. Berry tile: +10 HP (free). Forest: +10 wood (costs drain). Other: no-op. |
| 5 | craft_raft | Costs 100 wood. Reduces water/ocean drain. Fails if no wood or already have tool. |
| 6 | craft_rope | Costs 100 wood. Reduces rocky/mountain drain. |
| 7 | craft_shoes | Costs 100 wood. Reduces grassland drain after 10 consecutive steps. |

Crafting is one-time: you can hold at most one tool, and it can't be replaced.

### Terrain drain table

The HP cost of stepping onto each terrain, and how tools modify it:

| Terrain | Base drain | With raft | With rope | With shoes (10+ steps) |
|---------|-----------|-----------|-----------|----------------------|
| ocean | 16 | 8 | - | - |
| deep_water | 10 | 3 | - | - |
| water | 6 | 1 | - | - |
| beach | 1 | - | - | - |
| sandy | 1 | - | - | - |
| grassland | 1 | - | - | 0.5 |
| forest | 3 | - | - | - |
| rocky | 6 | - | 1 | - |
| mountains | 12 | - | 3 | - |

Source of truth: `src/cogniland/envs/tile_effects.py`

### Observations

The agent sees:

- **minimap** `[B, 3, 45, 45]` — RGB patch of the map centered on the agent, with Bresenham raycasting for line-of-sight occlusion. Unseen cells are black. Vision radius depends on current terrain (mountains: 22, forest: 5).
- **scalars** `[B, 6]` — compass direction to target (unit vector x,y), terrain index / 8, hp / 100, wood / 100, tool_id / 3.
- **task_embedding** `[B, 7]` — fixed orthogonal vector identifying the current task.

### Maps

Maps are pre-generated and stored as `.pt` files in `data/strategy/`. Each file contains:

```python
{
    "rgb": uint8 [N, 128, 128, 3],       # Rendered RGB map
    "heightmap": float32 [N, 128, 128],   # For occlusion computation
    "terrain_idx": int8 [N, 128, 128],    # Terrain class per cell (-1 = deadly border)
    "berry_mask": bool [N, 128, 128],     # Berry locations
    "biomes": list[str],                  # Biome name per map
    "seeds": list[int],                   # RNG seed per map
}
```

Train: 256 maps (64 per biome). Val/Test: 16 maps (4 per biome). Biomes: balanced, archipelago, grassland, highland.

To regenerate: `python scripts/generate_strategy_dataset.py`

### Adding a new task

Tasks are defined in `src/cogniland/envs/tasks.py`. A task is just a reward function:

```python
def _task_N_reward(mask, dones, info, config):
    """Return float array of rewards for envs where task_ids == N."""
    ...
```

Then add a dispatch line in `compute_task_reward()`:

```python
mask_N = task_ids == N
if mask_N.any():
    rewards[mask_N] = _task_N_reward(mask_N, dones, info, config)
```

Update `num_tasks` in `configs/env/cogniland.yaml`.

## Agent layer

**Location**: `src/cogniland/agents/`

An agent is a `@dataclass` containing closures over the network and optimizer. It is NOT a class hierarchy — there's no `BaseAgent` to subclass.

### The Agent dataclass

```python
@dataclass
class Agent:
    init: Callable[[PRNGKey], AgentState]
    train: Callable   # (state, env, rng, num_frames, task_ids=...) -> (state, metrics)
    evaluate: Callable # (state, env, rng, num_frames, task_ids=...) -> metrics
    select_action: Callable
    state_from_checkpoint: Callable
```

The factory function (`make_ppo_rnn`, `make_dreamerv3`, etc.) sets up the network and optimizer, then returns an `Agent` with those closures. The Trainer calls these methods without knowing what's inside.

### Adding a new agent

1. Create `src/cogniland/agents/my_agent.py`
2. Write a factory function:

```python
from cogniland.agents.registry import register_agent
from cogniland.agents.agent import Agent

@register_agent("my_agent")
def make_my_agent(config, obs_space, act_space) -> Agent:
    # Set up network, optimizer, etc.

    def init(rng):
        # Initialize params, return AgentState
        ...

    def train(state, env, rng, num_frames, progress_bar=None,
              checkpoint_callback=None, task_ids=None):
        # Collect rollout, update params
        # Return (new_state, metrics_dict)
        ...

    def evaluate(state, env, rng, num_frames, progress_bar=None,
                 task_ids=None):
        # Run policy without updates
        # Return metrics_dict
        ...

    return Agent(init=init, train=train, evaluate=evaluate, ...)
```

3. Create `configs/agent/my_agent.yaml`:

```yaml
agent:
  name: my_agent
  # ... hyperparameters
```

4. Run: `python scripts/train.py --env-config configs/env/cogniland.yaml --agent-config configs/agent/my_agent.yaml`

No changes to the Trainer or env code.

### Key contracts

The `train()` function must return `(new_state, metrics)` where `metrics` contains:

```python
{
    "episode_info": {
        "returned_episode_returns": np.ndarray,  # [B] episode returns (0 if not done)
        "returned_episode_lengths": np.ndarray,  # [B] episode lengths (0 if not done)
        "returned_episode": np.ndarray,          # [B] bool mask of which envs just finished
    },
    # Optional scalar losses (auto-logged by Trainer):
    "policy_loss": float,
    "value_loss": float,
    "entropy": float,
}
```

The `task_ids` kwarg is a numpy int array `[num_envs]`. Use it to look up task embeddings:

```python
task_emb = np.eye(config.task_embedding_dim, dtype=np.float32)[task_ids]
```

### Env API (from the agent's perspective)

```python
obs = env.reset()                        # dict with "minimap", "scalars", "task_embedding"
obs, rewards, dones, info = env.step(a)  # a is np.ndarray [B] of ints 0-7
```

The env returns numpy arrays. Convert to JAX at the agent boundary, convert actions back to numpy for `env.step()`.

## Trainer layer

**Location**: `src/cogniland/trainer/`

The Trainer runs a single training loop (not one per task). It:

1. Samples task assignments via `TaskSampler` (round-robin or random)
2. Calls `agent.train(state, env, rng, num_frames, task_ids=task_ids)`
3. Logs training metrics (episode returns, losses) to W&B
4. Periodically runs evaluation: for each task, calls `agent.evaluate()` with `task_ids=fixed(task_id)`
5. Logs per-task and aggregate eval metrics, prints a console table

### W&B metrics

| Key | Step metric | When |
|-----|------------|------|
| `train/reward`, `train/success`, `train/length` | `train_steps` | Per completed episode |
| `train/moving_avg_*` | `train_steps` | Rolling window |
| `train/<agent_key>` | `train_steps` | Per training segment |
| `eval/task_{i}/avg_*` | `train_frames` | Each eval checkpoint |
| `eval/aggregate/avg_*` | `train_frames` | Each eval checkpoint |

### Config system

Two YAMLs merged at startup (agent wins on conflicts):

```
env config    +    agent config    +    CLI dotlist    +    W&B sweep
(experiment)       (hyperparams)       (overrides)         (overrides)
```

Priority: sweep > CLI > agent > env.

## Sweep infrastructure

### SLURM cluster

```bash
# Create sweep + submit job array
./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_seeds.yaml -n 10 -r 1

# Dry run (prints command without submitting)
./scripts/launch_sweep.sh --dry-run configs/sweeps/ppo_rnn_seeds.yaml -n 10

# Reuse existing sweep
./scripts/launch_sweep.sh --sweep-id entity/project/abc123 -n 10 -r 1
```

### Local

```bash
wandb sweep configs/sweeps/ppo_rnn_seeds.yaml
python scripts/run_sweep.py <SWEEP_ID> --num-agents 5 --count 1 --gpus 0 1
```

### Sweep YAML structure

Seed benchmark (grid):

```yaml
method: grid
parameters:
  seed: {values: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]}
```

HP search (random):

```yaml
method: random
parameters:
  agent.lr: {distribution: log_uniform_values, min: 1e-5, max: 1e-3}
  agent.entropy_coef: {distribution: log_uniform_values, min: 0.001, max: 0.1}
```

## Testing

```bash
pytest tests/
```

The env layer has 24 tests covering task sampling, env step/reset, foraging, crafting, auto-reset, timeout, and reward computation.

## Checkpointing

Uses orbax. Only `train_state` (params + optimizer) is saved — runtime state (replay buffer, counters) is ephemeral.

```yaml
# In agent config
agent:
  checkpoint:
    enabled: true
    interval: 1000      # Save every N training steps
    keep_last: 3        # Keep last 3 checkpoints
    save_best: true     # Track best by eval return
```

Checkpoints are saved to `results/{wandb_run_id}/checkpoints/`.
