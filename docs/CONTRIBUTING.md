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
| 4 | forage | Stay in place. Berry tile: +10 HP (free). Forest: +10 wood (costs drain) |
| 5 | craft_raft | Costs 100 wood. Reduces water/ocean drain. |
| 6 | craft_rope | Costs 100 wood. Reduces rocky/mountain drain. |
| 7 | craft_shoes | Costs 100 wood. Reduces grassland drain after 10 consecutive steps. |

Crafting is one-time: you can hold at most one tool, and it can't be replaced.

### Terrain drain table

The HP cost of stepping onto each tile, and how tools modify it:

| Tile | Base drain | With raft | With rope | With shoes (10+ steps) |
|------|-----------|-----------|-----------|----------------------|
| ocean | 16 | 8 | - | - |
| deep_water | 10 | 3 | - | - |
| water | 6 | 1 | - | - |
| beach | 1 | - | - | - |
| sandy | 1 | - | - | - |
| grassland | 1 | - | - | 0.5 |
| forest | 2 | - | - | - |
| rocky | 6 | - | 1 | - |
| mountains | 12 | - | 3 | - |
| **berry** | **0** | - | - | - |

`berry` is an overlay on forest/beach tiles. Stepping onto it is free (0 drain) and `forage` on a berry tile heals +10 HP without paying drain for the step.

Source of truth: `src/cogniland/envs/tile_effects.py` + the berry branch in `src/cogniland/envs/env.py`.

### Observations

The agent sees a dict of three arrays (returned from both `reset()` and `step()`):

- **minimap** `int8 [B, 45, 45]` — egocentric 45×45 patch (radius 22) of **tile-class
  IDs**, not RGB. Exactly one label per cell; berry / target / deadly overlays override
  the base terrain. Vision radius depends on the agent's current terrain (mountains/ocean:
  22, forest: 10; full table in `configs/env/cogniland.yaml::env.terrain_vis_radius`),
  and line-of-sight occlusion is applied via a precomputed Bresenham
  visibility LUT. The 14 classes (from `NUM_TILE_CLASSES` in `src/cogniland/envs/env.py`):

  | ID | Class | Notes |
  |----|-------|-------|
  | 0  | `TILE_UNSEEN` | Occluded, outside visibility disk, or OOB |
  | 1  | ocean | |
  | 2  | deep_water | |
  | 3  | water | |
  | 4  | beach | |
  | 5  | sandy | |
  | 6  | grassland | |
  | 7  | forest | |
  | 8  | rocky | |
  | 9  | mountains | |
  | 10 | `TILE_BERRY` | Overlay on forest/beach |
  | 11 | `TILE_TARGET_YES` | Real target, if visible |
  | 12 | `TILE_TARGET_NO` | Decoy target, if visible |
  | 13 | `TILE_DEADLY` | 1-px deadly border |

  Agents are expected to embed this via a learned lookup table
  (`nn.Embed(14, embed_dim)` in the PPO-RNN trunk), not to treat it as pixels.
- **scalars** `float32 [B, 6]` — `[compass_x, compass_y, tile_class/9, hp/hp_max,
  wood/wood_max, tool_id/3]`. Compass is a unit vector from the agent toward the
  YES/NO target midpoint. `tile_class` of the current cell is 0..8 for base terrain and
  9 when the agent stands on a berry overlay.
- **task_embedding** `float32 [B, 7]` — one-hot of the current task id
  (`np.eye(7)[task_ids]`), injected by `MultiTaskEnvWrapper`.

RGB is **not** part of the agent obs; the env keeps the `rgb` key from the map dataset
only for trajectory visualisation. See the `# env obs is tile-idx` comment in
`CognilandEnv._get_obs`.

### Maps

Maps are pre-generated and stored as `.pt` files in `data/maps/`. Each file contains:

```python
{
    "rgb":            uint8   [N, 128, 128, 3],     # Trajectory-viz only, not fed to agent
    "heightmap":      float32 [N, 128, 128],        # For occlusion (also used offline)
    "terrain_idx":    int8    [N, 128, 128],        # 0..8 terrain class, -1 = deadly border
    "berry_mask":     bool    [N, 128, 128],        # Berry overlay locations
    "visibility_lut": uint8   [N, 128, 128, 254],   # Packed 45x45 Bresenham occlusion mask
                                                    # per cell. REQUIRED — _load_maps
                                                    # raises if missing.
    "biomes":         list[str],                    # Biome name per map
    "seeds":          list[int],                    # RNG seed per map
}
```

Train: 256 maps (64 per biome). Val/Test: 16 maps (4 per biome). Biomes: `balanced`,
`archipelago`, `grassland`, `highland`. `env.biome_filter` in
`configs/env/cogniland.yaml` subsets these (e.g. `[balanced]` → 64 training maps).

To regenerate: `python scripts/generate_dataset.py` (this also runs
`precompute_visibility.compute_visibility_luts` — required, since the env will refuse
to load a dataset without `visibility_lut`).

### Adding a new task

All tasks share one reward function: `compute_task_reward()` in
`src/cogniland/envs/tasks.py`. The shared base reward is:

```
r = -step_penalty
  + reach_bonus   * [reached YES or NO]
  + shaping_coef  * (ctg_prev - ctg_curr)      # PBRS on Dijkstra cost-to-go
  - death_penalty * [died]                     # sparse, terminal (default 0)
  + forage_berry_bonus * [action==4 on a berry tile]   # optional Markovian shaping
```

On top of that, task-specific bonuses are added via `task_ids` masks:

- **Tasks 1–3 (biome classification)**: `+correct_answer_bonus` when the reached target
  matches the biome question — see `_TASK_BIOME_QUESTION` (task 1 ↔ archipelago,
  2 ↔ grassland, 3 ↔ highland).
- **Tasks 4–6 (craft)**: `+craft_bonus` on the step the required tool is crafted — see
  `_TASK_CRAFT_TOOL` (task 4 ↔ raft, 5 ↔ rope, 6 ↔ shoes).
- **Task 0**: reach-target only (no extra bonus on top of the shared base).

To add a new task `N`, edit `compute_task_reward()` and add your mask-based bonus:

```python
mask_N = task_ids == N
if mask_N.any():
    # Read per-env signals from info (e.g. info["reached_yes"], info["crafted"])
    # and add a scalar bonus to rewards[mask_N].
    rewards[mask_N] += my_bonus * my_condition[mask_N]
```

Then add the task id to the `tasks:` list in `configs/env/cogniland.yaml` (e.g.
`tasks: [0, 4]`) and — if the task needs a success criterion visible in eval
metrics — extend `MultiTaskEnvWrapper._compute_task_success` to recognise it.
The one-hot task embedding has a fixed width of `TASK_EMBEDDING_DIM` (see
`src/cogniland/envs/tasks.py`, currently 7), so new task ids are valid
as long as they are `< TASK_EMBEDDING_DIM`.

## Agent layer

**Location**: `src/cogniland/agents/`

An agent is a `@dataclass` containing closures over the network and optimizer. It is NOT a class hierarchy — there's no `BaseAgent` to subclass.

### The Agent dataclass

```python
@dataclass
class Agent:
    init: Callable[[PRNGKey], AgentState]
    train: Callable   # (state, env, rng, num_train_frames, progress_bar=None,
                      #  checkpoint_callback=None, task_ids=None) -> (state, metrics)
    evaluate: Callable # (state, env, rng, num_eval_frames, progress_bar=None,
                       #  task_ids=None) -> metrics
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
from cogniland.envs.tasks import TASK_EMBEDDING_DIM
task_emb = np.eye(TASK_EMBEDDING_DIM, dtype=np.float32)[task_ids]
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

Raw values only — no rolling averages are maintained in code.

| Key | Step metric | When |
|-----|------------|------|
| `train/reward`, `train/success`, `train/length` | `train_steps` | Per completed episode |
| `train/task_{t}/{reward,success,length}` | `train_steps` | Per episode, scoped to task `t` |
| `train/biome_{b}/{reward,success,length}` | `train_steps` | Per episode, scoped to biome `b` |
| `train/<agent_key>` | `train_steps` | Per training segment |
| `eval/task_{i}/{reward,success,length,episodes}` | `train_frames` | Each eval checkpoint |
| `eval/aggregate/{reward,success,length}` | `train_frames` | Each eval checkpoint |

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

The env layer has 25 tests in `tests/test_env_layer.py` covering task sampling, env
step/reset, foraging, crafting, auto-reset, timeout, cost-to-go / PBRS shaping, and
reward computation.

## Checkpointing

Uses orbax. Only `train_state` (params + optimizer) is saved — runtime state (replay buffer, counters) is ephemeral.

```yaml
# In agent config (defaults from configs/agent/ppo_rnn.yaml)
agent:
  checkpoint:
    enabled: true
    save_best: true         # Track + save best-by-eval checkpoint
    save_last: true         # Always refresh 'last/'
    save_only_best: true    # Skip periodic step_* snapshots; only write 'best/' and 'last/'
    upload_to_wandb: false  # Upload best checkpoint as a W&B artifact
    checkpoint_dir: checkpoints
    # Advanced (not set in ppo_rnn.yaml, but respected by CheckpointCallback):
    # interval: 1000        # Periodic step_* save cadence (ignored when save_only_best=true)
    # keep_last: 3          # Rotation window for step_* checkpoints
```

Checkpoints are saved to `results/{wandb_run_id}/{checkpoint_dir}/`.
