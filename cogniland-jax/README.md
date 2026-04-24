# cogniland-jax

[Gymnax](https://github.com/RobertTLange/gymnax)-style of Cogniland environment. One agent on a 128×128 procedurally-generated
island must navigate to a target tile while managing HP, gathering wood from forests, foraging berries for healing, and crafting a raft / rope / shoes to cut terrain drain.

## Install

```bash
pip install -e cogniland-jax
# or: pip install -e .   (from inside the cogniland-jax/ dir)
```

Demo + pygame renderer are optional:

```bash
pip install -e 'cogniland-jax[demo]'
```

## Generate the map dataset

The env samples per-episode spawns / targets from a pre-computed pool of
island maps. Before training, demoing, or anything else, generate it:

```bash
# default splits: 64 train / 4 val / 4 test per biome (4 biomes in total)
python cogniland-jax/scripts/generate_dataset.py --output-dir data/maps --preview
```

Customise split sizes with `--train-per-biome`, `--val-per-biome`,
`--test-per-biome`. The script writes three `.pt` files:

```
data/maps/
├── train.pt       # terrain_idx + berry_mask + heightmap + visibility Look Up Table (LUT)
├── val.pt
└── test.pt
```

Full-default build ≈ 1.1 GB total and takes a couple of minutes on a
multi-core CPU (the visibility LUT precompute is CPU-bound and
parallelises across maps). For a tiny smoke dataset:

## Quickstart

```python
import jax
import jax.numpy as jnp
from cogniland_jax import CognilandEnv, EnvParams
from cogniland_jax.maps import load_map_arrays

# 1. Load a pre-generated map dataset (see above).
arrays = load_map_arrays("data/maps/train.pt", biome_filter=["balanced"])

# 2. Pack into a flax.struct EnvParams.
params = EnvParams.from_map_arrays(
    **arrays,
    difficulty=jnp.int32(1),   # 0=easy (<=20), 1=medium (<=50), 2=hard (inf)
)

# 3. Construct the env. `reset` / `step` are side-effect-free functions of (key, state, action, params).
env = CognilandEnv(default_params=params)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)

for _ in range(100):
    key, act_key, step_key = jax.random.split(key, 3)
    action = jax.random.randint(act_key, (), 0, env.num_actions)
    obs, state, reward, done, info = env.step(step_key, state, action, params)
    if bool(done):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)
```

## Observation / Action / Reward

| Field             | Shape / type                  | Notes |
|-------------------|-------------------------------|-------|
| `obs.minimap`     | `int8 [45, 45]`               | tile-class ids (0=unseen, 1–9=terrain, 10=berry, 11=TARGET_YES, 12=TARGET_NO, 13=deadly (map border)) |
| `obs.scalars`     | `float32 [6]`                 | `[compass_x, compass_y, tile_cls/10, hp/hp_max, wood/wood_max, tool/3]` — `tile_cls` raw values: 0..8 terrain, **10 berry** (so berry→1.0, mountains→0.8) |
| `obs.task_embedding` | `float32 [7]`              | one-hot task id |

Action space: `Discrete(8)` — 0–3 cardinal moves, 4 forage,
5–7 craft raft / rope / shoes.

Reward (shared base across all tasks):

```
r_base = -step_penalty
       + reach_bonus      · [reached YES or NO]
       + shaping_coef     · (ctg_prev - ctg_curr)      # Dijkstra cost-to-go on drain graph
       + hp_coef          · (hp_curr - hp_prev)        # HP delta (healing / drain)
       - death_penalty    · [died]
```

On top of the base the environment adds a **task-specific bonus** (see
the *Tasks* section below). All coefficients live on `EnvParams` and
can be overridden at construction time.

Termination: HP ≤ 0, step into a deadly border tile, reach either YES
or NO target, **or** `step_count ≥ params.max_steps` (truncation).

## Tasks

Each episode has a `task_id ∈ {0, …, 6}` broadcast into the agent via
the one-hot `task_embedding` channel on every obs. The task id is drawn
uniformly at reset; override per-env with
`state = state.replace(task_id=jnp.int32(t))` (or, when using the
batched wrapper, `env.set_tasks(task_ids)`). A map's biome is exposed
to the env (not the agent) through `params.biome_id[state.map_idx]`.

| id | name              | bonus (added to base)             | success |
|---:|-------------------|-----------------------------------|---------|
| 0  | `REACH`           | —                                 | reached YES or NO |
| 1  | `CLS_ARCHIPELAGO` | `+correct_answer_bonus` on a reach that matches the rule | correct pick |
| 2  | `CLS_GRASSLAND`   | same                              | correct pick |
| 3  | `CLS_HIGHLAND`    | same                              | correct pick |
| 4  | `CRAFT_RAFT`      | `+craft_bonus` on the craft step  | required tool held |
| 5  | `CRAFT_ROPE`      | same                              | required tool held |
| 6  | `CRAFT_SHOES`     | same                              | required tool held |

**Classification rule (tasks 1–3):** reach YES iff the map's biome
matches the task, otherwise reach NO. Reaching the *wrong* target still
costs/rewards as a normal reach — only the `correct_answer_bonus`
toggles on correctness.

**Craft rule (tasks 4–6):** agent must craft the specified tool
(`raft`/`rope`/`shoes`, costing 100 wood each). One-shot per episode:
the bonus fires on the step where `state.tool` transitions from `none`
to the target tool. `task_success` remains true for the rest of the
episode as long as the tool is held.

Defaults: `correct_answer_bonus = 100.0`, `craft_bonus = 100.0`. All
enums and tables live in `cogniland_jax/constants.py`:

```python
TASK_REACH = 0
TASK_CLS_ARCHIPELAGO, TASK_CLS_GRASSLAND, TASK_CLS_HIGHLAND = 1, 2, 3
TASK_CRAFT_RAFT, TASK_CRAFT_ROPE, TASK_CRAFT_SHOES = 4, 5, 6
TASK_BIOME_FOR_CLS  = jnp.array([-1, 1, 2, 3, -1, -1, -1])  # biome ids
TASK_TOOL_FOR_CRAFT = jnp.array([ 0, 0, 0, 0,  1,  2,  3])  # tool ids
```

## No auto-reset in `step`

`step_env` returns the raw terminal transition — `done=True` leaves the
state as-is. Wrap with an auto-reset helper (see `cogniland_jax.batched`
or the Craftax `AutoResetEnvWrapper`) if you need continuous rollouts.

## Difficulty bands

Difficulty caps the **maximum** Euclidean spawn-to-target distance. There
is no minimum — an agent may spawn right next to the target in any mode.

| Mode     | `difficulty` | Max spawn radius |
|----------|:------------:|-----------------:|
| easy     | `0`          | 20               |
| medium   | `1`          | 50               |
| hard     | `2`          | ∞ (map extent)   |

## Map validity constraints

Spawn and target positions are sampled with:

1. Centre cell on non-water land (`terrain_idx > 2`).
2. **7×7 box around the centre contains no water** — prevents 1-pixel
   islands where the agent is trapped.

Sampling is hierarchical on a **single sampled map**:
`target (×50) → spawn (×50)`. The map stays fixed for the episode; if
every target/spawn attempt fails we fall back to a grassland tile on
that same map — every real biome has at least one grassland cell, so
we never need to swap maps.

## Module layout

```
cogniland_jax/
├── constants.py      # tile/action enums, drain LUT, difficulty bands
├── state.py          # EnvParams + EnvState @flax.struct.dataclass
├── dynamics.py       # pure-JAX step kernels (movement / forage / craft / drain / ctg)
├── render.py         # minimap renderer, obs builder
├── env.py            # CognilandEnv(gymnax.Environment)
├── maps.py           # dataset loader (torch .pt → jnp)
├── batched.py        # numpy-API batched wrapper (vmap + auto-reset)
├── demo_pygame.py    # human-playable pygame demo (minimap main view + M toggle)
├── dataset.py        # build_dataset + load_map_arrays convenience re-exports
└── mapgen/           # self-contained map generation
    ├── simplexnoise/ # simplex noise core
    ├── terrain.py    # biome thresholds, heightmap pipeline, berry sampling
    ├── visibility.py # Bresenham raycast + bit-packed LUT
    └── build.py      # train/val/test split orchestrator
scripts/
└── generate_dataset.py   # CLI entrypoint
tests/
└── test_cogniland_env.py
```

## Smoke test

```bash
pytest cogniland-jax/tests            # or:
python cogniland-jax/tests/test_cogniland_env.py
```

