# Crusoe-Cogniland

Batched reinforcement learning environment for island navigation. An agent navigates procedurally generated islands (250×250 tiles, 9 terrain types) from spawn to target while managing health, resources, and movement costs. Built with PyTorch for batched tensor operations, with PPO training and WandB logging.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Crusoe-Cogniland.git
cd Crusoe-Cogniland
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate crusoe
```

### 3. Install the package

```bash
pip install -e .
```

This registers `cogniland` as an importable package. Required because Hydra changes the working directory at runtime.

### 4. Set up WandB (for experiment tracking)

```bash
wandb login
```

Paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize). To skip WandB, pass `logging.wandb.mode=disabled` to any training command.

## Usage

### Training

```bash
# Default PPO run
python train.py

# Smoke test — fast, no WandB
python train.py models.training.total_env_moves=20000 logging.wandb.mode=disabled

# Hard-mode environment
python train.py env=hard

# Override hyperparameters (Hydra syntax)
python train.py models.training.learning_rate=3e-4 models.training.parallel_envs=64
```

All config lives in `configs/`. Any value can be overridden from the CLI:

| Override | Values | Default |
|----------|--------|---------|
| `env` | `default`, `hard`, `map_strait`, `map_twin_peaks`, `map_river_delta`, `map_archipelago` | `default` |
| `models` | `ppo`, `ppo_mini` | `ppo` |
| `device` | `auto`, `cuda`, `cpu` | `auto` |
| `models.training.total_env_moves` | int | `2_500_000` |
| `models.training.parallel_envs` | int | `32` |
| `models.training.learning_rate` | float | `0.0003` |
| `logging.wandb.mode` | `online`, `offline`, `disabled` | `online` |

See `configs/models/ppo.yaml` for all PPO hyperparameters and `configs/env/default.yaml` for all reward coefficients and terrain effects.

### Evaluating a Model from WandB

```bash
python eval.py YOUR_RUN_ID
```

Fetches the exact frozen config and weights from WandB, runs `EvalRunner` with the deterministic policy, and prints all behavioral metrics. Works offline if `artifacts/{run_id}/ckpt_final.pt` exists locally.

### Interactive Demo (Human)

```bash
python human_demo.py [hard]
```

**Controls:** Arrow keys / WASD to move, Space to stay, R to reset, ESC to quit.

### Agent Demo (AI Playback)

```bash
python agent_demo.py
```

Load a trained checkpoint, click spawn and target on the map, watch the agent navigate with trajectory trail, minimap, and live stats.

**Flow:** Select checkpoint from `artifacts/` → click spawn (red) then target (green) → Enter to start.
**Controls:** +/− speed, P pause, R reset, ESC quit.

## Project Structure

```
Crusoe-Cogniland/
├── configs/
│   ├── config.yaml             # Top-level: device, logging, eval settings
│   ├── env/                    # default.yaml, hard.yaml, map_*.yaml
│   └── models/                 # ppo.yaml, ppo_mini.yaml
├── cogniland/
│   ├── env/
│   │   ├── constants.py        # TERRAIN_COSTS, TERRAIN_VISIBILITY, ACTION_DELTAS
│   │   ├── types.py            # EnvState (NamedTuple), EnvConfig (frozen dataclass)
│   │   ├── core.py             # Pure-function step logic (movement, terrain, minimap)
│   │   ├── reward.py           # compute_reward() — pure function
│   │   ├── islands.py          # Procedural map generation + batched reset
│   │   ├── pathfinding.py      # batch_astar() — vectorised A*
│   │   └── wrappers.py         # BatchedIslandEnv (auto-reset, obs dict, episode stats)
│   ├── models/
│   │   ├── __init__.py         # build_model(cfg) factory
│   │   └── ppo.py              # PPO: ActorCritic, RolloutBuffer, GAE, training loop
│   ├── eval/
│   │   ├── __init__.py         # Public API: EvalRunner, EpisodeResult, EvalResult, CognilandSummarizer
│   │   ├── runner.py           # EvalRunner — runs episodes, computes behavioral metrics
│   │   └── summarizer.py       # CognilandSummarizer — aggregates EvalResult → dict[str, float]
│   ├── simplexnoise/           # Bundled noise library for island generation
│   ├── logging.py              # WandBLogger + log_rollout_stats()
│   └── utils.py                # Checkpoints, render_trajectory, set_reproducibility
├── train.py                    # Hydra entry point
├── eval.py                     # Standalone evaluation by WandB run ID
├── human_demo.py               # Interactive PyGame demo
├── agent_demo.py               # AI playback demo
├── CLAUDE.md                   # Full architecture & metrics reference
├── environment.yml             # Conda environment
└── setup.py
```

## Terrain Types

| Level | Name | Move cost | Visibility (tiles) | Resource drain/step | HP drain if no resources |
|-------|------|-----------|--------------------|---------------------|--------------------------|
| 0 | Ocean | 0.5 | 18 | 3.0 | 6.0 |
| 1 | Deep Water | 0.75 | 15 | 2.0 | 4.0 |
| 2 | Water | 1.0 | 12 | 1.5 | 3.0 |
| 3 | Beach | 1.5 | 9 | 1.0 | 2.0 |
| 4 | Sandy | 2.0 | 9 | 1.0 | 2.0 |
| 5 | Grassland | 1.5 | 9 | 1.0 | 2.0 |
| 6 | Forest | 3.5 | 6 | 0 | — |
| 7 | Rocky | 3.5 | 15 | 1.5 | 3.0 |
| 8 | Mountains | 6.0 | 21 | 3.0 | 6.0 |

Resource drain applies every step. When resources hit 0, remaining drain becomes HP damage via `no_res_hp_multiplier` (default 2.0). HP drain when no resources = `drain × 2.0`.

### Land → Water transition

Entering any water tile (levels 0–2) from land (levels 3–8) costs **10 resources** (boat construction). Each resource point short deals **5 HP** damage instead.

### Forest mechanic (HP-first priority)

- HP below max: heals **+5 HP/step**, no resource gain.
- HP at max: gathers **+2 resources/step**, no further healing.

Forest is the main HP recovery zone (no passive regeneration in default mode).

## WandB Metrics

All metrics follow a `{split}/{mode}/env/{name}` namespace. Every behavioral metric is logged as `_mean` and `_std` across evaluation episodes.

```
train/env/episode_return_mean   — online rollout stats
train/env/success_rate
train/model/ppo/policy_loss
train/model/ppo/entropy
system/steps_per_second

val/det/env/success_rate        — deterministic policy, held-out maps
val/det/env/path_efficiency_mean
val/det/env/directness_mean
val/det/env/survival_margin_min_mean
val/det/env/exploration_mean
val/det/charts/terrain_distribution
val/det/tables/episodes

val/stoch/...                   — same keys, sampled policy
test/det/...                    — final eval at end of training
```

See `CLAUDE.md` for the full metric schema and behavioral metric formulas.

## Reproducibility

- All seeds controlled via `configs/env/default.yaml` → `seed: 42`
- `set_reproducibility()` pins PyTorch, NumPy, and Python RNG + CuDNN deterministic mode
- Eval maps are held-out at `seed + eval_seed_offset` (default +1000) — never seen during training

## Architecture Notes

- **JAX-ready**: `EnvState` is a NamedTuple (pytree), all step logic in pure functions
- **Batched**: all operations vectorised over the batch dimension, no Python loops in hot paths
- **GPU-friendly**: all tensors on device; map generation runs on CPU once at startup
- **Eval pipeline**: `EvalRunner → CognilandSummarizer → WandBLogger` — no WandB dependency in runner or summarizer; `eval.py` reuses runner without any wandb calls

See `CLAUDE.md` for full architecture documentation.
