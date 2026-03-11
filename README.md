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
python train.py [overrides]
```

All config lives in `configs/config.yaml` and any value can be overridden from CLI:

| Override | Values | Default |
|----------|--------|---------|
| `models` | `ppo`, `compass` | `ppo` |
| `env` | `default`, `hard`, `map_strait`, `map_forest_belt`, `map_twin_peaks`, `map_river_delta`, `map_archipelago` | `default` |
| `device` | `auto`, `cuda`, `cpu` | `auto` |
| `models.training.total_timesteps` | int | `10000000` |
| `models.training.num_envs` | int | `32` |
| `models.training.learning_rate` | float | `0.0003` |
| `logging.wandb.mode` | `online`, `offline`, `disabled` | `online` |

See `configs/config.yaml` for all PPO hyperparams and `configs/env/default.yaml` for all reward coefficients.

### Evaluating a Model from WandB

You can cleanly evaluate any historically trained model using its WandB Run ID (the 8-character string at the end of the run URL).

```bash
python eval.py YOUR_RUN_ID
```

This script is entirely decoupled from the local `configs/` folder. It automatically downloads the exact frozen configuration and weights that the model was originally trained with, guaranteeing no configuration mismatch errors. It works modularly for any model assuming they use `build_model(cfg)` and `model_state_dict`.

### Interactive Demo (Human)

```bash
python human_demo.py [hard]
```

**Controls:** Arrow keys / WASD to move, Space to stay, R to reset, ESC to quit.

### Agent Demo (AI Playback)

Load a trained checkpoint, visually pick spawn and target on the map, and watch the agent navigate step-by-step with trajectory trail, minimap, and live stats. Supports speed control and pause.

```bash
python agent_demo.py
```

**Flow:** Select a checkpoint from `artifacts/` → click on the map to place spawn (red) then target (green) → press Enter to start.

**Controls during playback:** +/− to change speed, P to pause, R to reset positions, ESC to quit.

## Project Structure

```
Crusoe-Cogniland/
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Top-level: training + logging params inlined
│   ├── env/                    # Environment configs (default, hard)
│   └── models/                 # Model configs (ppo, compass)
├── cogniland/                  # Main Python package
│   ├── env/                    # Environment engine
│   │   ├── constants.py        # TERRAIN_LEVELS, ACTIONS, palette, etc.
│   │   ├── types.py            # EnvState, StepResult, EnvConfig
│   │   ├── core.py             # Pure-function step logic
│   │   ├── reward.py           # Reward function
│   │   ├── islands.py          # Islands class + terrain generation
│   │   └── wrappers.py         # BatchedIslandEnv (auto-reset, obs dict)
│   ├── models/                 # Self-contained agents
│   │   ├── __init__.py         # build_model() factory
│   │   ├── ppo.py              # PPO: architecture + rollout + GAE + training loop + eval
│   │   └── compass.py          # Compass: greedy baseline + eval
│   ├── simplexnoise/           # Bundled noise library (for island generation)
│   ├── logging.py              # WandB logger + behavioral metrics
│   └── utils.py                # Checkpoints + reproducibility
├── train.py                    # Hydra entry point → build_model(cfg).train(cfg)
├── human_demo.py               # Interactive PyGame demo (manual play)
├── agent_demo.py               # AI playback demo (watch trained agent)
├── assets/
│   ├── images/                 # Reference island screenshots
│   └── maps/                   # Auto-generated PNG previews of custom maps
├── utils/
│   └── generate_map_assets.py  # Regenerate assets/maps/ from custom_maps.py
├── setup.py                    # Package definition (enables pip install -e .)
├── environment.yml             # Conda environment (reproducible)
└── requirements.txt            # Pip-only fallback
```

### Show custom maps

After adding or modifying a map in `cogniland/env/custom_maps.py`, regenerate the PNG previews:

```bash
python utils/generate_map_assets.py
```

This writes one PNG per map to `assets/maps/`, using the same terrain palette as the game.

### Adding a new model

1. Create `cogniland/models/your_model.py` with a class that has `.train(cfg)`
2. Add a branch in `cogniland/models/__init__.py`
3. Add `configs/models/your_model.yaml`

## Terrain Types

| Level | Name       | Move Cost | Visibility | Resource drain/step | HP/step (no resources) | Notes |
|-------|------------|-----------|------------|---------------------|------------------------|-------|
| 0     | Ocean      | 0.5       | 6          | −3.0                | −6.0                   | Fast travel, very expensive |
| 1     | Deep Water | 0.75      | 5          | −2.0                | −4.0                   | |
| 2     | Water      | 1.0       | 4          | −1.5                | −3.0                   | Highway if resourced |
| 3     | Beach      | 1.5       | 3          | −1.0                | −2.0                   | Coastal, but not free |
| 4     | Sandy      | 2.0       | 3          | −1.0                | −2.0                   | Desert |
| 5     | Grassland  | 1.5       | 3          | −1.0                | −2.0                   | Open land |
| 6     | Forest     | 3.5       | 2          | +5 HP/step **or** +2 res/step | —          | HP-first: heals until max HP, then gathers resources |
| 7     | Rocky      | 3.5       | 5          | −1.5                | −3.0                   | High-ground view advantage |
| 8     | Mountains  | 6.0       | 7          | −3.0                | −6.0                   | Full 15×15 strategic view |

Resource drain applies **every step** with no free window. HP/step when no resources = `drain × 2`.
**Visibility** is the radius in tiles; the minimap window is `2×visibility+1` across (max 15×15 at mountains).

### Land → Water transition

Entering any water tile (levels 0–2) from land (levels 3–8) costs **10 resources** as a boat-construction fee.
Each resource point short of 10 deals **5 HP** damage instead.
This makes ocean crossings a deliberate, resource-gated decision.

### Forest priority mechanic

- HP below max (100): forest heals **+10 HP/step**, no resource gain.
- HP at max (100): forest gathers **+2 resources/step**, no further healing.

Forest is the only HP source in the game (no passive regeneration).

## WandB Metrics

Training runs log the following to WandB:

- **Training:** policy_loss, value_loss, entropy, clipfrac, approx_kl, mean_reward, episode_length, SPS
- **Evaluation:** success_rate, mean_reward, mean_steps, final_hp, path_efficiency
- **Behavioral:** terrain distribution (water/forest/mountain %), resource management, HP score, directness ratio

See `docs/METRICS.md` for detailed metric definitions.

## Reproducibility

- All seeds are controlled via `configs/env/default.yaml` → `seed: 42`
- `set_reproducibility()` pins PyTorch, NumPy, and Python RNG seeds + CuDNN deterministic mode
- Checkpoints save full RNG state for exact resume

## Architecture Notes

- **JAX-ready**: Environment state is a NamedTuple (JAX pytree), step logic is pure functions
- **Batched**: All operations are vectorized over batch dimension
- **GPU-friendly**: All tensors on device, only island generation runs on CPU
- **Self-contained models**: Each model defines its own architecture + training loop — swap by changing `models=ppo` to `models=compass`

See `docs/ARCHITECTURE.md` for detailed architecture documentation.
