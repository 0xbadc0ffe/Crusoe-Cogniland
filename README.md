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

### Interactive Demo

```bash
python demo.py [hard]
```

**Controls:** Arrow keys / WASD to move, Space to stay, R to reset, ESC to quit.

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
│   │   └── wrappers.py         # BatchedIslandEnv + Gymnasium wrapper
│   ├── models/                 # Self-contained agents
│   │   ├── __init__.py         # build_model() factory
│   │   ├── ppo.py              # PPO: architecture + rollout + GAE + training loop + eval
│   │   └── compass.py          # Compass: greedy baseline + eval
│   ├── simplexnoise/           # Bundled noise library (for island generation)
│   ├── logging.py              # WandB logger + behavioral metrics
│   └── utils.py                # Checkpoints + reproducibility
├── train.py                    # Hydra entry point → build_model(cfg).train(cfg)
├── demo.py                     # Interactive PyGame demo
├── assets/images/              # Reference island screenshots
├── setup.py                    # Package definition (enables pip install -e .)
├── environment.yml             # Conda environment (reproducible)
└── requirements.txt            # Pip-only fallback
```

### Adding a new model

1. Create `cogniland/models/your_model.py` with a class that has `.train(cfg)`
2. Add a branch in `cogniland/models/__init__.py`
3. Add `configs/models/your_model.yaml`

## Terrain Types

| Level | Name       | Cost | Special Effects |
|-------|------------|------|-----------------|
| 0     | Ocean      | 0.5  | Drains 3.0 resources/turn; 30 HP damage; requires resources after 7 moves |
| 1     | Deep Water | 0.75 | Drains 2.0 resources/turn; 25 HP damage; requires resources after 7 moves |
| 2     | Water      | 1.0  | Drains 1.0 resource/turn; 15 HP damage; requires resources after 7 moves |
| 3     | Beach      | 2.5  | - |
| 4     | Sandy      | 2.5  | - |
| 5     | Grassland  | 1.8  | - |
| 6     | Forest     | 3.0  | Gain 3 resources + 2 HP per turn |
| 7     | Rocky      | 4.0  | Consumes 1.0 resource/turn |
| 8     | Mountains  | 8.0  | Consumes 2.5 resources/turn |

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
