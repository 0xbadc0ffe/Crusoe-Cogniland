# Crusoe-Cogniland

Batched reinforcement learning environment for island navigation. An agent navigates procedurally generated islands (250x250 tiles, 9 terrain types) from spawn to target while managing health, resources, and movement costs. Built with PyTorch for batched tensor operations, with PPO training and WandB logging.

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

This creates a `crusoe` environment with all dependencies (PyTorch, Gymnasium, Hydra, WandB, etc.) and installs the `cogniland` package in editable mode.

### 3. Verify the installation

```bash
pytest tests/ -v
```

### 4. Set up WandB (for experiment tracking)

```bash
wandb login
```

Paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize). Runs will log to the `cogniland` project (configured in `configs/logging/default.yaml`). To skip WandB, pass `logging.wandb.mode=disabled` to any training command.

## Usage

### Training

```bash
python scripts/train.py [overrides]
```

All config lives in `configs/` and any value can be overridden from CLI. Key parameters:

| Override | Values | Default |
|----------|--------|---------|
| `model` | `ppo`, `compass` | `ppo` |
| `env` | `default`, `easy`, `hard` | `default` |
| `device` | `auto`, `cuda`, `cpu` | `auto` |
| `training.total_timesteps` | int | `1000000` |
| `training.num_envs` | int | `32` |
| `training.learning_rate` | float | `0.0003` |
| `logging.wandb.mode` | `online`, `offline`, `disabled` | `online` |

See `configs/training/default.yaml` for all PPO hyperparams and `configs/env/default.yaml` for all reward coefficients.

### Evaluation

```bash
python scripts/evaluate.py model_path=checkpoints/ckpt_50.pt
```

### Interactive Demo

```bash
python scripts/demo.py [easy|hard]
```

**Controls:** Arrow keys / WASD to move, Space to stay, R to reset, ESC to quit.

## Project Structure

```
Crusoe-Cogniland/
├── configs/                    # Hydra config groups
│   ├── config.yaml             # Top-level defaults
│   ├── env/                    # Environment configs (default, easy, hard)
│   ├── model/                  # Model configs (ppo, compass)
│   ├── training/               # PPO hyperparameters
│   └── logging/                # WandB + trajectory settings
├── cogniland/                  # Main Python package
│   ├── env/                    # Environment engine
│   │   ├── constants.py        # TERRAIN_LEVELS, ACTIONS, etc.
│   │   ├── types.py            # EnvState, StepResult, EnvConfig (NamedTuples)
│   │   ├── core.py             # Pure-function step logic
│   │   ├── reward.py           # Reward function
│   │   ├── islands.py          # Islands class + terrain generation
│   │   └── wrappers.py         # BatchedIslandEnv + Gymnasium wrapper
│   ├── models/                 # Agent architectures
│   │   ├── compass_agent.py    # Compass baseline (greedy Manhattan)
│   │   └── ppo.py              # ActorCritic (CNN + MLP)
│   ├── training/               # Training infrastructure
│   │   ├── rollout.py          # RolloutBuffer, GAE
│   │   └── trainer.py          # PPO training loop
│   ├── logging/                # Logging
│   │   ├── wandb_logger.py     # WandB integration
│   │   └── metrics.py          # Behavioral metrics
│   └── utils/                  # Utilities
│       ├── checkpoint.py       # Save/load checkpoints
│       └── reproducibility.py  # Seed management
├── scripts/
│   ├── train.py                # Hydra training entry point
│   ├── evaluate.py             # Model evaluation
│   └── demo.py                 # PyGame demo launcher
├── game_demo.py                # Interactive PyGame demo
├── tests/                      # Test suite
├── lib/simplexnoise/           # Bundled noise library
├── environment.yml             # Conda environment (reproducible)
├── setup.py                    # Package definition
└── requirements.txt            # Pip-only fallback
```

## Terrain Types

| Level | Name       | Cost | Special Effects |
|-------|------------|------|-----------------|
| 0     | Ocean      | 0.5  | Requires resources after 7 moves |
| 1     | Deep Water | 0.75 | Requires resources after 7 moves |
| 2     | Water      | 1.0  | Requires resources after 7 moves |
| 3     | Beach      | 2.5  | - |
| 4     | Sandy      | 2.5  | - |
| 5     | Grassland  | 1.8  | - |
| 6     | Forest     | 3.0  | Gain 1 resource + 4 HP per turn |
| 7     | Rocky      | 4.0  | Consumes 0.25 resources per turn |
| 8     | Mountains  | 8.0  | Consumes 0.75 resources per turn |

## WandB Metrics

Training runs log the following to WandB:

- **Training:** policy_loss, value_loss, entropy, clipfrac, approx_kl, mean_reward, episode_length, SPS
- **Evaluation:** success_rate, mean_reward, mean_steps, final_hp, path_efficiency
- **Behavioral:** terrain distribution (water/forest/mountain %), resource management, HP score, directness ratio

## Reproducibility

- All seeds are controlled via `configs/env/default.yaml` → `seed: 42`
- `set_reproducibility()` pins PyTorch, NumPy, and Python RNG seeds + CuDNN deterministic mode
- Checkpoints save full RNG state for exact resume
- `tests/test_roundtrip.py` verifies deterministic trajectories (reference data for JAX migration)
- The conda `environment.yml` pins all dependency channels for environment reproducibility

## Architecture Notes

- **JAX-ready**: Environment state is a NamedTuple (JAX pytree), step logic is pure functions
- **Batched**: All operations are vectorized over batch dimension
- **GPU-friendly**: All tensors on device, only island generation runs on CPU
