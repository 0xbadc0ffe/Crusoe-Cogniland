# Cogniland

A multi-task reinforcement learning benchmark on procedurally generated maps.

An agent spawns on a 128x128 island and must reach a target while managing HP. Terrain drains health at different rates, forests provide wood, berries heal, and craftable tools (raft, rope, shoes) unlock efficient traversal of specific terrain families. The game is designed so that naive shortest-path navigation usually fails — survival requires strategic foraging, tool crafting, and terrain awareness.

The framework supports multiple RL agents (PPO-RNN, DreamerV3, STORM) through a shared `Agent` interface. All training, evaluation, and logging infrastructure is agent-agnostic.

## Setup

```bash
conda env create -f environment.yml
conda activate crusoe
pip install -e .

# Generate map datasets (one-time, ~30s)
python scripts/generate_dataset.py
```

## Quick start

```bash
# Train PPO-RNN (default: 5M frames, 32 parallel envs, W&B logging)
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml

# Smoke test (offline, fast)
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  --offline trainer.num_train_frames=20000

# Switch agent
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/dreamerv3.yaml
```

Any config value can be overridden from the command line using dotlist notation:

```bash
python scripts/train.py \
  --env-config configs/env/cogniland.yaml \
  --agent-config configs/agent/ppo_rnn.yaml \
  agent.lr=1e-4 agent.entropy_coef=0.05 seed=123
```

## Running experiments

### K-seed benchmark (SLURM cluster)

```bash
# Create sweep, submit 10 SLURM jobs (one per seed)
./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_seeds.yaml -n 10 -r 1
```

### K-seed benchmark (local, multi-GPU)

```bash
wandb sweep configs/sweeps/ppo_rnn_seeds.yaml
python scripts/run_sweep.py <SWEEP_ID> --num-agents 5 --count 1 --gpus 0 1
```

### Hyperparameter search

```bash
./scripts/launch_sweep.sh configs/sweeps/ppo_rnn_hpsearch.yaml -n 20 -r 5
```

### Play as human

```bash
python demo.py
```

## How it works

The framework has four layers. Each layer only talks to its immediate neighbors:

```
configs/           What to run (env params, agent hyperparams, sweep grids)
    |
Trainer            How to run (training loop, eval schedule, W&B logging)
    |
Agent              What to learn (network, optimizer, rollout collection, loss)
    |
Environment        What the world does (maps, terrain, HP drain, actions, obs)
```

**Environment** (`src/cogniland/envs/`) loads pre-generated 128x128 RGB maps and runs a batched game loop in numpy. Each step, the agent picks one of 8 actions (move, forage, craft). The env computes HP drain, foraging effects, and returns an RGB minimap observation with raycasted occlusion.

**Agent** (`src/cogniland/agents/`) is a pure-function dataclass with `init`, `train`, and `evaluate` methods. The network runs in JAX; the env boundary converts numpy<->jax. An agent never imports wandb or knows about the training schedule.

**Trainer** (`src/cogniland/trainer/`) orchestrates the loop: call `agent.train()` for N frames, periodically call `agent.evaluate()` on each task, log everything to W&B. It assigns tasks via `TaskSampler` and passes `task_ids` to the agent. The trainer never imports anything agent-specific.

**Config** (`configs/`) uses OmegaConf with two YAML files merged at startup: `env` (experiment setup) + `agent` (hyperparameters). Agent config wins on conflicts. CLI dotlist and W&B sweep overrides apply on top.

### Multi-task design

The environment supports multiple tasks sharing the same world. Tasks differ only in reward function. Currently task 0 (reach target) is implemented; the framework is wired for 7 tasks.

- The **Trainer** samples task assignments and passes `task_ids` to the agent
- The **Agent** receives a task embedding (one-hot vector) concatenated to its features
- **Evaluation** runs each task separately and logs per-task + aggregate metrics

### Available agents

| Agent | Config | Description |
|-------|--------|-------------|
| `ppo_rnn` | `configs/agent/ppo_rnn.yaml` | PPO with LSTM, CNN minimap encoder (JAX/Flax) |
| `dreamerv3` | `configs/agent/dreamerv3.yaml` | DreamerV3 world model with RSSM (JAX) |
| `storm` | `configs/agent/storm.yaml` | STORM world model with Transformer SSM (JAX) |

## Project structure

```
scripts/train.py               Entry point
configs/env/cogniland.yaml     Environment + experiment config
configs/agent/*.yaml           Agent hyperparameters
configs/sweeps/*.yaml          W&B sweep definitions

src/cogniland/
  envs/                        Batched numpy environment
    env.py              Game loop (8 actions, HP/wood/tools)
    tile_effects.py              Terrain drain table
    tasks.py                     Per-task reward functions
    multitask_wrapper.py         Reward routing + task embeddings
  agents/                      JAX agent implementations
    ppo_rnn.py                   PPO-RNN (Flax CNN+LSTM)
    dreamer.py                   DreamerV3
    storm.py                     STORM
    agent.py                     Agent dataclass (the interface)
    registry.py                  @register_agent + auto-discovery
    commons/                     Shared NN blocks, replay buffers
    policy/                      Actor-critic training
    world_models/                RSSM, TSSM implementations
  trainer/                     Training orchestration
    trainer.py                   Main loop + multi-task eval
    run_logger.py                W&B integration
    checkpoint.py                Orbax save/load
  config/                      Config loading, XLA setup
  metrics/                     Rolling stats tracker

data/maps/                 Pre-generated map datasets (.pt)
demo.py                        Playable human demo (pygame)
```

## License

MIT
