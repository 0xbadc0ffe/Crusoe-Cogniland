# Cogniland Nav

A small, fast 2-D navigation environment with a one-shot build commitment
(raft vs harness) — designed to study partially observable RL agents on a
procedurally generated grid world. PyTorch-only, baselines include
**PPO + GRU** and **DreamerV3**.

The agent always sees an *egocentric* RGB crop. It must reach a target
while choosing which build item to commit to — raft makes water easy,
harness makes rock easy, and neither helps on balanced maps. The build
decision is permanent.

## Layout

```
src/cogniland/
  nav/                         The environment
    nav_env.py                 CognilandNavEnv (gymnasium)
    mapgen.py                  procedural map generation
    renderer.py                pygame / numpy sprite renderer
    skills.py                  reward shaping + walkability + slip
    tiles.py                   tile id constants
    wrappers.py                TorchTensorWrapper (optional)
  nav_dreamer_video.py         imagination-video helper for Dreamer
  assets/sprites/              Crafter sprites used by the renderer

scripts/
  train_dreamer.py             DreamerV3 trainer (paper hyperparameters)
  train_ppo_gru.py             PPO + GRU trainer
  play_cogniland.py            Playable pygame demo
  play_ppo_gru.py              Evaluate / visualize a trained PPO policy
  profile_dreamer.py           Per-component wallclock profiler
```

## Setup

```bash
conda env create -f environment.yml
conda activate crusoe
pip install -e .
```

## Quick start

Play the env as a human:
```bash
python scripts/play_cogniland.py
```

Train PPO + GRU:
```bash
python scripts/train_ppo_gru.py \
  --total-timesteps 5_000_000 \
  --num-envs 32 --num-steps 128 \
  --env-size 64 --view-size 21 --tile-px 8 \
  --device cuda --wandb-project cogniland-nav
```

Train DreamerV3 (25M params default, ~paper hyperparameters):
```bash
python scripts/train_dreamer.py \
  --model-size medium \
  --view-size 21 --tile-px 8 \
  --total-env-steps 1_000_000 \
  --num-envs 4 --batch-size 16 --batch-length 64 \
  --train-ratio 4 --compile \
  --device cuda --wandb-project cogniland-nav
```

`--model-size small | medium | large | xlarge` selects 7M / 25M / 55M / 110M
world-model presets. `--compile` turns on torch.compile of the RSSM step
(~2x speedup, +20s compile time).

## Environment summary

- **Map**: 32 / 64 / 96 / 128 grid sizes, biomes `random | lake | rocky | balanced`.
- **Observation**: egocentric RGB crop (e.g. 21×21 tiles at tile_px=8 → 168×168).
- **Actions**: `Dict(move=Discrete(5), build_scalar=Box(-1,1))`. Move is
  up/down/left/right or build; build commits one item permanently —
  `build_scalar ≥ 0` makes a raft, `< 0` a harness.
- **Reward**: flat slack penalty + PBRS shaping over cost-to-go +
  reach-target sparse bonus. See `src/cogniland/nav/skills.py`.

## DreamerV3 implementation notes

`train_dreamer.py` is a self-contained PyTorch port that follows the
paper recipe closely (Hafner et al. 2023):

* **TwoHotDist** heads for reward and value (bounded, symlog-spaced bins
  — prevents the value-target spiral the old code suffered from).
* **Slow critic** with EMA + cross-entropy regularizer.
* **Percentile-EMA RetNorm** for actor advantage scaling.
* **RMSNorm + SiLU** activations, **AGC(0.3)** gradient clipping,
  **LaProp(eps=1e-20)** optimizer.
* **Discrete action space** internally (4 moves + build-raft + build-harness)
  to eliminate the unbounded `Normal.log_prob` path.
* **KL-balanced** dyn/rep losses with `free_nats=1`, `beta_rep=0.1`.
* Paper defaults: `gamma=0.997`, `lambda=0.95`, `entropy=3e-4`, `unimix=1%`.

Run `scripts/profile_dreamer.py` to see the wallclock breakdown — on an
RTX 4090 with the medium model the env stepping is ~7% of an outer tick
and the dominant costs are the WM backward (≈37%), the 64-step Python
RSSM rollout (≈14%), and the decoder forward (≈12%).
