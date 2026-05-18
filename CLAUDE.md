# Cogniland Nav — Architecture & Developer Guide

## Project overview

A small, fast 2-D POMDP navigation env with a one-shot build commitment
(raft vs harness vs none) and PyTorch baselines for **PPO+GRU** and
**DreamerV3**. The codebase is intentionally minimal — one file per
trainer, no hidden abstractions — so it's quick to iterate.

The agent starts on a procedurally generated map (32/64/96/128) and must
reach a target tile. Water and rock are universally walkable but
high-slip; the appropriate item (raft/harness) makes them no-slip. The
build is committed once and the agent only sees a binary `skill_active`
flag — *which* item it built is its own secret to remember through the
GRU/RSSM.

## Layout

```
src/cogniland/
  nav/                         the environment
    nav_env.py                 CognilandNavEnv (gymnasium.Env)
    mapgen.py                  procedural maps (OpenSimplex + Dijkstra)
    renderer.py                pygame / numpy sprite renderer
    skills.py                  reward + walkability + slip mechanics
    tiles.py                   tile id constants (GRASS/WATER/ROCK/...)
    wrappers.py                optional TorchTensorWrapper
  nav_dreamer_video.py         imagination-video helper used by train_dreamer
  assets/sprites/              Crafter sprites used by the renderer

scripts/
  train_dreamer.py             DreamerV3 trainer (paper hyperparameters)
  train_ppo_gru.py             PPO + GRU trainer (custom hybrid policy)
  play_cogniland.py            Playable pygame demo for humans
  play_ppo_gru.py              Evaluate / visualize a trained PPO policy
  profile_dreamer.py           Per-component wallclock profiler
```

## How to run

```bash
# Setup
conda env create -f environment.yml
conda activate crusoe
pip install -e .

# Smoke-test PPO
python scripts/train_ppo_gru.py --total-timesteps 100000 \
  --num-envs 4 --wandb-mode disabled

# Real PPO training
python scripts/train_ppo_gru.py --total-timesteps 5_000_000 \
  --num-envs 32 --num-steps 128 --device cuda

# Dreamer (25M default with --model-size medium)
python scripts/train_dreamer.py --total-env-steps 1_000_000 \
  --view-size 21 --tile-px 8 --batch-size 16 --batch-length 64 \
  --train-ratio 4 --compile --device cuda

# Profile Dreamer (find bottlenecks)
python scripts/profile_dreamer.py --model-size medium \
  --batch-size 16 --batch-length 64 --device cuda

# Play
python scripts/play_cogniland.py
```

## Environment mechanics

- **Action space**: `Dict(move=Discrete(5), build_scalar=Box(-1,1))`.
  - `move ∈ {0..3}` = up / down / left / right
  - `move = 4` = build; `build_scalar ≥ 0` → raft, `< 0` → harness.
  - Build is committed once. Subsequent build actions are no-ops.
- **Observation**: `Dict(image=Box uint8 [3, V*tile_px, V*tile_px],
  skill_active=Box[0,1])`. The agent sees an egocentric RGB crop and a
  single bit telling it whether *anything* has been built.
- **Reward** (see `skills.py`):
  - `SLACK_PENALTY = -0.005` paid every step.
  - `SHAPING_COEF * (ctg_old - ctg_new)` PBRS shaping, where ctg is the
    unit-cost Dijkstra distance to target under the agent's current item.
  - `REACH_BONUS = +1.0` sparse, paid on stepping onto the target.
- **Slip**: water and rock slip with prob 0.9 unless you carry the matching
  item; trees always slip 0.9; land slips 0.15 if you carry anything (the
  "weight tax" that makes the wrong skill strictly worse than carrying
  nothing).

## DreamerV3 (scripts/train_dreamer.py)

Single-file PyTorch port that follows the paper closely. Key choices:

| Component | Implementation |
|---|---|
| Encoder/decoder | 4-layer stride-2 CNN + 1×1 bottleneck. RMSNorm + SiLU. |
| RSSM | discrete stochastic latents (classes × codes) + GRU recurrent. unimix=1%. |
| Reward / value | **TwoHotDist** (255 bins, symlog-spaced). Bounded; this is what fixed the previous infinity blowup. |
| Slow critic | EMA decay 0.98 + cross-entropy slow-reg loss. |
| Actor scale | **RetNorm**: S = max(1, Per(R,95)−Per(R,5)) tracked by EMA(0.99). |
| Action space | discrete 6 (4 moves + 2 build variants). |
| KL loss | balanced dyn/rep with free_nats=1, β_rep=0.1. |
| Optim | LaProp(eps=1e-20) + AGC(0.3). |
| Defaults | γ=0.997, λ=0.95, entropy=3e-4, lr=4e-5. |

Model size presets (selected with `--model-size`):

| Preset | d | deter | cnn_d | codes | ~params (wm) |
|---|---|---|---|---|---|
| small  | 256 | 1024 | 16 | 16 | 7M |
| medium | 384 | 2048 | 24 | 24 | 25M (default) |
| large  | 512 | 3072 | 32 | 32 | 55M |
| xlarge | 768 | 4096 | 48 | 48 | 110M |

`--compile` turns on torch.compile of the RSSM step. On an RTX 4090 with
the medium model and a 21-tile view, this drops total step time from
≈400 ms to ≈155 ms (≈2.5x). See `profile_dreamer.py` for the breakdown.

## PPO + GRU (scripts/train_ppo_gru.py)

Custom PPO with a CNN trunk → GRU → hybrid policy head:

- Categorical over 5 moves
- Deterministic tanh-bounded **belief scalar** (the build signal),
  supervised by an MSE aux loss against the privileged `map_type` label
  (raft on lake, harness on rocky, 0 on balanced). The belief head is
  what the env reads for `build_scalar` — no information leak, the
  policy still has to infer the belief from the local observation.

Sufficient under the default hyperparameters (`ent_coef=3e-2`,
`shaping_coef=0.3`) to learn a >20% reach rate on the balanced biome
within 3M frames.

## Design invariants

- The agent never imports wandb.
- The env never imports torch.
- Train scripts are self-contained — no `from cogniland.trainer` imports;
  if you want shared infra you copy-paste rather than extract.
- Map generation is deterministic given a seed and `map_type`. Per-skill
  cost-to-go grids are cached on the `MapRecord`.

## Tests

```bash
pytest tests/
```

Two test files cover the nav env: `test_nav_env.py` checks the observation
and reward contracts; `test_nav_mapgen.py` checks map generation
invariants. `conftest.py` caches mapgen calls across the test session.
