# crafter_in_cogniland — Architecture & Developer Guide

Pure-JAX DreamerV3 + PyTorch PPO on a small POMDP navigation env with a
one-shot **build commitment** (raft vs harness). The repo is structured
to make research-grade experiments easy: shared W&B metrics across
algos, frozen-model loaders for mech interp, paper-aligned model size
presets, single-file trainers.

## Layout

```
src/cogniland/
  crafter_in_cogniland/         pure-JAX env (Gymnax-style)
    constants.py                tile / action / object ids
    state.py                    EnvState + EnvParams pytrees
    dynamics.py                 step logic (jnp.where-driven)
    render.py                   tile-id minimap + scalars
    env.py                      CrafterInCognilandEnv class
    maps.py                     numpy mapgen helper (loads to JAX)
  nav/                          PyTorch env (for PPO + demo)
  assets/sprites/               Crafter PNG sprites used by the renderer

purejaxwm/                      DreamerV3 algorithm library (vendored)
  dreamerv3/                    RSSM (block GRU), TwoHotDist, LaProp,
                                RetNorm, slow critic, lambda returns
  commons/                      Gymnax wrappers + dtype helpers

scripts/
  dreamerv3_crafter_in_cogniland.py  JAX Dreamer trainer
  viz_dreamer_trajectory.py          Roll out + visualise a frozen ckpt
  plot_dreamer_on_demo_maps.py       Dreamer trajectories on the 12 demo maps
  train_ppo_gru.py                   PyTorch PPO trainer
  play_ppo_gru.py                    Visualise a trained PPO policy
  plot_ppo_on_demo_maps.py           PPO trajectories on the 12 demo maps
  play_cogniland.py                  Playable pygame demo

tests/
  test_nav_env.py + test_nav_mapgen.py  PyTorch env contract
  purejaxwm/                            Algorithm-library unit tests
```

## How to run

```bash
conda env create -f environment.yml && conda activate crusoe
pip install -e .

# Dreamer (25M default — paper Table 3)
python scripts/dreamerv3_crafter_in_cogniland.py \
  --size 25M --total-env-steps 1_000_000 --num-envs 32 \
  --train-ratio 64 --wandb-mode online

# Quick smoke (12M, no wandb)
python scripts/dreamerv3_crafter_in_cogniland.py \
  --size 12M --total-env-steps 50000 --num-envs 16 \
  --train-ratio 16 --map-size 32 --view-size 11 --wandb-mode disabled

# PPO baseline
python scripts/train_ppo_gru.py --total-timesteps 5_000_000 \
  --num-envs 32 --num-steps 128 --device cuda

# Inspect a frozen Dreamer checkpoint (random eval-set rollouts)
python scripts/viz_dreamer_trajectory.py \
  --checkpoint runs/<run_id>/checkpoints/step_1000000 \
  --maps-path data/crafter_in_cogniland/train_256.pkl

# Dreamer trajectories on the 12 demo maps (4 maps × 3 biomes grid)
python scripts/plot_dreamer_on_demo_maps.py \
  --checkpoint runs/<run_id>/checkpoints/step_1000000

# PPO equivalent for cross-algo comparison
python scripts/plot_ppo_on_demo_maps.py \
  --checkpoint runs/<run_name>/checkpoints/final.pt

# Play
python scripts/play_cogniland.py
```

## On-disk layout

Both trainers write under a single ``runs/`` parent (per
``OUTPUT_PROTOCOL.md``). Run-id slugs carry the algo prefix so the tree
self-disambiguates:

```
runs/
  dreamerv3_<env>_size<S>_seed<N>_<ts>/
    config.json
    checkpoints/step_<env_step>/    (orbax pytree, params only)
    viz/                            (viz_dreamer_trajectory.py output)
    viz_demo_maps/                  (plot_dreamer_on_demo_maps.py output)
  ppo_gru_size<S>_seed<N>_<ts>/
    checkpoints/{iter<N>.pt, final.pt}
```

``scripts/play_cogniland.py`` scans only ``runs/`` for ``.pt`` files in
the AI-weights picker; the dreamer demo-map plotter still has no hookup
in the pygame demo (PPO-only — see ``_play_ai`` in play_cogniland.py).

## Environment mechanics

- **Action space**: `Discrete(6)` — 0/1/2/3 = up/down/left/right,
  4 = build_raft, 5 = build_harness. Build is committed once.
- **Observation** (JAX env):
  `{minimap: (V,V) int8, scalars: (5,) float32}` egocentric crop;
  scalars = `[compass_r, compass_c, active_obj/2, build_active, step/max]`.
  OOB is padded with the `OOB=6` tile id.
- **Reward**:
  `-0.005` slack per step
  `+ 0.01 · (ctg_prev − ctg_curr)` PBRS shaping (cells, unit-cost)
  `+ 1.0` on stepping into the target tile.
- **Slip**: water/rock slip with prob 0.9 unless you carry the matching
  item; trees always slip 0.9; land slips 0.15 if carrying anything.

## Dreamer implementation notes

The trainer follows the DreamerV3 paper closely via the in-tree
`purejaxwm` library:

| Component | Implementation |
|---|---|
| Encoder/decoder | 4-block MLP + RMSNorm + SiLU on the flat `V·V + 5` vector obs (no CNN — minimap is symbolic). |
| RSSM | block GRU + discrete stochastic latents. unimix=1%. |
| Reward / value heads | TwoHotDist over 255 symlog-spaced bins. |
| Slow critic | EMA decay 0.98 + cross-entropy slow-reg loss. |
| Actor scale | RetNorm: `S = max(1, Per(R,95) − Per(R,5))` tracked by EMA. |
| KL loss | balanced dyn/rep with `free_nats=1`, `β_rep=0.1`. |
| Optim | `LaProp(ε=1e-20)` + `AGC(0.3)`. |
| Defaults | γ=0.997, λ=0.95, entropy=3e-4, lr=4e-5. |

Size presets (`--size`) map directly to paper Table 3:

| size  | d   | deter (8d) | cnn_d / codes |
|-------|-----|------------|---------------|
| 12M   | 256 | 1024 (×4)  | 16            |
| 25M   | 384 | 3072       | 24 (default)  |
| 50M   | 512 | 4096       | 32            |
| 100M  | 768 | 6144       | 48            |
| 200M  |1024 | 8192       | 64            |
| 400M  |1536 |12288       | 96            |

The size always lands in `wandb.tags` as `size=<preset>`.

## Mech-interp workflow

`viz_dreamer_trajectory.load_frozen(ckpt_dir, cfg)` returns:

```python
{
  "encoder": _Encoder(...),
  "decoder": _Decoder(...),
  "rssm":    purejaxwm.dreamerv3.world_model.RSSM(...),
  "actor":   MLPHead(...),
  "critic":  MLPHead(...),
  "wm_params": {...},
  "ac_params": {...},
}
```

The apply-fns take params explicitly, so you can patch the params tree
(zero out a head, ablate a layer, swap a slow critic) without touching
the trainer.

## Shared W&B schema

Both trainers log into the same project (`crafter_in_cogniland`) with
matching metric names. Add `algo=<...>` and `size=<...>` as tags so the
cross-algo workspace charts are filtered correctly.

| Key                  | Where computed                            |
|----------------------|-------------------------------------------|
| `success/mean`       | reach rate this log interval              |
| `success/rolling100` | rolling reach rate over 100 episodes      |
| `return/mean`        | mean episode return                       |
| `return/rolling100`  | rolling mean over 100 episodes            |
| `rollout/episode_length` | mean episode length                  |
| `loss/*`             | per-component loss values                 |
| `perf/fps`           | wallclock fps                             |

## Design invariants

- The env never imports torch (JAX env is pure JAX; PyTorch env is
  numpy + torch).
- Trainers never import each other or share state — single-file scripts.
- Map generation is numpy + deterministic by seed. The dataset is
  pickled once at startup if the file doesn't exist.
- Final checkpoints are orbax pytrees with the *params only* (not opt
  state) — enough to load and analyse, not enough to resume training.

## Tests

```bash
pytest tests/                  # env + mapgen contract tests
pytest tests/purejaxwm/        # algorithm-library unit tests
```
