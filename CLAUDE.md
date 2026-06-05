# Crusoe-Cogniland — Architecture & Developer Guide

Pure-JAX DreamerV3 + PyTorch PPO on small POMDP navigation envs, built as a
substrate for mechanistic interpretability (belief/skill probing + steering).
The **active** env is `bridge_tunnel`; `crafter_in_cogniland` + `nav` are a
legacy/secondary cluster. See `docs/codebase_map.md` for the full navigation
guide and `configs/bridge_tunnel/REGISTRY.md` for the released agents.

## Layout

```
src/cogniland/
  bridge_tunnel/                ← ACTIVE env: ONE package, two variants
    tiles.py ctg.py mapgen.py   variant="bt" (base) | "btc" (implicit commitment
    env.py policy.py _solver.py  + 3 map categories). Discrete(6); PPO+GRU & DreamerV3.
    jax/                        pure-JAX port (Gymnax-style); EnvParams.commit static flag
  crafter_in_cogniland/  nav/   legacy/secondary (crafter JAX env + nav PyTorch env+mapgen)
  assets/sprites/               Crafter PNG sprites
purejaxwm/                      vendored DreamerV3 (RSSM, TwoHot, LaProp, RetNorm, …)

scripts/bridge_tunnel/          train_ppo / dreamerv3 (both --variant), eval, play, viz, sweeps
scripts/mechinterp/             build_activation_dataset, decode_dataset, replay_trajectory
scripts/crafter/  scripts/figures/   legacy crafter+nav scripts ; figure drawers

configs/bridge_tunnel/          experiment configs + REGISTRY.md
released_models/                frozen agents (+ as-trained yaml; git-LFS orbax)
data/                           procedural maps (val + regenerable jax train sets)
activation_datasets/  outputs/  mech-interp bundles ; ALL generated artifacts  [gitignored]
tests/                          env contract + JAX↔PyTorch parity + purejaxwm/
```

The PyTorch `BridgeTunnelEnv` and pure-JAX `bridge_tunnel.jax` env are proven
**bit-for-bit equivalent** for both variants (`tests/test_bridge_tunnel*parity.py`).

## How to run

```bash
conda env create -f environment.yml && conda activate crusoe
pip install -e .

# PPO+GRU  (--variant bt | btc)
python scripts/bridge_tunnel/train_ppo_bridge_tunnel.py \
  --config configs/bridge_tunnel/btc_ppo_onehot.yaml --device cuda

# DreamerV3 (25M; --variant bt | btc)
python scripts/bridge_tunnel/dreamerv3_bridge_tunnel.py \
  --variant btc --size 25M --decoder categorical \
  --total-env-steps 1_500_000 --num-envs 32 --train-ratio 64 --wandb-mode online

# evaluate / play
python scripts/bridge_tunnel/eval_bridge_tunnel_commit_ppo.py --checkpoint released_models/bridge_tunnel_commit/ppo_commit_onehot.pt
python scripts/bridge_tunnel/play_bridge_tunnel.py

# legacy crafter cluster
python scripts/crafter/dreamerv3_crafter_in_cogniland.py --size 25M --wandb-mode online

# Inspect a frozen Dreamer checkpoint
python scripts/crafter/viz_dreamer_trajectory.py \
  --checkpoint runs/<run_id>/checkpoints/step_1000000 \
  --maps-path data/crafter_in_cogniland/train_256.pkl

# Play
python scripts/crafter/play_cogniland.py
```

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
- **Slip** (land weight tax, 2026-05-28): water/rock slip 0.75 unless you
  carry the matching item; trees always slip 0.75; when **any** skill is
  committed, grass/sand/dirt all slip 0.50 (the weight tax). Bare-handed:
  sand/dirt slip 0.30, grass slips `SLIP_PROB_GRASS_NOSKILL` (default 0,
  sweep knob); the target never slips.

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
  pickled once at startup if the file doesn't exist. Training uses the
  legacy `simplex` noise generator (the `CognilandNavEnv` default, and
  `train_ppo_gru.py --generator simplex`); the structured `composed` /
  `components` generators are held out as a test set (passed explicitly,
  e.g. in the trajectory-grid eval scripts).
- Final checkpoints are orbax pytrees with the *params only* (not opt
  state) — enough to load and analyse, not enough to resume training.

## Tests

```bash
pytest tests/                  # env + mapgen contract tests
pytest tests/purejaxwm/        # algorithm-library unit tests
```
