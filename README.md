# crafter_in_cogniland

A small, fast 2-D POMDP navigation environment with a one-shot
**build commitment** (raft vs harness), shipped with two solid baselines:

* **DreamerV3 (JAX)** — `scripts/dreamerv3_crafter_in_cogniland.py`,
  built on the in-tree `purejaxwm/` algorithm library.
* **PPO + GRU (PyTorch)** — `scripts/train_ppo_gru.py`,
  uses the gymnasium PyTorch env in `src/cogniland/nav/`.

Both trainers log the *same* metric names (`success/mean`,
`success/rolling100`, `return/mean`, `return/rolling100`, …) into a
single W&B project (`crafter_in_cogniland` by default) so PPO and
Dreamer runs sit on the same chart side by side.

## Layout

```
src/cogniland/
  crafter_in_cogniland/         JAX env (Gymnax-style) — for Dreamer
    constants.py                tile / action / object ids
    state.py                    EnvState + EnvParams pytrees
    dynamics.py                 step logic (pure JAX)
    render.py                   tile-id minimap + scalars
    env.py                      CrafterInCognilandEnv class
    maps.py                     numpy mapgen helper (uses cogniland.nav)
  nav/                          PyTorch env — for PPO + the demo
  assets/sprites/               Crafter sprites (used by nav renderer)

purejaxwm/                      DreamerV3 algorithm library (in-tree)
  dreamerv3/                    RSSM, TwoHotDist, LaProp, RetNorm, …
  commons/                      Gymnax wrappers, dtype helpers

scripts/
  dreamerv3_crafter_in_cogniland.py    Train Dreamer (JAX)
  viz_dreamer_trajectory.py            Roll out a frozen Dreamer checkpoint
                                       and plot trajectories
  train_ppo_gru.py                     Train PPO + GRU (PyTorch)
  play_ppo_gru.py                      Visualise a trained PPO policy
  play_cogniland.py                    Playable pygame demo

tests/
  test_nav_env.py + test_nav_mapgen.py  PyTorch nav env contract
  purejaxwm/                            Algorithm library tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate crusoe
pip install -e .
```

## Quick start

### DreamerV3

```bash
# Smoke test
python scripts/dreamerv3_crafter_in_cogniland.py \
  --size 12M --total-env-steps 50000 --num-envs 16 --train-ratio 16 \
  --map-size 32 --view-size 11 --wandb-mode disabled

# Real run (25M default, ~1M env-steps, ~30 min on RTX 4090)
python scripts/dreamerv3_crafter_in_cogniland.py \
  --size 25M --total-env-steps 1_000_000 \
  --num-envs 32 --train-ratio 64 --wandb-mode online
```

Size presets follow DreamerV3 paper Table 3:
`12M | 25M | 50M | 100M | 200M | 400M`. The `size=...` value is added
as a W&B tag automatically.

### PPO + GRU

```bash
python scripts/train_ppo_gru.py \
  --total-timesteps 5_000_000 --num-envs 32 --num-steps 128 \
  --env-size 64 --view-size 21 --tile-px 8 \
  --device cuda --wandb-project crafter_in_cogniland
```

PPO logs `size=X.YM` tag automatically based on its actual param count.

### Inspect a trained Dreamer

```bash
python scripts/viz_dreamer_trajectory.py \
  --checkpoint runs/<run_id>/checkpoints/step_1000000 \
  --maps-path data/crafter_in_cogniland/train_64x64_n256.pkl \
  --n-episodes 8
# → runs/<run_id>/viz/trajectories.png, trajectories.json
```

### W&B sweeps (SLURM)

Two sweep configs ship in `configs/sweeps/`:

| Sweep | Axes | # runs |
|---|---|---|
| `ppo_gru_map_sizes.yaml`  | `env-size ∈ {32, 96}` | 2 |
| `dreamer_size_x_map.yaml` | `size ∈ {12M,25M,50M,100M}` × `map-size ∈ {32,96}` | 8 |

Cluster setup (do once):

```bash
# 1. Conda env on the cluster
conda env create -f environment.yml -p $CONDA_ENV
conda activate $CONDA_ENV
pip install -e .

# 2. Put your WANDB_API_KEY in $PROJECT_DIR/.env (one line: WANDB_API_KEY=...)

# 3. Pre-generate map datasets so agents don't race on lazy generation
python scripts/generate_maps.py --sizes 32 96

# 4. EDIT cluster-specific paths in scripts/job_sweep.slurm:
#    PROJECT_DIR, CONDA_ENV, --mail-user, --exclude
```

Launching:

```bash
# Submit the sweeps (each call creates a wandb sweep + a SLURM array)
./scripts/launch_sweep.sh configs/sweeps/ppo_gru_map_sizes.yaml
./scripts/launch_sweep.sh configs/sweeps/dreamer_size_x_map.yaml

# Override SLURM resources / parallelism:
./scripts/launch_sweep.sh configs/sweeps/dreamer_size_x_map.yaml \
  -n 16 -r 2 -t 12:00:00 -m 48G
```

Each run carries `algo=<…>`, `size=<…>`, `map=<…>` tags so the W&B
workspace can filter and group the cross-condition charts.

`viz_dreamer_trajectory.load_frozen(...)` returns the encoder, RSSM,
decoder, actor, and critic *as apply-fns over a single params pytree*,
which is the intended entry point for mech-interp probes (residual
streams, activations, etc.).

## Environment summary

- **Map**: pre-generated procedurally (numpy + simplex noise + Dijkstra
  validation). 64×64 default. 3 biomes: `balanced` / `lake` / `rocky`.
- **Action space**: `Discrete(6)` — up / down / left / right /
  build_raft / build_harness. Build is committed once per episode.
- **Observation (JAX env)**: `{minimap: (V,V) int8, scalars: (5,) float32}`
  where the scalars are
  `[compass_r, compass_c, active_obj/2, build_active, step/max]`.
  Egocentric crop, OOB padded with `OOB=6`.
- **Reward** (mirrors `cogniland.nav.skills`):
  `-0.005` slack per step
  `+ 0.01 · (ctg_prev - ctg_curr)` PBRS shaping
  `+ 1.0` on stepping into the target tile.
- **Slip**: water/rock slip with prob 0.9 unless you carry the matching
  item; trees always slip 0.9; land slips 0.15 with any item carried.

## Shared W&B logging schema

Both trainers log into the same project. The cross-algo charts in the
default W&B workspace key off these metric names:

| Key                  | Meaning                                  |
|----------------------|------------------------------------------|
| `success/mean`       | Mean reach-rate (this log interval)      |
| `success/rolling100` | Rolling success rate over 100 episodes   |
| `return/mean`        | Mean episode return                      |
| `return/rolling100`  | Rolling mean over 100 episodes           |
| `rollout/episode_length` | Mean episode length                  |
| `perf/fps`           | Wallclock throughput                     |

Tags always include `algo=<dreamerv3|ppo_gru>` and `size=<X.YM>`.

## Goals

- **>80% success** on Dreamer (medium model, 1–3M env steps).
- **Frozen checkpoints** that load deterministically for analysis
  (`viz_dreamer_trajectory.load_frozen`).
- **Trajectory + imagination plots** out of the box.
- **One W&B workspace** comparing PPO and Dreamer on shared metrics.
