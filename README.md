# crafter_in_cogniland

A small, fast 2-D POMDP navigation environment with a one-shot
**build commitment** (raft vs harness), shipped with two solid baselines:

* **DreamerV3 (JAX)** — `scripts/crafter/dreamerv3_crafter_in_cogniland.py`,
  built on the in-tree `purejaxwm/` algorithm library.
* **PPO + GRU (PyTorch)** — `scripts/crafter/train_ppo_gru.py`,
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
python scripts/crafter/dreamerv3_crafter_in_cogniland.py \
  --size 12M --total-env-steps 50000 --num-envs 16 --train-ratio 16 \
  --map-size 32 --view-size 11 --wandb-mode disabled

# Real run (25M default, ~1M env-steps, ~30 min on RTX 4090)
python scripts/crafter/dreamerv3_crafter_in_cogniland.py \
  --size 25M --total-env-steps 1_000_000 \
  --num-envs 32 --train-ratio 64 --wandb-mode online
```

Size presets follow DreamerV3 paper Table 3:
`12M | 25M | 50M | 100M | 200M | 400M`. The `size=...` value is added
as a W&B tag automatically.

### PPO + GRU

```bash
python scripts/crafter/train_ppo_gru.py \
  --total-timesteps 5_000_000 --num-envs 32 --num-steps 128 \
  --env-size 64 --view-size 21 --tile-px 8 \
  --device cuda --wandb-project crafter_in_cogniland
```

PPO logs `size=X.YM` tag automatically based on its actual param count.

### Inspect a trained Dreamer

```bash
python scripts/crafter/viz_dreamer_trajectory.py \
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
python scripts/crafter/generate_maps.py --sizes 32 96

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
  Mapgen mirrors lake↔rocky by swapping which noise field becomes
  water vs. rock; balanced has scattered narrow features that don't
  cross the spawn-target line.
- **Action space**: `Discrete(6)` — up / down / left / right /
  build_raft / build_harness. Build is committed once per episode.
- **Observation (JAX env)**: `{minimap: (V,V) int8, scalars: (4,) float32}`
  with scalars = `[compass_r, compass_c, build_active, step/max]`.
  *Which* item was built (raft vs harness) is intentionally **not**
  observable — only the binary `build_active` flag. The agent must
  remember its commitment from the build action it took, which makes
  this a partial-observability problem requiring recurrent memory
  (GRU for PPO, RSSM for Dreamer). Egocentric tile-id crop, OOB
  padded with `OOB=6`.
- **Reward** (mirrors `cogniland.nav.skills`):
  `-0.02` slack per step
  `+ 0.01 · (ctg_prev − ctg_curr)` PBRS shaping
  reach bonus disabled (`REACH_BONUS = 0`) — length-of-path drives the
  return so PPO/Dreamer keep tightening the route after `success → 1`.
- **Slip**: water/rock slip with prob 0.75 unless you carry the
  matching item; trees always slip 0.9; land slips 0.30 with any item
  carried (the "weight tax" that makes the wrong skill strictly worse
  than carrying nothing).

## How Dreamer trains in imagination

A common question — Dreamer's actor and critic never see real
observations directly. Their gradient signal comes from the world
model's predictions over short *imagined* trajectories.

Each train step does two passes:

1. **World-model pass** on a real replay sequence
   `(obs_{0..T}, action_{0..T-1}, reward_{0..T}, is_terminal_{0..T})`:
   - **Encode** each frame's `obs` through the MLP encoder
     (`crafter_in_cogniland` flattens the dict obs to `(V·V + 4,)`
     and runs it through a 4-block Dense + RMSNorm + SiLU stack).
   - **Roll the RSSM** with the encoded embeds: for each step,
     posterior `z_t = q(z_t | h_t, embed_t)` is sampled; the
     deterministic recurrent state evolves as
     `h_{t+1} = GRU(h_t, [z_t, a_t])`. Both `h_t` and the discrete
     `z_t` form the latent.
   - **Decode** each latent back to `(V·V + 4,)` for reconstruction;
     predict reward and continue flag with two heads. All three are
     trained against the real targets, plus KL between the prior
     `p(z_t | h_t)` and posterior `q(z_t | h_t, embed_t)`.

2. **Actor-critic pass entirely in imagination**:
   - Take the posterior states from step 1 as *starts* (T·B of them).
   - Roll the RSSM forward H=15 steps using **only the prior**:
     `z_{t+1} ~ p(z_{t+1} | h_{t+1})`. The encoder never runs in this
     pass — the agent imagines its own future without any pixel
     reconstruction.
   - At each imagined step the actor samples `a_t` from `π(a|h_t,z_t)`;
     the reward and continue heads predict the rewards/discounts;
     λ-returns are computed over the imagined trajectory.
   - The actor is updated with REINFORCE + return-normalised
     advantage; the critic regresses on λ-returns (TwoHot) with a
     slow-EMA target for stability.

So Dreamer's observable for AC training is the RSSM's *latent* state
`(h_t, z_t)`, not the raw `(minimap, scalars)`. The encoder/decoder
are only trained on real data — the actor/critic gradient flows
entirely through the world model's predictions. That's why we
shouldn't worry about the encoder being on a "thin" observation:
once the world model nails the dynamics, the agent's policy can be
arbitrarily complex on top of the learned latent.

In this repo's setup specifically:
- The **actor's input** is `[h_t (deter, ~3072-d), z_t (stoch×classes
  one-hot, ~576-d)]`. No raw pixels, no scalars.
- The encoder only runs on **real obs** during the WM-loss pass and
  during action-selection in the env stepping (to compute the
  posterior for the next step). It does **not** run inside imagination.
- Hiding `active_obj/2` from the scalars means the encoder gets
  ambiguous data (build_active=1 covers raft or harness), so the
  RSSM has to recover the missing bit from the *history* of inputs
  (the agent's previous build action lands in `last_action_oh`,
  which the RSSM consumes). Latent state ends up carrying the
  identity implicitly.

## Shared W&B logging schema

Both trainers log into the same project. The cross-algo charts in the
default W&B workspace key off these metric names:

| Key                          | Meaning                              |
|------------------------------|--------------------------------------|
| `success/mean`               | Mean reach-rate (this log interval)  |
| `success/rolling100`         | Rolling success rate over 100 eps    |
| `return/mean`                | Mean episode return                  |
| `return/rolling100`          | Rolling mean over 100 episodes       |
| `return/min_over_steps`      | Path efficiency: mean of `2·map_size / length` per episode. 1.0 = the corner-to-corner Manhattan bound, > 1 = the agent's path was shorter than that bound (typical). |
| `rollout/episode_length`     | Mean episode length                  |
| `skill_usage/<biome>/<skill>`| Per-cell scalar of the 3×3 matrix    |
| `skill_usage/matrix`         | 3×3 heatmap image (this log interval)|
| `perf/fps`                   | Wallclock throughput                 |

The skill-usage matrix rows are `grassland | rocky | lake`; columns
are `noskill | harness | raft`. Each row sums to 1, where the row
counts (n=…) on the y-axis show how many episodes finished on that
map type in the current log interval — it's a per-eval snapshot, not
a cumulative average.

Tags always include `algo=<dreamerv3|ppo_gru>`, `size=<X.YM>`, and
`map=<N>`.

## Goals

- **>80% success** on Dreamer (medium model, 1–3M env steps).
- **Frozen checkpoints** that load deterministically for analysis
  (`viz_dreamer_trajectory.load_frozen`).
- **Trajectory + imagination plots** out of the box.
- **One W&B workspace** comparing PPO and Dreamer on shared metrics.
