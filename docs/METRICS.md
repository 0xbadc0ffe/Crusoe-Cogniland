# Metrics Reference

All metrics logged to WandB during training and evaluation. Cross-references use `file.py:line`.

---

## Terminology

| Term | Meaning |
|------|---------|
| **update** | One PPO training iteration: collect rollout → GAE → minibatch updates. X-axis in WandB. |
| **global_step** | Cumulative env moves since training start: `update * num_envs * rollout_steps`. |
| **C_agent** | Terrain-weighted cost accumulated by the agent (`EnvState.cost`). Each move adds `move_cost[terrain]`. |
| **dijkstra_cost** | Optimal terrain-weighted traversal cost spawn→target, computed by forward Dijkstra at reset. |

---

## Training metrics (`train/`)

Logged every update. Source: `ppo.py:591` (PPO) / `recurrent_ppo.py:619` (RNN PPO).

### PPO algorithm (`train/model/`)

| Metric | What it tells you |
|--------|-------------------|
| `policy_loss` | Clipped surrogate objective. Should decrease then stabilise. |
| `value_loss` | Critic MSE against GAE returns. Should decrease. |
| `entropy` | Policy randomness `H(pi)`. Starts high, decreases as policy specialises. Too low = premature convergence. |
| `clipfrac` | Fraction of samples where PPO clipping activates. Healthy: 0.05-0.2. Consistently high → LR too large. |
| `approx_kl` | KL divergence approximation between old and new policy. > 0.03 consistently → reduce LR. |
| `explained_variance` | How well the value function predicts returns. 1.0 = perfect, < 0 = worse than mean. |
| `learning_rate` | Current LR after optional linear annealing. |

### Environment stats (`train/env/`)

Logged by `log_rollout_stats()` in `logging.py:24`. Only covers episodes that **completed** within the rollout window (partial episodes excluded — noisier than eval metrics).

| Metric | Source |
|--------|--------|
| `episode_return_mean` | Mean cumulative reward of completed episodes |
| `episode_length_mean` | Mean episode length (moves) |
| `success_rate` | Fraction that reached the target |

### Throughput

| Metric | Formula |
|--------|---------|
| `train/sps` | `global_step / wall_time` — moves processed per second |

---

## Evaluation metrics

Logged periodically during training (val) and once at end (test). Source: `runner.py:64` → `summarizer.py:24`.

Two policies evaluated in parallel:
- **det** — deterministic: `argmax pi(a|obs)` 
- **stoch** — stochastic: sampled from `pi(.|obs)`

Namespaces: `val_det/env/`, `val_stoch/env/`, `test_det/env/`, `test_stoch/env/`.

### Scalar metrics

| Metric | Range | What it measures | Implementation |
|--------|-------|-----------------|----------------|
| `success_rate` | [0, 1] | Fraction of episodes reaching target. Primary metric. | `summarizer.py:33` |
| `return_{mean,std,...}` | (-inf, +inf) | Cumulative reward per episode. | `summarizer.py:37` |
| `episode_length_{mean,...}` | [1, 1000] | Steps per episode. | `summarizer.py:38` |
| `directness_{mean,...}` | [0, 1] | Time efficiency: `dijkstra_cost / agent_cost`. 1.0 = optimal path. | `metrics.py:13` |
| `risk_exposure_{mean,...}` | [0, 1] | Ulcer Index of survival budget. RMS of relative drawdowns. | `metrics.py:25` |
| `exploration_{mean,...}` | [0, 1] | Fraction of land cells observed at least once. | `metrics.py:47` |
| `danger_fraction_{mean,...}` | [0, 1] | Fraction of steps with HP below danger threshold (50.0). | `metrics.py:39` |
| `min_hp_{mean,...}` | [0, 100] | Lowest HP at any step during episode. | `runner.py:189` |
| `final_hp_{mean,...}` | [0, 100] | HP at termination. | `runner.py:236` |
| `mean_hp_{mean,...}` | [0, 100] | Time-averaged HP. | `runner.py:265` |
| `final_resources_{mean,...}` | [0, 100] | Resources at termination. | `runner.py:263` |
| `mean_resources_{mean,...}` | [0, 100] | Time-averaged resources. | `runner.py:264` |
| `max_resources_{mean,...}` | [0, 100] | Peak resources during episode. | `runner.py:200` |

Each non-rate metric is logged as `{prefix}/{name}_mean`, `_std`, `_max`, `_min`. The `_std/_max/_min` variants go to `run.summary` only (no time-series plot). See `logging.py:186`.

### Terrain distribution

Logged as a growing WandB Table under `{namespace}/terrain_distribution` using custom Vega spec `crusoe/terrain_distribution`. Shows how terrain visit fractions evolve over training. Val splits only. See `logging.py:245`.

### Trajectory images

Logged as `wandb.Image` under `trajectories/env_{i}`. Deterministic policy only, first `max_saved_per_eval` episodes (default 3). Shows terrain-colored map + agent path + fog-of-war for unseen cells. See `utils.py:83`.

### Per-episode tables

Logged as `wandb.Table` under `{namespace}/tables/episodes` with columns: episode, outcome, return, episode_length, final_hp, trajectory. See `logging.py:217`, `summarizer.py:60`.

---

## Behavioral metrics — detailed specification

### Directness (time efficiency)

**What it measures:** How efficiently the agent traverses terrain relative to the optimal shortest-time path.

**Formula** (`metrics.py:22`):

```
D = dijkstra_cost / agent_cost
```

Where `dijkstra_cost` is the optimal terrain-weighted traversal cost from spawn to target (forward Dijkstra, computed at reset), and `agent_cost` is the accumulated `EnvState.cost` (sum of `move_cost[terrain]` along the agent's actual path).

**Range:** [0, 1], clamped. 1.0 = agent matched the optimal path. Lower values indicate detours, backtracking, or foraging stops that increased total terrain cost.

**Design choice:** Uses terrain-weighted cost rather than Manhattan distance. This means a detour through cheaper terrain (e.g., water shortcut) can score *higher* than a straight line through expensive terrain.

### Risk Exposure (Ulcer Index)

**What it measures:** The severity and duration of survival budget depletion over the episode. Based on the Ulcer Index from finance.

**Formula** (`metrics.py:25`, accumulated in `runner.py:209`):

```
u_t = resources_t + hp_t                          (survival budget at step t)
u_0 = init_resources + init_hp = 200              (initial budget)

rho = sqrt( (1/T) * sum_t ((u_0 - u_t) / u_0)^2 )
```

**Range:** [0, 1]. Low = healthy budget throughout. High = prolonged or acute depletion. Unlike a simple "min HP" metric, this captures *how long* the agent spent in a depleted state, not just the worst moment.

**Design choice:** The Ulcer Index penalises both depth and duration of drawdowns. An agent that drops to 50% budget for 100 steps scores worse than one that drops to 10% for 1 step (if the total squared-drawdown-sum is larger).

### Exploration (Coverage)

**What it measures:** Fraction of land cells the agent observed during the episode.

**Formula** (`metrics.py:47`):

```
C = |cells_observed ∩ land_cells| / |land_cells|
```

Visibility is determined by the minimap system (terrain-dependent radius + line-of-sight occlusion via channel 2). Accumulated via per-cell visibility counters in `runner.py:122`.

**Range:** [0, 1]. Full land coverage (1.0) is practically unachievable within 1000 steps on a 250x250 map. Typical values for successful agents are 0.05-0.15.

### Danger Fraction

**What it measures:** Fraction of episode steps with HP below the danger threshold.

**Formula** (`metrics.py:39`):

```
danger_fraction = (steps with hp_t < hp_danger_threshold) / episode_length
```

Default threshold: 50.0 (configurable via `logging.eval.hp_danger_threshold` in `main.yaml`).

---

## Test summary

At training end, `log_final_test_summary()` (`logging.py:304`) pushes all test metrics to `run.summary` (WandB runs table columns) and logs a readable `wandb.Table` under `test/summary_table` comparing det vs stoch on key metrics.

## Behavioral test evaluation

After the standard test eval, a behavioral eval runs the deterministic policy on each of the 9 hand-crafted maps from `custom_maps.py`. Results logged under `test/behavioral/{map_name}/success`, `/return`, `/episode_length`, plus trajectory images. See `ppo.py:438`.
