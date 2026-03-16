# Metrics Reference

All metrics logged to WandB during training and evaluation. The **x-axis** in WandB is the **update** number (one PPO optimisation pass over a rollout buffer), not raw timesteps.

---

## Terminology

| Term | Meaning |
|------|---------|
| **update** | One PPO training iteration: collect rollout → compute GAE → run minibatch updates. X-axis in WandB. |
| **move** | One agent action in the environment (up/down/left/right/stay). An episode is a sequence of moves. |
| **global\_step** | Cumulative number of moves across all parallel envs since training start: `update × num_envs × rollout_steps`. |
| **C\_agent** | Terrain-weighted cost accumulated by the agent during an episode (`EnvState.cost`). Each non-stay move adds `TERRAIN_COSTS[terrain_idx]` to this counter. |
| **A\*** | Optimal terrain-weighted shortest path computed by `batch_astar()`. The cost of the A* path equals the sum of `TERRAIN_COSTS` along the optimal route on the actual map used in the episode. |

---

## Training metrics (`train/`)

Logged every update. Computed from the rollout buffer collected from `num_envs` parallel environments over `rollout_steps` moves.

### PPO algorithm (`train/model/ppo/`)

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `policy_loss` | `mean( max(−Â·r_t, −Â·clip(r_t, 1−ε, 1+ε)) )` where `r_t = π_θ(a)/π_θ_old(a)`, `Â = GAE advantage` | Clipped surrogate objective. Should decrease then stabilise. |
| `value_loss` | `0.5 · mean( (V_pred − returns)² )` | Critic MSE against GAE returns. Should decrease. |
| `entropy` | `mean( H(π) ) = mean( −∑ π(a) log π(a) )` | Policy randomness. Starts high, decreases as policy specialises. Too low = premature convergence. |
| `clipfrac` | `mean( 𝟙[|r_t − 1| > ε] )` | Fraction of minibatch samples where PPO clipping activates. Healthy range: 0.05–0.2. If consistently high, LR may be too large. |
| `approx_kl` | `mean( r_t − 1 − log(r_t) )` | Second-order KL approximation between old and new policy. If > 0.03 consistently, consider reducing LR. |
| `learning_rate` | Current LR after optional linear annealing. | Tracks the LR schedule. |

### Environment stats during rollout (`train/env/`)

These are logged only for episodes that happen to **complete** within the current rollout window; partially-completed episodes are excluded. They are therefore noisier than eval metrics.

| Metric | What it tells you |
|--------|-------------------|
| `episode_return_mean` | Mean cumulative reward of completed training episodes in this rollout. |
| `episode_length_mean` | Mean episode length (moves) of completed training episodes. |
| `success_rate` | Fraction of completed training episodes where the agent reached the target. |

### Throughput

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `train/sps` | `global_step / wall_time` | Moves processed per second. Pure throughput; GPU-bound code shows higher values. |

---

## Evaluation metrics

Logged every 10 updates (val splits) and once at the end of training (test split). Two policies are evaluated in parallel:

- **det** — deterministic policy: action = `argmax π(a | obs)`.
- **stoch** — stochastic policy: action sampled from `π(· | obs)`.

Each produces a separate WandB section: `val_det/`, `val_stoch/`, `test_det/`.

### Scalar metric (`{split}_{mode}/env/`)

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `success_rate` | `n_success / n_episodes` | Primary performance metric — fraction of eval episodes where the agent reached the target. Logged as a plain scalar line. |

### Per-episode charts (`{split}_{mode}/env/`)

All remaining metrics are logged as **mean ± std** shaded-area charts (Vega preset `crusoe/eval_mean_std`). Each chart accumulates one data point per eval step, showing training progress over time.

| Metric | Range | What it tells you |
|--------|-------|-------------------|
| `return` | (−∞, +∞) | Cumulative reward per episode. Primary signal alongside success rate. |
| `episode_length` | [1, 1000] | How long episodes last. Successful episodes may be shorter (efficient path) or longer (cautious path). |
| `min_hp` | [0, 100] | Minimum HP observed at any step during the episode. Low values indicate the agent came close to death. |
| `final_hp` | [0, 100] | HP at episode termination (death → 0; success/timeout → positive). |
| `mean_hp` | [0, 100] | Running mean of HP across all steps: `(1/T) ∑_t hp_t`. |
| `danger_fraction` | [0, 1] | Fraction of moves with HP below the danger threshold (30 by default). See [Danger Fraction](#danger-fraction). |
| `final_resources` | [0, 100] | Resources at episode end. |
| `mean_resources` | [0, 100] | Running mean of resources: `(1/T) ∑_t res_t`. |
| `max_resources` | [0, 100] | Peak resources reached during the episode. |
| `directness` | [1, 100] | How efficiently the agent moved relative to the optimal path to its final position. See [Directness](#directness). |
| `survival_margin` | (0, 100] | Minimum over all steps of the ratio between current resources/HP and what would be needed to complete the remaining journey. See [Survival Margin](#survival-margin). |
| `exploration` | [0, 1] | Fraction of the 250×250 map cells the agent observed during the episode. See [Exploration (Coverage)](#exploration-coverage). |

### Terrain distribution (`{split}_{mode}/terrain_distribution`)

A Vega stacked area chart (preset `crusoe/terrain_distribution`) showing how terrain visit fractions evolve over training. The Y-axis is the **fraction of episode steps** spent on each terrain type (summing to 1 across terrains at each eval step). The chart accumulates rows across all eval steps.

---

## Behavioral metrics — detailed specification

### Directness

**What it measures:** How efficiently the agent moved in terms of terrain cost, compared to the optimal path to wherever it *ended up* — regardless of whether it reached the target.

**Why "to its final position":** Using the agent's final position rather than the target makes the metric well-defined for all episode outcomes (success, death, timeout). An agent that wandered far and then died is correctly penalised, even though it never reached the target.

**Computation:**

Let:
- `C_agent` = total terrain-weighted cost accumulated during the episode (`EnvState.cost`)
- `C_astar(spawn → final)` = optimal A* cost from spawn to the agent's **final position** on the episode's actual map

Then:

```
         C_agent
D = ─────────────────────────────
    C_agent − C_astar(spawn → final)
```

This formula measures the "overhead factor": if `C_astar = 0.9 · C_agent`, the agent wasted 10% of its movement cost versus the optimal route, giving `D = 1 / 0.1 = 10`. If the agent took exactly the optimal route, `C_agent = C_astar`, the denominator is zero and `D` is capped at 100.

**Boundary cases:**
- If `C_agent ≤ C_astar + ε` (agent at least as efficient as A*): `D = 100` (cap).
- If `C_agent = 0` (agent stayed still): `D = 100` by the cap rule.
- The cap of 100 prevents numerical explosion for nearly-optimal agents; it does not imply the agent scored perfectly on all metrics.

**Range:** [1, 100]. `D = 100` means the agent followed the optimal route (or stayed still). `D = 2` means the agent spent twice the terrain cost of the optimal route. `D = 1` is the theoretical minimum: the agent wasted all of its movement cost vs. the optimal path (only possible if `C_astar ≈ 0`).

**Implementation in code** (`runner.py`):
```python
directness = torch.where(
    final_cost > astar_to_final + 1e-6,
    (final_cost / (final_cost - astar_to_final)).clamp(max=100.0),
    torch.full_like(final_cost, 100.0),
)
```

---

### Survival Margin

**What it measures:** At every step during the episode, the agent's *projected viability* for completing the remaining journey — specifically whether its current HP and resources would be sufficient to cover the optimal remaining path. Reports the **worst-case ratio** across the entire episode.

**Intuition:** A survival margin > 1.0 means the agent was always carrying more HP/resources than needed (comfortable margin). A margin < 1.0 means there was a point where the agent was projected to run out of HP or resources before reaching the target via the shortest remaining route.

**Per-step computation:**

Let `dist_t` be the Euclidean distance from the agent to the target at step `t`, and `dist_0` the initial distance. The remaining A* cost is approximated as:

```
C_remaining(t) = C_astar(spawn → target) × (dist_t / dist_0)
```

This scales the initial optimal cost by the fraction of distance still remaining, giving a terrain-weighted estimate of the remaining journey cost.

From the environment's resource drain rates, two conversion factors are precomputed once from `EnvConfig`:

```
terrain_drains = [sea[0], sea[1], sea[2], land, land, land, 0.0, mtn[0], mtn[1]]
             = [3.0,    2.0,    1.5,   1.0, 1.0, 1.0,  0.0, 1.5,    3.0   ]  (defaults)

k_R  = mean(terrain_drains) / mean(TERRAIN_COSTS)   ≈ 0.767
k_HP = k_R × no_res_hp_multiplier                   ≈ 1.534  (multiplier default = 2.0)
```

These factors convert "expected remaining terrain cost" into "expected HP drain" and "expected resource drain" respectively.

The projected requirements at step `t` are:

```
Ĉ_HP(t) = C_remaining(t) × k_HP       (HP the agent would consume)
Ĉ_R(t)  = C_remaining(t) × k_R        (resources the agent would consume)
```

The per-step survival margin is the binding constraint (whichever resource is tighter):

```
SM_t = min( hp_t / (Ĉ_HP(t) + ε),   resources_t / (Ĉ_R(t) + ε) )
```

**Episode summary:**

```
survival_margin = min over all running steps t of SM_t
```

Only steps where the episode is still active (not done, not dead) contribute to the minimum.

**Boundary handling:**
- `inf` values (e.g. when `dist_0 = 0`) are clipped to 100.
- NaN values are set to 0.

**Range:** (0, 100]. Values > 1 indicate the agent was always well-provisioned. Values < 1 indicate a resource crisis at some point in the episode.

**Implementation in code** (`runner.py`, precomputed in `__init__`):
```python
c_remaining = astar_costs * (dist_to_target / initial_dist)
c_hat_hp = c_remaining * self._k_HP
c_hat_r  = c_remaining * self._k_R
sm_t = torch.minimum(
    current_hp / (c_hat_hp + 1e-6),
    current_resources / (c_hat_r + 1e-6),
)
survival_margin = torch.minimum(survival_margin, sm_t)  # running min
```

---

### Exploration (Coverage)

**What it measures:** What fraction of the 250×250 map the agent observed during the episode.

**Computation:**

An `observed` boolean tensor of shape `[n_episodes, H, W]` is maintained. At each step, for each running episode, all map cells within a disk of radius `vis_r` centred on the agent's current position are marked as observed:

```
observed[i, r+dr, c+dc] = True   for all (dr, dc) with dr²+dc² ≤ vis_r²
```

where `vis_r = TERRAIN_VISIBILITY[terrain_idx]` depends on the terrain the agent is currently standing on.

**Terrain visibility radii** (from `constants.py`):

| Terrain | Index | Visibility radius |
|---------|-------|------------------|
| ocean | 0 | 10 |
| deep\_water | 1 | 8 |
| water | 2 | 6 |
| beach | 3 | 4 |
| sandy | 4 | 4 |
| grassland | 5 | 4 |
| forest | 6 | 4 |
| rocky | 7 | 8 |
| mountains | 8 | 22 |

The episode exploration fraction is:

```
exploration = count(observed[i]) / (H × W)
```

**Range:** [0, 1]. 0 = agent never moved (or took only stay actions). Full map coverage (1.0) is practically unachievable within 1000 steps on a 250×250 map.

**Implementation in code** (`runner.py`):
```python
exploration = observed.sum(dim=(1, 2)).float() / (H * W)
```

Disk offsets for each unique visibility radius are precomputed in `EvalRunner.__init__` and cached as `_disk_offsets[vis_r]`.

---

### Danger Fraction

**What it measures:** Fraction of episode steps where the agent was in a critically low HP state.

```
danger_fraction = (steps with hp_t < hp_danger_threshold) / episode_length
```

Default threshold: `hp_danger_threshold = 30.0` (out of `max_hp = 100.0`).

**Range:** [0, 1]. 0 = agent never entered danger zone. 1 = agent was always in danger (e.g. spawned with low HP and never recovered).

---

### HP and Resource Metrics

Tracked per episode as running accumulators:

| Key | Description |
|-----|-------------|
| `min_hp` | `min_t(hp_t)` — lowest HP at any step |
| `final_hp` | HP at episode termination |
| `mean_hp` | `(1/T) ∑_t hp_t` — time-averaged HP |
| `final_resources` | Resources at episode termination |
| `mean_resources` | `(1/T) ∑_t res_t` — time-averaged resources |
| `max_resources` | `max_t(res_t)` — peak resources reached during episode |

---

## Terrain types

| Index | Name | Mov. cost | Visibility | Resource drain / step |
|-------|------|-----------|------------|----------------------|
| 0 | ocean | 0.5 | 10 | −3.0 |
| 1 | deep\_water | 0.75 | 8 | −2.0 |
| 2 | water | 1.0 | 6 | −1.5 |
| 3 | beach | 1.5 | 4 | −1.0 |
| 4 | sandy | 2.0 | 4 | −1.0 |
| 5 | grassland | 1.5 | 4 | −1.0 |
| 6 | forest | 3.5 | 4 | +2.0 res (or +5.0 HP if below max) |
| 7 | rocky | 3.5 | 8 | −1.5 |
| 8 | mountains | 4.0 | 22 | −3.0 |

Resource drain: each move drains resources by the listed amount. If resources reach 0, the shortfall is multiplied by `no_res_hp_multiplier` (default 2.0) and deducted from HP instead. Forest is the only regenerating terrain: it heals HP at 5.0/step until `max_hp`, then grants resources at 2.0/step.

---

## Trajectory images (`trajectories/`)

Logged as `wandb.Image` panels under `trajectories/env_0`, `trajectories/env_1`, etc. (deterministic policy only, first 4 eval episodes). Each image shows the full island map with the agent's path overlaid.

| Element | Meaning |
|---------|---------|
| Terrain colours | Blue = ocean/water, green = grassland/forest, tan = beach/sand, grey = rocky, white = mountains |
| Path line | Agent's route from spawn to final position |
| Green circle | Spawn position |
| Red cross | Final position |
| Gold star | Target position |
| Caption | Outcome + episode length |

---

## Reward function

The per-move reward is the sum of six components. All coefficients are set in `configs/env/default.yaml` and can be overridden via Hydra CLI.

| Component | Formula | Default | Purpose |
|-----------|---------|---------|---------|
| `r_dist` | `(dist_prev − dist_t) × coef` | coef = 0.25 | Dense signal: reward moving toward target. |
| `r_reach` | `+bonus` if agent reached target | +60.0 | Sparse: large reward for success. |
| `r_death` | `+penalty` if HP ≤ 0 | −40.0 | Sparse: penalise dying. |
| `r_time` | constant per move | −0.01 | Dense: discourage dawdling. |
| `r_hp` | `−coef × max(thresh − hp_t, 0)` | coef = 0.05, thresh = 50.0 | Soft penalty for dangerously low HP. |
| `r_resource` | `−coef × max(thresh − res_t, 0)` | coef = 0.05, thresh = 30.0 | Soft penalty for dangerously low resources. |

**Total per-move reward:**
```
r_t = r_dist + r_reach + r_death + r_time − r_hp − r_resource
```
