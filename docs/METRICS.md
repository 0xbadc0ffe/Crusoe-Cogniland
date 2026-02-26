# Metrics Reference

All metrics logged to WandB during training and evaluation. The **x-axis** in WandB is the **update** number (one PPO optimisation pass over a rollout buffer), not raw timesteps.

---

## Terminology

| Term | Meaning |
|------|---------|
| **update** | One PPO training iteration: collect rollout → compute GAE → run minibatch updates. X-axis in WandB. |
| **move** | One agent action in the environment (up/down/left/right/stay). An episode is a sequence of moves. |
| **global_step** | Cumulative number of moves across all parallel envs since training start: `update × num_envs × rollout_steps`. |

---

## Training metrics (`train/`)

Logged every update. Computed from the rollout buffer collected from `num_envs` parallel environments over `rollout_steps` moves.

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `train/policy_loss` | `mean(max(-A·r, -A·clip(r, 1±ε)))` where `r = π_new/π_old`, `A = GAE advantage` | PPO clipped surrogate objective. Should decrease then stabilise. |
| `train/value_loss` | `0.5 · mean((V_pred - returns)²)` | Critic's prediction error against GAE returns. Should decrease. |
| `train/entropy` | `mean(H(π))` — entropy of the action distribution | Policy randomness. Starts high, decreases as policy specialises. Too low = premature convergence. |
| `train/clipfrac` | Fraction of minibatch samples where `|r - 1| > clip_coef` | How often PPO's clipping activates. High = policy changing too fast. Healthy range: 0.05–0.2. |
| `train/approx_kl` | `mean((r - 1) - log(r))` — second-order KL approximation | Divergence between old and new policy. If consistently > 0.03, learning rate may be too high. |
| `train/learning_rate` | Current LR (with optional linear annealing) | Tracks the LR schedule. |
| `train/steps_per_second` | `global_step / wall_time` | Moves per second — throughput. |
| `train/mean_reward` | Mean episodic return across episodes completed during rollout | Online training performance. Noisy because it only includes episodes that happened to finish during the rollout window. |
| `train/mean_episode_length` | Mean episode length (moves) of completed episodes | How long episodes last during training. |
| `train/global_step` | `update × num_envs × rollout_steps` | Raw move count for reference. |

---

## Evaluation metrics (`eval-deterministic/`, `eval-stochastic/`)

Logged every `eval_interval` updates. Two modes run in parallel: **deterministic** (greedy argmax) and **stochastic** (sampled from policy). Each mode gets its own WandB section.

### Scalar metric

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `eval-{mode}/success_rate_mean` | `mean(reached)` | Primary performance metric (scalar line). |

### Density heatmaps (per mode)

All per-episode distributions are logged as **Plotly density heatmaps** (2D histogram with blue→red colorscale). Each eval appends a new column; the chart rebuilds with the full history. The Y-axis shows metric value bins, the X-axis shows training updates, and color intensity indicates the density of episodes in each bin.

| Panel key | What it tells you |
|-----------|-------------------|
| `eval-{mode}/return` | Per-episode return distribution. |
| `eval-{mode}/episode_length` | Per-episode length distribution. |
| `eval-{mode}/min_hp` | Worst-case HP during episode. |
| `eval-{mode}/final_hp` | HP at episode end. |
| `eval-{mode}/danger_fraction` | Fraction of steps in danger zone. |
| `eval-{mode}/final_resources` | Resources at episode end. |
| `eval-{mode}/max_resources` | Peak resources during episode (capped at `max_resources`). |
| `eval-{mode}/path_efficiency` | Bounded [0, 1]. `clamp(A* cost / agent cost, 0, 1)`. 1 = optimal path. 0 if A* finds no path. |
| `eval-{mode}/mean_resource` | Running mean of resources across trajectory. |
| `eval-{mode}/mean_hp` | Running mean of HP across trajectory. |

---

## Behavioral metrics (`behavioral-deterministic/`, `behavioral-stochastic/`)

Logged alongside eval metrics, computed **per mode**. Each mode gets its own WandB section. Derived from terrain visit counts and final agent state.

### Terrain distribution chart (`behavioral-{mode}/terrain_distribution`)

An interactive **Plotly stacked normalized area chart** (`groupnorm="percent"`) showing the fraction of moves spent on each of the 9 terrain types over training. Terrains are stacked bottom→top: ocean, deep_water, water, beach, sandy, grassland, forest, rocky, mountains — with palette-matched colors from `constants.py`.

**Aggregation**: Per-environment terrain visit fractions are averaged across all envs. `groupnorm="percent"` re-normalizes to sum to 100% at each X-point. Y-axis is fixed to [0, 100]%.

### Density heatmaps (per mode)

| Panel key | Source | What it tells you |
|-----------|--------|-------------------|
| `behavioral-{mode}/mean_resource` | Per-episode Welford running mean of resources | Average resource level across the trajectory. |
| `behavioral-{mode}/mean_hp` | Per-episode Welford running mean of HP | Average health across the trajectory. |
| `behavioral-{mode}/final_hp` | HP at episode end | End-of-episode health snapshot. |
| `behavioral-{mode}/final_resources` | Resources at episode end | End-of-episode resource snapshot. |
| `behavioral-{mode}/max_resources` | Peak resources during episode | Highest resource level reached (capped at `max_resources`). |
| `behavioral-{mode}/danger_fraction` | Steps with HP < threshold / total moves | Fraction of the episode spent in danger zone. |

---

## Trajectory images (`trajectories/`)

Logged as individual `wandb.Image` panels under `trajectories/ep_0`, `trajectories/ep_1`, etc. Each image shows the full island map with the agent's path overlaid.

| Element | Meaning |
|---------|---------|
| Terrain colours | Blue = ocean, green = grassland/forest, tan = beach/sand, white = mountain/snow |
| White outline + red line | Agent's path from start to end |
| Green circle | Spawn position |
| Red X | Final position |
| Gold star | Target position |
| Caption | `success` or `fail` + number of moves |

---

## Reward function breakdown

The per-move reward is the sum of six components, all configurable via `configs/env/default.yaml` or CLI overrides:

| Component | Formula | Default coef | Purpose |
|-----------|---------|-------------|---------|
| `r_dist` | `(prev_distance - current_distance) × coef` | 0.03 | Dense signal: move toward target |
| `r_reach` | `+bonus` if agent reaches target | +12.0 | Sparse: reward success |
| `r_death` | `+penalty` if HP ≤ 0 | -8.0 | Sparse: penalise dying |
| `r_time` | `penalty` (constant per move) | -0.1 | Dense: discourage dawdling |
| `r_hp` | `-coef × max(threshold - HP, 0)` | coef=0.05, thresh=35 | Penalise dangerously low HP |
| `r_resource` | `-coef × max(threshold - resources, 0)` | coef=0.02, thresh=25 | Penalise dangerously low resources |

**Total**: `r_dist + r_reach + r_death + r_time - r_hp - r_resource`
