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
| `train/sps` | `global_step / wall_time` | Moves per second — throughput. |
| `train/mean_reward` | Mean episodic return across episodes completed during rollout | Online training performance. Noisy because it only includes episodes that happened to finish during the rollout window. |
| `train/mean_episode_length` | Mean episode length (moves) of completed episodes | How long episodes last during training. |
| `train/global_step` | `update × num_envs × rollout_steps` | Raw move count for reference. |

---

## Evaluation metrics (`eval/`)

Logged every `eval_interval` updates. Computed from `eval_episodes` parallel episodes run to completion (success, death, or truncation at `max_steps`).

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `eval/success_rate` | `mean(reached)` — fraction of episodes where agent reached the target | Primary performance metric. |
| `eval/mean_reward` | `mean(Σ rewards)` over all eval episodes | Total episodic return, comparable across evals since all episodes run to termination. |
| `eval/mean_moves` | `mean(episode_length)` | Average moves per episode. Lower with higher success rate = agent is getting efficient. |
| `eval/mean_final_hp` | `mean(HP at episode end)` | Agent's health management. Low = risky strategy, high = conservative. |
| `eval/path_efficiency` | `mean(initial_distance_to_target / moves)` | How directly the agent moves toward the target. Higher = more efficient pathfinding. |

---

## Behavioral metrics (`behavioral/`)

Logged alongside eval metrics. Computed from terrain visit counts and final agent state across eval episodes. Also visualised as a bar chart in `behavior/profile`.

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| `behavioral/water_pct` | `mean(water_moves / total_moves)` — terrains 0–2 (ocean, deep water, water) | How much time the agent spends at sea. High = risky, costs HP without resources. |
| `behavioral/land_pct` | `mean(land_moves / total_moves)` — terrains 3–5 (beach, sandy, grassland) | Time on safe traversable land. |
| `behavioral/forest_pct` | `mean(forest_moves / total_moves)` — terrain 6 | Forest has high movement cost but provides resources. High = conservative resource-gathering policy. |
| `behavioral/mountain_pct` | `mean(mountain_moves / total_moves)` — terrains 7–8 (rocky, mountains) | Very costly to traverse. High = agent taking bad routes. |
| `behavioral/resource_held` | `mean(resources at episode end)` | End-of-episode resource level. High = hoarding, low = spending efficiently (or never collecting). |
| `behavioral/hp_score` | `mean(final_HP / 100)` | Normalised final health. 0 = dead, 1 = full health. Tracks survival strategy. |
| `behavioral/directness` | `mean(remaining_distance / moves)` | Remaining compass distance divided by moves taken. Lower = agent moved toward target efficiently. High = wandering or stuck. |

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

The per-move reward is the sum of five components, all configurable via `configs/env/*.yaml`:

| Component | Formula | Default coef | Purpose |
|-----------|---------|-------------|---------|
| `r_dist` | `(prev_distance - current_distance) × coef` | 1.0 | Dense signal: move toward target |
| `r_reach` | `+bonus` if agent reaches target | +10.0 | Sparse: reward success |
| `r_death` | `+penalty` if HP ≤ 0 | -5.0 | Sparse: penalise dying |
| `r_time` | `penalty` (constant per move) | -0.01 | Dense: discourage dawdling |
| `r_hp` | `-coef × max(threshold - HP, 0)` | coef=0.01, thresh=30 | Mild: penalise dangerously low HP |

**Total**: `r_dist + r_reach + r_death + r_time - r_hp`
