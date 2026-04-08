# Environment Specification

Cogniland is a navigation task on procedurally generated 250x250 island maps. An agent starts at a random spawn and must reach a target while managing HP and resources. Terrain imposes movement costs, resource drains, visibility constraints, and strategic opportunities (forest healing, mountain scouting, water shortcuts).

Cross-references use `file.py:line` format.

---

## MDP formulation

| Component | Description |
|-----------|-------------|
| **State space** | `minimap` [3, 45, 45] + `scalars` [5] — see [Observation space](#observation-space) |
| **Action space** | 5 discrete: up, down, right, left, forage |
| **Transition** | Deterministic 1-cell movement (forage = stay in place); terrain at destination determines effects |
| **Reward** | Dense cost-to-go progress + sparse success/death — see [Reward function](#reward-function) |
| **Termination** | Reached target (success), HP <= 0 (death), or steps >= 1000 (timeout) |

---

## Observation space

Built in `wrappers.py:96` from `EnvState`.

| Field | Shape | Description | Source |
|-------|-------|-------------|--------|
| `minimap` ch 0 | [45, 45] | Heightmap patch centred on agent, gated by visibility | `core.py:499` |
| `minimap` ch 1 | [45, 45] | Target indicator (1.0 at target cell if in view) | `core.py:500` |
| `minimap` ch 2 | [45, 45] | Binary visibility mask (1.0 = visible, 0.0 = occluded/out of range) | `core.py:501` |
| `compass_x, compass_y` | scalar x2 | Noisy unit vector pointing toward target | `core.py:48` |
| `terrain_idx` | scalar | Current terrain index / (num_terrains - 1), normalised to [0, 1] | `wrappers.py:108` |
| `resources` | scalar | Current resources / max_resources | `wrappers.py:109` |
| `hp` | scalar | Current HP / max_hp | `wrappers.py:110` |

**Minimap visibility:** The effective radius depends on terrain (see table below). Cells outside the radius show as 0. Within the radius, Bresenham raycasting from the agent's position determines line-of-sight occlusion (`core.py:343`). Mountains behind ridges are hidden from low-visibility terrain.

**Compass noise:** The compass direction is perturbed by uniform random rotation. Noise magnitude depends on curriculum stage: EXTRA_EASY=5deg, EASY=30deg, NORMAL=60deg (`islands.py:22`).

---

## Action space

Defined in `constants.py`.

| ID | Name | Delta (dy, dx) | Effect |
|----|------|----------------|--------|
| 0 | up | (-1, 0) | Move one cell north |
| 1 | down | (+1, 0) | Move one cell south |
| 2 | right | (0, +1) | Move one cell east |
| 3 | left | (0, -1) | Move one cell west |
| 4 | forage | (0, 0) | Stay in place; triggers forest gather if on forest tile |

**Forage action:** On forest tiles, triggers HP-first healing (+5 HP/step until max, then +5 res/step). On any other tile, forage is a wasted action (no movement, still pays terrain resource drain). See `core.py:256`.

---

## Terrain types

Defined in `configs/env/default.yaml` as a YAML list, parsed into `TerrainDef` tuples (`types.py:50`), compiled into tensor LUTs (`CompiledTerrainData`, `types.py:72`).

**Current defaults** (from `default.yaml`):

| ID | Name | Threshold | Move cost | Res rate/step | HP rate/step | Visibility | Tags | Strategic role |
|----|------|-----------|-----------|---------------|-------------|------------|------|----------------|
| 0 | ocean | 0.007 | 1.0 | -1.0 | 0 | 16 | water | Fast crossing; cheap per-step but boat cost at entry |
| 1 | deep_water | 0.025 | 1.25 | -0.5 | 0 | 12 | water | Moderate crossing |
| 2 | water | 0.05 | 1.5 | -0.2 | 0 | 10 | water | Shallow; cheapest water drain |
| 3 | beach | 0.06 | 1.75 | -1.0 | 0 | 7 | land | Transition zone |
| 4 | sandy | 0.1 | 2.0 | -1.0 | 0 | 7 | land | Slow, expensive; avoid |
| 5 | grassland | 0.25 | 2.25 | -1.0 | 0 | 7 | land | Standard traversal |
| 6 | forest | 0.6 | 3.0 | +5.0 | +5.0 | 5 | land, forest | Resource depot (forage action only) |
| 7 | rocky | 0.7 | 3.5 | -2.0 | 0 | 10 | land | Elevated visibility; high cost |
| 8 | mountains | 1.0 | 4.0 | -5.0 | 0 | 22 | land | Scouting post; best visibility, very expensive |

**How terrain effects work** (`core.py:229`):

1. **Resource drain:** Every step (including forage), the agent pays `|res_rate|` from resources for non-forest terrain. Forest tiles cost 0 resources to be on.
2. **Resource depletion → HP loss:** When resources hit 0, the shortfall is multiplied by `no_res_hp_multiplier` (default 2.0) and deducted from HP. See `core.py:251`.
3. **Forest forage:** Only triggers on `action == 4` (forage) while on a forest tile. Heals HP at +5.0/step until max_hp, then grants resources at +5.0/step. Moving through forest (actions 0-3) has zero drain and zero gain. See `core.py:257-266`.
4. **Land-to-water transition:** Entering any water tile from land costs 20 resources (boat construction). Missing resources convert to HP damage at `no_res_hp_multiplier` rate. See `core.py:269-277`.

**Break-even on water crossing:** Water saves ~0.75-1.25 res/step vs grassland. A crossing must save at least 16-27 land steps to be resource-neutral after the 20-resource boat cost.

---

## Resource and HP system

```
Agent starts with:  hp=100, resources=100          (default.yaml agent.init_hp, agent.init_resources)

Each step on non-forest terrain:
  drain = |res_rate[terrain]|                       (positive amount)
  actual_drain = min(resources, drain)
  resources -= actual_drain
  hp -= (drain - actual_drain) * no_res_hp_multiplier    core.py:250-251

Each forage action on forest:
  if hp < max_hp:
    hp += hp_rate (5.0)                              core.py:264
  else:
    resources += res_rate (5.0)                      core.py:266

Land → water transition (one-time):
  cost = min(resources, 20)
  resources -= cost
  hp -= (20 - cost) * no_res_hp_multiplier           core.py:275-277

Episode ends when hp <= 0 (death).
```

---

## Reward function

Defined in `core.py:121`. All coefficients in `configs/env/default.yaml` under `reward:`.

```
r_t = lambda_p * (J_t - J_{t+1})                  # cost-to-go progress
    - lambda_s                                       # per-step penalty
    + 1_reached * (reach_bonus + lambda_t * time*/time)  # success + time efficiency
    - 1_dead * lambda_d * reach_bonus                # death penalty
```

| Component | Formula | Default | Purpose |
|-----------|---------|---------|---------|
| Progress | `lambda_p * ctg_delta` | lambda_p = **0.3** | Dense: reward for reducing Dijkstra cost-to-go |
| Step penalty | `-lambda_s` | lambda_s = **0.02** | Dense: discourages dawdling |
| Success | `reach_bonus + lambda_t * (dijkstra_cost / agent_cost)` | reach_bonus = **150**, lambda_t = **100** | Sparse: primary goal + time-efficiency bonus |
| Death | `-lambda_d * reach_bonus` | lambda_d = **0.10** | Sparse: `-15` on death |

**Cost-to-go J_t:** Computed via reverse Dijkstra from the target at reset. Edge cost in the reward graph:

```
c(s → s') = max(move_cost(s') - res_rate(s'), 0.1) + beta_raft * 1_{land→water}
```

Where `beta_raft = 20.0` penalises land-to-water transitions. The `max(..., 0.1)` floor prevents negative edge weights from forest tiles (which would break Dijkstra). See `pathfinding.py:90`.

**time\*/time ratio:** `dijkstra_cost / agent_cost`, clamped to [0, 1]. `dijkstra_cost` is the forward Dijkstra optimal traversal cost from spawn to target. `agent_cost` is the accumulated `EnvState.cost` (sum of `move_cost[terrain]` along the agent's actual path). An agent that matches the optimal path gets `time*/time = 1.0`.

---

## Map generation

### Procedural maps

Generated by `islands.py:42` using simplex noise with square island filtering. 250x250 grid, center is always high-elevation land, ocean at perimeter. Params controlled by `map_generation:` in `default.yaml`.

### Dataset splits

Pre-generate with `scripts/generate_dataset.py`:

```bash
python scripts/generate_dataset.py --seed 42 --train 128 --val 16 --test 16
```

Produces `data/train_seed42_n128.pt`, `data/val_seed42_n16.pt`, `data/test_seed42_n16.pt`. Configured in `models/*.yaml` under `training.dataset`.

### Curriculum

Three stages, controlled by `curriculum_switch_steps` and `curriculum_switch_steps_2`:

| Stage | Spawn/target constraint | Compass noise | Triggered at |
|-------|------------------------|--------------|-------------|
| EXTRA_EASY | Within 25-cell radius of map center | 5 deg | Start (if curriculum enabled) |
| EASY | Within 50-cell radius of map center | 30 deg | `curriculum_switch_steps` (default 6M) |
| NORMAL | Uniform over all land cells | 60 deg | `curriculum_switch_steps_2` (default 16M) |

Compass noise increases with difficulty to force the agent to rely on the minimap rather than just following the compass. See `islands.py:22`.

---

## Episode lifecycle

```
reset()                                            islands.py:203
  → sample map from pool (random index)
  → sample spawn + target on land (curriculum-constrained)
  → forward Dijkstra from spawn (optimal traversal time)
  → reverse Dijkstra from target (cost-to-go maps)
  → init hp=100, resources=100, cost=0

for each step:                                     wrappers.py:58
  → agent selects action (0-4)
  → env_step(): movement, terrain effects, reward   core.py:24
  → check termination: hp<=0 (death), dist<1 (reached), steps>=1000 (timeout)

on done:                                           wrappers.py:86
  → auto-reset: new map + new spawn/target + new Dijkstra
  → episode stats (reward, length, reached) captured before reset
```

---

## Configuration reference

All env params in `configs/env/default.yaml`:

| Group | Key params | Defaults |
|-------|-----------|----------|
| Map generation | `size`, `scale`, `octaves`, `seed`, `filtering`, `sink_mode` | 250, 0.33, 6, 42, "square", 1 |
| Agent | `init_hp`, `max_hp`, `init_resources`, `max_resources`, `land_to_water_resource_cost`, `no_res_hp_multiplier` | 100, 100, 100, 100, 20, 2.0 |
| Minimap | `max_ray`, `occlude`, `clear_tolerance` | 22, true, 0.1 |
| Reward | `reach_bonus`, `lambda_p`, `lambda_s`, `lambda_t`, `lambda_d`, `beta_raft` | 150, 0.3, 0.02, 100, 0.10, 20 |
| Episode | `max_steps` | 1000 |
| Terrains | 9-entry list with name, threshold, move_cost, res_rate, hp_rate, visibility, color, tags | See table above |
