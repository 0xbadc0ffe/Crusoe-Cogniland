# Cogniland Environment

Cogniland is a navigation game built on procedurally generated island maps. An agent starts at a random spawn point and must reach a target position while keeping its HP above zero.

Maps are 250×250 grids generated via simplex noise.

---

## Example Map with a trajectory

<p align="center">
  <img src="../assets/images/trajectory_example.png" alt="Example Cogniland Map with Agent Trajectory" width="300"/>
</p>


---

## MDP Formulation

| Component | Description |
|-----------|----------------------------|
| **State space** | `minimap` [45×45, 2 ch] + `scalars` [6] — see observation table below |
| **Action space** | 5 discrete actions: up, down, left, right, stay |
| **Transition** | Deterministic grid movement; terrain type at destination determines HP/resource effects |
| **Reward** | Dense approach reward + sparse reach bonus/death penalty + shaping terms — see reward table below |
| **Termination** | Agent reaches target (success), HP ≤ 0 (death), or episode length > 1000 (timeout) |

---

## Observation Space

| Field | Shape | Description |
|-------|-------|-------------------------|
| `minimap` channel 0 | [45, 45] | Heightmap patch centred on the agent |
| `minimap` channel 1 | [45, 45] | Visibility mask (1 = observed, 0 = unseen) |
| `compass_x`, `compass_y` | scalar × 2 | Unit vector pointing toward target |
| `terrain_idx` | scalar | Current terrain index, normalised to [0, 1] |
| `resources` | scalar | Current resources / max resources |
| `hp` | scalar | Current HP / max HP |
| `visibility_range` | scalar | Current visibility radius / max radius |

---

## Action Space

| ID | Name| Effect |
|----|-----|--------|
| 0 | up   | Move one cell north |
| 1 | down | Move one cell south |
| 2 | right| Move one cell east |
| 3 | left | Move one cell west |
| 4 | stay | Remain in place; incurs terrain movement cost (time) and all terrain effects |

---

## Terrain Types

| ID | Name | Mov. cost | Visibility radius | Resource effect (per step) |
|----|-----------|-----------|-------------------|-------------------------------------|
| 0 | ocean      | 0.5 | 10 | −3.0 resources |
| 1 | deep_water | 0.75 | 8 | −2.0 resources |
| 2 | water      | 1.0 | 6 | −1.5 resources |
| 3 | beach      | 1.5 | 4 | −1.0 resources |
| 4 | sandy      | 2.0 | 4 | −1.0 resources |
| 5 | grassland  | 1.5 | 4 | −1.0 resources |
| 6 | forest     | 3.5 | 4 | +5 HP/step (until max HP), then +2 resources/step; no resource drain   |
| 7 | rocky      | 3.5 | 8 | −1.5 resources |
| 8 | mountains  | 4.0 | 22 | −3.0 resources |

When resources reach 0, the deficit is converted to HP loss (×2 multiplier by default).
Crossing from land to water costs 10 resources (boat construction); shortfall is deducted from HP at 5 HP per missing resource unit.

---

## Reward Function

| Term | Formula | Notes |
|------|-------------------|--------------|
| Approach | `(prev_dist − dist) × dist_coef` | Dense; `dist_coef = 0.35` |
| Reach bonus | `+12.0` | Sparse; on reaching target |
| Death penalty | `−8.0` | Sparse; on HP ≤ 0 |
| Time penalty | `−0.1` per step | Encourages efficiency |
| Low-HP shaping | `−0.05 × max(0, 35 − HP)` | Penalises being below HP threshold |
| Low-resource shaping | `−0.02 × max(0, 25 − resources)` | Penalises low resources |