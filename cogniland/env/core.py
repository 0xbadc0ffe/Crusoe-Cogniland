"""Pure-function environment step logic.

Every function takes state in, returns new state out.
No `self`, no mutation, no `copy.deepcopy`.
This module is the primary target for a future JAX migration:
swap `torch.*` -> `jnp.*` and these functions become jit-compilable.
"""

from __future__ import annotations

import torch

from cogniland.env.constants import (
    ACTIONS,
    ACTION_DELTAS,
    TERRAIN_COSTS,
    TERRAIN_THRESHOLDS,
)
from cogniland.env.types import EnvConfig, EnvState, StepResult


# ---------------------------------------------------------------------------
# Top-level step
# ---------------------------------------------------------------------------

def env_step(
    state: EnvState,
    action: torch.Tensor,
    world_map: torch.Tensor,
    target_pos: torch.Tensor,
    config: EnvConfig,
) -> StepResult:
    """Execute one batched step.  Pure function — no side effects."""
    from cogniland.env.reward import compute_reward

    old_terrain = state.terrain_lev.clone()
    prev_dist = torch.norm((state.position - target_pos).float(), dim=1)

    # 1. Movement
    new_state = apply_movement(state, action, config.size)

    # 2. Compass update
    compass = (new_state.position - target_pos).float()
    new_state = new_state._replace(compass=compass)

    # 3. Minimap update
    minimap = compute_minimap_batch(
        world_map, new_state.position, config.minimap_ray,
        config.minimap_occlude, config.minimap_min_clear_lv,
    )
    new_state = new_state._replace(minimap=minimap)

    # 4. Terrain level
    terrain_lev = compute_terrain_levels(world_map, new_state.position)
    new_state = new_state._replace(terrain_lev=terrain_lev)

    # 5. Terrain clock
    new_state = update_terrain_clock(new_state, old_terrain)

    # 6. Passive healing (easy mode only)
    if not config.hard_mode:
        new_state = new_state._replace(hp=new_state.hp + 1.0)

    # 7. Movement costs & terrain effects
    new_state = apply_movement_costs(new_state, old_terrain, action, config)
    new_state = apply_terrain_effects(new_state, action, config)

    # 8. Clamp
    hp = torch.clamp(new_state.hp, 0.0, config.max_hp)
    resources = torch.clamp(new_state.resources, min=0.0)
    new_state = new_state._replace(hp=hp, resources=resources)

    # 9. Terminal conditions
    alive = new_state.hp > 0
    dist_to_target = torch.norm(compass, dim=1)
    reached = dist_to_target < 1.0
    done = ~alive | reached

    # 10. Reward
    reward = compute_reward(new_state, alive, reached, dist_to_target, prev_dist, config)

    info = {
        "alive": alive,
        "reached": reached,
        "dist_to_target": dist_to_target,
    }
    return StepResult(state=new_state, reward=reward, done=done, info=info)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def apply_movement(state: EnvState, action: torch.Tensor, map_size: int) -> EnvState:
    """Apply movement action, clamp to map bounds."""
    device = state.position.device
    deltas = ACTION_DELTAS.to(device)[action]  # [B, 2]
    new_pos = torch.clamp(state.position + deltas, 0, map_size - 1)
    return state._replace(position=new_pos)


# ---------------------------------------------------------------------------
# Terrain queries  (vectorised — no Python loops)
# ---------------------------------------------------------------------------

def compute_terrain_levels(world_map: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Vectorised terrain-level lookup using searchsorted.

    Returns: [B] float terrain level indices (0-8).
    """
    device = positions.device
    thresholds = TERRAIN_THRESHOLDS.to(device)
    height_values = world_map[positions[:, 0], positions[:, 1]]  # [B]
    levels = torch.searchsorted(thresholds, height_values)        # [B]
    levels = torch.clamp(levels, 0, 8).float()
    return levels


def _get_terrain_group(terrain_levels: torch.Tensor) -> torch.Tensor:
    """Group terrain types for clock purposes.

    0 = water (0-2), 1 = flat land (3-5), 2 = forest (6), 3 = rocky/mountain (7-8).
    """
    groups = torch.zeros_like(terrain_levels)
    groups = torch.where(terrain_levels <= 2, torch.zeros_like(groups), groups)
    groups = torch.where((terrain_levels >= 3) & (terrain_levels <= 5), torch.ones_like(groups), groups)
    groups = torch.where(terrain_levels == 6, torch.full_like(groups, 2), groups)
    groups = torch.where(terrain_levels >= 7, torch.full_like(groups, 3), groups)
    return groups


def update_terrain_clock(state: EnvState, old_terrain: torch.Tensor) -> EnvState:
    """Increment clock if same terrain group, else reset to 1."""
    old_group = _get_terrain_group(old_terrain)
    new_group = _get_terrain_group(state.terrain_lev)
    same = old_group == new_group
    clock = torch.where(same, state.terrain_clock + 1, torch.ones_like(state.terrain_clock))
    return state._replace(terrain_clock=clock)


# ---------------------------------------------------------------------------
# Movement costs
# ---------------------------------------------------------------------------

def apply_movement_costs(
    state: EnvState, old_terrain: torch.Tensor, action: torch.Tensor, config: EnvConfig
) -> EnvState:
    """Apply base movement costs based on terrain (vectorised)."""
    device = state.position.device
    moving = action != ACTIONS["stay"]
    costs = TERRAIN_COSTS.to(device)
    terrain_idx = state.terrain_lev.long()
    step_cost = costs[terrain_idx]  # [B]
    step_cost = step_cost * moving.float()

    # Land-to-water transition penalty
    land_to_water = (old_terrain > 2) & (state.terrain_lev <= 2) & moving
    step_cost = step_cost + land_to_water.float() * 3.0

    return state._replace(cost=state.cost + step_cost)


# ---------------------------------------------------------------------------
# Terrain effects
# ---------------------------------------------------------------------------

def apply_terrain_effects(
    state: EnvState, action: torch.Tensor, config: EnvConfig
) -> EnvState:
    """Apply forest, sea, mountain, and hard-mode effects (vectorised)."""
    terrain = state.terrain_lev
    hp = state.hp.clone()
    resources = state.resources.clone()

    # --- Forest: +1 resource, +4 HP ---
    forest = terrain == 6
    resources = resources + forest.float()
    hp = hp + forest.float() * 4.0

    # --- Sea effects ---
    water = terrain <= 2
    exceeds_free = (state.terrain_clock > config.max_sea_movement_without_resources) & water

    # Resource cost by water level
    sea_res_cost = torch.zeros_like(resources)
    sea_res_cost = torch.where((terrain == 0) & exceeds_free, torch.full_like(sea_res_cost, 0.75), sea_res_cost)
    sea_res_cost = torch.where((terrain == 1) & exceeds_free, torch.full_like(sea_res_cost, 0.50), sea_res_cost)
    sea_res_cost = torch.where((terrain == 2) & exceeds_free, torch.full_like(sea_res_cost, 0.25), sea_res_cost)
    resources = resources - sea_res_cost

    # HP loss when out of resources at sea
    no_res_at_sea = (resources <= 0) & exceeds_free
    sea_hp_cost = torch.zeros_like(hp)
    sea_hp_cost = torch.where((terrain == 0) & no_res_at_sea, torch.full_like(sea_hp_cost, 25.0), sea_hp_cost)
    sea_hp_cost = torch.where((terrain == 1) & no_res_at_sea, torch.full_like(sea_hp_cost, 25.0), sea_hp_cost)
    sea_hp_cost = torch.where((terrain == 2) & no_res_at_sea, torch.full_like(sea_hp_cost, 10.0), sea_hp_cost)
    hp = hp - sea_hp_cost

    # --- Mountain effects ---
    mtn_res_cost = torch.zeros_like(resources)
    mtn_res_cost = torch.where(terrain == 7, torch.full_like(mtn_res_cost, 0.25), mtn_res_cost)
    mtn_res_cost = torch.where(terrain == 8, torch.full_like(mtn_res_cost, 0.75), mtn_res_cost)
    resources = resources - mtn_res_cost

    no_res_mtn = (resources <= 0) & (terrain >= 7)
    mtn_hp_cost = torch.zeros_like(hp)
    mtn_hp_cost = torch.where((terrain == 7) & no_res_mtn, torch.full_like(mtn_hp_cost, 5.0), mtn_hp_cost)
    mtn_hp_cost = torch.where((terrain == 8) & no_res_mtn, torch.full_like(mtn_hp_cost, 20.0), mtn_hp_cost)
    hp = hp - mtn_hp_cost

    # --- Hard mode ---
    if config.hard_mode:
        # Only spend 0.25 when agent has at least 0.25; else treat as no resources
        has_res = resources >= 0.25
        resources = torch.where(has_res, resources - 0.25, resources)
        hp = torch.where(has_res, hp + 1.0, hp - 0.5)

    return state._replace(hp=hp, resources=resources)


# ---------------------------------------------------------------------------
# Minimap (batched)
# ---------------------------------------------------------------------------

def compute_minimap_batch(
    world_map: torch.Tensor,
    positions: torch.Tensor,
    ray: int,
    occlude: bool,
    min_clear_lv: float,
) -> torch.Tensor:
    """Compute minimap for a batch of positions.

    Returns: [B, 1, 2*ray+1, 2*ray+1] channel-first float tensor.
    """
    B = positions.shape[0]
    size = world_map.shape[0]
    diameter = 2 * ray + 1
    maps = torch.zeros(B, 1, diameter, diameter, device=positions.device)

    for b in range(B):
        cy, cx = positions[b, 0].item(), positions[b, 1].item()
        # Compute slice bounds (may be out-of-bounds — we'll pad)
        y0, y1 = cy - ray, cy + ray + 1
        x0, x1 = cx - ray, cx + ray + 1

        # Source region (clamped)
        sy0 = max(y0, 0)
        sy1 = min(y1, size)
        sx0 = max(x0, 0)
        sx1 = min(x1, size)

        # Dest offsets in the output patch
        dy0 = sy0 - y0
        dy1 = dy0 + (sy1 - sy0)
        dx0 = sx0 - x0
        dx1 = dx0 + (sx1 - sx0)

        maps[b, 0, dy0:dy1, dx0:dx1] = world_map[sy0:sy1, sx0:sx1]

    return maps
