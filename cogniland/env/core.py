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
    TERRAIN_VISIBILITY,
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

    # 3. Terrain level (needed by minimap visibility)
    terrain_lev = compute_terrain_levels(world_map, new_state.position)
    new_state = new_state._replace(terrain_lev=terrain_lev)

    # 4. Minimap update
    minimap = compute_minimap_batch(
        world_map, new_state.position, config.minimap_max_ray,
        terrain_lev, config.minimap_occlude, config.minimap_min_clear_lv,
    )
    new_state = new_state._replace(minimap=minimap)

    # 5. Terrain clock
    new_state = update_terrain_clock(new_state, old_terrain)

    # 6. Passive healing (easy mode only)
    if not config.hard_mode:
        new_state = new_state._replace(hp=new_state.hp + config.passive_heal_rate)

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
    step_cost = step_cost + land_to_water.float() * config.land_to_water_penalty

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
    resources = resources + forest.float() * config.forest_resource_gain
    hp = hp + forest.float() * config.forest_hp_gain

    # --- Sea effects ---
    water = terrain <= 2
    exceeds_free = (state.terrain_clock > config.max_sea_movement_without_resources) & water

    # Resource cost by water level
    sea_res_cost = torch.zeros_like(resources)
    sea_res_cost = torch.where((terrain == 0) & exceeds_free, torch.full_like(sea_res_cost, config.sea_resource_costs[0]), sea_res_cost)
    sea_res_cost = torch.where((terrain == 1) & exceeds_free, torch.full_like(sea_res_cost, config.sea_resource_costs[1]), sea_res_cost)
    sea_res_cost = torch.where((terrain == 2) & exceeds_free, torch.full_like(sea_res_cost, config.sea_resource_costs[2]), sea_res_cost)
    resources = resources - sea_res_cost

    # HP loss when out of resources at sea
    no_res_at_sea = (resources <= 0) & exceeds_free
    sea_hp_cost = torch.zeros_like(hp)
    sea_hp_cost = torch.where((terrain == 0) & no_res_at_sea, torch.full_like(sea_hp_cost, config.sea_hp_costs[0]), sea_hp_cost)
    sea_hp_cost = torch.where((terrain == 1) & no_res_at_sea, torch.full_like(sea_hp_cost, config.sea_hp_costs[1]), sea_hp_cost)
    sea_hp_cost = torch.where((terrain == 2) & no_res_at_sea, torch.full_like(sea_hp_cost, config.sea_hp_costs[2]), sea_hp_cost)
    hp = hp - sea_hp_cost

    # --- Mountain effects ---
    mtn_res_cost = torch.zeros_like(resources)
    mtn_res_cost = torch.where(terrain == 7, torch.full_like(mtn_res_cost, config.mountain_resource_costs[0]), mtn_res_cost)
    mtn_res_cost = torch.where(terrain == 8, torch.full_like(mtn_res_cost, config.mountain_resource_costs[1]), mtn_res_cost)
    resources = resources - mtn_res_cost

    no_res_mtn = (resources <= 0) & (terrain >= 7)
    mtn_hp_cost = torch.zeros_like(hp)
    mtn_hp_cost = torch.where((terrain == 7) & no_res_mtn, torch.full_like(mtn_hp_cost, config.mountain_hp_costs[0]), mtn_hp_cost)
    mtn_hp_cost = torch.where((terrain == 8) & no_res_mtn, torch.full_like(mtn_hp_cost, config.mountain_hp_costs[1]), mtn_hp_cost)
    hp = hp - mtn_hp_cost

    # --- Hard mode ---
    if config.hard_mode:
        # Only spend 0.25 when agent has at least 0.25; else treat as no resources
        has_res = resources >= config.hard_mode_resource_drain
        resources = torch.where(has_res, resources - config.hard_mode_resource_drain, resources)
        hp = torch.where(has_res, hp + config.hard_mode_hp_gain, hp - config.hard_mode_hp_loss)

    return state._replace(hp=hp, resources=resources)


# ---------------------------------------------------------------------------
# Minimap (batched)
# ---------------------------------------------------------------------------

import functools


@functools.lru_cache(maxsize=None)
def _bresenham_rays(max_ray: int) -> torch.Tensor:
    """Pre-compute Bresenham rays from center to all perimeter cells.
    
    Returns:
        rays: [num_rays, max_len, 2] tensor of (dy, dx) offsets from center.
        lengths: [num_rays] tensor of valid lengths for each ray.
    """
    diameter = 2 * max_ray + 1
    center = max_ray
    
    # Get all perimeter coordinates
    perimeter = []
    for i in range(diameter):
        perimeter.append((0, i))
        perimeter.append((diameter - 1, i))
    for i in range(1, diameter - 1):
        perimeter.append((i, 0))
        perimeter.append((i, diameter - 1))
        
    rays = []
    for (y1, x1) in perimeter:
        # Bresenham from center to (y1, x1)
        y0, x0 = center, center
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        ray = []
        while True:
            ray.append((y0 - center, x0 - center))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        rays.append(ray)
        
    # Pad rays to same length for tensor
    max_len = max(len(r) for r in rays)
    ray_tensor = torch.zeros(len(rays), max_len, 2, dtype=torch.long)
    lengths = torch.tensor([len(r) for r in rays], dtype=torch.long)
    
    for i, r in enumerate(rays):
        for j, (dy, dx) in enumerate(r):
            ray_tensor[i, j, 0] = dy
            ray_tensor[i, j, 1] = dx
            
    return ray_tensor, lengths


def compute_occlusion_mask(patch: torch.Tensor, max_ray: int, min_clear_lv: float) -> torch.Tensor:
    """Compute binary visibility mask using raycasting from the center.
    
    Args:
        patch: [D, D] heightmap patch centered on the agent.
        max_ray: radius of the patch.
        min_clear_lv: height threshold that blocks vision.
        
    Returns:
        mask: [D, D] float tensor (1.0 = visible, 0.0 = occluded).
    """
    diameter = 2 * max_ray + 1
    device = patch.device
    mask = torch.ones(diameter, diameter, device=device)
    
    rays, lengths = _bresenham_rays(max_ray)
    rays = rays.to(device)
    lengths = lengths.to(device)
    
    num_rays = rays.shape[0]
    
    # Trace each ray
    for i in range(num_rays):
        length = lengths[i].item()
        blocked = False
        
        for j in range(1, length):  # skip center (j=0)
            dy = rays[i, j, 0].item()
            dx = rays[i, j, 1].item()
            py, px = max_ray + dy, max_ray + dx
            
            if blocked:
                mask[py, px] = 0.0
            else:
                height = patch[py, px].item()
                if height >= min_clear_lv:
                    blocked = True
                    
    return mask


def compute_minimap_batch(
    world_map: torch.Tensor,
    positions: torch.Tensor,
    max_ray: int,
    terrain_levels: torch.Tensor,
    occlude: bool,
    min_clear_lv: float,
) -> torch.Tensor:
    """Compute minimap with terrain-dependent visibility for a batch of positions.

    Returns: [B, 2, 2*max_ray+1, 2*max_ray+1] channel-first float tensor.
        Channel 0 = heightmap values (zero outside visibility circle)
        Channel 1 = binary visibility mask (1.0 inside, 0.0 outside)
    """
    B = positions.shape[0]
    size = world_map.shape[0]
    diameter = 2 * max_ray + 1
    device = positions.device

    maps = torch.zeros(B, 2, diameter, diameter, device=device)

    # Pre-compute distance grid from center (shared across batch)
    coords = torch.arange(diameter, device=device).float() - max_ray
    dy_grid, dx_grid = torch.meshgrid(coords, coords, indexing="ij")
    dist_grid = torch.sqrt(dy_grid ** 2 + dx_grid ** 2)  # [D, D]

    # Per-agent visibility radii
    vis_radii = TERRAIN_VISIBILITY.to(device)[terrain_levels.long()]  # [B]

    for b in range(B):
        vis_ray = vis_radii[b].item()
        cy, cx = positions[b, 0].item(), positions[b, 1].item()

        # Build visibility mask for this agent (distance-based)
        dist_mask = (dist_grid <= vis_ray).float()  # [D, D]

        # Compute slice bounds (may be out-of-bounds — we'll zero-pad)
        y0, y1 = cy - max_ray, cy + max_ray + 1
        x0, x1 = cx - max_ray, cx + max_ray + 1

        # Source region (clamped to map bounds)
        sy0 = max(y0, 0)
        sy1 = min(y1, size)
        sx0 = max(x0, 0)
        sx1 = min(x1, size)

        # Dest offsets in the output patch
        dy0 = sy0 - y0
        dy1 = dy0 + (sy1 - sy0)
        dx0 = sx0 - x0
        dx1 = dx0 + (sx1 - sx0)

        patch = torch.zeros(diameter, diameter, device=device)
        patch[dy0:dy1, dx0:dx1] = world_map[sy0:sy1, sx0:sx1]

        # Apply occlusion if enabled
        if occlude:
            occ_mask = compute_occlusion_mask(patch, max_ray, min_clear_lv)
            final_mask = dist_mask * occ_mask
        else:
            final_mask = dist_mask

        # Channel 0: heightmap * mask (only visible cells have values)
        maps[b, 0] = patch * final_mask

        # Channel 1: visibility mask
        maps[b, 1] = final_mask

    return maps
