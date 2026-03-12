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
    """Execute one batched step.  Pure function — no side effects.

    world_map: either [H, W] (shared) or [B, H, W] (per-env Level Replay).
    """
    from cogniland.env.reward import compute_reward

    old_terrain = state.terrain_lev.clone()
    prev_dist = torch.norm((state.position - target_pos).float(), dim=1)

    # 1. Movement
    new_state = apply_movement(state, action, config.size)

    # 2. Compass update — unit direction (pos − target), magnitude dropped
    compass_raw = (new_state.position - target_pos).float()           # [B, 2]
    compass_dist = torch.norm(compass_raw, dim=1, keepdim=True).clamp(min=1e-8)
    compass_unit = compass_raw / compass_dist                          # [B, 2]
    new_state = new_state._replace(compass=compass_unit)

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
    new_state = apply_movement_costs(new_state, action, config)
    new_state = apply_terrain_effects(new_state, old_terrain, action, config)

    # 8. Clamp
    hp = torch.clamp(new_state.hp, 0.0, config.max_hp)
    resources = torch.clamp(new_state.resources, 0.0, config.max_resources)
    new_state = new_state._replace(hp=hp, resources=resources)

    # 9. Terminal conditions
    alive = new_state.hp > 0
    dist_to_target = compass_dist.squeeze(1)
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

    Args:
        world_map: [H, W] shared or [B, H, W] per-env heightmap.
        positions: [B, 2] (row, col) positions.

    Returns: [B] float terrain level indices (0-8).
    """
    device = positions.device
    thresholds = TERRAIN_THRESHOLDS.to(device)
    if world_map.dim() == 3:
        # Per-env maps: index [b, row, col]
        b_idx = torch.arange(positions.shape[0], device=device)
        height_values = world_map[b_idx, positions[:, 0], positions[:, 1]]
    else:
        # Shared map
        height_values = world_map[positions[:, 0], positions[:, 1]]
    levels = torch.searchsorted(thresholds, height_values)
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
    state: EnvState, action: torch.Tensor, config: EnvConfig
) -> EnvState:
    """Apply base movement costs based on terrain (vectorised)."""
    device = state.position.device
    moving = action != ACTIONS["stay"]
    costs = TERRAIN_COSTS.to(device)
    terrain_idx = state.terrain_lev.long()
    step_cost = costs[terrain_idx]  # [B]
    step_cost = step_cost * moving.float()

    return state._replace(cost=state.cost + step_cost)


# ---------------------------------------------------------------------------
# Terrain effects
# ---------------------------------------------------------------------------

def apply_terrain_effects(
    state: EnvState, old_terrain: torch.Tensor, action: torch.Tensor, config: EnvConfig
) -> EnvState:
    """Apply forest, sea, mountain, and hard-mode effects (vectorised)."""
    device = state.position.device
    terrain = state.terrain_lev
    hp = state.hp.clone()
    resources = state.resources.clone()

    # --- Per-terrain resource drain (every step, no free window) ---
    # 0=ocean, 1=deep_water, 2=water, 3=beach, 4=sandy, 5=grassland,
    # 6=forest (handled separately), 7=rocky, 8=mountains
    terrain_res_costs = torch.tensor([
        config.sea_resource_costs[0],       # ocean
        config.sea_resource_costs[1],       # deep_water
        config.sea_resource_costs[2],       # water
        config.land_resource_drain,         # beach
        config.land_resource_drain,         # sandy
        config.land_resource_drain,         # grassland
        0.0,                               # forest
        config.mountain_resource_costs[0], # rocky
        config.mountain_resource_costs[1], # mountains
    ], device=device)
    res_drain = terrain_res_costs[terrain.long()]       # [B]
    actual_drain = torch.min(resources, res_drain)
    resources = resources - actual_drain
    hp = hp - (res_drain - actual_drain) * config.no_res_hp_multiplier

    # --- Forest: HP-first priority mechanic ---
    forest = terrain == 6
    at_max_hp = hp >= config.max_hp
    # Heal if below max HP
    hp = hp + forest.float() * (~at_max_hp).float() * config.forest_hp_gain
    # Only collect resources when at full HP
    resources = resources + forest.float() * at_max_hp.float() * config.forest_resource_gain

    # --- Land-to-water transition: costs resources; each missing resource = HP loss ---
    moving = action != ACTIONS["stay"]
    land_to_water = (old_terrain > 2) & (terrain <= 2) & moving
    resources_available = torch.clamp(resources, 0.0, config.land_to_water_resource_cost)
    resources_missing = config.land_to_water_resource_cost - resources_available
    resources = resources - land_to_water.float() * resources_available
    hp = hp - land_to_water.float() * resources_missing * config.land_to_water_hp_per_missing_res

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


def compute_occlusion_mask_batch(patches: torch.Tensor, max_ray: int, min_clear_lv: float) -> torch.Tensor:
    """Compute binary visibility mask in batch using raycasting from the center.
    
    Args:
        patches: [B, D, D] heightmap patches centered on the agents.
        max_ray: radius of the patches.
        min_clear_lv: height threshold that blocks vision.
        
    Returns:
        masks: [B, D, D] float tensor (1.0 = visible, 0.0 = occluded).
    """
    B, D, _ = patches.shape
    device = patches.device
    
    rays, lengths = _bresenham_rays(max_ray)
    rays = rays.to(device)
    lengths = lengths.to(device)
    
    num_rays, max_len, _ = rays.shape
    
    # Global indices for rays relative to patch top-left
    ray_y = max_ray + rays[..., 0]  # [num_rays, max_len]
    ray_x = max_ray + rays[..., 1]  # [num_rays, max_len]
    
    # Gather heights for all patches along all rays
    ray_heights = patches[:, ray_y, ray_x]  # [B, num_rays, max_len]
    
    # Find blocking cells
    blocks = (ray_heights >= min_clear_lv).float()
    
    # Cells *after* the first block are occluded.
    # cummax creates a mask of 1s starting from the first block.
    # By shifting right, the blocking cell itself remains 0 (visible).
    is_blocked = blocks.cummax(dim=2)[0]
    occluded = torch.cat([torch.zeros(B, num_rays, 1, device=device), is_blocked[:, :, :-1]], dim=2)
    
    # Mask to ignore padding in the ray sequences
    valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # We want final mask to be 1.0 (visible) by default, and set to 0.0 if occluded
    # We take the minimum visibility over all rays that visit a cell
    visible = 1.0 - occluded
    
    # Flatten the ray coordinates for scatter_reduce
    flat_y = ray_y.flatten()
    flat_x = ray_x.flatten()
    flat_indices = flat_y * D + flat_x # [num_rays * max_len]
    flat_indices = flat_indices.unsqueeze(0).expand(B, -1) # [B, num_rays * max_len]
    
    flat_visible = visible.reshape(B, -1) # [B, num_rays * max_len]
    flat_valid = valid_mask.reshape(1, -1).expand(B, -1)
    
    # Set invalid elements to 1.0 so min reduction ignores them
    flat_visible = torch.where(flat_valid, flat_visible, torch.ones_like(flat_visible))
    
    final_masks = torch.ones(B, D * D, device=device)
    final_masks.scatter_reduce_(1, flat_indices, flat_visible, reduce="amin", include_self=False)
    
    return final_masks.view(B, D, D)


def compute_minimap_batch(
    world_map: torch.Tensor,
    positions: torch.Tensor,
    max_ray: int,
    terrain_levels: torch.Tensor,
    occlude: bool,
    min_clear_lv: float,
) -> torch.Tensor:
    """Compute minimap with terrain-dependent visibility for a batch of positions.

    Args:
        world_map: [H, W] shared or [B, H, W] per-env heightmap.

    Returns: [B, 2, 2*max_ray+1, 2*max_ray+1] channel-first float tensor.
        Channel 0 = heightmap values (zero outside visibility circle)
        Channel 1 = binary visibility mask (1.0 inside, 0.0 outside)
    """
    B = positions.shape[0]
    per_env = world_map.dim() == 3
    size = world_map.shape[-1]  # works for both [H, W] and [B, H, W]
    diameter = 2 * max_ray + 1
    device = positions.device

    maps = torch.zeros(B, 2, diameter, diameter, device=device)
    patches = torch.zeros(B, diameter, diameter, device=device)

    # 1. Slice out patches sequentially
    for b in range(B):
        wm = world_map[b] if per_env else world_map  # [H, W]
        cy, cx = positions[b, 0].item(), positions[b, 1].item()

        y0, y1 = cy - max_ray, cy + max_ray + 1
        x0, x1 = cx - max_ray, cx + max_ray + 1

        sy0 = max(y0, 0)
        sy1 = min(y1, size)
        sx0 = max(x0, 0)
        sx1 = min(x1, size)

        dy0 = sy0 - y0
        dy1 = dy0 + (sy1 - sy0)
        dx0 = sx0 - x0
        dx1 = dx0 + (sx1 - sx0)

        if sy0 < sy1 and sx0 < sx1:
            patches[b, dy0:dy1, dx0:dx1] = wm[sy0:sy1, sx0:sx1]

    # Pre-compute distance grid from center
    coords = torch.arange(diameter, device=device).float() - max_ray
    dy_grid, dx_grid = torch.meshgrid(coords, coords, indexing="ij")
    dist_grid = torch.sqrt(dy_grid ** 2 + dx_grid ** 2)  # [D, D]

    # Batch distance visibility mask
    vis_radii = TERRAIN_VISIBILITY.to(device)[terrain_levels.long()]  # [B]
    dist_masks = (dist_grid.unsqueeze(0) <= vis_radii.view(B, 1, 1)).float()  # [B, D, D]

    # Batch occlusion mask
    if occlude:
        occ_masks = compute_occlusion_mask_batch(patches, max_ray, min_clear_lv)
        final_masks = dist_masks * occ_masks
    else:
        final_masks = dist_masks

    # Combine
    maps[:, 0] = patches * final_masks
    maps[:, 1] = final_masks

    return maps
