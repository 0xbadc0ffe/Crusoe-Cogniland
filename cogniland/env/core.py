"""Pure-function environment step logic.

Every function takes state in, returns new state out.
No `self`, no mutation, no `copy.deepcopy`.
This module is the primary target for a future JAX migration:
swap `torch.*` -> `jnp.*` and these functions become jit-compilable.
"""

from __future__ import annotations

import functools
import math

import torch

from cogniland.env.constants import ACTION_DELTAS
from cogniland.env.types import CompiledTerrainData, EnvConfig, EnvState, RewardConfig, StepResult


# ---------------------------------------------------------------------------
# Top-level step
# ---------------------------------------------------------------------------

def env_step(
    state: EnvState,
    action: torch.Tensor,
    world_map: torch.Tensor,
    target_pos: torch.Tensor,
    config: EnvConfig,
    compiled: CompiledTerrainData,
    cost_to_go_maps: torch.Tensor | None = None,
    compass_noise_deg: float = 0.0,
) -> StepResult:
    """Execute one batched step.  Pure function — no side effects.

    world_map: either [H, W] (shared) or [B, H, W] (per-env Level Replay).
    cost_to_go_maps: [B, H, W] reverse-Dijkstra maps for reward progress signal.
    """

    old_terrain = state.terrain_idx.clone()  # needed for land-to-water transition

    # 1. Save old cost-to-go for progress signal (J_t)
    old_ctg = state.cost_to_go.clone()

    # 2. Movement
    new_state = apply_movement(state, action, config.size)

    # 3. Compass update — unit direction (pos − target), magnitude dropped
    compass_raw = (target_pos - new_state.position).float()           # [B, 2] — points toward target
    compass_euclidean = torch.norm(compass_raw, dim=1, keepdim=True).clamp(min=1e-8)
    compass_unit = compass_raw / compass_euclidean                     # [B, 2]
    if compass_noise_deg > 0.0:
        max_rad = compass_noise_deg * math.pi / 180.0
        theta = (torch.rand(compass_unit.shape[0], device=compass_unit.device) * 2.0 - 1.0) * max_rad
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x = compass_unit[:, 0] * cos_t - compass_unit[:, 1] * sin_t
        y = compass_unit[:, 0] * sin_t + compass_unit[:, 1] * cos_t
        compass_unit = torch.stack([x, y], dim=1)
    new_state = new_state._replace(compass=compass_unit)

    # 4. Terrain level (needed by minimap visibility)
    terrain_idx = compute_terrain_levels(world_map, new_state.position, compiled)
    new_state = new_state._replace(terrain_idx=terrain_idx)

    # 5. Minimap update
    minimap = compute_minimap_batch(
        world_map, new_state.position, config.minimap_max_ray,
        terrain_idx, config.minimap_occlude, config.minimap_clear_tolerance,
        compiled, target_pos=target_pos,
    )
    new_state = new_state._replace(minimap=minimap)

    # 6. Movement costs & terrain effects
    new_state = apply_movement_costs(new_state, action, config, compiled)
    new_state = apply_terrain_effects(new_state, old_terrain, action, config, compiled)

    # 7. Clamp
    hp = torch.clamp(new_state.hp, 0.0, config.max_hp)
    resources = torch.clamp(new_state.resources, 0.0, config.max_resources)
    new_state = new_state._replace(hp=hp, resources=resources)

    # 8. Update cost-to-go from maps
    if cost_to_go_maps is not None:
        device = new_state.position.device
        b_idx = torch.arange(new_state.position.shape[0], device=device)
        new_ctg = cost_to_go_maps[b_idx, new_state.position[:, 0], new_state.position[:, 1]]
        new_state = new_state._replace(cost_to_go=new_ctg)

    # 9. Terminal conditions
    alive = new_state.hp > 0
    dist_to_target = (new_state.position - target_pos).float().abs().sum(dim=1)
    reached = dist_to_target < 1.0
    done = ~alive | reached

    # 10. Cost-to-go progress: J_t - J_{t+1}
    ctg_delta = old_ctg - new_state.cost_to_go

    # 11. Reward
    reward = compute_reward(
        ctg_delta=ctg_delta,
        cost=new_state.cost,
        dijkstra_cost=new_state.dijkstra_cost,
        alive=alive,
        reached=reached,
        rw=config.reward,
    )

    info = {
        "alive": alive,
        "reached": reached,
        "dist_to_target": dist_to_target,
    }
    return StepResult(state=new_state, reward=reward, done=done, info=info)


# ---------------------------------------------------------------------------
# Reward (pure function — part of the environment specification)
# ---------------------------------------------------------------------------

def compute_reward(
    ctg_delta: torch.Tensor,
    cost: torch.Tensor,
    dijkstra_cost: torch.Tensor,
    alive: torch.Tensor,
    reached: torch.Tensor,
    rw: RewardConfig,
) -> torch.Tensor:
    """Compute shaped reward. Pure function, no side effects.

    r_t = λ_p (J_t − J_{t+1})                           # cost-to-go progress
        − λ_s                                             # per-step penalty
        + 1_reached · (r_success + λ_t · time*/time)    # success reward with time bonus
        − 1_dead    · λ_d · r_success                   # death penalty

    time*/time ∈ [0,1]: Dijkstra optimal cost / actual agent cost
    """
    device = alive.device

    # Progress: reward proportional to decrease in cost-to-go
    r_progress = rw.lambda_p * ctg_delta

    # Per-step penalty
    r_step = -rw.lambda_s

    # Time-efficiency ratio: optimal time / actual time, clamped to [0, 1]
    time_ratio = torch.clamp(dijkstra_cost / (cost + 1e-6), 0.0, 1.0)

    r_success = torch.where(
        reached,
        torch.tensor(rw.reach_bonus, device=device) + rw.lambda_t * time_ratio,
        torch.zeros(1, device=device),
    )
    r_death = torch.where(
        ~alive,
        torch.tensor(-rw.lambda_d * rw.reach_bonus, device=device),
        torch.zeros(1, device=device),
    )

    return r_progress + r_step + r_success + r_death


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

def compute_terrain_levels(
    world_map: torch.Tensor,
    positions: torch.Tensor,
    compiled: CompiledTerrainData,
) -> torch.Tensor:
    """Vectorised terrain-level lookup using searchsorted.

    Args:
        world_map: [H, W] shared or [B, H, W] per-env heightmap.
        positions: [B, 2] (row, col) positions.
        compiled: pre-built terrain tensors.

    Returns: [B] float terrain level indices (0..N-1).
    """
    device = positions.device
    thresholds = compiled.thresholds.to(device)
    if world_map.dim() == 3:
        # Per-env maps: index [b, row, col]
        b_idx = torch.arange(positions.shape[0], device=device)
        height_values = world_map[b_idx, positions[:, 0], positions[:, 1]]
    else:
        # Shared map
        height_values = world_map[positions[:, 0], positions[:, 1]]
    levels = torch.searchsorted(thresholds, height_values)
    levels = torch.clamp(levels, 0, compiled.num_terrains - 1).float()
    return levels



# ---------------------------------------------------------------------------
# Movement costs
# ---------------------------------------------------------------------------

def apply_movement_costs(
    state: EnvState, action: torch.Tensor,
    config: EnvConfig, compiled: CompiledTerrainData,
) -> EnvState:
    """Apply base movement costs based on terrain (vectorised)."""
    device = state.position.device
    costs = compiled.move_costs.to(device)
    terrain_idx = state.terrain_idx.long()
    step_cost = costs[terrain_idx]  # [B]

    return state._replace(cost=state.cost + step_cost)


# ---------------------------------------------------------------------------
# Terrain effects
# ---------------------------------------------------------------------------

def apply_terrain_effects(
    state: EnvState, old_terrain: torch.Tensor,
    action: torch.Tensor, config: EnvConfig,
    compiled: CompiledTerrainData,
) -> EnvState:
    """Apply forest, sea, mountain, and hard-mode effects (vectorised)."""
    device = state.position.device
    terrain = state.terrain_idx
    hp = state.hp.clone()
    resources = state.resources.clone()

    # --- Per-terrain resource drain (negative res_rate) ---
    res_rate_table = compiled.res_rate.to(device)
    res_rate = res_rate_table[terrain.long()]             # [B], negative = drain
    drain = (-res_rate).clamp(min=0)                     # positive drain amount
    actual_drain = torch.min(resources, drain)
    resources = resources - actual_drain
    hp = hp - (drain - actual_drain) * config.agent.no_res_hp_multiplier

    # --- Forest: HP-first priority mechanic (positive res_rate / hp_rate) ---
    is_forest = compiled.is_forest.to(device)
    forest = is_forest[terrain.long()]
    hp_rate_table  = compiled.hp_rate.to(device)
    res_rate_gain  = res_rate.clamp(min=0)               # positive gain for forest

    at_max_hp = hp >= config.max_hp
    # Heal if below max HP
    hp = hp + forest.float() * (~at_max_hp).float() * hp_rate_table[terrain.long()]
    # Only collect resources when at full HP
    resources = resources + forest.float() * at_max_hp.float() * res_rate_gain

    # --- Land-to-water transition: costs resources; shortfall converts to HP ---
    is_water = compiled.is_water.to(device)
    old_is_water = is_water[old_terrain.long()]
    new_is_water = is_water[terrain.long()]
    land_to_water = (~old_is_water) & new_is_water

    resources_available = torch.clamp(resources, 0.0, config.agent.land_to_water_resource_cost)
    resources_missing = config.agent.land_to_water_resource_cost - resources_available
    resources = resources - land_to_water.float() * resources_available
    hp = hp - land_to_water.float() * resources_missing * config.agent.no_res_hp_multiplier

    return state._replace(hp=hp, resources=resources)


# ---------------------------------------------------------------------------
# Minimap (batched)
# ---------------------------------------------------------------------------

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


def compute_occlusion_mask_batch(patches: torch.Tensor, max_ray: int, clear_tolerance: float) -> torch.Tensor:
    """Compute binary visibility mask in batch using raycasting from the center.

    Args:
        patches: [B, D, D] heightmap patches centered on the agents.
        max_ray: radius of the patches.
        clear_tolerance: max height difference above the agent before vision is blocked.

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

    # Get the height of the agent at the center of each patch
    center_heights = patches[:, max_ray, max_ray]  # [B]

    # Find blocking cells: Blocked if the cell is taller than the agent + tolerance
    blocks = (ray_heights >= (center_heights.unsqueeze(1).unsqueeze(2) + clear_tolerance)).float()

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
    terrain_indices: torch.Tensor,
    occlude: bool,
    min_clear_lv: float,
    compiled: CompiledTerrainData,
    target_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute minimap with terrain-dependent visibility for a batch of positions.

    Args:
        world_map: [H, W] shared or [B, H, W] per-env heightmap.
        compiled: pre-built terrain tensors (visibility table).
        target_pos: optional [B, 2] int tensor of target positions.

    Returns: [B, 3, 2*max_ray+1, 2*max_ray+1] channel-first float tensor.
        Channel 0 = heightmap values (zero outside visibility circle)
        Channel 1 = target indicator (1.0 if target cell is in view, gated by visibility)
        Channel 2 = binary visibility mask (1.0 inside, 0.0 outside)
    """
    B = positions.shape[0]
    per_env = world_map.dim() == 3
    size = world_map.shape[-1]  # works for both [H, W] and [B, H, W]
    diameter = 2 * max_ray + 1
    device = positions.device

    # 1. Vectorized patch extraction via padding + unfold-style gather
    #    Pad the world map(s) by max_ray on each side so every position can
    #    be gathered without boundary checks.
    if per_env:
        # [B, H, W] → [B, H+2*max_ray, W+2*max_ray]
        padded = torch.nn.functional.pad(world_map, (max_ray, max_ray, max_ray, max_ray), value=0.0)
    else:
        # [H, W] → expand to [B, H+2*max_ray, W+2*max_ray]
        padded = torch.nn.functional.pad(world_map, (max_ray, max_ray, max_ray, max_ray), value=0.0)
        padded = padded.unsqueeze(0).expand(B, -1, -1)

    # Position in padded coordinates (original pos + max_ray is the center)
    # Top-left corner of the diameter×diameter patch = original position (no offset needed
    # because padding shifted everything by max_ray).
    cy = positions[:, 0]  # [B]
    cx = positions[:, 1]  # [B]

    # Build gather indices: for each batch element, a diameter×diameter grid of row/col offsets
    offsets = torch.arange(diameter, device=device)  # [D]
    # Row indices: cy[b] + offsets[i] for each b,i  → [B, D, 1] + [1, D, 1] broadcast
    row_idx = (cy.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(2).expand(B, diameter, diameter)  # [B, D, D]
    col_idx = (cx.unsqueeze(1) + offsets.unsqueeze(0)).unsqueeze(1).expand(B, diameter, diameter)  # [B, D, D]

    # Flatten spatial dims for gather, then reshape back
    padded_W = size + 2 * max_ray
    flat_idx = row_idx * padded_W + col_idx  # [B, D, D]
    padded_flat = padded.reshape(B, -1)  # [B, H_pad * W_pad]
    patches = torch.gather(padded_flat, 1, flat_idx.reshape(B, -1)).reshape(B, diameter, diameter)

    # Pre-compute distance grid from center
    coords = torch.arange(diameter, device=device).float() - max_ray
    dy_grid, dx_grid = torch.meshgrid(coords, coords, indexing="ij")
    dist_grid = torch.sqrt(dy_grid ** 2 + dx_grid ** 2)  # [D, D]

    # Batch distance visibility mask
    vis_radii = compiled.visibility.to(device)[terrain_indices.long()]  # [B]
    dist_masks = (dist_grid.unsqueeze(0) <= vis_radii.view(B, 1, 1)).float()  # [B, D, D]

    # Batch occlusion mask
    if occlude:
        occ_masks = compute_occlusion_mask_batch(patches, max_ray, min_clear_lv)
        final_masks = dist_masks * occ_masks
    else:
        final_masks = dist_masks

    # 2. Vectorized target indicator (3×3 block around target if within patch)
    target_mask = torch.zeros(B, diameter, diameter, device=device)
    if target_pos is not None:
        # Offset of target relative to patch top-left corner
        patch_y = target_pos[:, 0] - cy + max_ray  # [B]
        patch_x = target_pos[:, 1] - cx + max_ray  # [B]

        # Which envs have the target within the patch?
        in_view = (patch_y >= 0) & (patch_y < diameter) & (patch_x >= 0) & (patch_x < diameter)

        if in_view.any():
            # Build a 3×3 block around each visible target using scatter
            dy_offsets = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=device)
            dx_offsets = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=device)

            # Indices of envs with visible target
            view_idx = in_view.nonzero(as_tuple=True)[0]  # [V]
            vy = patch_y[view_idx].unsqueeze(1) + dy_offsets.unsqueeze(0)  # [V, 9]
            vx = patch_x[view_idx].unsqueeze(1) + dx_offsets.unsqueeze(0)  # [V, 9]

            # Clamp to patch bounds
            vy = vy.clamp(0, diameter - 1).long()
            vx = vx.clamp(0, diameter - 1).long()

            # Scatter 1.0 into target_mask
            # Expand batch indices: [V, 9]
            bi = view_idx.unsqueeze(1).expand_as(vy)
            target_mask[bi, vy, vx] = 1.0

    # Combine
    maps = torch.zeros(B, 3, diameter, diameter, device=device)
    maps[:, 0] = patches * final_masks
    maps[:, 1] = target_mask * final_masks  # target indicator, gated by visibility
    maps[:, 2] = final_masks

    return maps
