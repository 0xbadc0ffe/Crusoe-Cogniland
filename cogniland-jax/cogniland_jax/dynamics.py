"""Pure-JAX step kernels for the Cogniland env.

Every function here is trace-pure: no Python branching on array values,
no side effects. All dynamics are composed inside ``env.step_env`` —
these helpers take / return ``EnvState`` or intermediate arrays only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cogniland_jax import constants as C
from cogniland_jax.state import EnvParams, EnvState


# ── Drain helper ─────────────────────────────────────────────────────────

def drain_for(
    terrain_id: jnp.int32,
    tool: jnp.int32,
    consec_grass: jnp.int32,
) -> jnp.float32:
    """HP drain for stepping on `terrain_id` with `tool` equipped.

    Shoes' grassland override only kicks in once ``consec_grass >= SHOES_K``
    — otherwise grass drains its base 1 HP/step.
    """
    tool_c = jnp.clip(tool, 0, 3)
    terr_c = jnp.clip(terrain_id, 0, C.NUM_TERRAINS - 1)
    base = C.DRAIN_BY_TOOL[tool_c, terr_c]
    shoes_active = (
        (tool == C.TOOL_SHOES)
        & (terrain_id == C.GRASS_IDX)
        & (consec_grass >= C.SHOES_K)
    )
    return jnp.where(shoes_active, C.SHOES_DRAIN_GRASSLAND, base)


# ── Bellman–Ford cost-to-go ─────────────────────────────────────────────

def compute_ctg(
    terrain_idx_map: jnp.ndarray,     # [H, W] int8, -1 = deadly
    berry_mask_map: jnp.ndarray,      # [H, W] bool
    target_r: jnp.int32,
    target_c: jnp.int32,
    num_iters: int = C.MAP_SIZE * 2,
) -> jnp.ndarray:
    """Dijkstra-style cost-to-go on the HP-drain graph.

    We run fixed-iteration Bellman–Ford with 4-connectivity. Edge cost
    entering a cell = hp_drain[dest] (0 for berries, +inf for deadly),
    matching the numpy env's ``_build_cost_graph``. After ~diameter
    iterations the distances converge; we budget 2×MAP_SIZE to be safe.
    """
    H, W = terrain_idx_map.shape
    terrain_c = jnp.clip(terrain_idx_map, 0, C.NUM_TERRAINS - 1).astype(jnp.int32)
    base_cost = C.HP_DRAIN[terrain_c]                       # [H, W]
    cell_cost = jnp.where(berry_mask_map, 0.0, base_cost)
    deadly = terrain_idx_map == -1
    cell_cost = jnp.where(deadly, jnp.inf, cell_cost)

    # Initial dist: 0 at target, inf elsewhere.
    dist = jnp.full((H, W), jnp.inf, dtype=jnp.float32)
    dist = dist.at[target_r, target_c].set(0.0)

    def _relax(_, d):
        up    = jnp.roll(d, +1, axis=0).at[0, :].set(jnp.inf)
        down  = jnp.roll(d, -1, axis=0).at[-1, :].set(jnp.inf)
        left  = jnp.roll(d, +1, axis=1).at[:, 0].set(jnp.inf)
        right = jnp.roll(d, -1, axis=1).at[:, -1].set(jnp.inf)
        neighbour_best = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
        candidate = neighbour_best + cell_cost   # entering ``self`` from a neighbour
        return jnp.minimum(d, candidate)

    dist = jax.lax.fori_loop(0, num_iters, _relax, dist)
    return dist


# ── Water-free 7×7 box check (spawn/target validity) ──────────────────

def has_no_water_in_box(
    terrain_idx_map: jnp.ndarray,    # [H, W] int8
    r: jnp.int32,
    c: jnp.int32,
) -> jnp.bool_:
    """True iff the 7×7 box centred at (r, c) contains no water tile.

    Water is ``terrain_idx <= C.WATER_MAX_IDX``. Cells outside the map
    count as "not water" (they're not checkable; the 1-px deadly border
    already excludes spawns there).
    """
    H, W = terrain_idx_map.shape
    half = C.SPAWN_TARGET_CLEAR_HALF
    # Gather the box via dynamic slice (padded with sentinel 99 so OOB
    # neighbours don't trigger the water check).
    padded = jnp.pad(
        terrain_idx_map.astype(jnp.int32),
        ((half, half), (half, half)),
        mode="constant", constant_values=99,
    )
    box = jax.lax.dynamic_slice(padded, (r, c), (2 * half + 1, 2 * half + 1))
    return jnp.all(box > C.WATER_MAX_IDX)


# ── Spawn/target sampling (hierarchical) ─────────────────────────────

def _sample_target(
    key: jnp.ndarray,
    terrain_idx_map: jnp.ndarray,
) -> tuple[jnp.int32, jnp.int32, jnp.bool_]:
    """Draw one (yes_r, yes_c) candidate and validate it.

    Target is valid iff:
      - YES cell + 6 cells of the YES→NO row segment are land (not water,
        not deadly).
      - 7×7 water-free boxes around both YES and NO.
    """
    H, W = terrain_idx_map.shape
    k_r, k_c = jax.random.split(key)
    yr = jax.random.randint(k_r, (), 1, H - 1)
    yc = jax.random.randint(k_c, (), 1, W - 1 - C.TARGET_GAP)
    nr = yr
    nc = yc + C.TARGET_GAP

    def _cell_land(off):
        return terrain_idx_map[yr, yc + off] > C.WATER_MAX_IDX
    segment_ok = jnp.all(jax.vmap(_cell_land)(jnp.arange(C.TARGET_GAP + 1)))
    yes_box = has_no_water_in_box(terrain_idx_map, yr, yc)
    no_box = has_no_water_in_box(terrain_idx_map, nr, nc)
    ok = segment_ok & yes_box & no_box
    return yr, yc, ok


def _sample_spawn_around(
    key: jnp.ndarray,
    terrain_idx_map: jnp.ndarray,
    mid_r: jnp.int32,
    mid_c: jnp.int32,
    max_euclid: jnp.float32,
) -> tuple[jnp.int32, jnp.int32, jnp.bool_]:
    """Uniform-map spawn draw with a Euclidean-band rejection gate.

    We sample a cell uniformly over the whole 128×128 map (excluding
    the 1-px deadly border), then reject if farther than ``max_euclid``
    from the target midpoint, off-land, or the 7×7 box contains water.
    This keeps candidate density flat across the full map instead of
    piling up in corners when disc sampling clips out-of-bounds draws.
    """
    H, W = terrain_idx_map.shape
    key_r, key_c = jax.random.split(key)
    sr = jax.random.randint(key_r, (), 1, H - 1)
    sc = jax.random.randint(key_c, (), 1, W - 1)
    dr = (sr - mid_r).astype(jnp.float32)
    dc = (sc - mid_c).astype(jnp.float32)
    dist = jnp.sqrt(dr * dr + dc * dc)
    in_range = dist <= max_euclid
    on_land = terrain_idx_map[sr, sc] > C.WATER_MAX_IDX
    box_ok = has_no_water_in_box(terrain_idx_map, sr, sc)
    return sr, sc, in_range & on_land & box_ok


def _search_on_one_map(
    key: jnp.ndarray,
    terrain_idx_map: jnp.ndarray,
    max_euclid: jnp.float32,
) -> tuple[jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.bool_]:
    """Hierarchical search inside one map.

    Outer: up to TARGET_TRIES_PER_MAP target draws. For each valid target,
    inner: up to SPAWN_TRIES_PER_TARGET spawn draws inside the Euclidean
    ball around the target midpoint.

    Returns (sr, sc, yr, yc, nr, nc, found).
    """
    H, W = terrain_idx_map.shape

    def outer_cond(carry):
        t_try, _, found, *_ = carry
        return (~found) & (t_try < C.TARGET_TRIES_PER_MAP)

    def outer_body(carry):
        t_try, key, _, sr, sc, yr, yc, nr, nc = carry
        key, k_tgt = jax.random.split(key)
        yr_c, yc_c, target_ok = _sample_target(k_tgt, terrain_idx_map)
        nr_c = yr_c
        nc_c = yc_c + C.TARGET_GAP
        mid_r = yr_c
        mid_c = yc_c + C.TARGET_GAP // 2

        def inner_cond(icarry):
            s_try, _, sf, *_ = icarry
            return (~sf) & (s_try < C.SPAWN_TRIES_PER_TARGET)

        def inner_body(icarry):
            s_try, ikey, _, isr, isc = icarry
            ikey, k_s = jax.random.split(ikey)
            isr_c, isc_c, spawn_ok = _sample_spawn_around(
                k_s, terrain_idx_map, mid_r, mid_c, max_euclid,
            )
            isr_ = jnp.where(spawn_ok, isr_c, isr)
            isc_ = jnp.where(spawn_ok, isc_c, isc)
            return (s_try + 1, ikey, spawn_ok, isr_, isc_)

        # Only run inner search if target itself is valid. Use cond to
        # preserve a shape-stable carry; invalid target → immediately
        # return spawn_found=False so outer loop advances.
        def _with_valid_target(_):
            init_i = (jnp.int32(0), key, jnp.bool_(False),
                      jnp.int32(1), jnp.int32(1))
            _, _, sf, isr, isc = jax.lax.while_loop(inner_cond, inner_body, init_i)
            return sf, isr, isc

        def _skip(_):
            return jnp.bool_(False), jnp.int32(1), jnp.int32(1)

        spawn_found, isr, isc = jax.lax.cond(
            target_ok, _with_valid_target, _skip, operand=None,
        )
        accepted = target_ok & spawn_found
        # Commit the draw iff this iteration found a valid pair.
        sr_ = jnp.where(accepted, isr, sr)
        sc_ = jnp.where(accepted, isc, sc)
        yr_ = jnp.where(accepted, yr_c, yr)
        yc_ = jnp.where(accepted, yc_c, yc)
        nr_ = jnp.where(accepted, nr_c, nr)
        nc_ = jnp.where(accepted, nc_c, nc)
        return (t_try + 1, key, accepted, sr_, sc_, yr_, yc_, nr_, nc_)

    init = (jnp.int32(0), key, jnp.bool_(False),
            jnp.int32(1), jnp.int32(1),
            jnp.int32(1), jnp.int32(1),
            jnp.int32(1), jnp.int32(C.TARGET_GAP + 1))
    _, _, found, sr, sc, yr, yc, nr, nc = jax.lax.while_loop(
        outer_cond, outer_body, init,
    )
    return sr, sc, yr, yc, nr, nc, found


def _grass_fallback(
    terrain_idx_map: jnp.ndarray,     # [H, W] int8
    key: jnp.ndarray,
) -> tuple[jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32]:
    """Deterministic grassland-tile fallback when rejection exhausts.

    Returns (sr, sc, yr, yc, nr, nc) where spawn = any grassland tile,
    target YES = a different grassland tile (or the same if none other),
    and NO = ``TARGET_GAP`` columns to the right (may not itself be
    grass, but the env stays consistent). We don't enforce the 7×7
    water-free rule here — the point is avoiding deadly / water spawns
    on pathological maps.
    """
    H, W = terrain_idx_map.shape
    is_grass = (terrain_idx_map == C.GRASS_IDX).ravel()
    # If no grass at all (should never happen on real biomes) fall back
    # to the geometric centre.
    any_grass = jnp.any(is_grass)
    first_idx = jnp.argmax(is_grass)              # first grass cell
    last_idx = (is_grass.shape[0] - 1) - jnp.argmax(is_grass[::-1])

    def grass_coords(flat_idx):
        return (flat_idx // W).astype(jnp.int32), (flat_idx % W).astype(jnp.int32)

    sr, sc = grass_coords(first_idx)
    yr, yc = grass_coords(last_idx)
    # If no grass, park everything at the centre.
    centre = jnp.int32(H // 2)
    sr = jnp.where(any_grass, sr, centre)
    sc = jnp.where(any_grass, sc, centre)
    yr = jnp.where(any_grass, yr, centre)
    yc = jnp.where(any_grass, yc, centre - C.TARGET_GAP // 2)
    nr = yr
    nc = jnp.clip(yc + C.TARGET_GAP, 0, W - 1).astype(jnp.int32)
    return sr, sc, yr, yc, nr, nc


def sample_map_and_spawn_target(
    key: jnp.ndarray,
    terrain_idx: jnp.ndarray,        # [N, H, W] int8
    max_euclid: jnp.float32,
) -> tuple[jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32, jnp.int32]:
    """Pick one map, then search (target, spawn) on it.

    One map is drawn uniformly at reset. We then run
    ``_search_on_one_map`` (up to ``TARGET_TRIES_PER_MAP`` targets ×
    ``SPAWN_TRIES_PER_TARGET`` spawns). On total exhaustion we fall
    back to a grassland tile on the same map — every real biome has at
    least one grassland cell, so no map-resample is needed.
    """
    num_maps = terrain_idx.shape[0]
    key, k_map, k_search, k_fb = jax.random.split(key, 4)
    map_idx = jax.random.randint(k_map, (), 0, num_maps)
    map_terrain = terrain_idx[map_idx]

    sr_c, sc_c, yr_c, yc_c, nr_c, nc_c, found = _search_on_one_map(
        k_search, map_terrain, max_euclid,
    )

    fb_sr, fb_sc, fb_yr, fb_yc, fb_nr, fb_nc = _grass_fallback(
        map_terrain, k_fb,
    )
    sr = jnp.where(found, sr_c, fb_sr)
    sc = jnp.where(found, sc_c, fb_sc)
    yr = jnp.where(found, yr_c, fb_yr)
    yc = jnp.where(found, yc_c, fb_yc)
    nr = jnp.where(found, nr_c, fb_nr)
    nc = jnp.where(found, nc_c, fb_nc)
    return map_idx, sr, sc, yr, yc, nr, nc


# ── Step: movement action ────────────────────────────────────────────

def _apply_movement(
    state: EnvState,
    action: jnp.int32,
    terrain_idx_map: jnp.ndarray,
    berry_mask_map: jnp.ndarray,
) -> tuple[EnvState, jnp.bool_, jnp.bool_]:
    """Return (new_state, reached_yes, reached_no) after a movement action.

    ``action`` may be non-movement (≥4); we just don't move and don't
    apply drain in that case. Called only for in-bounds positions; the
    caller must guard with ``~state.terminated``.
    """
    is_move = action < 4
    delta = C.MOVE_DELTAS[jnp.clip(action, 0, 3)]

    H = W = C.MAP_SIZE
    new_r = jnp.where(is_move, state.pos_r + delta[0], state.pos_r)
    new_c = jnp.where(is_move, state.pos_c + delta[1], state.pos_c)
    in_bounds = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < W)
    new_r = jnp.where(in_bounds, new_r, state.pos_r)
    new_c = jnp.where(in_bounds, new_c, state.pos_c)

    terrain_here = terrain_idx_map[new_r, new_c]
    is_deadly = is_move & in_bounds & (terrain_here < 0)
    is_berry_here = berry_mask_map[new_r, new_c]

    # Consecutive grass streak (shoes activation).
    on_grass = terrain_here == C.GRASS_IDX
    new_consec = jnp.where(
        is_move & on_grass,
        state.consec_grass + 1,
        jnp.where(is_move, jnp.int32(0), state.consec_grass),
    )
    # Apply drain only on an actual move step and not on a berry tile.
    raw_drain = drain_for(terrain_here, state.tool, new_consec)
    drain = jnp.where(is_move & ~is_berry_here, raw_drain, 0.0)
    # Deadly cell → instant kill.
    drain = jnp.where(is_deadly, C.HP_MAX, drain)

    new_hp = jnp.maximum(state.hp - drain, 0.0)

    reached_yes = is_move & (new_r == state.yes_r) & (new_c == state.yes_c)
    reached_no = is_move & (new_r == state.no_r) & (new_c == state.no_c)

    new_state = state.replace(
        pos_r=new_r, pos_c=new_c,
        hp=new_hp, consec_grass=new_consec,
    )
    return new_state, reached_yes, reached_no


# ── Step: forage action ──────────────────────────────────────────────

def _apply_forage(
    state: EnvState,
    terrain_idx_map: jnp.ndarray,
    berry_mask_map: jnp.ndarray,
) -> EnvState:
    """Forage in place: +100 HP on berry, +wood on forest, else no-op.

    Berries are **non-consumable** (infinite): every forage on a berry
    tile heals to full HP regardless of prior forages. Forest forage
    costs the base drain. Standing on grassland during a forage still
    increments the shoes streak.
    """
    terrain_here = terrain_idx_map[state.pos_r, state.pos_c]
    is_berry = berry_mask_map[state.pos_r, state.pos_c]
    is_forest = ~is_berry & (terrain_here == C.FOREST_IDX)
    on_grass = ~is_berry & (terrain_here == C.GRASS_IDX)

    heal = jnp.where(is_berry, C.BERRY_HEAL, jnp.float32(0.0))
    new_consec = jnp.where(
        on_grass, state.consec_grass + 1, jnp.int32(0),
    )
    raw_drain = drain_for(terrain_here, state.tool, new_consec)
    drain = jnp.where(is_berry, jnp.float32(0.0), raw_drain)

    new_hp = jnp.clip(state.hp + heal - drain, 0.0, C.HP_MAX)
    added_wood = jnp.where(is_forest, C.FOREST_WOOD, jnp.int32(0))
    new_wood = jnp.clip(state.wood + added_wood, 0, C.WOOD_MAX)

    return state.replace(
        hp=new_hp,
        wood=new_wood,
        consec_grass=new_consec,
    )


# ── Step: craft action ───────────────────────────────────────────────

def _apply_craft(
    state: EnvState,
    action: jnp.int32,
    terrain_idx_map: jnp.ndarray,
    berry_mask_map: jnp.ndarray,
) -> tuple[EnvState, jnp.int32]:
    """Craft raft/rope/shoes at the current tile.

    Requires ``state.tool == 0`` and ``wood >= CRAFT_COST``. Also applies
    the standing-tile drain like any step. Returns (new_state, crafted_tool_id).
    """
    tool_id = C.ACTION_TO_TOOL[action]
    can_craft = (state.tool == C.TOOL_NONE) & (state.wood >= C.CRAFT_COST) & (tool_id > 0)
    new_tool = jnp.where(can_craft, tool_id, state.tool)
    new_wood = jnp.where(can_craft, state.wood - C.CRAFT_COST, state.wood)

    # Apply standing-tile drain with the (possibly newly updated) tool.
    terrain_here = terrain_idx_map[state.pos_r, state.pos_c]
    is_berry = berry_mask_map[state.pos_r, state.pos_c]
    on_grass = ~is_berry & (terrain_here == C.GRASS_IDX)
    new_consec = jnp.where(
        on_grass, state.consec_grass + 1, jnp.int32(0),
    )
    raw_drain = drain_for(terrain_here, new_tool, new_consec)
    drain = jnp.where(is_berry, 0.0, raw_drain)
    new_hp = jnp.maximum(state.hp - drain, 0.0)

    crafted = jnp.where(can_craft, tool_id, jnp.int32(0))
    return state.replace(
        tool=new_tool, wood=new_wood,
        hp=new_hp, consec_grass=new_consec,
    ), crafted


# ── One full step (movement / forage / craft dispatch) ──────────────

def env_step_core(
    state: EnvState,
    action: jnp.int32,
    params: EnvParams,
) -> tuple[EnvState, jnp.float32, jnp.bool_, dict]:
    """Core step logic — dispatches on action id, updates state, computes
    reward, returns ``(new_state, reward, done, info)``.

    `info` includes reach flags, HP snapshots, ctg_prev/curr, and crafted
    tool id for downstream loggers.
    """
    terrain_idx_map = params.terrain_idx[state.map_idx]    # [H, W] int8
    berry_mask_map = params.berry_mask[state.map_idx]       # [H, W] bool

    hp_prev = state.hp
    ctg_prev = state.ctg[state.pos_r, state.pos_c]

    is_move = action < 4
    is_forage = action == C.ACTION_FORAGE
    is_craft = action >= 5

    # Movement branch (always compute; we'll blend).
    moved_state, reached_yes, reached_no = _apply_movement(
        state, action, terrain_idx_map, berry_mask_map,
    )
    foraged_state = _apply_forage(state, terrain_idx_map, berry_mask_map)
    crafted_state, crafted_tool = _apply_craft(
        state, action, terrain_idx_map, berry_mask_map,
    )

    pick = lambda a, b, c: jnp.where(is_move, a, jnp.where(is_forage, b, c))
    new_pos_r = pick(moved_state.pos_r, foraged_state.pos_r, crafted_state.pos_r)
    new_pos_c = pick(moved_state.pos_c, foraged_state.pos_c, crafted_state.pos_c)
    new_hp = pick(moved_state.hp, foraged_state.hp, crafted_state.hp)
    new_wood = pick(moved_state.wood, foraged_state.wood, crafted_state.wood)
    new_tool = pick(moved_state.tool, foraged_state.tool, crafted_state.tool)
    new_consec = pick(
        moved_state.consec_grass,
        foraged_state.consec_grass,
        crafted_state.consec_grass,
    )
    reached_yes = reached_yes & is_move
    reached_no = reached_no & is_move
    crafted_tool = jnp.where(is_craft, crafted_tool, jnp.int32(0))

    # Termination signals.
    died = new_hp <= 0.0
    reached = reached_yes | reached_no
    new_steps = state.steps + 1
    truncated = new_steps >= params.max_steps
    done = died | reached | truncated

    ctg_curr = state.ctg[new_pos_r, new_pos_c]

    # ── Base reward (shared by every task) ─────────────────────────────
    r_step = -params.step_penalty
    r_reach = params.reach_bonus * reached.astype(jnp.float32)
    finite = jnp.isfinite(ctg_prev) & jnp.isfinite(ctg_curr)
    progress = jnp.where(finite, ctg_prev - ctg_curr, jnp.float32(0.0))
    r_shape = params.shaping_coef * progress
    r_hp = params.hp_coef * (new_hp - hp_prev)
    r_death = -params.death_penalty * died.astype(jnp.float32)

    # ── Task-specific bonuses ──────────────────────────────────────────
    # Tasks 1-3 (biome classification): bonus on reaching the correct
    # target for the map's biome.  Task 0 and 4-6 set biome_for_task=-1
    # so ``is_cls_task`` is False and r_cls=0.
    biome_id = params.biome_id[state.map_idx]
    biome_for_task = C.TASK_BIOME_FOR_CLS[state.task_id]
    is_cls_task = biome_for_task >= 0
    biome_match = biome_id == biome_for_task
    correct_cls = (reached_yes & biome_match) | (reached_no & ~biome_match)
    r_cls = jnp.where(
        is_cls_task & correct_cls,
        params.correct_answer_bonus,
        jnp.float32(0.0),
    )

    # Tasks 4-6 (craft): bonus on the step the required tool is crafted.
    tool_for_task = C.TASK_TOOL_FOR_CRAFT[state.task_id]
    is_craft_task = tool_for_task > 0
    crafted_right_tool = crafted_tool == tool_for_task
    r_craft = jnp.where(
        is_craft_task & crafted_right_tool,
        params.craft_bonus,
        jnp.float32(0.0),
    )

    reward = r_step + r_reach + r_shape + r_hp + r_death + r_cls + r_craft

    # ── Per-task success flag ──────────────────────────────────────────
    # Task 0     : reached YES or NO at all.
    # Task 1-3   : reached the biome-correct target.
    # Task 4-6   : the required tool is currently held (one-shot craft).
    task_success = jnp.where(
        is_cls_task,
        correct_cls & reached,
        jnp.where(
            is_craft_task,
            new_tool == tool_for_task,
            reached,
        ),
    )

    new_state = state.replace(
        pos_r=new_pos_r, pos_c=new_pos_c,
        hp=new_hp, wood=new_wood, tool=new_tool,
        consec_grass=new_consec,
        steps=new_steps,
        terminated=done,
        last_action=action,
        crafted_this_step=crafted_tool,
    )

    info = {
        "reached": reached,
        "reached_yes": reached_yes,
        "reached_no": reached_no,
        "died": died,
        "truncated": truncated,
        "hp_prev": hp_prev,
        "hp_curr": new_hp,
        "ctg_prev": ctg_prev,
        "ctg_curr": ctg_curr,
        "crafted": crafted_tool,
        "biome_id": biome_id,
        "task_id": state.task_id,
        "task_success": task_success,
    }
    return new_state, reward, done, info
