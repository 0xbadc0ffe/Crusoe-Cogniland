"""Tests for the zebra_nav env.

Contract:
  * mapgen produces 32×32 maps with the right tile vocabulary, exactly
    ``n_stripes`` stripes, a cue per stripe, and a non-obsidian path from
    spawn to target.
  * env mechanics — move faces+steps, blocked moves still face, PLACE only
    affects WATER → WOOD, MINE only affects ROCK → GRASS, **obsidian is
    inviolable**, reaching TARGET pays the reach bonus and terminates.
  * a hand-scripted solver can reach the target through the natural mix of
    mining and bridging (the env is fully solvable end-to-end).
"""
from __future__ import annotations

import numpy as np
import pytest

from cogniland.zebra_nav import ZebraNavEnv, generate_zebra_map, tiles as T
from cogniland.zebra_nav.env import (
    A_DOWN, A_LEFT, A_MINE, A_PLACE, A_RIGHT, A_UP,
    F_DOWN, F_LEFT, F_RIGHT, F_UP,
)
from cogniland.zebra_nav.mapgen import is_reachable


# ─────────────────────────── mapgen contract ─────────────────────────────

ORIENTATIONS = ("diagonal", "vertical")
# 7-wide walls fit 4 across the diagonal's long axis, but only ~3 across the 32
# columns of a vertical map (with grass gaps + clear mid-left/right endpoints).
N_STRIPES = {"diagonal": 4, "vertical": 3}


def _coords(rec):
    """Return (wall_coord_grid, side_coord_grid, mid) for the record's
    orientation: diagonal uses t=r-c / s=r+c; vertical uses c / r."""
    H, W = rec.terrain.shape
    rr = np.arange(H)[:, None]; cc = np.arange(W)[None, :]
    if rec.orientation == "vertical":
        return cc * np.ones_like(rr), rr * np.ones_like(cc), float(H // 2)
    return rr - cc, rr + cc, (H + W - 2) / 2.0


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_map_shape_and_corners(seed: int, orientation: str):
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    assert rec.terrain.shape == (32, 32)
    assert rec.orientation == orientation
    if orientation == "diagonal":
        assert rec.spawn == (31, 0) and rec.target == (0, 31)
    else:
        assert rec.spawn == (16, 0) and rec.target == (16, 31)
    assert rec.terrain[rec.spawn] == T.GRASS
    assert rec.terrain[rec.target] == T.TARGET


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_n_stripes_and_cues(seed: int, orientation: str):
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    n = N_STRIPES[orientation]
    assert len(rec.stripe_centers) == n
    assert len(rec.stripe_thinner) == n
    assert len(rec.cue_positions) >= 3
    for r, c, tile in rec.cue_positions:
        assert tile in (T.CUE_WATER_THIN, T.CUE_ROCK_THIN)
        assert rec.terrain[r, c] == tile


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_target_reachable_through_minable_bridgeable(seed: int, orientation: str):
    """With mining + bridging, the target is always reachable: BFS over all
    non-obsidian cells finds the goal."""
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    assert is_reachable(rec)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_thinner_side_really_is_thinner(seed: int, orientation: str):
    """The thinner segment is genuinely narrower to cross: counting the WATER
    vs ROCK cells inside each wall's crossing band, the side flagged as
    ``thinner`` has strictly fewer cells (3-wide vs 7-wide windows)."""
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    wall, _side, _mid = _coords(rec)
    for k, C in enumerate(rec.stripe_centers):
        band = np.abs(wall - C) <= 3
        water_n = int(((rec.terrain == T.WATER) & band).sum())
        rock_n = int(((rec.terrain == T.ROCK) & band).sum())
        if rec.stripe_thinner[k] == "water":
            assert water_n < rock_n
        else:
            assert rock_n < water_n


@pytest.mark.parametrize("seed", range(8))
def test_vertical_rectangular_32x64(seed: int):
    """Vertical maps support a 32×64 layout with 4 walls, reachable, and with
    NO obsidian on the top/bottom sides (only the central divider): the top and
    bottom rows contain no obsidian."""
    rec = generate_zebra_map(seed=seed, n_stripes=4, orientation="vertical", width=64)
    assert rec.terrain.shape == (32, 64)
    assert rec.spawn == (16, 0) and rec.target == (16, 63)
    assert len(rec.stripe_centers) == 4
    assert is_reachable(rec)
    # no obsidian on the top or bottom edge rows (sides have water/rock, not obsidian)
    assert not np.any(rec.terrain[0] == T.OBSIDIAN)
    assert not np.any(rec.terrain[-1] == T.OBSIDIAN)


@pytest.mark.parametrize("seed", range(8))
def test_natural_map_contract(seed: int):
    """Natural maps: spawn at centre-left, the whole right wall is the goal
    (TARGET), a mix of water + rock + grass, NO obsidian (everything crossable),
    and reachable from spawn to the goal wall."""
    rec = generate_zebra_map(seed=seed, orientation="natural", size=48)
    H, W = rec.terrain.shape
    assert rec.orientation == "natural"
    assert rec.spawn == (H // 2, 0)
    assert rec.terrain[rec.spawn] == T.GRASS
    # default goal = the whole right wall (touch it anywhere to win) → diverse endpoints
    assert np.all(rec.terrain[:, W - 1] == T.TARGET)
    assert len(rec.goal_cells) == H
    # a positive goal_half instead makes only a central door the goal
    rec2 = generate_zebra_map(seed=seed, orientation="natural", size=32, width=64, goal_half=4)
    gc = rec2.terrain[:, 63]
    assert np.any(gc == T.TARGET) and not np.all(gc == T.TARGET) and gc[16] == T.TARGET
    assert not np.any(rec.terrain == T.OBSIDIAN)            # no inviolable walls
    for tile in (T.WATER, T.ROCK, T.GRASS):
        assert np.any(rec.terrain == tile)
    assert is_reachable(rec)


def test_natural_rectangular_and_coverage():
    """Natural maps support rectangular sizes and have a sensible terrain mix.
    (Raw water/rock coverage is below the quantile target because the left/right
    edge bands and the sand/dirt fringes convert some of it.)"""
    rec = generate_zebra_map(seed=1, orientation="natural", size=32, width=64,
                             water_frac=0.15, rock_frac=0.25)
    assert rec.terrain.shape == (32, 64)
    assert rec.spawn == (16, 0)
    assert 0.05 <= (rec.terrain == T.WATER).mean() <= 0.22
    assert 0.08 <= (rec.terrain == T.ROCK).mean() <= 0.32


def test_natural_edge_bands_clear():
    """The left and right 10-column bands carry no obstacles (only walkable
    tiles / the goal wall), so spawn and the goal approach are clear."""
    rec = generate_zebra_map(seed=3, orientation="natural", size=32, width=64)
    H, W = rec.terrain.shape
    obstacles = (T.WATER, T.ROCK, T.TREE, T.OBSIDIAN)
    left = rec.terrain[:, :10]
    right = rec.terrain[:, W - 10:W - 1]      # exclude the goal-wall column
    assert not np.isin(left, obstacles).any()
    assert not np.isin(right, obstacles).any()


def test_natural_sand_and_dirt_present():
    """Cosmetic SAND (around water) and DIRT (around rock) appear and are
    walkable look-alikes of grass."""
    rec = generate_zebra_map(seed=2, orientation="natural", size=32, width=64)
    assert np.any(rec.terrain == T.SAND)
    assert np.any(rec.terrain == T.DIRT)
    assert T.is_walkable(T.SAND) and T.is_walkable(T.DIRT)


@pytest.mark.parametrize("orientation", ("diagonal", "vertical"))
def test_thick_side_is_5050_build_vs_mine(orientation: str):
    """Across many walls the THICK (costly) side is water (→BUILD) ~50% and rock
    (→MINE) ~50%, drawn independently — the agent can't win by always mining or
    always bridging; it must read the cue."""
    build_thick = mine_thick = 0
    for s in range(2000):
        rec = generate_zebra_map(seed=s, orientation=orientation)
        for thin in rec.stripe_thinner:
            if thin == "rock":      # water thick -> costly route is BUILD
                build_thick += 1
            else:                   # rock  thick -> costly route is MINE
                mine_thick += 1
    frac = build_thick / (build_thick + mine_thick)
    assert 0.47 <= frac <= 0.53, f"thick=BUILD fraction {frac:.3f} not ~50/50"


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_central_divider_is_obsidian(seed: int, orientation: str):
    """The path centre of every wall is an obsidian divider, so the agent can't
    go straight through and must pick a window."""
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    wall, side, mid = _coords(rec)
    for C in rec.stripe_centers:
        centre = (wall == C) & (np.abs(side - mid) <= 1.0)
        if centre.any():
            assert np.all(rec.terrain[centre] == T.OBSIDIAN)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_each_wall_has_one_water_and_one_rock_window(seed: int, orientation: str):
    """Every wall offers a water crossing on one side of the path centre and a
    rock crossing on the other — both present, opposite sides."""
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    wall, side, mid = _coords(rec)
    for C in rec.stripe_centers:
        band = np.abs(wall - C) <= 3
        water = band & (rec.terrain == T.WATER)
        rock = band & (rec.terrain == T.ROCK)
        assert water.any() and rock.any()
        assert side[water].max() < mid < side[rock].min()


@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("seed", range(8))
def test_stripes_cannot_be_skirted(seed: int, orientation: str):
    """Every in-bounds cell on a wall's centre line is non-grass, so the agent
    must always mine/bridge to cross (no walking around)."""
    rec = generate_zebra_map(seed=seed, n_stripes=N_STRIPES[orientation], orientation=orientation)
    wall, _side, _mid = _coords(rec)
    for C in rec.stripe_centers:
        assert not np.any(rec.terrain[wall == C] == T.GRASS), \
            f"wall {C} has a grass gap on its centre line"


# ─────────────────────────── env mechanics ───────────────────────────────


def _fresh_env(seed: int = 0) -> ZebraNavEnv:
    env = ZebraNavEnv(seed=seed)
    env.reset()
    return env


def test_move_faces_even_when_blocked():
    env = _fresh_env(seed=0)
    # facing right initially; press DOWN — would go off the map → blocked,
    # but the agent should still face down.
    obs, r, term, trunc, info = env.step(A_DOWN)
    assert info["facing"] == F_DOWN
    assert info["blocked"] is True
    assert info["position"] == env._record.spawn       # unmoved


def test_place_no_op_on_non_water():
    env = _fresh_env(seed=0)
    # facing right onto grass at spawn → PLACE should do nothing
    env.step(A_RIGHT)                                   # face right, move OK
    obs, r, term, trunc, info = env.step(A_PLACE)
    assert info["placed"] is False
    assert info["mined"] is False


def test_mine_no_op_on_non_rock():
    env = _fresh_env(seed=0)
    env.step(A_RIGHT)
    obs, r, term, trunc, info = env.step(A_MINE)
    assert info["mined"] is False


def test_place_turns_water_into_wood():
    """Construct a tiny custom map: agent facing one water cell, PLACE → WOOD."""
    from cogniland.zebra_nav.mapgen import MapRecord
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.WATER
    rec = MapRecord(terrain=terr, spawn=(2, 2), target=(4, 4),
                    stripe_centers=[], stripe_thinner=[], cue_positions=[], seed=0)
    env = ZebraNavEnv(size=5, n_stripes=0, view_size=3, map_record=rec, max_steps=10)
    env.reset()
    env.step(A_RIGHT)                                   # face right, blocked by water
    assert env._terrain[2, 3] == T.WATER
    obs, r, term, trunc, info = env.step(A_PLACE)
    assert info["placed"] is True
    assert env._terrain[2, 3] == T.WOOD
    # now walking right onto the bridge works
    obs, r, term, trunc, info = env.step(A_RIGHT)
    assert info["position"] == (2, 3)


def test_mine_turns_rock_into_grass():
    from cogniland.zebra_nav.mapgen import MapRecord
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.ROCK
    rec = MapRecord(terrain=terr, spawn=(2, 2), target=(4, 4),
                    stripe_centers=[], stripe_thinner=[], cue_positions=[], seed=0)
    env = ZebraNavEnv(size=5, n_stripes=0, view_size=3, map_record=rec, max_steps=10)
    env.reset()
    env.step(A_RIGHT)
    obs, r, term, trunc, info = env.step(A_MINE)
    assert info["mined"] is True
    assert env._terrain[2, 3] == T.GRASS


def test_obsidian_is_inviolable():
    from cogniland.zebra_nav.mapgen import MapRecord
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.OBSIDIAN
    rec = MapRecord(terrain=terr, spawn=(2, 2), target=(4, 4),
                    stripe_centers=[], stripe_thinner=[], cue_positions=[], seed=0)
    env = ZebraNavEnv(size=5, n_stripes=0, view_size=3, map_record=rec, max_steps=10)
    env.reset()
    env.step(A_RIGHT)
    _, _, _, _, info_mine = env.step(A_MINE)
    _, _, _, _, info_place = env.step(A_PLACE)
    assert info_mine["mined"] is False and info_place["placed"] is False
    assert env._terrain[2, 3] == T.OBSIDIAN              # unchanged


def test_target_terminates_with_reach_bonus():
    from cogniland.zebra_nav.mapgen import MapRecord
    terr = np.full((3, 3), T.GRASS, dtype=np.int8)
    terr[0, 2] = T.TARGET
    rec = MapRecord(terrain=terr, spawn=(2, 0), target=(0, 2),
                    stripe_centers=[], stripe_thinner=[], cue_positions=[], seed=0)
    env = ZebraNavEnv(size=3, n_stripes=0, view_size=3, map_record=rec, max_steps=20,
                      shaping_coef=0.0)
    env.reset()
    for a in (A_RIGHT, A_RIGHT, A_UP, A_UP):
        obs, r, term, trunc, info = env.step(a)
    assert info["reached_target"] is True
    assert term is True
    assert info["position"] == (0, 2)
    assert info["episode_return"] == pytest.approx(
        4 * env.slack_penalty + env.reach_bonus
    )


# ─────────────────────── solvability via solver ──────────────────────────


def test_handcrafted_solver_reaches_target():
    """A simple deterministic solver (walk toward target, mine/place when
    blocked by the matching tile, avoid obsidian) reaches the goal — proves
    the env is end-to-end solvable from the *natural* maps."""
    from cogniland.zebra_nav._solver import scripted_solve
    env = ZebraNavEnv(seed=0, max_steps=600)
    obs, info = env.reset()
    steps, reached = scripted_solve(env)
    assert reached, f"solver failed after {steps} steps"
    assert env._step_count <= env.max_steps
