"""Tests for the zebra_nav env (natural-only vocabulary).

Contract:
  * mapgen produces natural maps with the right 9-tile vocabulary (no
    obsidian / cue tiles), a centre-left spawn, a goal on the right wall, a
    mix of water/rock/grass, and a TREE-passable path from spawn to target.
  * trees are heavily biased toward the top & bottom walls.
  * env mechanics — move faces+steps, blocked moves still face, PLACE only
    affects WATER → WOOD, MINE only affects ROCK → GRASS, **TREE is
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


# ─────────────────────────── vocabulary ──────────────────────────────────


def test_tile_vocabulary():
    """9 contiguous tile ids, no obsidian/cue, TREE the sole inviolable."""
    assert T.NUM_TILES == 9
    assert (T.GRASS, T.WATER, T.ROCK, T.WOOD, T.TARGET, T.OOB, T.TREE,
            T.SAND, T.DIRT) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert T.TILE_COLORS.shape == (9, 3)
    assert set(T.TILE_NAMES) == set(range(9))
    assert T.INVIOLABLE == (T.TREE,)
    assert not hasattr(T, "OBSIDIAN")
    assert not hasattr(T, "CUE_WATER_THIN")
    assert not hasattr(T, "CUE_ROCK_THIN")
    for t in (T.GRASS, T.WOOD, T.TARGET, T.SAND, T.DIRT):
        assert T.is_walkable(t)
    for t in (T.WATER, T.ROCK, T.TREE, T.OOB):
        assert not T.is_walkable(t)


# ─────────────────────────── mapgen contract ─────────────────────────────


def test_non_natural_orientation_rejected():
    for bad in ("diagonal", "vertical", "mixed"):
        with pytest.raises(ValueError):
            generate_zebra_map(seed=0, orientation=bad)


@pytest.mark.parametrize("seed", range(8))
def test_natural_map_contract(seed: int):
    """Natural maps: spawn at centre-left, the whole right wall is the goal
    (TARGET) by default, a mix of water + rock + grass, NO inviolable wall
    isolating spawn, and reachable from spawn to the goal wall."""
    rec = generate_zebra_map(seed=seed, orientation="natural", size=48)
    H, W = rec.terrain.shape
    assert rec.orientation == "natural"
    assert rec.spawn == (H // 2, 0)
    assert rec.terrain[rec.spawn] == T.GRASS
    # default goal = the whole right wall (touch it anywhere to win)
    assert np.all(rec.terrain[:, W - 1] == T.TARGET)
    assert len(rec.goal_cells) == H
    # a positive goal_half instead makes only a central door the goal
    rec2 = generate_zebra_map(seed=seed, orientation="natural", size=32, width=64, goal_half=4)
    gc = rec2.terrain[:, 63]
    assert np.any(gc == T.TARGET) and not np.all(gc == T.TARGET) and gc[16] == T.TARGET
    for tile in (T.WATER, T.ROCK, T.GRASS):
        assert np.any(rec.terrain == tile)
    # all tile ids are in the new vocabulary
    assert rec.terrain.min() >= 0 and rec.terrain.max() < T.NUM_TILES
    assert is_reachable(rec)


def test_central_door_default():
    """The library default (goal_half=1) yields a 3-cell centre door."""
    rec = generate_zebra_map(seed=5, orientation="natural", size=32, width=64, goal_half=1)
    gc = rec.terrain[:, 63]
    assert int((gc == T.TARGET).sum()) == 3
    assert gc[15] == T.TARGET and gc[16] == T.TARGET and gc[17] == T.TARGET


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
    obstacles = (T.WATER, T.ROCK, T.TREE)
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


def test_trees_hug_top_and_bottom():
    """Tree patches cluster heavily near the top & bottom walls and are sparse
    in the vertical middle third. Aggregated over many seeds (with a high
    tree_frac so there are enough trees to measure)."""
    H = 32
    top = bot = mid = 0
    for s in range(40):
        rec = generate_zebra_map(seed=s, orientation="natural", size=H, width=64,
                                  tree_frac=0.06)
        rows = np.where(rec.terrain == T.TREE)[0]
        top += int((rows < H * 0.25).sum())
        bot += int((rows >= H * 0.75).sum())
        mid += int(((rows >= H / 3) & (rows < 2 * H / 3)).sum())
    edges = top + bot
    assert edges > 0, "no trees produced at all"
    # the top/bottom quartiles should hold far more tree cells than the middle
    # third (heavy wall-hugging forest cover).
    assert edges > 3 * max(mid, 1), f"edge trees {edges} not >> middle trees {mid}"


# ─────────────────────────── env mechanics ───────────────────────────────


def _fresh_env(seed: int = 0) -> ZebraNavEnv:
    env = ZebraNavEnv(seed=seed)
    env.reset()
    return env


def test_env_rejects_non_natural_orientation():
    for bad in ("diagonal", "vertical", "mixed"):
        with pytest.raises(ValueError):
            ZebraNavEnv(orientation=bad)


def test_move_faces_even_when_blocked():
    # custom map: agent boxed below by a tree → DOWN is blocked but still faces.
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[3, 2] = T.TREE                                 # cell below spawn (2,2)
    env = _custom_env(terr, (2, 2), (4, 4))
    obs, r, term, trunc, info = env.step(A_DOWN)
    assert info["facing"] == F_DOWN
    assert info["blocked"] is True
    assert info["position"] == (2, 2)                   # unmoved


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


def _custom_env(terr, spawn, target, **kw):
    from cogniland.zebra_nav.mapgen import MapRecord
    rec = MapRecord(terrain=terr, spawn=spawn, target=target, seed=0)
    H = terr.shape[0]
    env = ZebraNavEnv(size=H, width=terr.shape[1], view_size=3,
                      map_record=rec, max_steps=20, **kw)
    env.reset()
    return env


def test_place_turns_water_into_wood():
    """Construct a tiny custom map: agent facing one water cell, PLACE → WOOD."""
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.WATER
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)                                   # face right, blocked by water
    assert env._terrain[2, 3] == T.WATER
    obs, r, term, trunc, info = env.step(A_PLACE)
    assert info["placed"] is True
    assert env._terrain[2, 3] == T.WOOD
    # now walking right onto the bridge works
    obs, r, term, trunc, info = env.step(A_RIGHT)
    assert info["position"] == (2, 3)


def test_mine_turns_rock_into_grass():
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.ROCK
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)
    obs, r, term, trunc, info = env.step(A_MINE)
    assert info["mined"] is True
    assert env._terrain[2, 3] == T.GRASS


def test_tree_is_inviolable():
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.TREE
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)
    _, _, _, _, info_mine = env.step(A_MINE)
    _, _, _, _, info_place = env.step(A_PLACE)
    assert info_mine["mined"] is False and info_place["placed"] is False
    assert env._terrain[2, 3] == T.TREE                 # unchanged
    # cannot walk onto a tree either
    _, _, _, _, info_move = env.step(A_RIGHT)
    assert info_move["blocked"] is True
    assert info_move["position"] == (2, 2)


def test_target_terminates_with_reach_bonus():
    terr = np.full((3, 3), T.GRASS, dtype=np.int8)
    terr[0, 2] = T.TARGET
    env = _custom_env(terr, (2, 0), (0, 2), shaping_coef=0.0)
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
    blocked by the matching tile, avoid trees) reaches the goal — proves the
    env is end-to-end solvable from the *natural* maps."""
    from cogniland.zebra_nav._solver import scripted_solve
    env = ZebraNavEnv(seed=0, max_steps=600)
    obs, info = env.reset()
    steps, reached = scripted_solve(env)
    assert reached, f"solver failed after {steps} steps"
    assert env._step_count <= env.max_steps
