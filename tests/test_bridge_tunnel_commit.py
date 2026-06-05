"""Tests for the bridge_tunnel_commit env (commit mechanics + 3 map categories).

Contract:
  * 8 actions; commitment is a single irreversible slot. BUILD/MINE are no-ops
    until the matching COMMIT; re-committing or committing the other tool after a
    commit does nothing.
  * observation scalars are (7,) = [facing one-hot, step/max, commit_build, commit_mine].
  * three map categories (balanced / lakes / rocky) with the right water:rock bias,
    each winnable under its intended commitment.
  * commitment-aware PBRS: committing the wrong tool on a one-sided map raises the
    cost-to-go at spawn (or makes the goal unreachable).
  * a hand-scripted commit-then-BFS solver reaches the goal on every category.
"""
from __future__ import annotations

import numpy as np
import pytest

from cogniland.bridge_tunnel import (
    BridgeTunnelCommitEnv, generate_commit_map, is_winnable, tiles as T,
)
from cogniland.bridge_tunnel.env import (
    A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE,
    COMMIT_NONE, COMMIT_BUILD, COMMIT_MINE, F_RIGHT,
)
N_SCALARS = 7   # btc obs: facing one-hot(4) + step/max + commit_build + commit_mine
from cogniland.bridge_tunnel.mapgen import (
    CATEGORIES, MapRecord, _can_reach_goal, _CATEGORY_FRACS,
)


# ─────────────────────────── mapgen / categories ─────────────────────────


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_winnable_and_labelled(category):
    for s in range(6):
        rec = generate_commit_map(seed=s, category=category)
        assert rec.category == category
        assert rec.orientation == "natural"
        assert rec.spawn == (16, 0)
        assert is_winnable(rec)
        # intended commitment is winnable
        bo = _can_reach_goal(rec.terrain, rec.spawn, frozenset({T.WATER}))
        mo = _can_reach_goal(rec.terrain, rec.spawn, frozenset({T.ROCK}))
        if category == "balanced":
            assert bo and mo
        elif category == "lakes":
            assert bo
        else:
            assert mo


def test_category_bias():
    """lakes are water-dominated, rocky rock-dominated, balanced ~even."""
    cov = {}
    for cat in CATEGORIES:
        w = r = 0.0
        for s in range(12):
            rec = generate_commit_map(seed=s, category=cat)
            w += (rec.terrain == T.WATER).mean()
            r += (rec.terrain == T.ROCK).mean()
        cov[cat] = (w / 12, r / 12)
    assert cov["lakes"][0] > 1.8 * cov["lakes"][1]      # water >> rock
    assert cov["rocky"][1] > 1.8 * cov["rocky"][0]      # rock  >> water
    assert 0.5 < cov["balanced"][0] / cov["balanced"][1] < 2.0


def test_deterministic_by_seed():
    a = generate_commit_map(seed=3, category="lakes")
    b = generate_commit_map(seed=3, category="lakes")
    assert np.array_equal(a.terrain, b.terrain)


def test_make_split_balanced():
    from cogniland.bridge_tunnel.mapgen import make_split
    recs = make_split(n_per_category=4)
    assert len(recs) == 12
    counts = {c: sum(r.category == c for r in recs) for c in CATEGORIES}
    assert all(v == 4 for v in counts.values())


# ─────────────────────────── env mechanics ───────────────────────────────


def _custom_env(terr, spawn, target, category="balanced", **kw):
    rec = MapRecord(terrain=terr, spawn=spawn, target=target, seed=0, category=category)
    H, W = terr.shape
    env = BridgeTunnelCommitEnv(size=H, width=W, view_size=3,
                                map_record=rec, max_steps=40, **kw)
    env.reset()
    return env


def test_obs_shapes_and_action_space():
    env = BridgeTunnelCommitEnv(seed=0)
    obs, info = env.reset()
    assert env.action_space.n == 6
    assert obs["scalars"].shape == (N_SCALARS,)
    assert info["commit"] == COMMIT_NONE
    # before commit, facing one-hot set, both commit flags zero
    assert obs["scalars"][4 + 1] == 0.0 and obs["scalars"][4 + 2] == 0.0


def test_build_noop_on_grass_does_not_commit():
    """BUILD facing a non-water tile while uncommitted is a harmless no-op and
    does NOT commit."""
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)                                   # face grass
    _, _, _, _, info = env.step(A_BUILD)
    assert info["placed"] is False and info["commit"] == COMMIT_NONE


def test_first_build_commits_and_locks_mine():
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.WATER
    terr[3, 2] = T.ROCK                                 # a rock to try mining later
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)                                   # face the water
    _, _, _, _, info = env.step(A_BUILD)                # first build → commit + bridge
    assert info["placed"] is True and info["committed_now"] is True
    assert info["commit"] == COMMIT_BUILD and env._terrain[2, 3] == T.WOOD
    # mine is now locked: facing the rock and mining does nothing
    env.step(A_DOWN)                                    # face the rock below
    _, _, _, _, info = env.step(A_MINE)
    assert info["mined"] is False and env._terrain[3, 2] == T.ROCK


def test_first_mine_commits_and_locks_build():
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.ROCK
    terr[3, 2] = T.WATER
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)                                   # face the rock
    _, _, _, _, info = env.step(A_MINE)                 # first mine → commit + mine
    assert info["mined"] is True and info["commit"] == COMMIT_MINE
    assert env._terrain[2, 3] == T.GRASS
    # build is now locked
    env.step(A_DOWN)                                    # face the water below
    _, _, _, _, info = env.step(A_BUILD)
    assert info["placed"] is False and env._terrain[3, 2] == T.WATER


def test_commit_flag_in_scalars():
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.ROCK
    env = _custom_env(terr, (2, 2), (4, 4))
    env.step(A_RIGHT)
    obs, *_ = env.step(A_MINE)                          # commit to mine
    # scalars[-2]=commit_build, scalars[-1]=commit_mine
    assert obs["scalars"][-1] == 1.0 and obs["scalars"][-2] == 0.0


def test_commit_cost_and_illegal_penalty():
    """One-time commit cost on the first successful build/mine; penalty for the
    LOCKED opposite skill. Obstacles are placed adjacent to spawn so moves are
    blocked (the agent only re-faces, staying put)."""
    terr = np.full((5, 5), T.GRASS, dtype=np.int8)
    terr[4, 4] = T.TARGET
    terr[2, 3] = T.WATER                                # right of spawn
    terr[1, 2] = T.ROCK                                 # above spawn
    env = _custom_env(terr, (2, 2), (4, 4), shaping_coef=0.0, build_cost=0.0,
                      commit_cost=0.05, illegal_penalty=0.02, slack_penalty=-0.01)
    # reset faces right → BUILD bridges the water and commits (slack + commit_cost)
    _, r, _, _, info = env.step(A_BUILD)
    assert info["committed_now"] is True and info["commit"] == COMMIT_BUILD
    assert env._pos == (2, 2)                           # build never moves
    assert r == pytest.approx(-0.01 - 0.05)
    # face the rock above (move blocked by rock → stays, just re-faces; slack only)
    _, r, _, _, _ = env.step(A_UP)
    assert r == pytest.approx(-0.01) and env._pos == (2, 2)
    # MINE the rock while committed to build → locked → no-op + illegal penalty
    _, r, _, _, info = env.step(A_MINE)
    assert info["mined"] is False and env._terrain[1, 2] == T.ROCK
    assert r == pytest.approx(-0.01 - 0.02)


def test_target_terminates_with_reach_bonus():
    terr = np.full((3, 3), T.GRASS, dtype=np.int8)
    terr[0, 2] = T.TARGET
    env = _custom_env(terr, (2, 0), (0, 2), shaping_coef=0.0)
    for a in (A_RIGHT, A_RIGHT, A_UP, A_UP):
        obs, r, term, trunc, info = env.step(a)
    assert info["reached_target"] is True and term is True
    assert info["episode_return"] == pytest.approx(4 * env.slack_penalty + env.reach_bonus)


def test_ctg_wrong_commit_costs_more():
    """On a lakes map, committing to MINE raises the spawn cost-to-go above the
    pre-commit / build potential (the wall of water becomes impassable)."""
    rec = generate_commit_map(seed=2, category="lakes")
    ctg = BridgeTunnelCommitEnv._compute_all_ctg(rec.terrain, rec.target)  # (3,H,W)
    sp = rec.spawn
    none_c, build_c, mine_c = ctg[0][sp], ctg[1][sp], ctg[2][sp]
    assert build_c < mine_c       # building (cross water) is cheaper than mining
    assert none_c <= build_c      # with both tools it's at least as cheap


# ─────────────────────── solvability via solver ──────────────────────────


@pytest.mark.parametrize("category", CATEGORIES)
def test_handcrafted_solver_reaches_target(category):
    from cogniland.bridge_tunnel._solver import scripted_solve
    for s in range(4):
        rec = generate_commit_map(seed=s, category=category)
        env = BridgeTunnelCommitEnv(map_record=rec, max_steps=800)
        env.reset()
        steps, reached = scripted_solve(env)
        assert reached, f"solver failed on {category} seed {s} after {steps} steps"
