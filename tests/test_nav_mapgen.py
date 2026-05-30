"""Tests for cogniland.nav.mapgen — generator contract & invariants.

All tests use the session-cached ``cached_map`` fixture (see ``conftest.py``)
so each (size, map_type, seed) is generated *once* per test run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cogniland.nav import skills as sk
from cogniland.nav.mapgen import cost_to_go_unit, shortest_path_cost
from cogniland.nav.tiles import GRASS, TARGET

SIZES = (32, 64, 96, 128)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("map_type", ("lake", "rocky"))
def test_generate_each_size_and_type(cached_map, size: int, map_type: str):
    rec = cached_map(size, map_type, 0)
    assert rec.terrain.shape == (size, size)
    assert rec.map_type == map_type


@pytest.mark.parametrize("size", SIZES)
def test_spawn_in_bottom_left_and_on_grass(cached_map, size: int):
    rec = cached_map(size, "lake", 0)
    zone = max(3, size // 4)
    r, c = int(rec.spawn[0]), int(rec.spawn[1])
    assert size - zone <= r < size
    assert 0 <= c < zone
    assert rec.terrain[r, c] == GRASS


@pytest.mark.parametrize("size", SIZES)
def test_target_in_top_right_and_marked(cached_map, size: int):
    rec = cached_map(size, "rocky", 0)
    zone = max(3, size // 4)
    r, c = int(rec.target[0]), int(rec.target[1])
    assert 0 <= r < zone
    assert size - zone <= c < size
    assert rec.terrain[r, c] == TARGET


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("map_type", ("lake", "rocky"))
def test_oracle_costs_all_finite(cached_map, size: int, map_type: str):
    rec = cached_map(size, map_type, 0)
    assert math.isfinite(rec.no_skill_cost)
    assert math.isfinite(rec.raft_cost)
    assert math.isfinite(rec.harness_cost)


# Generator semantics (composed): each map is a structured composition — a
# corner-anchored barrier + a secondary feature, NO cost-inequality validation
# (the env is a POMDP). The barrier is a sparse analytic feature, not dominant
# fill, but the biome still decides which material the *main* barrier is:
# lake → water barrier (river), rocky → rock barrier (ridge), balanced → a mix.
from cogniland.nav.tiles import ROCK, WATER


def _fracs(terrain):
    n = terrain.size
    return (terrain == WATER).sum() / n, (terrain == ROCK).sum() / n


@pytest.mark.parametrize("size", SIZES)
def test_lake_is_water_dominant(cached_map, size: int):
    water, rock = _fracs(cached_map(size, "lake", 0).terrain)
    assert water > rock
    assert water > 0.04            # the barrier (a river) is genuinely present


@pytest.mark.parametrize("size", SIZES)
def test_rocky_is_rock_dominant(cached_map, size: int):
    water, rock = _fracs(cached_map(size, "rocky", 0).terrain)
    assert rock > water
    assert rock > 0.04             # the barrier (a ridge) is genuinely present


@pytest.mark.parametrize("size", SIZES)
def test_balanced_has_both_terrains(cached_map, size: int):
    water, rock = _fracs(cached_map(size, "balanced", 0).terrain)
    assert water > 0.02 and rock > 0.02


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("map_type", ("lake", "rocky", "balanced"))
def test_costs_finite_and_label_valid(cached_map, size: int, map_type: str):
    rec = cached_map(size, map_type, 0)
    assert all(math.isfinite(x) for x in
               (rec.no_skill_cost, rec.raft_cost, rec.harness_cost))
    assert rec.correct_object in (sk.NONE, sk.RAFT, sk.HARNESS)


def test_oracle_matches_dijkstra_recompute(cached_map):
    rec = cached_map(64, "lake", 0)
    expect_no_skill = shortest_path_cost(
        rec.terrain, tuple(rec.spawn), tuple(rec.target), sk.NONE
    )
    expect_raft = (
        shortest_path_cost(rec.terrain, tuple(rec.spawn), tuple(rec.target), sk.RAFT)
        + 1.0
    )
    assert np.isclose(expect_no_skill, rec.no_skill_cost)
    assert np.isclose(expect_raft, rec.raft_cost)


def test_ctg_array_present_and_consistent(cached_map):
    """Stored ctg arrays must equal a fresh Dijkstra recomputation."""
    rec = cached_map(32, "lake", 0)
    fresh = cost_to_go_unit(rec.terrain, tuple(rec.target), sk.RAFT)
    np.testing.assert_array_equal(rec.ctg_raft, fresh.astype(np.float32))
