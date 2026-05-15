"""Tests for CognilandNavEnv — obs / action contract, slip + reward mechanics.

Where possible we inject a cached MapRecord into the env (`map_record=` arg)
so each test reuses the session-cached map. The few tests that exercise
deterministic geometry build their own synthetic record.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniland.nav import CognilandNavEnv
from cogniland.nav import skills as sk
from cogniland.nav.mapgen import MapRecord, cost_to_go_unit
from cogniland.nav.renderer import SpriteSheet
from cogniland.nav.tiles import GRASS, TARGET, WATER

SIZES = (32, 64, 96, 128)


def _action(move: int, scalar: float = 0.0):
    return {"move": move, "build_scalar": np.array([scalar], np.float32)}


def _make_env(cached_map, size: int, map_type: str = "lake", **kwargs):
    rec = cached_map(size, map_type, 0)
    return CognilandNavEnv(
        size=size, map_type=map_type, tile_px=8, map_record=rec, **kwargs
    )


@pytest.mark.parametrize("size", SIZES)
def test_reset_shapes_symbolic_default(cached_map, size: int):
    env = _make_env(cached_map, size, view_size=11)
    obs, info = env.reset()
    # Symbolic is now the default — image is absent.
    assert "image" not in obs
    assert obs["semantic"].shape == (11, 11)
    assert obs["semantic"].dtype == np.int8
    assert obs["skill_active"].shape == (1,)
    assert obs["skill_active"].dtype == np.float32
    assert obs["skill_active"][0] == 0.0
    assert info["active_object"] == "none"


def test_rgb_obs_mode_includes_image(cached_map):
    rec = cached_map(32, "lake", 0)
    env = CognilandNavEnv(size=32, view_size=11, tile_px=8, obs_mode="rgb", map_record=rec)
    obs, _ = env.reset()
    assert obs["image"].shape == (3, 11 * 8, 11 * 8)
    assert "semantic" not in obs


def test_both_obs_mode_returns_both(cached_map):
    rec = cached_map(32, "lake", 0)
    env = CognilandNavEnv(size=32, view_size=11, tile_px=8, obs_mode="both", map_record=rec)
    obs, _ = env.reset()
    assert obs["image"].shape == (3, 11 * 8, 11 * 8)
    assert obs["semantic"].shape == (11, 11)


def test_build_raft_then_persists(cached_map):
    env = _make_env(cached_map, 32, map_type="lake")
    env.reset()
    obs, r, term, trunc, info = env.step(_action(4, +0.5))
    assert obs["skill_active"][0] == 1.0
    assert info["active_object"] == "raft"
    # Build pays SLACK_PENALTY only (no shaping, unit-ctg unchanged).
    assert r == pytest.approx(sk.SLACK_PENALTY, abs=1e-6)
    obs, _, _, _, info = env.step(_action(0))
    assert info["active_object"] == "raft"
    assert obs["skill_active"][0] == 1.0


def test_build_harness_with_negative_scalar(cached_map):
    env = _make_env(cached_map, 32, map_type="rocky")
    env.reset()
    obs, _, _, _, info = env.step(_action(4, -0.5))
    assert info["active_object"] == "harness"
    assert obs["skill_active"][0] == 1.0


def test_second_build_is_noop_with_penalty(cached_map):
    env = _make_env(cached_map, 32)
    env.reset()
    _, r1, *_, info1 = env.step(_action(4, +0.9))
    assert info1["active_object"] == "raft"
    _, r2, *_, info2 = env.step(_action(4, -0.9))
    assert info2["invalid_build"] is True
    assert info2["active_object"] == "raft"
    assert r2 == pytest.approx(sk.SLACK_PENALTY, abs=1e-6)


def test_obs_hides_object_identity(cached_map):
    env_a = _make_env(cached_map, 32)
    env_b = _make_env(cached_map, 32)
    obs_a0, _ = env_a.reset()
    obs_b0, _ = env_b.reset()
    np.testing.assert_array_equal(obs_a0["semantic"], obs_b0["semantic"])
    obs_a, *_ = env_a.step(_action(4, +0.7))
    obs_b, *_ = env_b.step(_action(4, -0.7))
    assert obs_a["skill_active"][0] == obs_b["skill_active"][0] == 1.0
    np.testing.assert_array_equal(obs_a["semantic"], obs_b["semantic"])


def _synthetic_record(size: int = 10, with_water: bool = True) -> MapRecord:
    """Tiny deterministic map for the collision + reach tests."""
    terrain = np.full((size, size), GRASS, dtype=np.int8)
    if with_water:
        terrain[5, 5] = WATER  # blocked tile next to spawn (5, 4)
    spawn = (size - 1, 0)
    target = (0, size - 1)
    terrain[target] = TARGET
    ctg_none = cost_to_go_unit(terrain, target, sk.NONE).astype(np.float32)
    ctg_raft = cost_to_go_unit(terrain, target, sk.RAFT).astype(np.float32)
    ctg_harness = cost_to_go_unit(terrain, target, sk.HARNESS).astype(np.float32)
    return MapRecord(
        terrain=terrain,
        spawn=np.array(spawn, np.int32),
        target=np.array(target, np.int32),
        map_type="lake",
        correct_object=sk.RAFT,
        no_skill_cost=1.0,
        raft_cost=0.5,
        harness_cost=1.5,
        constraints_passed=True,
        seed=0,
        ctg_none=ctg_none,
        ctg_raft=ctg_raft,
        ctg_harness=ctg_harness,
    )


def test_collision_into_blocked_tile():
    """Walk off the map → collision, slack penalty only."""
    rec = _synthetic_record(size=10)
    env = CognilandNavEnv(size=10, tile_px=8, view_size=5, map_record=rec, max_steps=10)
    env.reset()
    # spawn = (9, 0); moving left walks off-map.
    obs, reward, term, trunc, info = env.step(_action(2))
    assert info["collision"] is True
    assert info["position"] == (9, 0)
    assert reward == pytest.approx(sk.SLACK_PENALTY, abs=1e-6)


def test_reach_target_terminates_with_reward():
    """Walk a Manhattan path on a synthetic grass-only map."""
    rec = _synthetic_record(size=10, with_water=False)
    env = CognilandNavEnv(
        size=10, tile_px=8, view_size=5, map_record=rec, max_steps=40
    )
    env.reset()
    # spawn = (9, 0), target = (0, 9). Walk up 9 times then right 9 times.
    # Land never slips, so the path is deterministic.
    last_info = {}
    for _ in range(9):
        _, _, term, trunc, info = env.step(_action(0))  # up
        last_info = info
        assert not term
    for _ in range(9):
        _, r, term, trunc, info = env.step(_action(3))  # right
        last_info = info
        if term:
            assert info["reached_target"] is True
            # reach step pays slack + toward shaping + reach bonus
            expected = sk.SLACK_PENALTY + sk.SHAPING_COEF + sk.REACH_BONUS
            assert r == pytest.approx(expected, abs=1e-6)
            return
    pytest.fail(f"agent did not reach target; final info: {last_info}")


def test_truncation_at_max_steps():
    rec = _synthetic_record(size=10)
    env = CognilandNavEnv(size=10, tile_px=8, view_size=5, map_record=rec, max_steps=4)
    env.reset()
    info: dict = {}
    for i in range(4):
        _, r, term, trunc, info = env.step(_action(1))  # down (off-map for spawn=(9,0))
    assert (term is True) or (trunc is True)
    assert info["step"] == 4


def test_action_space_sample_is_accepted(cached_map):
    env = _make_env(cached_map, 32)
    env.reset()
    for _ in range(5):
        env.step(env.action_space.sample())


def test_slip_keeps_agent_in_place(monkeypatch):
    """Force the slip table to 100% on water; verify the agent stays put."""
    monkeypatch.setattr(sk, "SLIP_PROB_DEFAULT", 1.0)
    # Custom map: spawn right next to a water tile.
    from cogniland.nav.tiles import GRASS, TARGET
    size = 6
    terrain = np.full((size, size), GRASS, dtype=np.int8)
    terrain[3, 4] = WATER
    target = (0, size - 1)
    terrain[target] = TARGET
    rec = MapRecord(
        terrain=terrain,
        spawn=np.array([3, 3], np.int32),
        target=np.array(target, np.int32),
        map_type="lake",
        correct_object=sk.RAFT,
        no_skill_cost=1.0, raft_cost=0.5, harness_cost=1.5,
        constraints_passed=True, seed=0,
        ctg_none=cost_to_go_unit(terrain, target, sk.NONE).astype(np.float32),
        ctg_raft=cost_to_go_unit(terrain, target, sk.RAFT).astype(np.float32),
        ctg_harness=cost_to_go_unit(terrain, target, sk.HARNESS).astype(np.float32),
    )
    env = CognilandNavEnv(size=size, tile_px=8, view_size=5, map_record=rec, max_steps=10)
    env.reset()
    # spawn (3, 3). Step right onto water at (3, 4) → 100% slip.
    pos_before = env._pos
    _, r, *_, info = env.step(_action(3))
    assert info["slipped"] is True
    assert env._pos == pos_before
    assert r == pytest.approx(sk.SLACK_PENALTY, abs=1e-6)


def test_sprite_sheet_loads():
    SpriteSheet(tile_px=16)
