"""Tests for the environment layer: TaskSampler, CognilandEnv, reward computation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cogniland.envs.task_sampler import TaskSampler
from cogniland.envs.env import CognilandEnv, MINIMAP_DIAMETER, NUM_ACTIONS
from cogniland.envs.tasks import compute_task_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VAL_MAPS = str(PROJECT_ROOT / "data/maps/val.pt")


def _make_config(**overrides):
    """Build a minimal config namespace matching what CognilandEnv expects."""
    env = SimpleNamespace(
        max_steps=200,
        min_spawn_target_manhattan=60,
        terrain_vis_radius={
            "ocean": 22, "deep_water": 18, "water": 14,
            "beach": 12, "sandy": 12, "grassland": 12,
            "forest": 10, "rocky": 18, "mountains": 22,
        },
        occlude=False,  # Disable raycasting for fast tests
        train_maps=VAL_MAPS,
        val_maps=VAL_MAPS,
        num_parallel_envs=4,
        num_parallel_envs_eval=2,
    )
    for k, v in overrides.items():
        setattr(env, k, v)
    reward = SimpleNamespace(
        reach_bonus=10.0,
        step_penalty=0.01,
        shaping_coef=0.05,
    )
    return SimpleNamespace(env=env, seed=42, tasks=[0], reward=reward)


# ---------------------------------------------------------------------------
# TaskSampler
# ---------------------------------------------------------------------------

class TestTaskSampler:
    def test_round_robin_single_task(self):
        ts = TaskSampler(task_ids=[0], num_envs=4, mode="round_robin")
        result = ts.sample()
        assert result.shape == (4,)
        assert (result == 0).all()

    def test_round_robin_multi_task(self):
        ts = TaskSampler(task_ids=[0, 1, 2], num_envs=6, mode="round_robin")
        r1 = ts.sample()
        assert r1.shape == (6,)
        # First batch: [0, 1, 2, 0, 1, 2]
        np.testing.assert_array_equal(r1, [0, 1, 2, 0, 1, 2])

        # Second batch continues the cycle
        r2 = ts.sample()
        np.testing.assert_array_equal(r2, [0, 1, 2, 0, 1, 2])

    def test_round_robin_wrap(self):
        ts = TaskSampler(task_ids=[0, 1, 2], num_envs=5, mode="round_robin")
        r1 = ts.sample()
        np.testing.assert_array_equal(r1, [0, 1, 2, 0, 1])
        r2 = ts.sample()
        np.testing.assert_array_equal(r2, [2, 0, 1, 2, 0])

    def test_round_robin_non_contiguous_task_ids(self):
        ts = TaskSampler(task_ids=[0, 3, 5], num_envs=6, mode="round_robin")
        r1 = ts.sample()
        np.testing.assert_array_equal(r1, [0, 3, 5, 0, 3, 5])

    def test_fixed(self):
        ts = TaskSampler(task_ids=[0, 1], num_envs=3, mode="round_robin")
        result = ts.fixed(2)
        assert result.shape == (3,)
        assert (result == 2).all()

    def test_random_mode(self):
        ts = TaskSampler(task_ids=[0, 1, 2, 3], num_envs=100, mode="random")
        rng = np.random.default_rng(0)
        result = ts.sample(rng)
        assert result.shape == (100,)
        assert result.min() >= 0
        assert result.max() < 4
        # With 100 samples from 4 tasks, very likely all tasks appear
        assert len(set(result.tolist())) == 4

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            TaskSampler(task_ids=[0], num_envs=1, mode="invalid")

    def test_empty_task_list(self):
        with pytest.raises(ValueError):
            TaskSampler(task_ids=[], num_envs=1)


# ---------------------------------------------------------------------------
# CognilandEnv
# ---------------------------------------------------------------------------

class TestCognilandEnv:
    @pytest.fixture
    def env(self):
        config = _make_config()
        return CognilandEnv(config, VAL_MAPS, num_envs=4)

    def test_reset_shapes(self, env):
        obs = env.reset(seed=42)
        assert "minimap" in obs
        assert "scalars" in obs
        assert obs["minimap"].shape == (4, 6, MINIMAP_DIAMETER, MINIMAP_DIAMETER)
        assert obs["scalars"].shape == (4, 6)
        assert obs["minimap"].dtype == np.float32
        assert obs["scalars"].dtype == np.float32

    def test_reset_scalar_ranges(self, env):
        obs = env.reset(seed=42)
        s = obs["scalars"]
        # Compass is a unit vector — check magnitude ~1
        compass_mag = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
        np.testing.assert_allclose(compass_mag, 1.0, atol=1e-4)

        # terrain_idx / 8 in [0, 1]
        assert (s[:, 2] >= 0).all() and (s[:, 2] <= 1).all()

        # hp/100 should be 1.0 at start
        np.testing.assert_allclose(s[:, 3], 1.0)

        # wood/100 should be 0 at start
        np.testing.assert_allclose(s[:, 4], 0.0)

        # tool/3 should be 0 at start
        np.testing.assert_allclose(s[:, 5], 0.0)

    def test_minimap_range(self, env):
        obs = env.reset(seed=42)
        assert obs["minimap"].min() >= 0.0
        assert obs["minimap"].max() <= 1.0

    def test_state_arrays(self, env):
        env.reset(seed=42)
        assert env.pos_r.shape == (4,)
        assert env.hp.shape == (4,)
        assert (env.hp == 100.0).all()
        assert (env.wood == 0).all()
        assert (env.tool == 0).all()
        assert (env.done == False).all()

    def test_step_movement(self, env):
        env.reset(seed=42)
        initial_r = env.pos_r.copy()
        initial_c = env.pos_c.copy()

        # All envs move down (action=1, delta=(1,0))
        actions = np.full(4, 1, dtype=np.int32)
        obs, rewards, dones, info = env.step(actions)

        assert obs["minimap"].shape == (4, 6, MINIMAP_DIAMETER, MINIMAP_DIAMETER)
        assert obs["scalars"].shape == (4, 6)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)

        # Positions should have changed (unless at boundary)
        # At least check steps incremented
        assert (env.steps >= 1).all()

    def test_step_forage_on_berry(self, env):
        env.reset(seed=42)

        # Find a berry tile in the first map
        mi = env.map_idx[0]
        berry_locs = np.argwhere(env._berry_mask[mi])
        if len(berry_locs) == 0:
            pytest.skip("No berries on this map")

        # Place env 0 on a berry tile and reduce HP first
        br, bc = int(berry_locs[0, 0]), int(berry_locs[0, 1])
        env.pos_r[0] = br
        env.pos_c[0] = bc
        env.hp[0] = 50.0

        # Forage (action=4)
        actions = np.array([4, 0, 0, 0], dtype=np.int32)
        obs, rewards, dones, info = env.step(actions)

        # HP should have increased (berry heals +10)
        assert env.hp[0] == 60.0 or dones[0]  # unless env was auto-reset

    def test_step_craft(self, env):
        env.reset(seed=42)

        # Give env 0 enough wood
        env.wood[0] = 100

        # Craft raft (action=5)
        actions = np.array([5, 0, 0, 0], dtype=np.int32)
        obs, rewards, dones, info = env.step(actions)

        # Tool should be 1 (raft), wood should be 0
        assert env.tool[0] == 1
        assert env.wood[0] == 0

    def test_craft_without_wood(self, env):
        env.reset(seed=42)
        env.wood[0] = 50  # Not enough

        actions = np.array([5, 0, 0, 0], dtype=np.int32)
        env.step(actions)

        # Should fail — tool stays 0
        assert env.tool[0] == 0
        assert env.wood[0] == 50

    def test_craft_already_has_tool(self, env):
        env.reset(seed=42)
        env.wood[0] = 200
        env.tool[0] = 1  # Already has raft

        actions = np.array([6, 0, 0, 0], dtype=np.int32)  # Try craft rope
        env.step(actions)

        # Should fail — tool stays raft
        assert env.tool[0] == 1
        assert env.wood[0] == 200

    def test_action_space(self, env):
        assert env.action_space() == 8

    def test_num_envs(self, env):
        assert env.num_envs == 4

    def test_auto_reset_on_death(self, env):
        env.reset(seed=42)

        # Kill env 0
        env.hp[0] = 1.0
        # Move to force a drain that kills
        actions = np.array([1, 0, 0, 0], dtype=np.int32)
        obs, rewards, dones, info = env.step(actions)

        if dones[0]:
            # After auto-reset, env 0 should have fresh state
            assert env.hp[0] == 100.0
            assert env.steps[0] == 0
            assert info["returned_episode"][0]

    def test_episode_timeout(self, env):
        env.reset(seed=42)
        env.steps[0] = 199  # max_steps=200

        actions = np.array([4, 4, 4, 4], dtype=np.int32)  # forage (no-op on most tiles)
        obs, rewards, dones, info = env.step(actions)

        # Env 0 should be done due to timeout
        assert dones[0] or env.steps[0] == 0  # Either done or already reset


# ---------------------------------------------------------------------------
# Reward computation (Task 0)
# ---------------------------------------------------------------------------

class TestTask0Reward:
    def _make_info(self, B, reached=None, ctg_prev=None, ctg_curr=None):
        if reached is None:
            reached = np.zeros(B, dtype=bool)
        if ctg_prev is None:
            ctg_prev = np.full(B, 100.0, dtype=np.float32)
        if ctg_curr is None:
            ctg_curr = ctg_prev.copy()  # no progress by default
        return {
            "reached": reached,
            "alive": np.ones(B, dtype=bool),
            "dist_to_target": np.full(B, 50.0, dtype=np.float32),
            "initial_dist": np.full(B, 100.0, dtype=np.float32),
            "ctg_prev": ctg_prev,
            "ctg_curr": ctg_curr,
            "ctg_spawn": ctg_prev.copy(),
        }

    def test_step_penalty(self):
        config = _make_config()
        task_ids = np.zeros(2, dtype=np.int32)
        base_rewards = np.zeros(2, dtype=np.float32)
        dones = np.zeros(2, dtype=bool)
        info = self._make_info(2)

        rewards = compute_task_reward(task_ids, base_rewards, dones, info, config)
        # No progress (ctg_curr == ctg_prev) so shaping = 0; only step penalty
        np.testing.assert_allclose(rewards, -0.01)

    def test_reach_bonus(self):
        config = _make_config()
        task_ids = np.zeros(2, dtype=np.int32)
        base_rewards = np.zeros(2, dtype=np.float32)
        dones = np.array([True, False], dtype=bool)
        # Env 0: reached → ctg_curr=0, ctg_prev=100 → progress=100
        info = self._make_info(
            2,
            reached=np.array([True, False]),
            ctg_prev=np.array([100.0, 100.0], dtype=np.float32),
            ctg_curr=np.array([0.0, 100.0], dtype=np.float32),
        )

        rewards = compute_task_reward(task_ids, base_rewards, dones, info, config)
        # Env 0: -0.01 + 10 + 0.05 * 100 = 14.99
        assert rewards[0] == pytest.approx(-0.01 + 10.0 + 0.05 * 100.0, abs=1e-4)
        # Env 1: just -0.01 (no progress)
        assert rewards[1] == pytest.approx(-0.01, abs=1e-4)

    def test_pbrs_progress_step(self):
        config = _make_config()
        task_ids = np.zeros(1, dtype=np.int32)
        base_rewards = np.zeros(1, dtype=np.float32)
        dones = np.zeros(1, dtype=bool)
        # Made progress: cost-to-go dropped from 50 to 47
        info = self._make_info(
            1,
            ctg_prev=np.array([50.0], dtype=np.float32),
            ctg_curr=np.array([47.0], dtype=np.float32),
        )

        rewards = compute_task_reward(task_ids, base_rewards, dones, info, config)
        # -step_penalty + shaping_coef * 3 = -0.01 + 0.05 * 3 = 0.14
        assert rewards[0] == pytest.approx(-0.01 + 0.05 * 3.0, abs=1e-4)

    def test_pbrs_ignores_infinite_ctg(self):
        """Stepping onto a deadly tile -> ctg_curr = inf; shaping must be 0."""
        config = _make_config()
        task_ids = np.zeros(1, dtype=np.int32)
        base_rewards = np.zeros(1, dtype=np.float32)
        dones = np.array([True], dtype=bool)
        info = self._make_info(
            1,
            ctg_prev=np.array([50.0], dtype=np.float32),
            ctg_curr=np.array([np.inf], dtype=np.float32),
        )

        rewards = compute_task_reward(task_ids, base_rewards, dones, info, config)
        # Only step penalty applies
        assert rewards[0] == pytest.approx(-0.01, abs=1e-4)

    def test_non_task0_returns_zero(self):
        config = _make_config()
        task_ids = np.array([1, 2], dtype=np.int32)
        base_rewards = np.zeros(2, dtype=np.float32)
        dones = np.zeros(2, dtype=bool)
        info = self._make_info(2)

        rewards = compute_task_reward(task_ids, base_rewards, dones, info, config)
        np.testing.assert_array_equal(rewards, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
