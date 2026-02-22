"""Tests for environment step logic."""

import torch
import pytest

from cogniland.env.types import EnvConfig, EnvState
from cogniland.env.islands import Islands
from cogniland.env.core import (
    apply_movement,
    compute_terrain_levels,
    env_step,
)
from cogniland.env.constants import ACTIONS


@pytest.fixture
def config():
    return EnvConfig(size=50, seed=42, device="cpu", max_steps=100)


@pytest.fixture
def env(config):
    return Islands(config)


def test_reset_produces_valid_state(env):
    state, target = env.reset(batch_size=4, seed=0)
    assert state.position.shape == (4, 2)
    assert state.minimap.shape[0] == 4
    assert state.minimap.shape[1] == 1  # channel dim
    assert state.hp.shape == (4,)
    assert target.shape == (4, 2)
    # Positions should be on land
    for i in range(4):
        h = env.world_map[state.position[i, 0], state.position[i, 1]]
        assert h.item() > 0.05  # above water threshold


def test_step_returns_correct_types(env):
    state, target = env.reset(2, seed=1)
    action = torch.tensor([ACTIONS["up"], ACTIONS["down"]])
    result = env.step(state, action, target)
    assert result.state.position.shape == (2, 2)
    assert result.reward.shape == (2,)
    assert result.done.shape == (2,)
    assert "alive" in result.info


def test_stay_action_does_not_move(env):
    state, target = env.reset(1, seed=2)
    pos_before = state.position.clone()
    action = torch.tensor([ACTIONS["stay"]])
    result = env.step(state, action, target)
    assert torch.equal(result.state.position, pos_before)


def test_movement_clamps_to_bounds(config):
    env = Islands(config)
    state, target = env.reset(1, seed=3)
    # Force position to edge
    state = state._replace(position=torch.tensor([[0, 0]]))
    action = torch.tensor([ACTIONS["up"]])
    result = env.step(state, action, target)
    assert result.state.position[0, 0].item() >= 0


def test_compute_terrain_levels_vectorized(env):
    positions = torch.tensor([[25, 25], [0, 0], [49, 49]])
    levels = compute_terrain_levels(env.world_map, positions)
    assert levels.shape == (3,)
    assert (levels >= 0).all() and (levels <= 8).all()


def test_batch_consistency(env):
    """Running the same action on two identical states should produce identical results."""
    state, target = env.reset(1, seed=10)
    # Duplicate
    state2 = state._replace(
        position=state.position.repeat(2, 1),
        minimap=state.minimap.repeat(2, 1, 1, 1),
        compass=state.compass.repeat(2, 1),
        terrain_lev=state.terrain_lev.repeat(2),
        terrain_clock=state.terrain_clock.repeat(2),
        resources=state.resources.repeat(2),
        hp=state.hp.repeat(2),
        cost=state.cost.repeat(2),
    )
    target2 = target.repeat(2, 1)
    action = torch.tensor([ACTIONS["right"], ACTIONS["right"]])
    result = env.step(state2, action, target2)
    assert torch.allclose(result.state.hp[0], result.state.hp[1])
    assert torch.allclose(result.reward[0], result.reward[1])
