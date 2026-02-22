"""Tests for the reward function."""

import torch
import pytest

from cogniland.env.types import EnvConfig, EnvState
from cogniland.env.reward import compute_reward


@pytest.fixture
def config():
    return EnvConfig()


def _make_state(hp=75.0, device="cpu"):
    """Helper to create a minimal EnvState."""
    B = 1
    return EnvState(
        position=torch.zeros(B, 2, dtype=torch.long, device=device),
        minimap=torch.zeros(B, 1, 51, 51, device=device),
        compass=torch.zeros(B, 2, device=device),
        terrain_lev=torch.tensor([5.0], device=device),
        terrain_clock=torch.tensor([0.0], device=device),
        resources=torch.tensor([0.0], device=device),
        hp=torch.tensor([hp], device=device),
        cost=torch.tensor([0.0], device=device),
    )


def test_reach_bonus(config):
    state = _make_state()
    alive = torch.tensor([True])
    reached = torch.tensor([True])
    dist = torch.tensor([0.5])
    prev_dist = torch.tensor([1.0])
    r = compute_reward(state, alive, reached, dist, prev_dist, config)
    assert r.item() > 0  # Should get positive reward for reaching target


def test_death_penalty(config):
    state = _make_state(hp=0.0)
    alive = torch.tensor([False])
    reached = torch.tensor([False])
    dist = torch.tensor([50.0])
    prev_dist = torch.tensor([50.0])
    r = compute_reward(state, alive, reached, dist, prev_dist, config)
    assert r.item() < 0  # Should get negative reward for dying


def test_distance_reward(config):
    state = _make_state()
    alive = torch.tensor([True])
    reached = torch.tensor([False])
    dist = torch.tensor([40.0])
    prev_dist = torch.tensor([50.0])
    r_closer = compute_reward(state, alive, reached, dist, prev_dist, config)

    dist2 = torch.tensor([60.0])
    r_farther = compute_reward(state, alive, reached, dist2, prev_dist, config)

    assert r_closer > r_farther  # Moving closer should give higher reward


def test_low_hp_penalty(config):
    state_high = _make_state(hp=80.0)
    state_low = _make_state(hp=10.0)
    alive = torch.tensor([True])
    reached = torch.tensor([False])
    dist = torch.tensor([50.0])
    prev_dist = torch.tensor([50.0])
    r_high = compute_reward(state_high, alive, reached, dist, prev_dist, config)
    r_low = compute_reward(state_low, alive, reached, dist, prev_dist, config)
    assert r_high > r_low  # Low HP should reduce reward
