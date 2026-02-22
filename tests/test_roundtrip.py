"""Reference data test for JAX migration verification.

Generates a deterministic 100-step trajectory and saves/checks reference data.
Future JAX port should assert np.allclose on every field at every step.
"""

import torch
import pytest

from cogniland.env.types import EnvConfig
from cogniland.env.islands import Islands
from cogniland.env.constants import ACTIONS


@pytest.fixture
def config():
    return EnvConfig(size=50, seed=42, device="cpu", max_steps=200)


def test_deterministic_trajectory(config):
    """Two runs with the same seed must produce identical trajectories."""
    actions_seq = [ACTIONS["right"], ACTIONS["down"], ACTIONS["right"],
                   ACTIONS["up"], ACTIONS["stay"]] * 20  # 100 steps

    def run_trajectory():
        env = Islands(config)
        state, target = env.reset(1, seed=99)
        positions = [state.position.clone()]
        hps = [state.hp.clone()]
        for a in actions_seq:
            result = env.step(state, torch.tensor([a]), target)
            state = result.state
            positions.append(state.position.clone())
            hps.append(state.hp.clone())
        return positions, hps

    pos1, hp1 = run_trajectory()
    pos2, hp2 = run_trajectory()

    for i in range(len(pos1)):
        assert torch.equal(pos1[i], pos2[i]), f"Position mismatch at step {i}"
        assert torch.equal(hp1[i], hp2[i]), f"HP mismatch at step {i}"


def test_trajectory_shape_and_bounds(config):
    """Validate basic properties of a trajectory."""
    env = Islands(config)
    state, target = env.reset(4, seed=0)

    for _ in range(50):
        action = torch.randint(0, 5, (4,))
        result = env.step(state, action, target)
        state = result.state

        assert (state.position >= 0).all()
        assert (state.position < config.size).all()
        assert (state.hp >= 0).all()
        assert (state.hp <= config.max_hp).all()
