"""Regression test for imagination/replay bootstrap alignment.

The actor samples actions from the current imagined state `s_t`, while `imagine_scan`
emits the successor state `s_{t+1}`. `imag_loss` must therefore compute rewards,
values, and `returns_start` on the current-state grid `(s_0, ..., s_{T-1})`, not on
the raw successor-state grid `(s_1, ..., s_T)`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from purejaxwm.dreamerv3.behavior import imag_loss, RetNorm  # noqa: E402
from purejaxwm.dreamerv3.world_model import State  # noqa: E402
from purejaxwm.dreamerv3.distributions import symlog  # noqa: E402


class _FakeRSSM:
    """Deterministic one-dimensional latent dynamics: s_{t+1} = s_t + 1."""

    def apply(self, params, state, action, embed=None, is_first=None, training=True, rngs=None):
        del params, action, embed, is_first, training, rngs
        return State(
            deter=state.deter + 1.0,
            stoch=state.stoch,
            logits=state.logits,
        )


class _FakeActorHead:
    """Single-action head so categorical sampling is deterministic."""

    def apply(self, params, feat):
        del params
        return jnp.zeros((*feat.shape[:-1], 1), dtype=jnp.float32)


def _value_to_twohot_logits(values, num_bins=255, low=-20.0, high=20.0):
    """Construct logits such that TwoHotDist(logits).mean() ≈ values."""
    v_sl = symlog(values)
    pos = jnp.clip((v_sl - low) / (high - low) * (num_bins - 1), 0.0, num_bins - 1.0)
    below = jnp.floor(pos).astype(jnp.int32)
    above = jnp.minimum(below + 1, num_bins - 1)
    w_above = pos - jnp.floor(pos)
    w_below = 1.0 - w_above
    target = (
        jax.nn.one_hot(below, num_bins) * w_below[..., None]
        + jax.nn.one_hot(above, num_bins) * w_above[..., None]
    )
    return jnp.log(target + 1e-10)


class _FakeCriticHead:
    """Returns TwoHot logits encoding V(s) ≈ feat[..., 0]."""

    def apply(self, params, feat):
        del params
        return _value_to_twohot_logits(feat[..., 0])


class _FakeRewardDist:
    def __init__(self, mean):
        self._mean = mean

    def mean(self):
        return self._mean


def test_returns_start_is_aligned_with_imagination_start_state():
    rssm = _FakeRSSM()
    actor_head = _FakeActorHead()
    critic_head = _FakeCriticHead()

    init_state = State(
        deter=jnp.array([[0.0]], dtype=jnp.float32),
        stoch=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        logits=jnp.zeros((1, 1, 1), dtype=jnp.float32),
    )

    total, (aux, _) = imag_loss(
        {"actor": None, "critic": None},
        slow_critic_params=None,
        rssm=rssm,
        rssm_params=None,
        actor_head=actor_head,
        critic_head=critic_head,
        init_state=init_state,
        reward_head_apply=lambda feat: _FakeRewardDist(feat[..., 0]),
        cont_head_apply=lambda feat: jnp.full(feat.shape[:-1], 80.0),
        retnorm=RetNorm.initial(),
        action_dim=1,
        horizon=3,
        gamma=1.0,
        gae_lambda=0.0,
        entropy_coef=0.0,
        slow_reg_coef=0.0,
        percentile_lo=5.0,
        percentile_hi=95.0,
        slowtar=False,
        unimix=0.0,
        rng=jax.random.PRNGKey(0),
    )

    del total

    # With correct alignment:
    #   states for rewards/values are s0=0, s1=1, s2=2, bootstrap uses s3=3
    #   lam=0 => return_start = r(s0) + V(s1) = 0 + 1 = 1
    #
    # TwoHotDist discretization introduces small error (~0.01), so we use atol=0.05.
    # The old off-by-one bug computed on successor states (s1, s2, s3), producing ~3.
    np.testing.assert_allclose(np.asarray(aux.returns_start), np.array([1.0]), atol=0.05)
