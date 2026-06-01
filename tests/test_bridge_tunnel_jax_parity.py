"""Parity test: pure-JAX bridge_tunnel == PyTorch BridgeTunnelEnv.

This is the acceptance gate for the JAX port that drives DreamerV3. For several
validation maps and several fixed pseudo-random action sequences, we step BOTH
envs from the same reset and assert identical:

    * minimap (V, V) int8
    * scalars (5,) float32
    * reward  (within 1e-5)
    * terminated / truncated

at EVERY step. If this passes, the Dreamer agent is training on exactly the
task the PyTorch PPO ``natural_agent`` was trained on.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from cogniland.bridge_tunnel_jax import (  # noqa: E402
    EnvParams,
    BridgeTunnelJaxEnv,
    records_to_arrays,
)

VAL_MAPS = Path(__file__).resolve().parents[1] / "data" / "bridge_tunnel" / "val_maps.pkl"

# natural_agent.yaml task params
TASK = dict(
    view_size=21,
    max_steps=800,
    slack_penalty=-0.01,
    reach_bonus=3.0,
    shaping_coef=0.015,
    build_cost=0.0,
    gamma=0.997,
)


def _load_records():
    if not VAL_MAPS.exists():
        pytest.skip(f"no val maps at {VAL_MAPS}")
    with VAL_MAPS.open("rb") as f:
        d = pickle.load(f)
    return d["records"], d["kwargs"]


def _single_map_params(rec) -> EnvParams:
    """EnvParams holding exactly one map → JAX reset always picks idx 0."""
    arrays = records_to_arrays([rec])
    return EnvParams.from_map_arrays(**arrays, **TASK)


def test_parity_obs_reward_done():
    records, kwargs = _load_records()
    H = kwargs["size"]
    W = kwargs["width"]

    rng_actions = np.random.default_rng(12345)
    n_action_seqs = 4
    seq_len = 250

    jax_env = BridgeTunnelJaxEnv()

    n_checked = 0
    for mi, rec in enumerate(records[:6]):
        params = _single_map_params(rec)

        for si in range(n_action_seqs):
            # PyTorch oracle on this exact map
            pt = BridgeTunnelEnv(
                size=H, width=W, map_record=rec,
                view_size=TASK["view_size"], max_steps=TASK["max_steps"],
                slack_penalty=TASK["slack_penalty"], reach_bonus=TASK["reach_bonus"],
                shaping_coef=TASK["shaping_coef"], build_cost=TASK["build_cost"],
                gamma=TASK["gamma"],
            )
            pt_obs, _ = pt.reset()

            # JAX env (single map → idx 0). Call the *_env methods directly so
            # EnvParams (which holds arrays) isn't treated as a static JIT arg.
            jx_obs, jx_state = jax_env.reset_env(jax.random.PRNGKey(0), params)

            # reset obs must match
            np.testing.assert_array_equal(
                np.asarray(jx_obs["minimap"]), pt_obs["minimap"],
                err_msg=f"map {mi} seq {si} reset minimap")
            np.testing.assert_allclose(
                np.asarray(jx_obs["scalars"]), pt_obs["scalars"], atol=1e-6,
                err_msg=f"map {mi} seq {si} reset scalars")

            # bias toward moves so we actually traverse + occasionally build
            actions = rng_actions.choice(
                6, size=seq_len, p=[0.22, 0.18, 0.12, 0.28, 0.10, 0.10])

            key = jax.random.PRNGKey(0)
            for t, a in enumerate(actions):
                a = int(a)
                pt_obs, pt_r, pt_term, pt_trunc, _ = pt.step(a)

                key, sub = jax.random.split(key)
                jx_obs, jx_state, jx_r, jx_done, jx_info = jax_env.step_env(
                    sub, jx_state, a, params)
                jx_term = bool(jx_info["is_terminal"])
                jx_trunc = bool(jx_done) and not jx_term

                ctx = f"map {mi} seq {si} step {t} action {a}"
                np.testing.assert_array_equal(
                    np.asarray(jx_obs["minimap"]), pt_obs["minimap"],
                    err_msg=f"{ctx}: minimap")
                np.testing.assert_allclose(
                    np.asarray(jx_obs["scalars"]), pt_obs["scalars"], atol=1e-6,
                    err_msg=f"{ctx}: scalars")
                np.testing.assert_allclose(
                    float(jx_r), float(pt_r), atol=1e-5,
                    err_msg=f"{ctx}: reward jax={float(jx_r)} pt={float(pt_r)}")
                assert jx_term == pt_term, f"{ctx}: terminated jax={jx_term} pt={pt_term}"
                assert jx_trunc == pt_trunc, f"{ctx}: truncated jax={jx_trunc} pt={pt_trunc}"
                n_checked += 1

                if pt_term or pt_trunc:
                    break

    assert n_checked > 1000, f"only checked {n_checked} steps"


def test_ctg_matches_oracle():
    """The precomputed ctg field equals BridgeTunnelEnv._compute_ctg exactly."""
    records, _ = _load_records()
    for rec in records[:6]:
        arrays = records_to_arrays([rec])
        oracle = BridgeTunnelEnv._compute_ctg(rec.terrain, rec.target).astype(np.float32)
        np.testing.assert_array_equal(arrays["ctg"][0], oracle)
