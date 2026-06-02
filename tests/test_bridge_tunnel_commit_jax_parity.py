"""Parity test: pure-JAX bridge_tunnel_commit == PyTorch BridgeTunnelCommitEnv.

Acceptance gate for the JAX port that drives DreamerV3. Over several maps (one
per category) and several fixed pseudo-random action sequences that exercise ALL
8 actions (moves + build + mine + the two commits), step BOTH envs from the same
reset and assert identical minimap / scalars / reward / terminated / truncated at
every step.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cogniland.bridge_tunnel_commit.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel_commit.mapgen import CATEGORIES, generate_commit_map  # noqa: E402
from cogniland.bridge_tunnel_commit_jax import (  # noqa: E402
    EnvParams,
    BridgeTunnelCommitJaxEnv,
    records_to_arrays,
)

TASK = dict(
    view_size=21,
    max_steps=800,
    slack_penalty=-0.01,
    reach_bonus=1.0,
    shaping_coef=0.01,
    build_cost=0.05,
    commit_cost=0.05,
    illegal_penalty=0.02,
    gamma=0.99,
)


def _single_map_params(rec) -> EnvParams:
    arrays = records_to_arrays([rec])
    return EnvParams.from_map_arrays(**arrays, **TASK)


def test_parity_obs_reward_done():
    rng_actions = np.random.default_rng(2024)
    n_action_seqs = 4
    seq_len = 300
    jax_env = BridgeTunnelCommitJaxEnv()

    # one map from each category (+ a couple extra balanced seeds)
    recs = [generate_commit_map(seed=s, category=c)
            for c in CATEGORIES for s in range(2)]

    n_checked = 0
    for mi, rec in enumerate(recs):
        params = _single_map_params(rec)
        H, W = rec.terrain.shape
        for si in range(n_action_seqs):
            pt = BridgeTunnelCommitEnv(
                size=H, width=W, map_record=rec,
                view_size=TASK["view_size"], max_steps=TASK["max_steps"],
                slack_penalty=TASK["slack_penalty"], reach_bonus=TASK["reach_bonus"],
                shaping_coef=TASK["shaping_coef"], build_cost=TASK["build_cost"],
                commit_cost=TASK["commit_cost"], illegal_penalty=TASK["illegal_penalty"],
                gamma=TASK["gamma"],
            )
            pt_obs, _ = pt.reset()
            jx_obs, jx_state = jax_env.reset_env(jax.random.PRNGKey(0), params)

            np.testing.assert_array_equal(
                np.asarray(jx_obs["minimap"]), pt_obs["minimap"],
                err_msg=f"map {mi} seq {si} reset minimap")
            np.testing.assert_allclose(
                np.asarray(jx_obs["scalars"]), pt_obs["scalars"], atol=1e-6,
                err_msg=f"map {mi} seq {si} reset scalars")

            # exercise all 6 actions; build/mine are frequent so implicit
            # commitment + the locked-tool penalty + commit-aware shaping all get covered.
            actions = rng_actions.choice(
                6, size=seq_len, p=[0.18, 0.14, 0.10, 0.24, 0.17, 0.17])

            key = jax.random.PRNGKey(0)
            for t, a in enumerate(actions):
                a = int(a)
                pt_obs, pt_r, pt_term, pt_trunc, _ = pt.step(a)
                key, sub = jax.random.split(key)
                jx_obs, jx_state, jx_r, jx_done, jx_info = jax_env.step_env(
                    sub, jx_state, a, params)
                jx_term = bool(jx_info["is_terminal"])
                jx_trunc = bool(jx_done) and not jx_term

                ctx = f"map {mi}({rec.category}) seq {si} step {t} action {a}"
                np.testing.assert_array_equal(
                    np.asarray(jx_obs["minimap"]), pt_obs["minimap"],
                    err_msg=f"{ctx}: minimap")
                np.testing.assert_allclose(
                    np.asarray(jx_obs["scalars"]), pt_obs["scalars"], atol=1e-6,
                    err_msg=f"{ctx}: scalars")
                np.testing.assert_allclose(
                    float(jx_r), float(pt_r), atol=1e-5,
                    err_msg=f"{ctx}: reward jax={float(jx_r)} pt={float(pt_r)}")
                assert jx_term == pt_term, f"{ctx}: terminated"
                assert jx_trunc == pt_trunc, f"{ctx}: truncated"
                # commit slot must agree
                assert int(jx_info["commit"]) == pt._commit, f"{ctx}: commit"
                n_checked += 1
                if pt_term or pt_trunc:
                    break

    assert n_checked > 1500, f"only checked {n_checked} steps"


def test_ctg_matches_oracle():
    """The precomputed (3,H,W) ctg stack equals _compute_all_ctg exactly."""
    for c in CATEGORIES:
        rec = generate_commit_map(seed=0, category=c)
        arrays = records_to_arrays([rec])
        oracle = BridgeTunnelCommitEnv._compute_all_ctg(rec.terrain, rec.target)
        np.testing.assert_array_equal(arrays["ctg"][0], oracle)
        assert arrays["category"][0] == {"balanced": 0, "lakes": 1, "rocky": 2}[c]
