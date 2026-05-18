"""Prove that purejaxwm's jax.lax.scan-based lambda_returns matches the legacy
Python-loop lambda_return when given identical inputs.

The legacy implementation lives at:
  /cluster/raid/home/joonsu/Projects/uzh-rl-course/cl/agents/commons/returns.py

The purejaxwm implementation lives at:
  purejaxwm/dreamerv3/returns.py

Key differences being verified:
  - jax.lax.scan (reverse) vs Python for-loop (reversed range)
  - (T, B) layout (purejaxwm) vs (B, T) layout (legacy)
  - Pre-multiplied disc array vs separate disc scalar + term array
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Inline legacy lambda_return (faithful port)
# ---------------------------------------------------------------------------

def _legacy_lambda_return(last, term, rew, val, boot, disc, lam):
    """Exact port of cl/agents/commons/returns.py::lambda_return.

    All inputs: (B, T). Returns: (B, T-1).
    The original casts booleans to float32 for JAX arithmetic; here inputs are
    already float64 so we skip the lossy cast to test pure algorithmic equivalence.
    """
    B, T = rew.shape
    rets = [boot[:, -1]]
    live = (1 - term.astype(np.float64))[:, 1:] * disc
    cont = (1 - last.astype(np.float64))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], axis=1)


# ---------------------------------------------------------------------------
# Inline purejaxwm lambda_returns (pure numpy mirror)
# ---------------------------------------------------------------------------

def _purejaxwm_lambda_returns(rewards, values, continues, lam):
    """Numpy mirror of purejaxwm/dreamerv3/returns.py::lambda_returns.

    rewards: (T, B), values: (T+1, B), continues: (T, B). Returns: (T, B).
    """
    T, B = rewards.shape
    rets = np.zeros((T, B), dtype=np.float64)
    next_ret = values[-1].copy()
    for t in reversed(range(T)):
        r = rewards[t]
        cont = continues[t]
        v_next = values[t + 1]
        next_ret = r + cont * ((1 - lam) * v_next + lam * next_ret)
        rets[t] = next_ret
    return rets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_contdisc_true_equivalence():
    """With contdisc=true (disc=1), both must produce identical returns."""
    rng = np.random.default_rng(42)
    B, T = 8, 16
    lam = 0.95
    disc = 1.0  # contdisc=true

    rewards = rng.normal(0, 1, (B, T))
    values = rng.normal(0, 1, (B, T))
    continuations = rng.uniform(0.9, 1.0, (B, T))

    last = np.zeros((B, T))
    term = 1 - continuations

    legacy_ret = _legacy_lambda_return(
        last=last, term=term, rew=rewards, val=values, boot=values,
        disc=disc, lam=lam,
    )

    # purejaxwm format: (T, B).
    # Legacy slices [:, 1:] internally, processing physical timesteps 1..T-1.
    # To match: pass rewards[1:], continues[1:] (physical steps 1..T-1),
    # and the FULL values array as the (T+1, B) values argument so that
    # values[t+1] = val at physical timestep t+1.
    rew_tb = rewards.T       # (T, B)
    val_tb = values.T        # (T, B) — already has T entries, last = bootstrap
    cont_tb = continuations.T
    disc_arr = disc * cont_tb

    pure_ret = _purejaxwm_lambda_returns(rew_tb[1:], val_tb, disc_arr[1:], lam)

    # legacy returns: (B, T-1); pure returns: (T-1, B)
    np.testing.assert_allclose(
        legacy_ret, pure_ret.T, atol=1e-10,
        err_msg="contdisc=true: returns diverged"
    )


def test_contdisc_false_equivalence():
    """With contdisc=false (disc=gamma), both must produce identical returns."""
    rng = np.random.default_rng(77)
    B, T = 4, 12
    lam = 0.95
    gamma = 0.997

    rewards = rng.normal(1, 0.5, (B, T))
    values = rng.normal(1, 0.5, (B, T))
    continuations = rng.uniform(0.95, 1.0, (B, T))

    last = np.zeros((B, T))
    term = 1 - continuations

    legacy_ret = _legacy_lambda_return(
        last=last, term=term, rew=rewards, val=values, boot=values,
        disc=gamma, lam=lam,
    )

    rew_tb = rewards.T
    val_tb = values.T
    cont_tb = continuations.T
    disc_arr = gamma * cont_tb

    pure_ret = _purejaxwm_lambda_returns(rew_tb[1:], val_tb, disc_arr[1:], lam)

    np.testing.assert_allclose(
        legacy_ret, pure_ret.T, atol=1e-10,
        err_msg="contdisc=false: returns diverged"
    )


def test_jax_scan_matches_numpy():
    """Verify the actual JAX lax.scan implementation matches the numpy mirror."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("SKIP: jax not available")
        return

    from purejaxwm.dreamerv3.behavior import lambda_returns

    rng_np = np.random.default_rng(55)
    T, B = 15, 8
    lam = 0.95

    rewards = rng_np.normal(0, 1, (T, B)).astype(np.float32)
    values = rng_np.normal(0, 1, (T + 1, B)).astype(np.float32)
    continues = rng_np.uniform(0.9, 1.0, (T, B)).astype(np.float32)

    np_ret = _purejaxwm_lambda_returns(rewards, values, continues, lam)
    jax_ret = np.asarray(lambda_returns(
        jnp.array(rewards), jnp.array(values), jnp.array(continues), lam
    ))

    np.testing.assert_allclose(
        jax_ret, np_ret.astype(np.float32), atol=1e-5,
        err_msg="JAX lax.scan vs numpy mirror diverged"
    )


def test_terminal_handling():
    """Episodes with termination (cont=0) should zero out future bootstrapping."""
    T, B = 5, 1
    lam = 0.95

    rewards = np.ones((T, B))
    values = np.ones((T + 1, B)) * 10.0
    continues = np.ones((T, B))
    continues[2, 0] = 0.0  # terminate at step 2

    ret = _purejaxwm_lambda_returns(rewards, values, continues, lam)

    # after termination at t=2, ret[2] = r[2] + 0 * (...) = 1.0
    assert abs(ret[2, 0] - 1.0) < 1e-10, f"terminated step should be reward only, got {ret[2, 0]}"
    # step 3 and 4 continue normally from values
    assert ret[3, 0] > 1.0, "post-termination steps should bootstrap from values"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("contdisc=true equivalence", test_contdisc_true_equivalence),
        ("contdisc=false equivalence", test_contdisc_false_equivalence),
        ("JAX scan matches numpy", test_jax_scan_matches_numpy),
        ("terminal handling", test_terminal_handling),
    ]
    print("\n" + "=" * 80)
    print("TEST: purejaxwm lambda_returns ≡ legacy lambda_return")
    print("=" * 80)
    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, True))
            print(f"  PASS: {name}")
        except Exception as e:
            results.append((name, False))
            print(f"  FAIL: {name} — {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
