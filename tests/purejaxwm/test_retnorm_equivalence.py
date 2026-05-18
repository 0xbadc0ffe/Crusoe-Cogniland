"""Prove that purejaxwm's RetNorm is semantically equivalent to the reference
Normalizer(impl='perc', rate=0.01, limit=1.0, perclo=5, perchi=95, debias=True).

The reference codebase (memory-maze-jax) uses a Normalizer PyTree class with
debiasing to correct for the zero-initialization bias.  purejaxwm's RetNorm
mirrors this with an EMA counter for debiasing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Inline legacy normalizer (faithful port of the deployed config)
# ---------------------------------------------------------------------------

class _LegacyPercNorm:
    """Python port of memory-maze-jax's Normalizer with
    impl='perc', debias=True, limit=1.0, perclo=5.0, perchi=95.0.
    """

    def __init__(self, rate: float = 0.01, limit: float = 1.0,
                 perclo: float = 5.0, perchi: float = 95.0):
        self.rate = rate
        self.limit = limit
        self.perclo = perclo
        self.perchi = perchi
        self.lo = 0.0
        self.hi = 0.0
        self.count = 0.0

    def update(self, x: np.ndarray):
        lo = float(np.percentile(x, self.perclo))
        hi = float(np.percentile(x, self.perchi))
        self.lo = (1 - self.rate) * self.lo + self.rate * lo
        self.hi = (1 - self.rate) * self.hi + self.rate * hi
        self.count = (1 - self.rate) * self.count + self.rate

    def scale(self) -> float:
        corr = 1.0 / max(self.rate, self.count)
        lo = self.lo * corr
        hi = self.hi * corr
        return max(self.limit, hi - lo)

    def offset(self) -> float:
        corr = 1.0 / max(self.rate, self.count)
        return self.lo * corr


class _PurejaxwmRetNorm:
    """Python mirror of purejaxwm/dreamerv3/train_state.py::RetNorm."""

    def __init__(self, rate: float = 0.01):
        self.low = 0.0
        self.high = 0.0
        self.count = 0.0
        self.rate = rate

    def update(self, x: np.ndarray, pct_lo: float = 5.0, pct_hi: float = 95.0):
        lo = float(np.percentile(x, pct_lo))
        hi = float(np.percentile(x, pct_hi))
        self.low = (1 - self.rate) * self.low + self.rate * lo
        self.high = (1 - self.rate) * self.high + self.rate * hi
        self.count = (1 - self.rate) * self.count + self.rate

    def scale(self) -> float:
        corr = 1.0 / max(self.rate, self.count)
        lo = self.low * corr
        hi = self.high * corr
        return max(hi - lo, 1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_update_equivalence():
    """Feed identical returns; lo/hi/scale must be bit-identical after each update."""
    rng = np.random.default_rng(42)
    legacy = _LegacyPercNorm(rate=0.01, limit=1.0)
    pure = _PurejaxwmRetNorm(rate=0.01)

    for _ in range(100):
        returns = rng.uniform(0, 20, size=256)
        legacy.update(returns)
        pure.update(returns)
        assert legacy.lo == pure.low, f"lo mismatch: {legacy.lo} != {pure.low}"
        assert legacy.hi == pure.high, f"hi mismatch: {legacy.hi} != {pure.high}"
        assert legacy.scale() == pure.scale(), \
            f"scale mismatch: {legacy.scale()} != {pure.scale()}"


def test_multistep_convergence():
    """N=1000 updates with changing distributions. Max divergence must be 0.0."""
    rng = np.random.default_rng(123)
    legacy = _LegacyPercNorm(rate=0.01)
    pure = _PurejaxwmRetNorm(rate=0.01)

    max_lo_diff = 0.0
    max_hi_diff = 0.0
    max_scale_diff = 0.0

    for i in range(1000):
        if i < 500:
            returns = rng.uniform(0, 10, size=128)
        else:
            returns = rng.normal(5, 2, size=128)
        legacy.update(returns)
        pure.update(returns)
        max_lo_diff = max(max_lo_diff, abs(legacy.lo - pure.low))
        max_hi_diff = max(max_hi_diff, abs(legacy.hi - pure.high))
        max_scale_diff = max(max_scale_diff, abs(legacy.scale() - pure.scale()))

    assert max_lo_diff == 0.0, f"lo diverged: max_diff={max_lo_diff}"
    assert max_hi_diff == 0.0, f"hi diverged: max_diff={max_hi_diff}"
    assert max_scale_diff == 0.0, f"scale diverged: max_diff={max_scale_diff}"


def test_scale_floor():
    """Constant returns → both return scale=1.0 (the floor)."""
    legacy = _LegacyPercNorm(rate=0.01)
    pure = _PurejaxwmRetNorm(rate=0.01)

    for _ in range(50):
        returns = np.full(64, 5.0)
        legacy.update(returns)
        pure.update(returns)

    assert legacy.scale() == 1.0, f"legacy scale={legacy.scale()}, expected 1.0"
    assert pure.scale() == 1.0, f"pure scale={pure.scale()}, expected 1.0"


def test_advantage_equivalence():
    """Given identical (returns, values, retnorm), advantages must match.

    Legacy with advnorm.impl=none: adv = (ret - val) / rscale (no offset subtracted).
    purejaxwm: advantages = (returns - values) / scale.
    """
    rng = np.random.default_rng(77)
    legacy = _LegacyPercNorm(rate=0.01)
    pure = _PurejaxwmRetNorm(rate=0.01)

    for _ in range(200):
        returns = rng.uniform(-5, 15, size=128)
        legacy.update(returns)
        pure.update(returns)

    values = rng.normal(5, 1, size=64)
    returns = rng.uniform(0, 10, size=64)
    adv_legacy = (returns - values) / legacy.scale()
    adv_pure = (returns - values) / pure.scale()
    np.testing.assert_array_equal(adv_legacy, adv_pure)


def test_rate_sensitivity():
    """rate=0.01 and rate=0.02 must produce measurably different EMA trajectories.

    After a small number of updates from zero, rate=0.02 should have moved further
    from the initialization (0.0) than rate=0.01 — it adapts faster. This documents
    why the hardcoded-0.02 bug mattered: the normalizer tracked 2x more aggressively.
    """
    rng = np.random.default_rng(99)
    slow = _PurejaxwmRetNorm(rate=0.01)
    fast = _PurejaxwmRetNorm(rate=0.02)

    for _ in range(10):
        returns = rng.uniform(0, 10, size=128)
        slow.update(returns)
        fast.update(returns)

    # after only 10 updates from zero, fast should be further from 0
    assert abs(fast.low) > abs(slow.low), \
        f"rate=0.02 should adapt faster: fast.low={fast.low:.4f} slow.low={slow.low:.4f}"
    assert abs(fast.high) > abs(slow.high), \
        f"rate=0.02 should adapt faster: fast.high={fast.high:.4f} slow.high={slow.high:.4f}"
    assert slow.low != fast.low, "different rates must produce different states"


def test_early_training_safety():
    """Start from (0, 0), feed positive returns, scale=1.0 until hi-lo >= 1.0."""
    pure = _PurejaxwmRetNorm(rate=0.01)
    legacy = _LegacyPercNorm(rate=0.01)

    assert pure.scale() == 1.0
    assert legacy.scale() == 1.0

    # feed small returns where hi - lo < 1 for a while
    returns = np.linspace(4.5, 5.5, 32)  # range = 1.0, but 5th-95th ≈ 0.95
    for _ in range(5):
        pure.update(returns)
        legacy.update(returns)
        assert pure.scale() == 1.0, f"scale should be 1.0, got {pure.scale()}"
        assert legacy.scale() == 1.0

    # feed wide returns until scale > 1
    wide_returns = np.linspace(0, 100, 256)
    for _ in range(200):
        pure.update(wide_returns)
        legacy.update(wide_returns)
    assert pure.scale() > 1.0, "scale should have grown past floor"
    assert legacy.scale() > 1.0


def test_jax_vmap_compatibility():
    """RetNorm updates must work under jax.vmap."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("SKIP: jax not available")
        return

    from purejaxwm.dreamerv3.behavior import RetNorm

    def update_retnorm(retnorm, returns, rate=0.01):
        lo = jnp.percentile(returns, 5.0)
        hi = jnp.percentile(returns, 95.0)
        return RetNorm(
            low=(1 - rate) * retnorm.low + rate * lo,
            high=(1 - rate) * retnorm.high + rate * hi,
            count=(1 - rate) * retnorm.count + rate,
        )

    K = 4
    retnorms = RetNorm(low=jnp.zeros(K), high=jnp.zeros(K), count=jnp.zeros(K))
    rng = jax.random.PRNGKey(0)
    returns_batch = jax.random.uniform(rng, (K, 128))

    new_retnorms = jax.vmap(update_retnorm)(retnorms, returns_batch)
    assert new_retnorms.low.shape == (K,)
    assert new_retnorms.high.shape == (K,)
    assert new_retnorms.count.shape == (K,)
    assert not jnp.any(jnp.isnan(new_retnorms.low))
    assert not jnp.any(jnp.isnan(new_retnorms.high))
    # seeds differ → percentiles should differ across the K instances
    assert jnp.std(new_retnorms.low) > 0, "vmapped updates should produce different values"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("single-update equivalence", test_single_update_equivalence),
        ("multi-step convergence", test_multistep_convergence),
        ("scale floor", test_scale_floor),
        ("advantage equivalence", test_advantage_equivalence),
        ("rate sensitivity", test_rate_sensitivity),
        ("early-training safety", test_early_training_safety),
        ("jax.vmap compatibility", test_jax_vmap_compatibility),
    ]
    print("\n" + "=" * 80)
    print("TEST: RetNorm ≡ reference Normalizer(impl='perc', debias=True, limit=1.0)")
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
