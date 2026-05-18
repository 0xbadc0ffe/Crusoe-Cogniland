"""Prove that the JAX-native `train_debt` schedule is semantically equivalent to the
legacy Python-side `RatioTracker`.

If someone sets `train_ratio=256` (faithful DreamerV3 on a hard env like Craftax) and
we were silently dispatching 16 updates instead, the agent would be ~16x
under-trained. This test exists specifically to prevent that class of silent drift.

We reproduce:
  - legacy_tracker(ratio).__call__(step) — a Python port of cl/agents/utils.py
  - train_debt_step(debt, rate) — a Python mirror of the JAX carry update

…and compare their cumulative update counts over N simulated outer steps for
`replay_ratio ∈ {16, 32, 64, 128, 256}`. Divergence must be O(1) (at most the +1
init-call difference in the legacy implementation).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

BATCH_SIZE = 16
SEQ_LEN = 64
BATCH_STEPS = BATCH_SIZE * SEQ_LEN      # 1024
NUM_ENVS = 16
N_CALLS = 2000


class _LegacyRatioTracker:
    """Bit-faithful Python port of cl/agents/utils.py::RatioTracker."""

    def __init__(self, ratio: float):
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def _simulate_legacy(replay_ratio: int, num_calls: int, step_increment: int) -> int:
    """Run the legacy tracker for `num_calls` outer iterations. Returns total updates."""
    ratio = replay_ratio / BATCH_STEPS
    tracker = _LegacyRatioTracker(ratio)
    step = 0
    total = 0
    for _ in range(num_calls):
        step += step_increment
        total += tracker(step)
    return total


def _simulate_train_debt(replay_ratio: int, num_calls: int, step_increment: int) -> int:
    """Run the JAX-equivalent train_debt schedule (pure Python simulation of the
    carry update in dreamerv3_craftax.py). Returns total updates."""
    rate_per_outer = (replay_ratio / BATCH_STEPS) * step_increment
    debt = 0.0
    total = 0
    for _ in range(num_calls):
        debt += rate_per_outer
        num_updates = math.floor(debt)
        debt -= num_updates
        total += num_updates
    return total


def _run_one(replay_ratio: int, step_inc: int = NUM_ENVS) -> dict:
    legacy_total = _simulate_legacy(replay_ratio, N_CALLS, step_increment=step_inc)
    jax_total = _simulate_train_debt(replay_ratio, N_CALLS, step_increment=step_inc)

    # Mathematical target: every collected transition owes `ratio` gradient updates,
    # where ratio = replay_ratio / BATCH_STEPS. Over `N_CALLS * step_inc` transitions:
    rate_per_outer = (replay_ratio / BATCH_STEPS) * step_inc
    expected = N_CALLS * rate_per_outer                      # real-valued ideal

    # Legacy has a one-time init offset: on the very first call it returns 1 regardless
    # of what floor(rate_per_outer) would be, so it may under-count by up to
    # (floor(rate_per_outer) - 1) on call #1 when rate_per_outer > 1.
    # After that, both converge to the same per-call rate.
    max_init_offset = max(0, math.floor(rate_per_outer) - 1)

    # Truth: train_debt matches `floor(N_CALLS * rate_per_outer)` exactly.
    # Legacy matches 1 + floor((N_CALLS - 1) * rate_per_outer).
    # So the legitimate tolerance is `max_init_offset + 1` (one for floor rounding).
    tol = max_init_offset + 1
    diff = abs(legacy_total - jax_total)

    # `train_debt` MUST hit the math target exactly (within floor rounding)
    jax_matches_math = abs(jax_total - math.floor(expected)) <= 1
    ok = diff <= tol and jax_matches_math
    return dict(
        ok=ok, legacy=legacy_total, jax_total=jax_total, expected=expected,
        diff=diff, tol=tol, rate_per_outer=rate_per_outer,
        jax_matches_math=jax_matches_math,
    )


def test_ratio_equivalence():
    print("\n" + "=" * 80)
    print("TEST: train_debt ≡ legacy RatioTracker over replay_ratio ∈ {16, 32, 64, 128, 256}")
    print(f"        ({N_CALLS} outer calls, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, num_envs={NUM_ENVS})")
    print("        legacy has a 1-time init offset: +1 on first call regardless of rate;")
    print("        train_debt is the more mathematically correct impl.")
    print("=" * 80)
    total_transitions = N_CALLS * NUM_ENVS
    print(f"  total transitions simulated: {total_transitions}")

    ok_all = True
    for rr in (16, 32, 64, 128, 256):
        r = _run_one(rr)
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"  rr={rr:4d} | expected={r['expected']:7.1f} | "
            f"legacy={r['legacy']:6d} | train_debt={r['jax_total']:6d} | "
            f"diff={r['diff']:3d} tol={r['tol']:3d} | {status}"
        )
        ok_all &= r["ok"]

    return ok_all


def test_ratio_equivalence_various_increments():
    """Confirm equivalence holds across different step_increment (num_envs) values."""
    print("\n" + "=" * 80)
    print("TEST: equivalence stable across step_increment (num_envs) values")
    print("=" * 80)
    ok_all = True
    for step_inc in (1, 8, 16, 64):
        for rr in (16, 256):
            r = _run_one(rr, step_inc=step_inc)
            status = "PASS" if r["ok"] else "FAIL"
            print(
                f"  step_inc={step_inc:3d}, rr={rr:4d} | "
                f"legacy={r['legacy']:6d} train_debt={r['jax_total']:6d} | "
                f"diff={r['diff']:3d} tol={r['tol']:3d} | {status}"
            )
            ok_all &= r["ok"]
    return ok_all


def main() -> int:
    results = [
        ("core ratios 16..256", test_ratio_equivalence()),
        ("step_increment sweep",  test_ratio_equivalence_various_increments()),
    ]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
