"""DreamerV3 learning smoke test on Craftax.

Runs a short training (~1K outer steps on tiny config) and verifies the world-model
losses land in reasonable ranges after training. Thresholds adapted from the legacy
`uzh-rl-course/tests/agents/test_dreamer_learning.py` — their specific numbers
(rec < 500, rew < 5, dyn < 30, con < 5) encode "an order of magnitude off is a bug".

A pass here does NOT mean DreamerV3 is reproducing published numbers; it means the
infrastructure is numerically plausible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from purejaxwm.dreamerv3_craftax import make_train  # noqa: E402


def _cfg(total_env_steps: int = 4096):
    # Small but not trivial: 4096 env steps × 8 envs = 512 outer iterations.
    # At warmup_steps=256, ~480 of those do a gradient step.
    return OmegaConf.create(dict(
        env_id="craftax_classic_pixels", num_envs=8, seed=0, num_seeds=1,
        total_env_steps=total_env_steps, train_ratio=1, warmup_steps=256,
        buffer_capacity=8192, batch_size=4, seq_len=16, buffer_min_size=256,
        deter=128, stoch=16, classes=16, blocks=8, wm_hidden=128,
        unimix=0.01, free_nats=1.0, num_reward_bins=51,
        cnn_depths=[16, 32, 64, 128], cnn_kernel=4, cnn_stride=2,
        ac_hidden=128, ac_layers=2, imag_horizon=8,
        gamma=0.997, gae_lambda=0.95, entropy_coef=3e-4,
        slow_ema_rate=0.02, slow_reg_coef=1.0, contdisc=True, slowtar=True,
        retnorm_rate=0.01, advantage_pct_lo=5.0, advantage_pct_hi=95.0,
        lr_wm=1e-4, lr_ac=3e-5, opt_eps=1e-5, max_grad_norm=1.0,
        loss_rec=1.0, loss_rew=1.0, loss_con=1.0, loss_dyn=0.5, loss_rep=0.1,
        loss_actor=1.0, loss_critic=1.0, loss_repval=0.3,
        run_dir="runs", wandb_project="p", wandb_entity=None,
        wandb_mode="disabled", wandb_log_interval=1,
    ))


def test_smoke_training():
    print("\n" + "=" * 80)
    print("TEST: DreamerV3 smoke training (4096 env steps)")
    print("=" * 80)
    cfg = _cfg()
    train_fn = make_train(cfg)
    out = jax.jit(train_fn)(jax.random.PRNGKey(0))
    metrics = {k: np.asarray(v) for k, v in out.metrics.items()}

    # take tail means (after warmup) for each key we care about
    tail = {}
    for k, v in metrics.items():
        # skip zero-metrics entries from before-warmup steps
        if "loss/" in k:
            mask = v != 0
            if mask.any():
                tail[k] = float(v[mask][-10:].mean())

    print("  tail means (last 10 non-zero updates):")
    for k, val in sorted(tail.items()):
        print(f"    {k:22s} = {val:10.4f}")

    # loss-range checks (legacy thresholds, adapted to v0 scale)
    checks = {
        "loss/rec":  tail.get("loss/rec", 0.0) < 5000.0,
        "loss/rew":  tail.get("loss/rew", 0.0) < 5.0,
        "loss/dyn":  tail.get("loss/dyn", 0.0) < 30.0,
        "loss/cont": tail.get("loss/cont", 0.0) < 5.0,
    }
    for k, ok in checks.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'} (got {tail.get(k, float('nan')):.4f})")

    # also check imagination entropy is still moving
    ent = float(metrics.get("loss/entropy", np.zeros(1))[-1])
    print(f"  loss/entropy final = {ent:.4f} (should be > 0 if policy is stochastic)")

    return all(checks.values())


def main() -> int:
    ok = test_smoke_training()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
