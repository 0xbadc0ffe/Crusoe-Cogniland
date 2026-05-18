"""Principle 8 gate: `jax.vmap(train)` over seed keys must Just Work.

See DESIGN.md Principle 8 & its self-test. A pass here is the defining requirement of
a reference implementation in `purejaxwm/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from purejaxwm.dreamerv3_craftax import make_train  # noqa: E402


def _tiny_cfg():
    return OmegaConf.create(dict(
        env_id="craftax_classic_pixels", num_envs=4, seed=0, num_seeds=1,
        total_env_steps=32, train_ratio=1, warmup_steps=0,
        buffer_capacity=256, batch_size=2, seq_len=8, buffer_min_size=16,
        deter=32, stoch=4, classes=4, blocks=4, wm_hidden=32,
        unimix=0.01, free_nats=1.0, num_reward_bins=51,
        cnn_depths=[8, 16, 16, 16], cnn_kernel=4, cnn_stride=2,
        ac_hidden=32, ac_layers=1, imag_horizon=4,
        gamma=0.997, gae_lambda=0.95, entropy_coef=3e-4,
        slow_ema_rate=0.02, slow_reg_coef=1.0, contdisc=True, slowtar=True,
        retnorm_rate=0.01, advantage_pct_lo=5.0, advantage_pct_hi=95.0,
        lr_wm=1e-4, lr_ac=3e-5, opt_eps=1e-5, max_grad_norm=1.0,
        loss_rec=1.0, loss_rew=1.0, loss_con=1.0, loss_dyn=0.5, loss_rep=0.1,
        loss_actor=1.0, loss_critic=1.0, loss_repval=0.3,
        run_dir="runs", wandb_project="p", wandb_entity=None,
        wandb_mode="disabled", wandb_log_interval=1,
    ))


def test_vmap_over_seeds():
    print("\n" + "=" * 80)
    print("TEST: jax.vmap(train) over 3 seed keys")
    print("=" * 80)
    cfg = _tiny_cfg()
    train_fn = make_train(cfg)
    K = 3
    rngs = jax.random.split(jax.random.PRNGKey(0), K)
    out = jax.jit(jax.vmap(train_fn))(rngs)

    # every metric leaf should have a leading axis of K
    ok = True
    for mname, arr in out.metrics.items():
        arr_np = np.asarray(arr)
        if arr_np.shape[0] != K:
            print(f"  FAIL: {mname} leading axis is {arr_np.shape[0]}, expected {K}")
            ok = False
        if not np.isfinite(arr_np).all():
            print(f"  FAIL: {mname} contains non-finite values")
            ok = False
    if ok:
        any_key = next(iter(out.metrics))
        print(f"  {K} seeds vmapped, every metric shape[0] == {K}")
        print(f"  sample ({any_key}): {np.asarray(out.metrics[any_key])[0, :3]} ...")
    print(f"  Principle 8 gate: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = test_vmap_over_seeds()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
