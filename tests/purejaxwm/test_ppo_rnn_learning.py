"""PPO-RNN + Impala-ResNet smoke test on Craftax.

Runs a short PPO-RNN training with the config HPs and verifies: (1) the training
function compiles cleanly, (2) losses stay finite, (3) mean episode return increases
above zero (Craftax gives partial reward for tree-chopping very early in training).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "baselines"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from ppo_rnn_craftax import Config, make_train  # noqa: E402


def _cfg() -> Config:
    # ~ 65K env steps (64 num_envs * 64 num_steps * 16 updates)
    return Config(
        env_id="craftax_classic_pixels", num_envs=64, optimistic_reset_ratio=16,
        seed=0, num_seeds=1, total_env_steps=64 * 64 * 16,
        num_steps=64, update_epochs=2, num_minibatches=4,
        gamma=0.99, gae_lambda=0.8, clip_eps=0.2,
        ent_coef=0.01, vf_coef=0.5,
        lr=2e-4, anneal_lr=True, max_grad_norm=1.0,
        impala_channels=(16, 32, 32), impala_num_blocks=2,
        mlp_hidden=128, gru_hidden=128, ac_hidden=128, ac_res_blocks=1,
        run_dir="runs", wandb_project="p", wandb_mode="offline", log_interval=1,
    )


def test_ppo_rnn_smoke():
    print("\n" + "=" * 80)
    print("TEST: PPO-RNN + Impala-ResNet smoke training")
    print("=" * 80)
    cfg = _cfg()
    print(f"  num_envs={cfg.num_envs}, num_steps={cfg.num_steps}, num_updates={cfg.num_updates}")
    train_fn = make_train(cfg)
    out = jax.jit(train_fn)(jax.random.PRNGKey(0))
    metrics = {k: np.asarray(v) for k, v in out.metrics.items()}

    print("  final metrics (last 5 updates):")
    for k in sorted(metrics):
        v = metrics[k][-5:]
        print(f"    {k:18s} = {v}")

    # finiteness
    ok_finite = all(np.isfinite(v).all() for v in metrics.values())
    # return > 0 at any point (very lax — Craftax gives reward for first tree chop)
    ret = metrics["return/mean"]
    ok_ret = bool((ret > 0).any())

    print(f"  finite metrics: {'PASS' if ok_finite else 'FAIL'}")
    print(f"  return/mean > 0 at some point: {'PASS' if ok_ret else 'FAIL'}")
    return ok_finite and ok_ret


def main() -> int:
    ok = test_ppo_rnn_smoke()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
