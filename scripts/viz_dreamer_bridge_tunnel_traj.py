#!/usr/bin/env python3
"""200-trajectory stochastic-policy grids for a trained DreamerV3 bridge_tunnel agent.

The sibling of ``scripts/bridge_tunnel_traj_grid.py`` (the PPO renderer): for each of
``--n-maps`` natural map seeds (a subplot), samples ``--n-traj`` rollouts from
the **stochastic** DreamerV3 actor on that fixed map and overlays them with low
alpha so the spread of paths is visible. Paths are dark blue; cells where a MINE
(rock→grass) succeeded are yellow, cells where a PLACE (water→wood, bridge)
succeeded are red. Each subplot is titled with its success rate; the suptitle
shows the overall success rate.

It reconstructs the world model + actor EXACTLY as ``dreamerv3_bridge_tunnel.py``
built them (config loaded from ``<run>/config.json``), restores the orbax
PyTree checkpoint (params only), and runs the SAME per-step inference path as
the trainer's ``_rollout_step`` (encode → RSSM observe → unimix actor logits →
categorical sample), carrying the posterior as the next RSSM state and masking
the last action to zeros + ``is_first=True`` on the first step of each episode.

It uses the SAME 6 maps the PPO grid uses — ``generate_bridge_tunnel_map(size=32,
width=64, seed=eval_seed_start+j, orientation="natural", water_frac=0.14,
rock_frac=0.14, tree_frac=0.03, goal_half=4)`` — so the two PNGs are
apples-to-apples (the JAX env's dataset is generated from the same numpy
generator, proven bit-for-bit in tests/test_bridge_tunnel_jax_parity.py).

    python scripts/viz_dreamer_bridge_tunnel_traj.py \\
        --checkpoint runs/dreamerv3_bridge_tunnel_natural_25M_1780145070/checkpoints/step_1000000 \\
        --n-maps 6 --n-traj 200 --out outputs/previews/dreamer_natural_traj.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

# GPU courtesy: don't grab the whole device (must be set before importing jax).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))   # so `purejaxwm` resolves

from cogniland.bridge_tunnel import generate_bridge_tunnel_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel_jax import (  # noqa: E402
    EnvParams,
    BridgeTunnelJaxEnv,
    constants as C,
    records_to_arrays,
)
from cogniland.bridge_tunnel_jax.render import build_obs  # noqa: E402
from cogniland.bridge_tunnel_jax import dynamics as dyn  # noqa: E402

import purejaxwm.dreamerv3.behavior as ac  # noqa: E402
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM  # noqa: E402
from purejaxwm.commons import resolve_dtype  # noqa: E402

# Natural-maps task kwargs — the canonical natural_agent.yaml task, identical to
# the dataset's NATURAL_KWARGS and to the PPO natural_agent checkpoint args.
NATURAL_KWARGS = dict(
    size=32, width=64, orientation="natural",
    water_frac=0.14, rock_frac=0.14, tree_frac=0.03, goal_half=1,
)
SCALAR_DIM = 5

# facing-id → (dr, dc); matches env F_UP/F_DOWN/F_LEFT/F_RIGHT = 0/1/2/3.
_FACE_DELTA = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)


import flax.linen as nn  # noqa: E402


# Exact copy of dreamerv3_bridge_tunnel.BridgeTunnelEncoder so the restored params (named
# Dense_0..Dense_4 + RMSNorm_0..RMSNorm_4) bind without an import of the trainer.
class BridgeTunnelEncoder(nn.Module):
    hidden: int
    num_layers: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return jax.nn.silu(x)


# Decoder/encoder mode of the loaded checkpoint; set from cfg in main().
_DECODER_MODE = "mse"


def _flatten_obs(obs: dict) -> jax.Array:
    """Matches FlattenObsWrapper._flatten (mse: scalar/NUM_TILES; categorical: one-hot)."""
    if _DECODER_MODE == "categorical":
        oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
        mm = oh.reshape(*oh.shape[:-3], -1)
    else:
        mm = (obs["minimap"].astype(jnp.float32) / float(C.NUM_TILES))
        mm = mm.reshape(*mm.shape[:-2], -1)
    return jnp.concatenate([mm, obs["scalars"].astype(jnp.float32)], axis=-1)


def _single_map_params(seed: int, cfg: dict) -> EnvParams:
    """Build an EnvParams holding exactly ONE natural map (the PPO-grid seed)."""
    rec = generate_bridge_tunnel_map(seed=seed, **NATURAL_KWARGS)
    arrays = records_to_arrays([rec])   # stacks the single record → (1, H, W) etc.
    return EnvParams.from_map_arrays(
        **arrays,
        max_steps=cfg["max_steps"],
        view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"],
        reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"],
        build_cost=cfg["build_cost"],
        gamma=cfg["gamma"],
    ), rec


def _build_model(cfg: dict):
    compute_dtype = resolve_dtype(cfg.get("compute_dtype", "float32"))
    param_dtype = jnp.float32
    encoder = BridgeTunnelEncoder(
        hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
        embed_dim=cfg["wm_hidden"], dtype=compute_dtype, param_dtype=param_dtype,
    )
    rssm = RSSM(
        deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
        hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"],
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    actor_head = MLPHead(
        hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
        out_dim=C.NUM_ACTIONS, outscale=0.01,
        dtype=compute_dtype, param_dtype=param_dtype,
    )
    return encoder, rssm, actor_head


def batched_rollout(encoder, rssm, actor_head, wm_params, ac_params,
                    env: BridgeTunnelJaxEnv, params: EnvParams,
                    n_traj: int, max_steps: int, key) -> tuple:
    """Roll ``n_traj`` stochastic rollouts on one fixed map in lockstep using a
    jax.lax.scan. Returns (positions[T+1, n_traj, 2], reached[n_traj] bool,
    placed[T, n_traj] bool, mined[T, n_traj] bool, face_after[T, n_traj]).

    The single-map EnvParams has num_maps==1, so reset deterministically lands
    on that one map for every parallel episode."""
    action_dim = C.NUM_ACTIONS

    # Use the un-jitted reset_env/step_env (params is a normal pytree arg here;
    # the public reset/step treat params as a static_argnum, which an array-
    # carrying EnvParams can't satisfy). vmap over the parallel episodes.
    key, kr = jax.random.split(key)
    reset_keys = jax.random.split(kr, n_traj)
    obs0, state0 = jax.vmap(env.reset_env, in_axes=(0, None))(reset_keys, params)

    rssm_state = rssm.initial_state((n_traj,))
    last_action = jnp.zeros((n_traj, action_dim))
    last_is_first = jnp.ones((n_traj,), dtype=bool)
    done = jnp.zeros((n_traj,), dtype=bool)

    def scan_step(carry, _):
        state, obs, rssm_state, last_action, last_is_first, done, key = carry

        # mask last_action to zeros where is_first (exactly as the trainer does).
        action_masked = jnp.where(
            last_is_first[..., None], jnp.zeros_like(last_action), last_action
        )
        flat = _flatten_obs(obs)

        key, s_stoch, s_pol, s_step = jax.random.split(key, 4)
        embed = encoder.apply(wm_params["encoder"], flat)
        _, posterior = rssm.apply(
            wm_params["rssm"], rssm_state, action_masked, embed, last_is_first,
            rngs={"stoch": s_stoch},
        )
        feat = posterior.features()
        logits = ac.unimix_logits(actor_head.apply(ac_params["actor"], feat))
        action_idx = jax.random.categorical(s_pol, logits)
        action_oh = jax.nn.one_hot(action_idx, action_dim)

        step_keys = jax.random.split(s_step, n_traj)
        next_obs, next_state, reward, done_next, info = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None)
        )(step_keys, state, action_idx, params)

        # Don't step finished episodes: freeze a done episode's state/obs so its
        # path stops where it terminated (no resets — one episode each).
        def _sel(a_next, a_prev):
            mask = done.reshape(done.shape + (1,) * (a_next.ndim - 1))
            return jnp.where(mask, a_prev, a_next)

        next_state = jax.tree_util.tree_map(_sel, next_state, state)
        next_obs = jax.tree_util.tree_map(_sel, next_obs, obs)
        # placed/mined only count for still-active episodes this step.
        placed = info["placed"] & (~done)
        mined = info["mined"] & (~done)
        reached_now = info["reached_target"] & (~done)
        new_done = done | done_next

        out = {
            "pos": jnp.stack([next_state.agent_r, next_state.agent_c], axis=-1),
            "placed": placed, "mined": mined,
            "face": next_state.facing, "reached": reached_now,
        }
        new_carry = (next_state, next_obs, posterior, action_oh,
                     jnp.zeros((n_traj,), dtype=bool), new_done, key)
        return new_carry, out

    carry = (state0, obs0, rssm_state, last_action, last_is_first, done, key)
    _, outs = jax.lax.scan(scan_step, carry, None, length=max_steps)

    pos0 = jnp.stack([state0.agent_r, state0.agent_c], axis=-1)[None]  # (1,n,2)
    positions = jnp.concatenate([pos0, outs["pos"]], axis=0)           # (T+1,n,2)
    reached = outs["reached"].any(axis=0)                              # (n,)
    return (np.asarray(positions), np.asarray(reached),
            np.asarray(outs["placed"]), np.asarray(outs["mined"]),
            np.asarray(outs["face"]))


def _decision_points(positions, placed, mined, face):
    """Recover the faced cell where each successful PLACE/MINE happened.

    PLACE/MINE never move the agent and keep facing, so the faced cell at step
    t is (agent_pos[t+1] + face_delta[face[t]]). positions has the post-step
    position at index t+1."""
    mine_pts, bridge_pts = [], []
    Tn, n = placed.shape
    for t in range(Tn):
        pos_t = positions[t + 1]   # post-step agent pos
        for ev_mask, store in ((mined[t], mine_pts), (placed[t], bridge_pts)):
            idx = np.nonzero(ev_mask)[0]
            for i in idx:
                d = _FACE_DELTA[face[t, i]]
                store.append((pos_t[i, 0] + d[0], pos_t[i, 1] + d[1]))
    return mine_pts, bridge_pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("runs/dreamerv3_bridge_tunnel_natural_25M_1780145070/"
                                "checkpoints/step_1000000"))
    p.add_argument("--n-maps", type=int, default=6)
    p.add_argument("--n-traj", type=int, default=200)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--out", type=Path, default=Path("outputs/previews/dreamer_natural_traj.png"))
    args = p.parse_args()

    ckpt_dir = args.checkpoint.resolve()
    cfg_path = ckpt_dir.parent.parent / "config.json"
    cfg = json.loads(cfg_path.read_text())
    run_name = ckpt_dir.parent.parent.name
    global _DECODER_MODE
    _DECODER_MODE = cfg.get("decoder", "mse")

    print(f"[load] config {cfg_path}")
    print(f"[load] checkpoint {ckpt_dir}")
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    wm_params = jax.tree_util.tree_map(jnp.asarray, payload["wm_params"])
    ac_params = jax.tree_util.tree_map(jnp.asarray, payload["ac_params"])

    encoder, rssm, actor_head = _build_model(cfg)
    env = BridgeTunnelJaxEnv()   # params passed explicitly per panel (single map)

    key = jax.random.PRNGKey(0)

    aspect = NATURAL_KWARGS["width"] / NATURAL_KWARGS["size"]
    ncol = 2 if aspect >= 1.5 else 3
    nrow = int(np.ceil(args.n_maps / ncol))
    pw = 3.4 * max(1.0, aspect * 0.6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * pw, nrow * 3.0))
    axes = np.atleast_1d(axes).flatten()

    all_succ = []
    for j in range(args.n_maps):
        seed = args.eval_seed_start + j
        params, rec = _single_map_params(seed, cfg)
        key, sub = jax.random.split(key)
        positions, reached, placed, mined, face = batched_rollout(
            encoder, rssm, actor_head, wm_params, ac_params,
            env, params, args.n_traj, args.max_steps, sub,
        )
        succ = float(reached.mean())
        all_succ.append(succ)
        mine_pts, bridge_pts = _decision_points(positions, placed, mined, face)

        ax = axes[j]
        ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
        for i in range(args.n_traj):
            a = positions[:, i, :]
            jit = (np.random.rand(*a.shape) - 0.5) * 0.6   # tiny jitter to spread overlap
            ax.plot(a[:, 1] + jit[:, 1], a[:, 0] + jit[:, 0],
                    color="darkblue", lw=0.7, alpha=0.04 if reached[i] else 0.08)
        if mine_pts:
            m = np.array(mine_pts)
            ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
        if bridge_pts:
            b = np.array(bridge_pts)
            ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
        ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=28, marker="s",
                   edgecolors="k", zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seed {seed}  succ {succ:.0%}", fontsize=8)
        print(f"  map {j} (seed {seed}): succ {succ:.0%}  "
              f"mine={len(mine_pts)} bridge={len(bridge_pts)}")
    for j in range(args.n_maps, len(axes)):
        axes[j].axis("off")

    overall = float(np.mean(all_succ))
    fig.suptitle(f"{run_name}  ·  {args.n_traj} stochastic rollouts/map  ·  "
                 f"success {overall:.0%}  ·  "
                 f"path=darkblue  mine=yellow  bridge=red", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"success={overall:.2%}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
