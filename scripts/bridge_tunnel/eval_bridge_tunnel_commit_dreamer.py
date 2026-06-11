#!/usr/bin/env python3
"""Evaluate a trained bridge_tunnel_commit DreamerV3 agent: the 3×3 belief→skill
commit matrix + a per-category trajectory grid (the Dreamer sibling of
``eval_bridge_tunnel_commit_ppo.py``).

Reconstructs the world model + actor exactly as ``dreamerv3_bridge_tunnel_commit.py``
built them (config from ``<run>/config.json``), restores the orbax checkpoint,
and runs the SAME per-step inference as the trainer's rollout (encode → RSSM
observe → unimix actor logits → categorical sample). Held-out maps come from
``generate_commit_map(seed=10000+, category=...)`` per category.

    python scripts/bridge_tunnel/eval_bridge_tunnel_commit_dreamer.py \\
        --checkpoint runs/dreamerv3_commit_25M_<ts>/checkpoints/step_1500000 \\
        --out-prefix paper/figures/bridge_tunnel_commit/dreamer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from cogniland.bridge_tunnel import generate_commit_map, tiles as T  # noqa: E402
from cogniland.bridge_tunnel.mapgen import CATEGORIES  # noqa: E402
from cogniland.bridge_tunnel.jax import (  # noqa: E402
    EnvParams, BridgeTunnelCommitJaxEnv, constants as C, records_to_arrays,
)
import purejaxwm.dreamerv3.behavior as ac  # noqa: E402
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM  # noqa: E402
from purejaxwm.commons import resolve_dtype  # noqa: E402

_FACE_DELTA = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int32)
COMMIT_NAMES = ["none", "commit_build", "commit_mine"]
_DECODER_MODE = "categorical"
# path colour by commitment state: none=blue, build=yellow, mine=purple
_COMMIT_COLORS = {0: "#1f5fd0", 1: "#ffd000", 2: "#a800e6"}


def _draw_commit_path(ax, pos, cm, reached):
    """Draw one trajectory as segments coloured by commitment (blue=none,
    yellow=build, orange=mine). Segment pos[k]->pos[k+1] uses cm[k+1]."""
    pos = np.asarray(pos, dtype=float)
    cm = np.asarray(cm)
    if len(pos) < 2:
        return
    jit = (np.random.rand(*pos.shape) - 0.5) * 0.6
    xy = np.stack([(pos + jit)[:, 1], (pos + jit)[:, 0]], axis=1)
    segs = np.stack([xy[:-1], xy[1:]], axis=1)
    colors = [_COMMIT_COLORS.get(int(c), "gray") for c in cm[1:]]
    lc = LineCollection(segs, colors=colors, linewidths=0.6,
                        alpha=0.045 if reached else 0.07)
    ax.add_collection(lc)


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
            x = nn.Dense(self.hidden, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return jax.nn.silu(x)


def _flatten_obs(obs):
    if _DECODER_MODE == "categorical":
        oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
        mm = oh.reshape(*oh.shape[:-3], -1)
    else:
        mm = (obs["minimap"].astype(jnp.float32) / float(C.NUM_TILES))
        mm = mm.reshape(*mm.shape[:-2], -1)
    return jnp.concatenate([mm, obs["scalars"].astype(jnp.float32)], axis=-1)


def _params_for(rec, cfg):
    arrays = records_to_arrays([rec])
    return EnvParams.from_map_arrays(
        **arrays, max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"], gamma=cfg["gamma"])


def _build_model(cfg):
    compute_dtype = resolve_dtype(cfg.get("compute_dtype", "float32"))
    encoder = BridgeTunnelEncoder(hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                  embed_dim=cfg["wm_hidden"], dtype=compute_dtype)
    rssm = RSSM(deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
                hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"], dtype=compute_dtype)
    actor_head = MLPHead(hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
                         out_dim=C.NUM_ACTIONS, outscale=0.01, dtype=compute_dtype)
    return encoder, rssm, actor_head


def batched_rollout(encoder, rssm, actor_head, wm_params, ac_params, env, params,
                    n_traj, max_steps, key):
    """``n_traj`` stochastic rollouts on one fixed map. Returns
    (positions[T+1,n,2], reached[n], placed[T,n], mined[T,n], face[T,n], commit[T,n])."""
    action_dim = C.NUM_ACTIONS
    key, kr = jax.random.split(key)
    reset_keys = jax.random.split(kr, n_traj)
    obs0, state0 = jax.vmap(env.reset_env, in_axes=(0, None))(reset_keys, params)
    rssm_state = rssm.initial_state((n_traj,))
    last_action = jnp.zeros((n_traj, action_dim))
    last_is_first = jnp.ones((n_traj,), dtype=bool)
    done = jnp.zeros((n_traj,), dtype=bool)

    def scan_step(carry, _):
        state, obs, rssm_state, last_action, last_is_first, done, key = carry
        action_masked = jnp.where(last_is_first[..., None], jnp.zeros_like(last_action), last_action)
        flat = _flatten_obs(obs)
        key, s_stoch, s_pol, s_step = jax.random.split(key, 4)
        embed = encoder.apply(wm_params["encoder"], flat)
        _, posterior = rssm.apply(wm_params["rssm"], rssm_state, action_masked, embed,
                                  last_is_first, rngs={"stoch": s_stoch})
        feat = posterior.features()
        logits = ac.unimix_logits(actor_head.apply(ac_params["actor"], feat))
        action_idx = jax.random.categorical(s_pol, logits)
        action_oh = jax.nn.one_hot(action_idx, action_dim)
        step_keys = jax.random.split(s_step, n_traj)
        next_obs, next_state, reward, done_next, info = jax.vmap(
            env.step_env, in_axes=(0, 0, 0, None))(step_keys, state, action_idx, params)

        def _sel(a_next, a_prev):
            mask = done.reshape(done.shape + (1,) * (a_next.ndim - 1))
            return jnp.where(mask, a_prev, a_next)
        next_state = jax.tree_util.tree_map(_sel, next_state, state)
        next_obs = jax.tree_util.tree_map(_sel, next_obs, obs)
        placed = info["placed"] & (~done)
        mined = info["mined"] & (~done)
        reached_now = info["reached_target"] & (~done)
        new_done = done | done_next
        out = {"pos": jnp.stack([next_state.agent_r, next_state.agent_c], axis=-1),
               "placed": placed, "mined": mined, "face": next_state.facing,
               "reached": reached_now, "commit": next_state.commit}
        new_carry = (next_state, next_obs, posterior, action_oh,
                     jnp.zeros((n_traj,), dtype=bool), new_done, key)
        return new_carry, out

    carry = (state0, obs0, rssm_state, last_action, last_is_first, done, key)
    _, outs = jax.lax.scan(scan_step, carry, None, length=max_steps)
    pos0 = jnp.stack([state0.agent_r, state0.agent_c], axis=-1)[None]
    positions = jnp.concatenate([pos0, outs["pos"]], axis=0)
    reached = outs["reached"].any(axis=0)
    return (np.asarray(positions), np.asarray(reached), np.asarray(outs["placed"]),
            np.asarray(outs["mined"]), np.asarray(outs["face"]), np.asarray(outs["commit"]))


def _decision_and_commit_points(positions, placed, mined, face, commit):
    """faced cells of successful PLACE/MINE + the cell where each episode first
    committed (commit transitions 0→nonzero)."""
    mine_pts, bridge_pts, commit_pts = [], [], []
    Tn, n = placed.shape
    prev = np.zeros(n, dtype=commit.dtype)
    for t in range(Tn):
        pos_t = positions[t + 1]
        for ev_mask, store in ((mined[t], mine_pts), (placed[t], bridge_pts)):
            for i in np.nonzero(ev_mask)[0]:
                d = _FACE_DELTA[face[t, i]]
                store.append((pos_t[i, 0] + d[0], pos_t[i, 1] + d[1]))
        newly = np.nonzero((prev == 0) & (commit[t] != 0))[0]
        for i in newly:
            commit_pts.append((pos_t[i, 0], pos_t[i, 1]))
        prev = commit[t]
    return mine_pts, bridge_pts, commit_pts


def plot_matrix(matrix, succ, title, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(COMMIT_NAMES, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{c}\n(succ {succ[c]:.0%})" for c in CATEGORIES], fontsize=10)
    ax.set_xlabel("committed skill", fontsize=11)
    ax.set_ylabel("map category (belief)", fontsize=11)
    for i in range(3):
        for j in range(3):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if v < 0.6 else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of episodes")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, default=Path("paper/figures/bridge_tunnel_commit/dreamer"))
    p.add_argument("--matrix-maps", type=int, default=20)
    p.add_argument("--matrix-traj", type=int, default=16)
    p.add_argument("--grid-seeds", type=int, default=4)
    p.add_argument("--grid-traj", type=int, default=120)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()

    ckpt_dir = args.checkpoint.resolve()
    cfg = json.loads((ckpt_dir.parent.parent / "config.json").read_text())
    tag = ckpt_dir.parent.parent.name
    global _DECODER_MODE
    _DECODER_MODE = cfg.get("decoder", "categorical")
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    wm_params = jax.tree_util.tree_map(jnp.asarray, payload["wm_params"])
    ac_params = jax.tree_util.tree_map(jnp.asarray, payload["ac_params"])
    encoder, rssm, actor_head = _build_model(cfg)
    env = BridgeTunnelCommitJaxEnv()
    key = jax.random.PRNGKey(0)
    gh = cfg.get("goal_half", 1)

    def make_map(cat, seed):
        return generate_commit_map(size=cfg["map_size"], width=cfg["map_width"], seed=seed,
                                   category=cat, tree_frac=0.03,
                                   goal_half=(gh if (gh is not None and gh >= 0) else None))

    # --- 3×3 commit matrix ---
    counts = np.zeros((3, 3), dtype=np.float64)
    succ = {c: [] for c in CATEGORIES}
    for ci, cat in enumerate(CATEGORIES):
        for j in range(args.matrix_maps):
            rec = make_map(cat, args.eval_seed_start + j)
            params = _params_for(rec, cfg)
            key, sub = jax.random.split(key)
            _, reached, _, _, _, commit = batched_rollout(
                encoder, rssm, actor_head, wm_params, ac_params, env, params,
                args.matrix_traj, args.max_steps, sub)
            fc = commit[-1]
            for v in fc:
                counts[ci, int(v)] += 1
            succ[cat].extend(reached.tolist())
    matrix = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    succ = {c: float(np.mean(v)) if v else 0.0 for c, v in succ.items()}
    print("commit matrix (rows=category, cols=none/build/mine):")
    for i, c in enumerate(CATEGORIES):
        print(f"  {c:9s} {matrix[i]}  succ={succ[c]:.2%}")
    plot_matrix(matrix, succ, f"DreamerV3  ·  belief→skill commit matrix\n{tag}",
                Path(str(args.out_prefix) + "_commit_matrix.png"))

    # --- trajectory grid ---
    fig, axes = plt.subplots(len(CATEGORIES), args.grid_seeds,
                             figsize=(args.grid_seeds * 3.0, len(CATEGORIES) * 2.0))
    axes = np.asarray(axes).reshape(len(CATEGORIES), args.grid_seeds)
    for ci, cat in enumerate(CATEGORIES):
        for sj in range(args.grid_seeds):
            rec = make_map(cat, args.eval_seed_start + sj)
            params = _params_for(rec, cfg)
            key, sub = jax.random.split(key)
            positions, reached, placed, mined, face, commit = batched_rollout(
                encoder, rssm, actor_head, wm_params, ac_params, env, params,
                args.grid_traj, args.max_steps, sub)
            mine_pts, bridge_pts, commit_pts = _decision_and_commit_points(
                positions, placed, mined, face, commit)
            ax = axes[ci, sj]
            ax.imshow(T.TILE_COLORS[rec.terrain], interpolation="nearest")
            for i in range(positions.shape[1]):
                cm_i = np.concatenate([[0], commit[:, i]])     # align with T+1 positions
                _draw_commit_path(ax, positions[:, i, :], cm_i, reached[i])
            if mine_pts:
                m = np.array(mine_pts); ax.scatter(m[:, 1], m[:, 0], color="yellow", s=6, alpha=0.18, zorder=3, linewidths=0)
            if bridge_pts:
                b = np.array(bridge_pts); ax.scatter(b[:, 1], b[:, 0], color="red", s=6, alpha=0.18, zorder=3, linewidths=0)
            ax.scatter([rec.spawn[1]], [rec.spawn[0]], color="white", s=22, marker="s", edgecolors="k", zorder=5)
            fcl = commit[-1]
            fb = float((fcl == 1).mean()); fm = float((fcl == 2).mean()); fn = float((fcl == 0).mean())
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{cat} s{args.eval_seed_start+sj}  succ {reached.mean():.0%}\n"
                         f"build {fb:.0%}/mine {fm:.0%}/none {fn:.0%}", fontsize=7)
    fig.suptitle(f"DreamerV3 bridge_tunnel_commit  ·  {tag}  ·  {args.grid_traj} rollouts/map  ·  "
                 f"line=commitment (blue none / yellow build / purple mine)  ·  "
                 f"dots: build=red mine=yellow", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(str(args.out_prefix) + "_traj.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
