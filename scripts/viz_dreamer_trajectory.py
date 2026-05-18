"""Roll out a frozen Dreamer checkpoint and visualise trajectories.

Loads a checkpoint saved by ``scripts/dreamerv3_crafter_in_cogniland.py``,
runs the policy deterministically (argmax) on N validation maps, and
writes:

* ``<out_dir>/trajectories.png``   — per-map top-down map + agent path
* ``<out_dir>/trajectories.json``  — per-map success / length / actions
* ``<out_dir>/imaginations.png``   — top row: real obs sequence after
  decoding a frame; bottom row: imagined obs sequence after H rollout
  steps from the same start. Lets you eyeball world-model quality.

This is the analysis entry point for the mech-interp workflow — the
``frozen_model`` returned by ``load_frozen()`` exposes the encoder,
RSSM, decoder, and actor as Flax apply-fns over a single params pytree,
so you can wire it into your own probes without re-training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogniland.crafter_in_cogniland import (
    CrafterInCognilandEnv, EnvParams, load_map_arrays,
    constants as C, build_obs,
)
from cogniland.nav.tiles import TILE_COLORS

from purejaxwm.dreamerv3 import behavior as ac
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM
import flax.linen as nn


# ─────────────────────────────────────────────────────────────
# minimal re-construction of the model graph; the values live in
# the checkpoint, this script just builds compatible apply-fns.
# ─────────────────────────────────────────────────────────────


class _Encoder(nn.Module):
    hidden: int; num_layers: int; embed_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False)(x)
            x = nn.RMSNorm()(x)
            x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False)(x)
        x = nn.RMSNorm()(x)
        return jax.nn.silu(x)


class _Decoder(nn.Module):
    hidden: int; num_layers: int; out_dim: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False)(x)
            x = nn.RMSNorm()(x)
            x = jax.nn.silu(x)
        return nn.Dense(self.out_dim, use_bias=True)(x)


def load_frozen(ckpt_dir: Path, cfg: dict):
    """Restore params + return apply-fns wired to those params.

    Returns a dict with keys: ``encoder``, ``decoder``, ``rssm``, ``actor``,
    ``critic``, ``params``. Apply-fns take ``(input,)`` only — params are
    closed over for ergonomic mech-interp inspection.
    """
    flat_dim = cfg["view_size"] * cfg["view_size"] + 5
    encoder = _Encoder(cfg["enc_hidden"], cfg["enc_layers"], cfg["wm_hidden"])
    decoder = _Decoder(cfg["enc_hidden"], cfg["enc_layers"], flat_dim)
    rssm = RSSM(
        deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
        hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"],
    )
    actor = MLPHead(cfg["ac_hidden"], cfg["ac_layers"], C.NUM_ACTIONS, 0.01)
    critic = MLPHead(cfg["ac_hidden"], cfg["ac_layers"], cfg["num_reward_bins"], 0.0)

    # Use orbax to load the params payload
    restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir.resolve()))
    wm_params = restored["wm_params"]
    ac_params = restored["ac_params"]
    return {
        "encoder": encoder, "decoder": decoder, "rssm": rssm,
        "actor": actor, "critic": critic,
        "wm_params": wm_params, "ac_params": ac_params,
    }


def rollout_one_episode(env, env_params, params_dict, rng, max_steps=300):
    """Run one greedy episode; return (states, actions, rewards, reached)."""
    encoder = params_dict["encoder"]
    rssm = params_dict["rssm"]
    actor = params_dict["actor"]
    wm = params_dict["wm_params"]
    ac_p = params_dict["ac_params"]

    rng, sub = jax.random.split(rng)
    obs, state = env.reset(sub, env_params)
    states = [state]
    actions = []
    rewards = []
    rssm_state = rssm.initial_state((1,))
    is_first = jnp.ones((1,), dtype=bool)
    last_action_oh = jnp.zeros((1, C.NUM_ACTIONS))
    done = False
    reached = False

    for t in range(max_steps):
        rng, s_enc, s_pol, s_step = jax.random.split(rng, 4)
        # flatten obs to match training pipeline
        mm = obs["minimap"].astype(jnp.float32)[None] / float(C.NUM_TERRAIN_TILES)
        sc = obs["scalars"].astype(jnp.float32)[None]
        flat = jnp.concatenate([mm.reshape(1, -1), sc], axis=-1)
        embed = encoder.apply(wm["encoder"], flat)
        _, posterior = rssm.apply(
            wm["rssm"], rssm_state, last_action_oh, embed, is_first,
            rngs={"stoch": s_enc},
        )
        rssm_state = posterior
        feat = posterior.features()
        logits = actor.apply(ac_p["actor"], feat)
        action_idx = jnp.argmax(logits, axis=-1)        # deterministic
        a_oh = jax.nn.one_hot(action_idx, C.NUM_ACTIONS)
        obs, state, r, done, info = env.step(s_step, state, action_idx[0], env_params)
        states.append(state)
        actions.append(int(action_idx[0]))
        rewards.append(float(r))
        last_action_oh = a_oh
        is_first = jnp.zeros((1,), dtype=bool)
        if bool(info["reached_target"]):
            reached = True
        if bool(done):
            break
    return states, actions, rewards, reached


def plot_trajectory(ax, env_params, state_hist, reached):
    map_idx = int(state_hist[0].map_idx)
    terrain = np.asarray(env_params.terrain[map_idx])
    target = np.asarray(env_params.target[map_idx])
    rgb = TILE_COLORS[terrain]
    ax.imshow(rgb)
    rs = [int(s.agent_r) for s in state_hist]
    cs = [int(s.agent_c) for s in state_hist]
    ax.plot(cs, rs, "-", color="white", linewidth=1.5, alpha=0.8)
    ax.plot(cs[0], rs[0], "o", color="cyan", markersize=6)
    ax.plot(cs[-1], rs[-1], "o", color="magenta", markersize=6)
    ax.plot(target[1], target[0], "*", color="yellow", markersize=12)
    ax.set_title(f"map={map_idx} reached={reached} steps={len(state_hist)}",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="path to runs/<id>/checkpoints/step_<N>")
    parser.add_argument("--config", type=Path, default=None,
                        help="config.json from same run; defaults to ../../config.json")
    parser.add_argument("--maps-path", required=True,
                        help="pkl produced by save_map_arrays")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.config is None:
        args.config = args.checkpoint.parents[1] / "config.json"
    if args.out_dir is None:
        args.out_dir = args.checkpoint.parents[1] / "viz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())

    print(f"loading checkpoint from {args.checkpoint} ...")
    pd = load_frozen(args.checkpoint, cfg)

    arrays = load_map_arrays(args.maps_path)
    env_params = EnvParams.from_map_arrays(
        **arrays, max_steps=cfg["max_steps"], view_size=cfg["view_size"],
    )
    env = CrafterInCognilandEnv(default_params=env_params)

    rng = jax.random.PRNGKey(args.seed)
    n = args.n_episodes
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    results = []
    n_reached = 0
    for i in range(n):
        rng, sub = jax.random.split(rng)
        states, actions, rewards, reached = rollout_one_episode(
            env, env_params, pd, sub, max_steps=args.max_steps,
        )
        plot_trajectory(axes[i], env_params, states, reached)
        results.append({
            "episode": i,
            "map_idx": int(states[0].map_idx),
            "reached": bool(reached),
            "steps": len(states),
            "total_reward": float(sum(rewards)),
            "actions": list(map(int, actions)),
        })
        n_reached += int(reached)
        print(f"ep {i}: map={int(states[0].map_idx)} reached={reached} "
              f"steps={len(states)} reward={sum(rewards):+.2f}")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    success_rate = n_reached / n
    fig.suptitle(
        f"Greedy rollouts — success {n_reached}/{n} ({100*success_rate:.0f}%)"
    )
    fig.tight_layout()
    out_png = args.out_dir / "trajectories.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    out_json = args.out_dir / "trajectories.json"
    out_json.write_text(json.dumps({
        "n_episodes": n, "n_reached": n_reached, "success_rate": success_rate,
        "episodes": results,
    }, indent=2))
    print(f"wrote {out_png}, {out_json}")


if __name__ == "__main__":
    main()
