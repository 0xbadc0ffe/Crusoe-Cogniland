#!/usr/bin/env python3
"""Build a DreamerV3 activation dataset for bridge_tunnel_commit — the Dreamer
sibling of ``build_activation_dataset.py`` (PPO). Same bundle layout, same held-out
maps and per-step labels, so the analysis pipeline (scripts/mechinterp/analysis)
runs unchanged and PPO↔Dreamer are directly comparable.

Activation sources captured per timestep (the RSSM latent = Dreamer's belief state):
  * rssm_deter        (deter_dim, e.g. 3072)  — deterministic recurrent state (belief carrier)
  * rssm_stoch_logits (stoch*classes, e.g. 576) — categorical posterior logits
  * enc_embed         (wm_hidden, e.g. 384)   — encoder output

Plus full obs (minimap, scalars), actor action_probs, critic value, and the same
belief/skill/strategy labels as the PPO bundle (category, commit_state,
final_commit, segment, decisions, …).

    python scripts/mechinterp/build_dreamer_activation_dataset.py \
        --checkpoint released_models/bridge_tunnel_commit/dreamer_commit_categorical/checkpoints/step_6000000 \
        --out-dir activation_datasets/btc_dreamer --n-traj 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

# reuse the PPO builder's label machinery + map list (identical labels + maps)
from mechinterp.build_activation_dataset import (  # noqa: E402
    _env_cfg, _map_geometry, _belief_row, _strategy, _build_map_list, _FACE_NAME)

from cogniland.bridge_tunnel.jax import (  # noqa: E402
    EnvParams, BridgeTunnelCommitJaxEnv, constants as C, records_to_arrays)
import purejaxwm.dreamerv3.behavior as ac  # noqa: E402
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM  # noqa: E402
from purejaxwm.dreamerv3.distributions import TwoHotDist  # noqa: E402
from purejaxwm.commons import resolve_dtype  # noqa: E402

_DECODER_MODE = "categorical"


# ───────────────────────── model (mirrors the eval / trainer) ─────────────
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


def _build_model(cfg):
    dt = resolve_dtype(cfg.get("compute_dtype", "float32"))
    encoder = BridgeTunnelEncoder(hidden=cfg["enc_hidden"], num_layers=cfg["enc_layers"],
                                  embed_dim=cfg["wm_hidden"], dtype=dt)
    rssm = RSSM(deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
                hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"], dtype=dt)
    actor = MLPHead(hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
                    out_dim=C.NUM_ACTIONS, outscale=0.01, dtype=dt)
    critic = MLPHead(hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
                     out_dim=cfg["num_reward_bins"], dtype=dt)
    return encoder, rssm, actor, critic


def _params_for(rec, cfg):
    arrays = records_to_arrays([rec])
    return EnvParams.from_map_arrays(
        **arrays, max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"], gamma=cfg["gamma"])


# ───────────────────────── capture rollout (vmap + scan) ──────────────────
def _make_rollout(models, wm_params, ac_params, env):
    encoder, rssm, actor, critic = models

    def rollout(params, n_traj, max_steps, key):
        key, kr = jax.random.split(key)
        obs0, state0 = jax.vmap(env.reset_env, in_axes=(0, None))(
            jax.random.split(kr, n_traj), params)
        rssm_state = rssm.initial_state((n_traj,))
        last_action = jnp.zeros((n_traj, C.NUM_ACTIONS))
        last_is_first = jnp.ones((n_traj,), dtype=bool)
        done = jnp.zeros((n_traj,), dtype=bool)

        def step(carry, _):
            state, obs, rssm_state, last_action, last_is_first, done, key = carry
            am = jnp.where(last_is_first[..., None], 0.0, last_action)
            flat = _flatten_obs(obs)
            key, s_stoch, s_pol, s_step = jax.random.split(key, 4)
            embed = encoder.apply(wm_params["encoder"], flat)
            _, post = rssm.apply(wm_params["rssm"], rssm_state, am, embed, last_is_first,
                                 rngs={"stoch": s_stoch})
            feat = post.features()
            logits = ac.unimix_logits(actor.apply(ac_params["actor"], feat))
            probs = jax.nn.softmax(logits, axis=-1)
            value = TwoHotDist(critic.apply(ac_params["critic"], feat)).mean()
            a = jax.random.categorical(s_pol, logits)
            nobs, nstate, _, done_next, info = jax.vmap(
                env.step_env, in_axes=(0, 0, 0, None))(
                jax.random.split(s_step, n_traj), state, a, params)

            def _sel(nx, pv):
                m = done.reshape(done.shape + (1,) * (nx.ndim - 1))
                return jnp.where(m, pv, nx)
            nstate = jax.tree_util.tree_map(_sel, nstate, state)
            nobs = jax.tree_util.tree_map(_sel, nobs, obs)
            out = {
                "deter": post.deter.astype(jnp.float16),
                "stoch_logits": post.logits.reshape(n_traj, -1).astype(jnp.float16),
                "embed": embed.astype(jnp.float16),
                "probs": probs.astype(jnp.float16),
                "value": value.astype(jnp.float32),
                "action": a.astype(jnp.int32),
                "minimap": obs["minimap"].astype(jnp.int8),
                "scalars": obs["scalars"].astype(jnp.float16),
                "pos_r": state.agent_r, "pos_c": state.agent_c, "facing": state.facing,
                "commit": state.commit, "commit_after": nstate.commit,
                "reached_now": info["reached_target"] & (~done),
                "already_done": done,
            }
            carry = (nstate, nobs, post, jax.nn.one_hot(a, C.NUM_ACTIONS),
                     jnp.zeros((n_traj,), bool), done | done_next, key)
            return carry, out

        carry = (state0, obs0, rssm_state, last_action, last_is_first, done, key)
        _, outs = jax.lax.scan(step, carry, None, length=max_steps)
        return outs

    return rollout


# ───────────────────────── per-traj label rows ────────────────────────────
_COMMIT = ("none", "build", "mine")


def _episode_rows(outs_np, i, geo, cfg, map_id, map_seed, category, traj_id, traj_seed,
                  min_cross, approach_window, near_radius, T):
    reached_now = outs_np["reached_now"][:, i]
    already = outs_np["already_done"][:, i]
    valid = ~already                                   # steps actually taken
    end = int(np.argmax(reached_now)) if reached_now.any() else int(valid.sum()) - 1
    end = max(end, 0)
    reached = bool(reached_now.any())
    L = end + 1

    pos = np.stack([outs_np["pos_r"][:L, i], outs_np["pos_c"][:L, i]], 1).astype(int)
    facing = outs_np["facing"][:L, i].astype(int)
    commit_pre = outs_np["commit"][:L, i].astype(int)
    commit_after = outs_np["commit_after"][:L, i].astype(int)
    anames = ["up", "down", "left", "right", "build", "mine"]
    path = [(int(p[0]), int(p[1])) for p in pos]
    seg, did, decisions = _strategy(path, geo, T, map_id, traj_id, traj_seed,
                                    min_cross, approach_window, near_radius)

    committed = (commit_pre == 0) & (commit_after != 0)
    commit_step = int(np.argmax(committed)) if committed.any() else -1
    final_commit = _COMMIT[int(commit_after[end])]
    dom = {"lakes": "build", "rocky": "mine"}.get(category)

    rows, acts = [], {"deter": [], "stoch_logits": [], "embed": [], "probs": [],
                      "minimap": [], "scalars": []}
    for k in range(L):
        a = int(outs_np["action"][k, i])
        cs = _COMMIT[int(commit_pre[k])]
        row = {
            "map_id": map_id, "map_seed": map_seed, "traj_id": traj_id,
            "traj_seed": traj_seed, "t": k, "pos_r": int(pos[k, 0]), "pos_c": int(pos[k, 1]),
            "facing": int(facing[k]), "facing_name": _FACE_NAME[int(facing[k])],
            "action": a, "action_name": anames[a], "value": float(outs_np["value"][k, i]),
            "category": category, "commit_state": cs,
            "committed_now": bool(committed[k]),
            "segment": seg[k], "decision_id": did[k],
            "reached": reached, "ep_len": L, "commit_step": commit_step,
            "time_since_commit": (k - commit_step) if (commit_step >= 0 and k >= commit_step) else -1,
            "final_commit": final_commit,
            "correct_commit": (cs == dom) if (dom is not None and cs != "none") else (cs != "none"),
        }
        row.update(_belief_row(geo, (int(pos[k, 0]), int(pos[k, 1])), int(facing[k]), T))
        rows.append(row)
        acts["deter"].append(outs_np["deter"][k, i])
        acts["stoch_logits"].append(outs_np["stoch_logits"][k, i])
        acts["embed"].append(outs_np["embed"][k, i])
        acts["probs"].append(outs_np["probs"][k, i])
        acts["minimap"].append(outs_np["minimap"][k, i])
        acts["scalars"].append(outs_np["scalars"][k, i])
    return rows, decisions, acts


# ─────────────────────────────────── main ─────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--maps-per-category", type=int, default=30)
    p.add_argument("--categories", default="balanced,lakes,rocky")
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--n-traj", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--min-cross", type=int, default=2)
    p.add_argument("--approach-window", type=int, default=8)
    p.add_argument("--near-radius", type=int, default=3)
    args = p.parse_args()

    ckpt_dir = args.checkpoint.resolve()
    cfg = json.loads((ckpt_dir.parent.parent / "config.json").read_text())
    global _DECODER_MODE
    _DECODER_MODE = cfg.get("decoder", "categorical")
    payload = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    wm_params = jax.tree_util.tree_map(jnp.asarray, payload["wm_params"])
    ac_params = jax.tree_util.tree_map(jnp.asarray, payload["ac_params"])
    models = _build_model(cfg)
    env = BridgeTunnelCommitJaxEnv()
    rollout = _make_rollout(models, wm_params, ac_params, env)

    ecfg = _env_cfg("bridge_tunnel_commit")
    T = ecfg["T"]
    natkw = dict(size=cfg["map_size"], width=cfg["map_width"], tree_frac=0.03,
                 goal_half=cfg.get("goal_half", 1))

    class _A:  # adapter for the shared map-list builder
        is_commit = True
    a2 = _A(); a2.categories = args.categories; a2.maps_per_category = args.maps_per_category
    a2.seed_start = args.seed_start; a2._natkw = natkw; a2.maps = None
    maps = _build_map_list({"is_commit": True, "gen": ecfg["gen"]}, a2)
    print(f"[setup] {len(maps)} held-out maps · {args.n_traj} rollouts/map · deter={cfg['deter']}", flush=True)

    all_rows, all_dec = [], []
    chunks = {"deter": [], "stoch_logits": [], "embed": [], "probs": [],
              "minimap": [], "scalars": []}
    key = jax.random.PRNGKey(0)
    for mi, m in enumerate(maps):
        geo = _map_geometry(m["rec"], T, ecfg["ctg_fn"])
        params = _params_for(m["rec"], cfg)
        key, sub = jax.random.split(key)
        outs = rollout(params, args.n_traj, args.max_steps, sub)
        outs_np = {k: np.asarray(v) for k, v in outs.items()}
        for i in range(args.n_traj):
            traj_seed = m["map_seed"] * 100000 + i
            rows, dec, acts = _episode_rows(
                outs_np, i, geo, cfg, m["map_id"], m["map_seed"], m["category"], i,
                traj_seed, args.min_cross, args.approach_window, args.near_radius, T)
            all_rows.extend(rows); all_dec.extend(dec)
            for kk in chunks:
                chunks[kk].append(np.asarray(acts[kk]))
        if (mi + 1) % 10 == 0:
            print(f"  map {mi+1}/{len(maps)}  rows so far={len(all_rows)}", flush=True)

    N = len(all_rows)
    for i, row in enumerate(all_rows):
        row["row_id"] = i
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── activations.h5 ──
    cat = lambda k: np.concatenate(chunks[k], 0)
    h5 = {
        "row_id": np.arange(N, dtype=np.int64),
        "rssm_deter": cat("deter"), "rssm_stoch_logits": cat("stoch_logits"),
        "enc_embed": cat("embed"), "action_probs": cat("probs"),
        "minimap": cat("minimap").astype(np.int8), "scalars": cat("scalars"),
    }
    import h5py
    with h5py.File(out_dir / "activations.h5", "w") as f:
        for k, v in h5.items():
            f.create_dataset(k, data=v, compression="gzip", compression_opts=4)

    # ── labels + decisions ──
    pd.DataFrame(all_rows).to_parquet(out_dir / "labels.parquet", index=False)
    pd.DataFrame(all_dec).to_parquet(out_dir / "decisions.parquet", index=False)

    # ── maps.npz ──
    terr = np.stack([np.asarray(m["rec"].terrain) for m in maps]).astype(np.int8)
    np.savez_compressed(
        out_dir / "maps.npz",
        terrain=terr,
        spawn=np.stack([np.asarray(m["rec"].spawn) for m in maps]).astype(np.int32),
        target=np.stack([np.asarray(m["rec"].target) for m in maps]).astype(np.int32),
        goal_mask=np.stack([(np.asarray(m["rec"].terrain) == T.TARGET) for m in maps]),
        map_seed=np.array([m["map_seed"] for m in maps], np.int64),
        category=np.array([m["category"] for m in maps]))

    # ── manifest ──
    sha = hashlib.sha256(ckpt_dir.read_bytes() if ckpt_dir.is_file()
                         else str(sorted(p.name for p in ckpt_dir.iterdir())).encode()).hexdigest()[:10]
    palette = np.asarray(T.TILE_COLORS, dtype=np.uint8).tolist()
    manifest = {
        "env": "bridge_tunnel_commit", "algo": "dreamerv3",
        "checkpoint": str(args.checkpoint), "agent_sha": sha,
        "decoder": _DECODER_MODE, "n_tiles": int(C.NUM_TILES),
        "view_size": int(cfg["view_size"]), "max_steps": int(args.max_steps),
        "n_scalars": int(h5["scalars"].shape[1]), "n_actions": int(C.NUM_ACTIONS),
        "action_names": ["up", "down", "left", "right", "build", "mine"],
        "natural_kwargs": natkw, "n_maps": len(maps), "n_traj_per_map": args.n_traj,
        "n_rows": N, "n_decisions": len(all_dec),
        "activation_sites": {"rssm_deter": int(cfg["deter"]),
                             "rssm_stoch_logits": int(cfg["stoch"] * cfg["classes"]),
                             "enc_embed": int(cfg["wm_hidden"])},
        "obs_stored": {"minimap": [int(cfg["view_size"]), int(cfg["view_size"]), "int8"],
                       "scalars": [int(h5["scalars"].shape[1]), "float16"]},
        "tile_names": {str(i): n for i, n in enumerate(
            ["grass", "water", "rock", "wood", "target", "oob", "tree", "sand", "dirt"])},
        "tile_colors": palette, "is_commit": True,
        "traj_seed_formula": "map_seed*100000 + traj_id",
        "reproduce": ("jax.random.PRNGKey(0) split per map; DreamerV3 encode→RSSM.observe"
                      "→unimix actor→categorical sample. Needs the repo (JAX/purejaxwm)."),
        "files": {"activations": "activations.h5", "labels": "labels.parquet",
                  "decisions": "decisions.parquet", "maps": "maps.npz"},
        "column_dictionary": {
            "rssm_deter": "RSSM deterministic state (belief carrier)",
            "rssm_stoch_logits": "RSSM categorical posterior logits (flattened)",
            "enc_embed": "encoder output", "category": "map category = belief label",
            "commit_state": "none/build/mine at step t", "final_commit": "commit at episode end"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ship the standalone renderer (works for Dreamer bundles too)
    import shutil
    shutil.copy(_ROOT / "activation_datasets" / "btc_ppo" / "decode_dataset.py",
                out_dir / "decode_dataset.py")
    (out_dir / "REPRODUCE.md").write_text(
        "# DreamerV3 bridge_tunnel_commit activation dataset\n\n"
        "Same layout as `btc_ppo` (same held-out maps + per-step labels) so the\n"
        "analysis pipeline runs unchanged. Activation sources = the RSSM latent\n"
        "(`rssm_deter`, `rssm_stoch_logits`) + `enc_embed`.\n\n"
        "`python decode_dataset.py --row N` renders a labelled frame (no repo needed).\n"
        "Reproducing the activations needs the repo (JAX + purejaxwm + the checkpoint);\n"
        "see `manifest.reproduce`.\n")

    print(f"\nwrote {out_dir}  rows={N}  maps={len(maps)}  decisions={len(all_dec)}")
    df = pd.DataFrame(all_rows)
    print("on-disk:", f"{(out_dir/'activations.h5').stat().st_size/1e6:.1f} MB")
    print("final_commit by category:\n",
          df.groupby(["category"])["final_commit"].value_counts().to_string())
    print(f"success: {df.groupby(['map_id','traj_id'])['reached'].first().mean():.1%}")


if __name__ == "__main__":
    main()
