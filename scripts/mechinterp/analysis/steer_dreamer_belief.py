#!/usr/bin/env python3
"""Belief steering on the DreamerV3 RSSM `deter` (the clean belief carrier).

Inject the belief axis (mean[deter|lakes] − mean[deter|rocky]) into the RSSM
deterministic state during rollouts and ask: does pushing the agent's *belief*
toward lakes / rocky causally change which *skill* it commits (build / mine),
while it still solves the env? Two injection sites:

  feature   add to the deter the ACTOR reads this step, but DON'T carry it
            forward (read-only / transient — the PPO-actor-input analog)
  recurrent  add to the carried deter (compounds through the RSSM)

Evaluated per map category on held-out maps; the cleanest test is BALANCED maps
(both obstacles present, so the commit is a free choice). Logs one W&B run.

    python -m scripts.mechinterp.analysis.steer_dreamer_belief \
        --checkpoint released_models/bridge_tunnel_commit/dreamerv3_commit/checkpoints/step_6000000 \
        --dataset activation_datasets/btc_dreamer --wandb-mode online
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp

from .bundle import ActivationBundle
from . import geometry as G, style

import sys
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
from cogniland.bridge_tunnel import generate_commit_map, tiles as TT       # noqa: E402
from cogniland.bridge_tunnel.jax import (                                   # noqa: E402
    EnvParams, BridgeTunnelCommitJaxEnv, constants as C, records_to_arrays)
import purejaxwm.dreamerv3.behavior as ac                                   # noqa: E402
from purejaxwm.dreamerv3.world_model import MLPHead, RSSM                    # noqa: E402
from purejaxwm.commons import resolve_dtype                                  # noqa: E402

CAT = ["rocky", "balanced", "lakes"]
_DECODER = "categorical"


class Enc(nn.Module):
    hidden: int; num_layers: int; embed_dim: int; dtype: jnp.dtype = jnp.float32
    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden, use_bias=False, dtype=self.dtype)(x)
            x = nn.RMSNorm(dtype=self.dtype)(x); x = jax.nn.silu(x)
        x = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype)(x)
        return jax.nn.silu(nn.RMSNorm(dtype=self.dtype)(x))


def _flat(obs):
    oh = jax.nn.one_hot(obs["minimap"].astype(jnp.int32), C.NUM_TILES)
    mm = oh.reshape(*oh.shape[:-3], -1)
    return jnp.concatenate([mm, obs["scalars"].astype(jnp.float32)], -1)


def _params_for(rec, cfg):
    return EnvParams.from_map_arrays(
        **records_to_arrays([rec]), max_steps=cfg["max_steps"], view_size=cfg["view_size"],
        slack_penalty=cfg["slack_penalty"], reach_bonus=cfg["reach_bonus"],
        shaping_coef=cfg["shaping_coef"], build_cost=cfg["build_cost"], gamma=cfg["gamma"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", default="activation_datasets/btc_dreamer")
    ap.add_argument("--per-cat", type=int, default=8)
    ap.add_argument("--n-traj", type=int, default=16)
    ap.add_argument("--alphas", type=float, nargs="*", default=[-4, -2, -1, 1, 2, 4])
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--out-dir", default="outputs/analysis_belief_steer")
    ap.add_argument("--wandb-project", default="bridge_tunnel_geometry")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--run-name", default="belief-steer-dreamer")
    args = ap.parse_args()

    import wandb, pandas as pd
    style.apply_theme()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- belief axis from the dataset deter (mean lakes − mean rocky) ----
    b = ActivationBundle(args.dataset)
    rng = np.random.default_rng(0)
    lab = b.labels
    samp = lab.groupby("category", observed=True, group_keys=False).apply(
        lambda g: g.sample(min(len(g), 20000), random_state=0)).reset_index(drop=True) \
        if False else lab.iloc[rng.choice(len(lab), min(60000, len(lab)), replace=False)]
    Xd = b.load_activations("rssm_deter", samp["row_id"])
    catd = samp["category"].to_numpy()
    b_raw = Xd[catd == "lakes"].mean(0) - Xd[catd == "rocky"].mean(0)
    gnorm = float(np.linalg.norm(b_raw))
    print(f"belief axis ||lakes-rocky|| = {gnorm:.2f}  (deter dim {Xd.shape[1]})")
    b_jnp = jnp.asarray(b_raw, jnp.float32)

    # ---- model ----
    ck = args.checkpoint.resolve()
    cfg = json.loads((ck.parent.parent / "config.json").read_text())
    dt = resolve_dtype(cfg.get("compute_dtype", "float32"))
    enc = Enc(cfg["enc_hidden"], cfg["enc_layers"], cfg["wm_hidden"], dt)
    rssm = RSSM(deter_dim=cfg["deter"], stoch_size=cfg["stoch"], classes=cfg["classes"],
                hidden=cfg["wm_hidden"], unimix=cfg["unimix"], blocks=cfg["blocks"], dtype=dt)
    actor = MLPHead(hidden=cfg["ac_hidden"], num_layers=cfg["ac_layers"],
                    out_dim=C.NUM_ACTIONS, outscale=0.01, dtype=dt)
    pay = ocp.PyTreeCheckpointer().restore(str(ck))
    wm = jax.tree_util.tree_map(jnp.asarray, pay["wm_params"])
    acp = jax.tree_util.tree_map(jnp.asarray, pay["ac_params"])
    env = BridgeTunnelCommitJaxEnv()

    def rollout(params, n_traj, max_steps, key, alpha, site):
        key, kr = jax.random.split(key)
        obs0, st0 = jax.vmap(env.reset_env, in_axes=(0, None))(jax.random.split(kr, n_traj), params)
        rs = rssm.initial_state((n_traj,))
        la = jnp.zeros((n_traj, C.NUM_ACTIONS)); lif = jnp.ones((n_traj,), bool)
        done = jnp.zeros((n_traj,), bool)

        def step(carry, _):
            st, obs, rs, la, lif, done, key = carry
            am = jnp.where(lif[..., None], 0.0, la)
            key, s1, s2, s3 = jax.random.split(key, 4)
            embed = enc.apply(wm["encoder"], _flat(obs))
            _, post = rssm.apply(wm["rssm"], rs, am, embed, lif, rngs={"stoch": s1})
            precommit = (st.commit == 0)
            steer = (alpha * b_jnp)[None, :] * precommit[:, None].astype(jnp.float32)
            deter_act = post.deter + steer
            feat = jnp.concatenate([deter_act, post.flat_stoch()], -1)
            logits = ac.unimix_logits(actor.apply(acp["actor"], feat))
            a = jax.random.categorical(s2, logits)
            post_carry = post._replace(deter=deter_act) if site == "recurrent" else post
            nobs, nst, _, dn, info = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(
                jax.random.split(s3, n_traj), st, a, params)

            def sel(nx, pv):
                m = done.reshape(done.shape + (1,) * (nx.ndim - 1))
                return jnp.where(m, pv, nx)
            nst = jax.tree_util.tree_map(sel, nst, st); nobs = jax.tree_util.tree_map(sel, nobs, obs)
            out = {"commit": nst.commit, "reached": info["reached_target"] & (~done)}
            return (nst, nobs, post_carry, jax.nn.one_hot(a, C.NUM_ACTIONS),
                    jnp.zeros((n_traj,), bool), done | dn, key), out

        _, outs = jax.lax.scan(step, (st0, obs0, rs, la, lif, done, key), None, length=max_steps)
        return np.asarray(outs["commit"][-1]), np.asarray(outs["reached"].any(0))

    # ---- eval maps (held-out, seed 10000+) ----
    gh = cfg.get("goal_half", 1)
    maps = []
    for ci, c in enumerate(CAT):
        for j in range(args.per_cat):
            rec = generate_commit_map(size=cfg["map_size"], width=cfg["map_width"],
                                      seed=10000 + ci * 100000 + j, category=c, tree_frac=0.03,
                                      goal_half=(gh if gh and gh >= 0 else None))
            maps.append((c, _params_for(rec, cfg)))
    print(f"{len(maps)} held-out maps  ·  sites=feature,recurrent  ·  alphas={args.alphas}")

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, name=args.run_name,
                     tags=["btc", "dreamer", "belief-steering"],
                     config=dict(alphas=args.alphas, per_cat=args.per_cat, n_traj=args.n_traj,
                                 belief_gap=gnorm))
    key = jax.random.PRNGKey(0)
    rows = []
    conds = [("baseline", "feature", 0.0)]
    for site in ["feature", "recurrent"]:
        conds += [(site, site, a) for a in args.alphas]
    for ci, (name, site, alpha) in enumerate(conds):
        per = {c: {"build": 0, "mine": 0, "none": 0, "reach": [], "n": 0} for c in CAT}
        for (c, params) in maps:
            key, sub = jax.random.split(key)
            commit, reached = rollout(params, args.n_traj, args.max_steps, sub, alpha, site)
            per[c]["build"] += int((commit == 1).sum())
            per[c]["mine"] += int((commit == 2).sum())
            per[c]["none"] += int((commit == 0).sum())
            per[c]["reach"].extend(reached.tolist()); per[c]["n"] += len(commit)
        for c in CAT:
            d = per[c]; n = max(d["n"], 1)
            rows.append(dict(site=site, alpha=alpha, category=c,
                             p_build=d["build"] / n, p_mine=d["mine"] / n, p_none=d["none"] / n,
                             skill_axis=(d["build"] - d["mine"]) / n,
                             reach=float(np.mean(d["reach"]))))
        bal = [r for r in rows if r["site"] == site and r["alpha"] == alpha and r["category"] == "balanced"][0]
        print(f"  [{ci+1}/{len(conds)}] {name:9s} a={alpha:+.1f}  bal build={bal['p_build']:.2f} "
              f"mine={bal['p_mine']:.2f} reach={bal['reach']:.2f}", flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "belief_steer.csv", index=False)
    run.log({"belief_steer/table": wandb.Table(dataframe=tab)})
    base = {r["category"]: r for r in rows if r["site"] == "feature" and r["alpha"] == 0.0}
    _figs(run, out, tab, base)
    run.summary.update({
        f"{s}/balanced_skill_swing": float(
            tab[(tab.site == s) & (tab.category == "balanced")]["skill_axis"].max()
            - tab[(tab.site == s) & (tab.category == "balanced")]["skill_axis"].min())
        for s in ["feature", "recurrent"]})
    run.finish(); print("DONE")


def _figs(run, out, tab, base):
    import wandb
    for site in ["feature", "recurrent"]:
        fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
        for c in CAT:
            d = tab[(tab.site == site) & (tab.category == c)].sort_values("alpha")
            ax[0].plot(d.alpha, d.skill_axis, "-o", color=style.CATEGORY_COLORS[c], label=c)
            ax[1].plot(d.alpha, d.reach, "-o", color=style.CATEGORY_COLORS[c], label=c)
        ax[0].axhline(0, color="#999", lw=0.7)
        ax[0].set_title(f"{site}: commit P(build)−P(mine) vs belief α\n(α>0 → lakes, α<0 → rocky)")
        ax[1].set_title(f"{site}: success (reach)"); ax[1].set_ylim(0, 1.03)
        for a_ in ax:
            a_.set_xlabel("belief injection α (× lakes−rocky gap)")
            a_.grid(True, color=style.GRIDC); a_.set_facecolor(style.PANEL); a_.legend(fontsize=8)
        fig.suptitle(f"Belief steering on Dreamer rssm_deter — {site} site", fontweight="bold")
        fig.tight_layout()
        p = out / f"belief_steer_{site}.png"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        run.log({f"belief_steer/{site}": wandb.Image(str(p))}); plt.close(fig)


if __name__ == "__main__":
    main()
