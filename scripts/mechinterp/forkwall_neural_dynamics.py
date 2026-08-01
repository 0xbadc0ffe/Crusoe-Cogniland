#!/usr/bin/env python3
"""Learning dynamics of the fork_wall recurrent controller, in the style of
Huang, Singh & Rajan (RLC 2024) but adapted to a discrete-latent POMDP.

Three departures from the pendulum setting, each forced by the task:

  * ``x = 0`` is meaningless here — a zero one-hot minimap is off-manifold. We
    therefore compute INPUT-CONDITIONED slow points, holding the observation
    embedding fixed at canonical views (open corridor / facing water / facing
    rock) taken from real rollouts.

  * Naive kinetic-energy descent WANDERS OFF the data manifold into quiescent
    regions the agent never visits, which yields a spurious all-|lambda|~1
    spectrum and integration times of ~1e4 steps against ~100-step episodes.
    We add an on-manifold penalty pulling candidates toward the convex region
    of visited states, and report how far each solution sits from real data.

  * The pendulum's circular state gives a ring; fork_wall's latent is a
    discrete 3-way category accumulated from evidence, so the prediction is an
    integrator (line/plane) manifold, not a ring. We characterise the found set
    by its participation-ratio dimensionality instead of assuming a shape.

Per checkpoint we report: number and dimensionality of slow points, the top
stimulus-integration timescales tau_i = |1/ln|lambda_i|| from the recurrence
Jacobian, and the fraction of eigenvalues that are marginal (|lambda| > 0.99).

    python scripts/mechinterp/forkwall_neural_dynamics.py --seeds 1 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bridge_tunnel"))

from cogniland.bridge_tunnel import tiles as T  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from cogniland.bridge_tunnel.mapgen import generate_commit_map  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402

CATS = ("balanced", "lakes", "rocky")


def participation_ratio(X):
    """Effective dimensionality: (sum lambda)^2 / sum lambda^2 of the covariance."""
    if len(X) < 2:
        return float("nan")
    ev = np.linalg.eigvalsh(np.cov(X - X.mean(0), rowvar=False))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


@torch.no_grad()
def collect_states_and_inputs(policy, cargs, view_size, env_size, env_width, device,
                              commit, n_traj=8, seed=90_000, max_steps=400):
    """Real hidden states + canonical observation embeddings for conditioning."""
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    hs = []
    for cat in CATS:
        rec = generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                  tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                  fork_wall=True, passage_half=cargs.get("passage_half", 1),
                                  wall_margin=cargs.get("wall_margin", 1))
        o = batched_rollout_steered(policy, rec, n_traj, view_size, max_steps, device,
                                    commit=commit, steer_fn=None, collect_hidden=True)
        if o["hiddens"] is not None:
            hs.append(o["hiddens"])
    H = np.concatenate(hs)

    # canonical inputs: place the agent on open grass, in front of water, and in
    # front of rock, on a balanced map; embed each observation once.
    rec = generate_commit_map(size=env_size, width=env_width, seed=seed, category="balanced",
                              tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                              fork_wall=True, passage_half=cargs.get("passage_half", 1),
                              wall_margin=cargs.get("wall_margin", 1))
    Hh, Ww = rec.terrain.shape
    targets = {"open": T.GRASS, "water": T.WATER, "rock": T.ROCK}
    inputs = {}
    for name, tile in targets.items():
        cand = np.argwhere(rec.terrain == tile)
        cand = [p for p in cand if 2 < p[1] < Ww - 3]
        if not cand:
            continue
        r, c = cand[len(cand) // 2]
        env = BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww, view_size=view_size,
                                    max_steps=max_steps, commit=commit)
        env.reset()
        env._pos = [int(np.clip(r, 1, Hh - 2)), int(np.clip(c - 1, 1, Ww - 2))]
        ob = env._make_obs()
        mm = torch.from_numpy(ob["minimap"])[None].to(device)
        sc = torch.from_numpy(ob["scalars"])[None].to(device)
        inputs[name] = policy._encode({"minimap": mm, "scalars": sc})
    return H, inputs


def find_slow_points(policy, H, x_fixed, device, n_init=192, iters=1500,
                     lr=0.05, manifold_w=1e-3, q_tol=1e-5):
    """Minimise kinetic energy with an ON-MANIFOLD penalty.

    q(h) = 1/2||F(h,x) - h||^2 + manifold_w * dist(h, nearest real state)^2

    The penalty keeps candidates in the region the agent actually occupies;
    without it the optimiser drifts into quiescent off-manifold territory where
    every point is trivially slow.
    """
    gru = policy.gru
    Hd = gru.hidden_size
    Href = torch.tensor(H, dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(H), size=min(n_init, len(H)), replace=False)
    h = torch.tensor(H[idx], dtype=torch.float32, device=device).clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)

    def F(z):
        _, hn = gru(x_fixed.expand(z.shape[0], -1)[None], z[None].contiguous())
        return hn.squeeze(0)

    for _ in range(iters):
        opt.zero_grad()
        q = 0.5 * ((F(h) - h) ** 2).sum(-1)
        d2 = torch.cdist(h, Href).min(dim=1).values ** 2
        (q + manifold_w * d2).sum().backward()
        opt.step()

    with torch.no_grad():
        qf = 0.5 * ((F(h) - h) ** 2).sum(-1)
        dist = torch.cdist(h, Href).min(dim=1).values
    hf = h.detach()
    keep = qf < q_tol
    return hf, qf, dist, keep


def jacobian_taus(policy, h0, x_fixed, device, top=8):
    gru = policy.gru

    def F1(z):
        _, hn = gru(x_fixed[None], z[None, None].contiguous())
        return hn.reshape(-1)

    J = torch.autograd.functional.jacobian(F1, h0.clone().requires_grad_(True))
    ev = np.linalg.eigvals(J.detach().cpu().numpy())
    mag = np.sort(np.abs(ev))[::-1]
    stable = mag[mag < 1.0]
    taus = np.abs(1.0 / np.log(np.clip(stable, 1e-12, 1 - 1e-12)))
    return mag, taus[:top]



@torch.no_grad()
def collect_state_input_pairs(policy, cargs, view_size, env_size, env_width, device,
                              commit, n_traj=6, seed=90_000, max_steps=400, cap=400):
    """Real (h_t, x_t) pairs off trajectories — the ON-MANIFOLD expansion points.

    The pendulum paper linearises at fixed points, but its own derivation holds
    at any expansion point (h^e, x^e). In fork_wall the agent never sits near a
    fixed point (states move ~60% of their norm per step under a 441-dim
    observation that refreshes every step), so trajectory states are the honest
    place to linearise: they are where the computation actually happens.
    """
    from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv
    gh = cargs.get("goal_half", 0)
    gh = gh if (gh is not None and gh >= 0) else None
    Hs, Xs = [], []
    for cat in CATS:
        rec = generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                  tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                  fork_wall=True, passage_half=cargs.get("passage_half", 1),
                                  wall_margin=cargs.get("wall_margin", 1))
        Hh, Ww = rec.terrain.shape
        envs = [BridgeTunnelCommitEnv(map_record=rec, size=Hh, width=Ww, view_size=view_size,
                                      max_steps=max_steps, commit=commit) for _ in range(n_traj)]
        obs = [e.reset()[0] for e in envs]
        h = torch.zeros(1, n_traj, policy.gru_hidden, device=device)
        done = torch.zeros(n_traj, device=device)
        active = np.ones(n_traj, dtype=bool)
        for _ in range(max_steps):
            mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
            sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
            feat = policy._encode({"minimap": mm.squeeze(0), "scalars": sc.squeeze(0)})
            Hs.append(h.squeeze(0)[active].cpu().numpy())
            Xs.append(feat[active].cpu().numpy())
            _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
            x = h.squeeze(0)
            logits, _ = policy._heads(x)
            acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            for i, e in enumerate(envs):
                if not active[i]:
                    continue
                o, _, term, trunc, _ = e.step(int(acts[i]))
                obs[i] = o
                if term or trunc:
                    active[i] = False
            done = torch.zeros(n_traj, device=device)
            if not active.any():
                break
    H = np.concatenate(Hs); X = np.concatenate(Xs)
    if len(H) > cap:
        idx = np.random.default_rng(0).choice(len(H), size=cap, replace=False)
        H, X = H[idx], X[idx]
    return H, X


def trajectory_taus(policy, H, X, device, n_points=48, top=5):
    """Recurrence-Jacobian spectrum at REAL (h, x) pairs.

    Returns (median tau per mode, frac of |lambda| > 0.99, median spectral radius).
    """
    gru = policy.gru
    rng = np.random.default_rng(0)
    idx = rng.choice(len(H), size=min(n_points, len(H)), replace=False)
    taus, margs, radii = [], [], []
    for i in idx:
        h0 = torch.tensor(H[i], dtype=torch.float32, device=device)
        x0 = torch.tensor(X[i], dtype=torch.float32, device=device)

        def F1(z):
            _, hn = gru(x0[None, None], z[None, None].contiguous())
            return hn.reshape(-1)

        J = torch.autograd.functional.jacobian(F1, h0.clone().requires_grad_(True))
        ev = np.abs(np.linalg.eigvals(J.detach().cpu().numpy()))
        ev = np.sort(ev)[::-1]
        radii.append(float(ev[0]))
        margs.append(float((ev > 0.99).mean()))
        st = ev[ev < 1.0][:top]
        t = np.abs(1.0 / np.log(np.clip(st, 1e-12, 1 - 1e-12)))
        taus.append(np.pad(t, (0, max(0, top - len(t))), constant_values=np.nan))
    taus = np.vstack(taus)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(taus, axis=0)
    return med, float(np.median(margs)), float(np.median(radii))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-prefix", default="ppo_gru_forkwall_noaux_dense_seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 3])
    p.add_argument("--ckpt-root", type=Path, default=REPO / "outputs/ppo_checkpoints")
    p.add_argument("--context", default="open", choices=["open", "water", "rock"])
    p.add_argument("--n-init", type=int, default=192)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--manifold-w", type=float, default=1e-3)
    p.add_argument("--max-ckpts", type=int, default=0, help="0 = all")
    p.add_argument("--out-prefix", type=Path, default=REPO / "paper/figures/forkwall_dynamics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.enabled = False        # fused GRU can't backward in eval()

    results = {}
    for seed in args.seeds:
        d = args.ckpt_root / f"{args.run_prefix}{seed}"
        cks = sorted(d.glob("iter*.pt"),
                     key=lambda p_: int(re.search(r"iter(\d+)", p_.name).group(1)))
        cks.append(d / "final.pt")
        if args.max_ckpts:
            step = max(1, len(cks) // args.max_ckpts)
            cks = cks[::step]
        print(f"\n=== seed {seed}: {len(cks)} checkpoints ===")
        rows = []
        for ck in cks:
            if not ck.exists():
                continue
            it = 0 if ck.name == "iter0.pt" else (
                10**9 if ck.name == "final.pt"
                else int(re.search(r"iter(\d+)", ck.name).group(1)))
            policy, cargs, view_size, env_size, env_width = _load_policy(ck, device)
            commit = False if cargs.get("no_commit", False) else None
            H, inputs = collect_states_and_inputs(policy, cargs, view_size, env_size,
                                                  env_width, device, commit)
            if args.context not in inputs:
                continue
            x = inputs[args.context]
            # PRIMARY: linearise at real (h, x) pairs — on-manifold by construction
            Ht, Xt = collect_state_input_pairs(policy, cargs, view_size, env_size,
                                               env_width, device, commit)
            tau_traj, marg_traj, radius_traj = trajectory_taus(policy, Ht, Xt, device)
            hf, qf, dist, keep = find_slow_points(policy, H, x, device,
                                                  n_init=args.n_init, iters=args.iters,
                                                  manifold_w=args.manifold_w)
            P = hf[keep].cpu().numpy() if keep.any() else hf.cpu().numpy()
            h0 = hf[int(qf.argmin())]
            mag, taus = jacobian_taus(policy, h0, x, device)
            step_motion = float(np.median(
                np.linalg.norm(H[1:] - H[:-1], axis=1))) if len(H) > 1 else float("nan")
            row = {
                "iteration": it, "global_step": int(torch.load(ck, map_location="cpu",
                                                               weights_only=False)["global_step"]),
                "n_converged": int(keep.sum()), "n_init": int(len(qf)),
                "q_min": float(qf.min()), "q_median": float(qf.median()),
                "slowpoint_dim": participation_ratio(P),
                "dist_to_data_median": float(dist.median()),
                "state_norm_median": float(np.median(np.linalg.norm(H, axis=1))),
                "traj_step_motion": step_motion,
                "taus": [float(t) for t in taus],
                "frac_marginal": float((mag > 0.99).mean()),
                "taus_traj": [float(t) for t in tau_traj],
                "frac_marginal_traj": marg_traj,
                "spectral_radius_traj": radius_traj,
                "top_eig": [float(v) for v in mag[:6]],
            }
            rows.append(row)
            print(f"  iter {it:>6}  q_min {row['q_min']:.2e}  conv {row['n_converged']:3d}/"
                  f"{row['n_init']}  dim {row['slowpoint_dim']:5.2f}  "
                  f"d(data) {row['dist_to_data_median']:5.2f}  "
                  f"tauFP {taus[0] if len(taus) else float('nan'):8.1f}  "
                  f"| TRAJ tau1 {tau_traj[0]:7.2f} tau2 {tau_traj[1]:6.2f} "
                  f"rho {radius_traj:.3f} marg {marg_traj:.3f}", flush=True)
        results[seed] = rows

    jp = Path(str(args.out_prefix) + f"_{args.context}.json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps({"context": args.context, "results": results}, indent=2))
    print(f"\nsaved {jp}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for (seed, rows), col in zip(results.items(), colors):
        if not rows:
            continue
        it = np.array([r["iteration"] for r in rows], dtype=float)
        it[it > 1e8] = it[it < 1e8].max() * 1.15 if (it < 1e8).any() else 1
        ax = axes[0]
        for k in range(3):
            tv = [r["taus_traj"][k] if len(r.get("taus_traj", [])) > k else np.nan
                  for r in rows]
            ax.plot(it, tv, "o-", ms=3, lw=1.2, color=col, alpha=1 - 0.25 * k,
                    label=f"seed {seed} mode{k+1}" if k == 0 else None)
        ax = axes[1]
        ax.plot(it, [r["slowpoint_dim"] for r in rows], "o-", ms=3, color=col,
                label=f"seed {seed}")
        ax = axes[2]
        ax.plot(it, [r.get("spectral_radius_traj", np.nan) for r in rows], "o-", ms=3,
                color=col, label=f"seed {seed}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("training iteration"); axes[0].set_ylabel(r"$\tau$ (steps)")
    axes[0].set_title(r"(A) integration times $\tau_i$ (on-manifold)", fontsize=11)
    axes[0].axhline(100, color="crimson", ls="--", lw=1,
                    label="episode length (~100)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.15)
    axes[1].set_xscale("log"); axes[1].set_xlabel("training iteration")
    axes[1].set_ylabel("participation ratio")
    axes[1].set_title("(B) slow-point manifold dimensionality", fontsize=11)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.15)
    axes[2].set_xscale("log"); axes[2].set_xlabel("training iteration")
    axes[2].set_ylabel(r"spectral radius $\rho(J^{rec})$")
    axes[2].axhline(1.0, color="crimson", ls="--", lw=1, label="|λ| = 1")
    axes[2].set_title("(C) recurrence gain along trajectories", fontsize=11)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.15)
    fig.suptitle(f"fork_wall recurrent dynamics over training — input context: "
                 f"{args.context}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    op = Path(str(args.out_prefix) + f"_{args.context}.png")
    fig.savefig(op, dpi=150)
    print(f"saved {op}")


if __name__ == "__main__":
    main()
