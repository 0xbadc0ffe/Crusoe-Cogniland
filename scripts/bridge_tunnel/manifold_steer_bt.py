#!/usr/bin/env python
"""Experiment 3 -- manifold steering of the r2dreamer bridge_tunnel belief.

Adapts the Goodfire causalab manifold-steering method (SplineManifold + geodesic
manifold-trace path) to the Dreamer belief state. Hypothesis: the category
belief lies on a curved 1D manifold, not a straight direction -- so steering
ALONG the manifold (interpolate the intrinsic coordinate, decode) should
produce coherent intermediate beliefs and a smooth decision sweep, while
straight-line (linear) steering in ambient belief space leaves the manifold and
produces incoherent "superposition" beliefs (the Fig-7 mountain-car result).

Conceptual coordinate  z = sign(category) * corridor_progress
    rocky  -> z in [0, -1]   (top door)
    lakes  -> z in [0, +1]   (bottom door)
    z ~ 0  = neutral belief at spawn, before evidence.
(balanced episodes are held out and overlaid as validation.)

Run (r2dreamer env, PYTHONPATH=src):
  python scripts/bridge_tunnel/manifold_steer_bt.py \
      --checkpoint external/r2dreamer/runs/forkwall_nocommit/latest.pt \
      --out outputs/bridge_tunnel_forkwall/manifold_steer_data.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "external" / "r2dreamer"))
sys.path.insert(0, str(_REPO / "scripts" / "bridge_tunnel"))

import dreamer_belief_report_r2d as R  # agent loading, env, tile helpers
from manifold_spline import SplineManifold1D

CATEGORIES = R.CATEGORIES
WATER, ROCK = R.WATER, R.ROCK
VIEW = R.VIEW
A_UP, A_DOWN = R.A_UP, R.A_DOWN
CAT_SIGN = {"rocky": -1.0, "lakes": +1.0, "balanced": 0.0}


def feat_of(stoch, deter):
    return np.concatenate([np.asarray(stoch).reshape(-1), np.asarray(deter).reshape(-1)])


def split_feat(feat_t, stoch_shape):
    """(B, ambient) -> (stoch (B,S,K), deter (B,D))."""
    sdim = int(np.prod(stoch_shape))
    stoch = feat_t[:, :sdim].reshape(feat_t.shape[0], *stoch_shape)
    deter = feat_t[:, sdim:]
    return stoch, deter


@torch.no_grad()
def collect_belief_trajectory(agent, device, n_per_category, seed0=5_000_000):
    """Roll out episodes; at EVERY corridor step before the fork, record the
    belief feat + its z coordinate. Returns dict per category:
      {feats:(T,ambient), z:(T,), stoch_shape}
    plus a small set of per-step minimaps for the viewer."""
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv

    def make_env(cat, seed):
        kw = dict(R.ENV_KW); kw["categories"] = (cat,)
        return BridgeTunnelEnv(seed=seed, **kw)

    out = {c: {"feats": [], "z": [], "cat": []} for c in CATEGORIES}
    stoch_shape = None
    t0 = time.time()
    for cat in CATEGORIES:
        for i in range(n_per_category):
            seed = seed0 + CATEGORIES.index(cat) * 1_000_000 + i
            env = make_env(cat, seed)
            raw, info = env.reset(seed=seed)
            wall_col = env._record.wall_col
            state = agent.get_initial_state(1)
            step = 0
            prev_first = True
            traj_feats, traj_z = [], []
            while True:
                vec = R.flatten_obs(raw)
                obs = {"vector": torch.as_tensor(vec, device=device, dtype=torch.float32)[None],
                       "is_first": torch.as_tensor([prev_first], device=device)}
                action, state = agent.act(obs, state, eval=True)
                prev_first = False
                stoch_np = state["stoch"][0].cpu().numpy()
                deter_np = state["deter"][0].cpu().numpy()
                if stoch_shape is None:
                    stoch_shape = stoch_np.shape
                # agent position BEFORE this action was applied (col along corridor)
                pos = env._traj[-1]
                col = pos[1]
                progress = min(1.0, max(0.0, col / max(1, wall_col)))
                # only log corridor steps up to the wall (evidence-accumulation phase)
                if col <= wall_col:
                    traj_feats.append(feat_of(stoch_np, deter_np))
                    traj_z.append(CAT_SIGN[cat] * progress)
                a = int(action.argmax(-1).item())
                raw, reward, term, trunc, info = env.step(a)
                step += 1
                if term or trunc or step >= 400:
                    break
            out[cat]["feats"].extend(traj_feats)
            out[cat]["z"].extend(traj_z)
            out[cat]["cat"].extend([cat] * len(traj_feats))
        print(f"  [collect] {cat}: {len(out[cat]['feats'])} belief states "
              f"({time.time()-t0:.0f}s)", flush=True)
    for c in CATEGORIES:
        out[c]["feats"] = np.stack(out[c]["feats"]) if out[c]["feats"] else np.zeros((0, 1))
        out[c]["z"] = np.array(out[c]["z"])
    return out, stoch_shape


def fit_manifold(traj, n_bins=15, z_lo=-1.0, z_hi=1.0, smoothness=1.0, device="cuda:0"):
    """Bin z over the DECISIVE categories (rocky+lakes), take mean belief per
    bin -> control points; fit a natural cubic smoothing spline."""
    feats = np.concatenate([traj["rocky"]["feats"], traj["lakes"]["feats"]], 0)
    zs = np.concatenate([traj["rocky"]["z"], traj["lakes"]["z"]], 0)
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    centers, means = [], []
    for b in range(n_bins):
        m = (zs >= edges[b]) & (zs < edges[b + 1] if b < n_bins - 1 else zs <= edges[b + 1])
        if m.sum() >= 3:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(feats[m].mean(0))
    centers = np.array(centers)
    means = np.stack(means)
    manifold = SplineManifold1D(
        torch.as_tensor(centers), torch.as_tensor(means), smoothness=smoothness
    ).to(device)
    return manifold, centers, means


@torch.no_grad()
def decode_coherence(agent, device, feats_t, stoch_shape):
    """Decode belief feats -> per-cell tile vectors; coherence = mean over cells
    of the top-1 softmax prob (crisp one-hot ~ high; smeared superposition ~ low).
    Returns (coherence:(B,), grids:(B,V,V), water_frac:(B,), rock_frac:(B,))."""
    stoch, deter = split_feat(feats_t, stoch_shape)
    dec = agent.decoder(stoch, deter)["vector"].mode()      # (B, 3974)
    K = R.NUM_TILES
    cells = dec[:, : VIEW * VIEW * K].reshape(-1, VIEW, VIEW, K)
    probs = torch.softmax(cells, dim=-1)
    top1 = probs.max(dim=-1).values                          # (B,V,V)
    coherence = top1.mean(dim=(1, 2))                        # (B,)
    grids = cells.argmax(dim=-1)                             # (B,V,V)
    water = (grids == WATER).float().mean(dim=(1, 2))
    rock = (grids == ROCK).float().mean(dim=(1, 2))
    return (coherence.cpu().numpy(), grids.cpu().numpy(),
            water.cpu().numpy(), rock.cpu().numpy())


@torch.no_grad()
def actor_decision(agent, device, feats_t):
    """P(up), P(down) and argmax action for each belief feat (fed straight to
    the actor; get_feat is just concat, so the raw feat == actor input)."""
    dist = agent.actor(feats_t)
    logits = dist.logits if hasattr(dist, "logits") else None
    if logits is None:
        # fall back: use mode one-hot
        mode = dist.mode
        probs = mode
    else:
        probs = torch.softmax(logits, dim=-1)
    p_up = probs[:, A_UP].cpu().numpy()
    p_down = probs[:, A_DOWN].cpu().numpy()
    act = dist.mode.argmax(-1).cpu().numpy()
    return p_up, p_down, act


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="external/r2dreamer/runs/forkwall_nocommit/latest.pt")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/manifold_steer_data.json")
    ap.add_argument("--raw-cache", default="outputs/bridge_tunnel_forkwall/manifold_steer_raw.pkl")
    ap.add_argument("--decoder-cache", default="outputs/bridge_tunnel_forkwall/belief_report_raw.pkl",
                    help="reuse the decoder-training buffer from the exp1/2 rollout")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-category", type=int, default=60)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--smoothness", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--decoder-steps", type=int, default=6000)
    ap.add_argument("--z-endpoint", type=float, default=0.9)
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = pathlib.Path(args.raw_cache)
    t0 = time.time()

    agent, config = R.load_agent(args.checkpoint, args.device)
    print(f"[main] agent loaded ({time.time()-t0:.1f}s)")

    # train the reconstruction decoder (reuse the exp1/2 decoder buffer)
    with open(args.decoder_cache, "rb") as f:
        decoder_buf = pickle.load(f)["decoder_buf"]
    R.train_decoder(agent, args.device, decoder_buf, steps=args.decoder_steps)

    if args.skip_collect and raw_path.exists():
        with open(raw_path, "rb") as f:
            cache = pickle.load(f)
        traj, stoch_shape = cache["traj"], cache["stoch_shape"]
        print(f"[main] reusing cached belief trajectory")
    else:
        traj, stoch_shape = collect_belief_trajectory(agent, args.device, args.n_per_category)
        with open(raw_path, "wb") as f:
            pickle.dump({"traj": traj, "stoch_shape": stoch_shape}, f)
        print(f"[main] belief trajectory cached -> {raw_path}")

    manifold, centers, means = fit_manifold(
        traj, n_bins=args.n_bins, smoothness=args.smoothness, device=args.device)
    print(f"[main] manifold fit: {len(centers)} control points over z in [{centers.min():.2f},{centers.max():.2f}]")

    zA, zB = -args.z_endpoint, +args.z_endpoint
    lin = manifold.linear_path(zA, zB, args.n_steps)          # (S, ambient)
    man = manifold.manifold_trace_path(zA, zB, args.n_steps)  # (S, ambient)

    # off-manifold distance along each path
    off_lin = manifold.off_manifold_distance(lin).cpu().numpy()
    off_man = manifold.off_manifold_distance(man).cpu().numpy()

    # decode coherence + grids + terrain fracs
    coh_lin, grids_lin, w_lin, r_lin = decode_coherence(agent, args.device, lin, stoch_shape)
    coh_man, grids_man, w_man, r_man = decode_coherence(agent, args.device, man, stoch_shape)

    # actor decision sweep (P(down)-P(up) = "how much toward the lakes/bottom door")
    pu_lin, pd_lin, act_lin = actor_decision(agent, args.device, lin)
    pu_man, pd_man, act_man = actor_decision(agent, args.device, man)

    # sanity: coherence of REAL belief states (upper bound) vs midpoint
    real_feats = np.concatenate([traj["rocky"]["feats"], traj["lakes"]["feats"]], 0)
    ridx = np.random.default_rng(0).choice(real_feats.shape[0], size=min(400, real_feats.shape[0]), replace=False)
    coh_real, _, _, _ = decode_coherence(
        agent, args.device, torch.as_tensor(real_feats[ridx], device=args.device, dtype=torch.float32), stoch_shape)

    # balanced-overlay: where do balanced beliefs project on the manifold?
    bal_feats = traj["balanced"]["feats"]
    if bal_feats.shape[0] > 0:
        bidx = np.random.default_rng(1).choice(bal_feats.shape[0], size=min(300, bal_feats.shape[0]), replace=False)
        u_bal, _ = manifold.encode_to_nearest_point(
            torch.as_tensor(bal_feats[bidx], device=args.device, dtype=torch.float32))
        bal_u = u_bal[:, 0].cpu().numpy().tolist()
    else:
        bal_u = []

    t_values = np.linspace(0, 1, args.n_steps)
    bundle = dict(
        checkpoint=args.checkpoint,
        z_endpoints=[zA, zB],
        n_steps=args.n_steps,
        n_bins=len(centers),
        smoothness=args.smoothness,
        control_z=centers.tolist(),
        t_values=t_values.tolist(),
        real_coherence_mean=float(coh_real.mean()),
        real_coherence_std=float(coh_real.std()),
        balanced_projected_u=bal_u,
        linear=dict(
            off_manifold=off_lin.tolist(), coherence=coh_lin.tolist(),
            water=w_lin.tolist(), rock=r_lin.tolist(),
            p_up=pu_lin.tolist(), p_down=pd_lin.tolist(), action=act_lin.tolist(),
            grids=grids_lin.tolist(),
        ),
        manifold=dict(
            off_manifold=off_man.tolist(), coherence=coh_man.tolist(),
            water=w_man.tolist(), rock=r_man.tolist(),
            p_up=pu_man.tolist(), p_down=pd_man.tolist(), action=act_man.tolist(),
            grids=grids_man.tolist(),
        ),
    )
    print("[main] off-manifold  linear mean=%.3f  manifold mean=%.3f" % (off_lin.mean(), off_man.mean()))
    print("[main] coherence     linear mean=%.3f  manifold mean=%.3f  (real=%.3f)"
          % (coh_lin.mean(), coh_man.mean(), coh_real.mean()))
    print("[main] mid-path coherence  linear=%.3f  manifold=%.3f"
          % (coh_lin[args.n_steps // 2], coh_man[args.n_steps // 2]))

    with open(out_path, "w") as f:
        json.dump(to_jsonable(bundle), f)
    print(f"[main] wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB), total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
