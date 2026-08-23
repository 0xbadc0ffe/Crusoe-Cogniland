#!/usr/bin/env python
"""Recompute manifold-steering measurements from the cached belief trajectory
(no re-collection). Fixes the coherence metric (raw per-cell margin instead of
softmax, which saturates on an MSE-trained decoder) and adds a PCA projection
of the belief cloud + fitted spline + both steering paths for visualization."""
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

import dreamer_belief_report_r2d as R
import manifold_steer_bt as MS
from manifold_spline import SplineManifold1D

VIEW, K = R.VIEW, R.NUM_TILES
WATER, ROCK = R.WATER, R.ROCK


@torch.no_grad()
def decode_full(agent, device, feats_t, stoch_shape):
    """Return raw decoded per-cell tile vectors + derived crispness metrics.
    coherence(margin) = mean over cells of (top1_raw - top2_raw): a crisp
    one-hot reconstruction has margin ~1, a smeared superposition ~0."""
    stoch, deter = MS.split_feat(feats_t, stoch_shape)
    dec = agent.decoder(stoch, deter)["vector"].mode()
    cells = dec[:, : VIEW * VIEW * K].reshape(-1, VIEW, VIEW, K)   # raw
    sorted_vals, _ = cells.sort(dim=-1, descending=True)
    margin = (sorted_vals[..., 0] - sorted_vals[..., 1]).mean(dim=(1, 2))  # (B,)
    grids = cells.argmax(dim=-1)
    water = (grids == WATER).float().mean(dim=(1, 2))
    rock = (grids == ROCK).float().mean(dim=(1, 2))
    return margin.cpu().numpy(), grids.cpu().numpy(), water.cpu().numpy(), rock.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="external/r2dreamer/runs/forkwall_nocommit/latest.pt")
    ap.add_argument("--raw-cache", default="outputs/bridge_tunnel_forkwall/manifold_steer_raw.pkl")
    ap.add_argument("--decoder-cache", default="outputs/bridge_tunnel_forkwall/belief_report_raw.pkl")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/manifold_steer_data.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--smoothness", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--z-endpoint", type=float, default=0.9)
    ap.add_argument("--decoder-steps", type=int, default=6000)
    args = ap.parse_args()

    t0 = time.time()
    agent, config = R.load_agent(args.checkpoint, args.device)
    with open(args.decoder_cache, "rb") as f:
        decoder_buf = pickle.load(f)["decoder_buf"]
    R.train_decoder(agent, args.device, decoder_buf, steps=args.decoder_steps)

    with open(args.raw_cache, "rb") as f:
        cache = pickle.load(f)
    traj, stoch_shape = cache["traj"], cache["stoch_shape"]

    manifold, centers, means = MS.fit_manifold(
        traj, n_bins=args.n_bins, smoothness=args.smoothness, device=args.device)
    print(f"[recompute] manifold: {len(centers)} control points, z in [{centers.min():.2f},{centers.max():.2f}]")

    zA, zB = -args.z_endpoint, +args.z_endpoint
    lin = manifold.linear_path(zA, zB, args.n_steps)
    man = manifold.manifold_trace_path(zA, zB, args.n_steps)

    off_lin = manifold.off_manifold_distance(lin).cpu().numpy()
    off_man = manifold.off_manifold_distance(man).cpu().numpy()
    m_lin, g_lin, w_lin, r_lin = decode_full(agent, args.device, lin, stoch_shape)
    m_man, g_man, w_man, r_man = decode_full(agent, args.device, man, stoch_shape)

    # real-belief margin (upper bound) and per-bin real margin along z
    feats_all = np.concatenate([traj["rocky"]["feats"], traj["lakes"]["feats"]], 0)
    z_all = np.concatenate([traj["rocky"]["z"], traj["lakes"]["z"]], 0)
    rng = np.random.default_rng(0)
    ridx = rng.choice(feats_all.shape[0], size=min(600, feats_all.shape[0]), replace=False)
    m_real, _, _, _ = decode_full(
        agent, args.device, torch.as_tensor(feats_all[ridx], device=args.device, dtype=torch.float32), stoch_shape)

    # PCA fit on the MANIFOLD CURVE itself (its intrinsic principal plane), not
    # the raw corridor cloud -- the cloud's variance is dominated by corridor
    # position, so a cloud-PCA hides the low-variance category structure the
    # manifold captures. Projecting onto the curve's own principal plane shows
    # the manifold as a curve and the real cloud as its (diffuse) scatter.
    dense_u = torch.linspace(float(centers.min()), float(centers.max()), 120, device=args.device).unsqueeze(1)
    dense_curve = manifold.decode(dense_u).cpu().numpy()
    mu = dense_curve.mean(0)
    _, S, Vt = np.linalg.svd(dense_curve - mu, full_matrices=False)
    PC = Vt[:3]
    def proj(x): return ((np.asarray(x) - mu) @ PC.T)
    curve_xy = proj(dense_curve)
    cloud_idx = rng.choice(feats_all.shape[0], size=min(800, feats_all.shape[0]), replace=False)
    cloud_xy = proj(feats_all[cloud_idx])
    cloud_z = z_all[cloud_idx]
    knots_xy = proj(means)
    lin_xy = proj(lin.cpu().numpy())
    man_xy = proj(man.cpu().numpy())
    # variance of the belief cloud captured by the manifold's own principal plane
    cloud_var_total = float(((feats_all[cloud_idx] - feats_all[cloud_idx].mean(0)) ** 2).sum(1).mean())

    t_values = np.linspace(0, 1, args.n_steps)
    bundle = dict(
        checkpoint=args.checkpoint,
        z_endpoints=[zA, zB], n_steps=args.n_steps, n_bins=len(centers),
        smoothness=args.smoothness, control_z=centers.tolist(), t_values=t_values.tolist(),
        real_margin_mean=float(m_real.mean()), real_margin_std=float(m_real.std()),
        pca=dict(
            explained=(S[:3] ** 2 / (S ** 2).sum()).tolist(),
            cloud_xy=cloud_xy[:, :2].tolist(), cloud_z=cloud_z.tolist(),
            knots_xy=knots_xy[:, :2].tolist(),
            curve_xy=curve_xy[:, :2].tolist(),
            linear_xy=lin_xy[:, :2].tolist(), manifold_xy=man_xy[:, :2].tolist(),
        ),
        linear=dict(off_manifold=off_lin.tolist(), margin=m_lin.tolist(),
                    water=w_lin.tolist(), rock=r_lin.tolist(), grids=g_lin.tolist()),
        manifold=dict(off_manifold=off_man.tolist(), margin=m_man.tolist(),
                      water=w_man.tolist(), rock=r_man.tolist(), grids=g_man.tolist()),
    )
    print("[recompute] off-manifold  linear mean=%.3f (mid=%.3f)  manifold=%.3f"
          % (off_lin.mean(), off_lin[args.n_steps // 2], off_man.mean()))
    print("[recompute] margin        linear mean=%.3f (mid=%.3f)  manifold mean=%.3f (mid=%.3f)  real=%.3f"
          % (m_lin.mean(), m_lin[args.n_steps // 2], m_man.mean(), m_man[args.n_steps // 2], m_real.mean()))
    print("[recompute] PCA explained (3 PCs):", [round(x, 3) for x in bundle["pca"]["explained"]])

    with open(args.out, "w") as f:
        json.dump(MS.to_jsonable(bundle), f)
    print(f"[recompute] wrote {args.out} ({pathlib.Path(args.out).stat().st_size/1e6:.2f} MB), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
