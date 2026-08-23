#!/usr/bin/env python
"""Experiment 3b -- steer the FORK decision and decode the imagined future.

Take the model at a balanced-map fork, where its belief is genuinely ~50/50
(top vs bottom door), and steer that belief toward "commit top" / "commit
bottom". Then IMAGINE forward (roll the RSSM prior + actor) and decode the
predicted future. The static-belief decode was robust to off-manifold
excursions (Exp 3), but the DYNAMICS (img_step) are trained only on
on-manifold beliefs -- so rolling forward from an off-manifold (linear-steered)
start should compound the error and corrupt the predicted future, while a
manifold-steered start should imagine a coherent future that reaches the
intended door.

Conceptual coordinate  d = door_sign(eventual_door) * progress
    top door    -> d in [0, -1]
    bottom door -> d in [0, +1]
    d ~ 0        = neutral belief at spawn.
The manifold is fit through binned d over ALL non-timeout episodes; the balanced
fork belief (the 50/50 start) projects near the pinch at d~0.

Run (r2dreamer env, PYTHONPATH=src):
  python scripts/bridge_tunnel/manifold_fork_steer.py \
      --checkpoint r2dreamer_model/runs/forkwall_nocommit/latest.pt \
      --out outputs/bridge_tunnel_forkwall/manifold_fork_data.json
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

import dreamer_belief_report_r2d as R
import manifold_steer_bt as MS
from manifold_spline import SplineManifold1D

VIEW, K = R.VIEW, R.NUM_TILES
WATER, ROCK, TARGET = R.WATER, R.ROCK, 4
A_UP, A_DOWN, A_LEFT, A_RIGHT = R.A_UP, R.A_DOWN, R.A_LEFT, R.A_RIGHT


def feat_of(stoch, deter):
    return np.concatenate([np.asarray(stoch).reshape(-1), np.asarray(deter).reshape(-1)])


@torch.no_grad()
def collect(agent, device, n_per_category, seed0=6_000_000, max_steps=200):
    """Full episodes past the wall. Per step: belief feat + d = door_sign*progress.
    Also collect balanced pre-decision (50/50) beliefs as steering starts."""
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv

    def make_env(cat, seed):
        kw = dict(R.ENV_KW); kw["categories"] = (cat,)
        return BridgeTunnelEnv(seed=seed, **kw)

    feats_all, d_all = [], []
    balanced_fork = []   # (stoch, deter) at the pre-decision step on balanced maps
    stoch_shape = None
    t0 = time.time()
    for cat in R.CATEGORIES:
        for i in range(n_per_category):
            seed = seed0 + R.CATEGORIES.index(cat) * 1_000_000 + i
            env = make_env(cat, seed)
            raw, info = env.reset(seed=seed)
            wall_col = env._record.wall_col
            state = agent.get_initial_state(1)
            first = True
            steps = []   # (stoch, deter, pos, action)
            while True:
                vec = R.flatten_obs(raw)
                obs = {"vector": torch.as_tensor(vec, device=device, dtype=torch.float32)[None],
                       "is_first": torch.as_tensor([first], device=device)}
                action, state = agent.act(obs, state, eval=True)
                first = False
                s_np = state["stoch"][0].cpu().numpy()
                d_np = state["deter"][0].cpu().numpy()
                if stoch_shape is None:
                    stoch_shape = s_np.shape
                pos = env._traj[-1]
                a = int(action.argmax(-1).item())
                steps.append((s_np, d_np, pos, a))
                raw, reward, term, trunc, info = env.step(a)
                if term or trunc or len(steps) >= max_steps:
                    break
            # eventual door
            reached = bool(info.get("reached_any_target", False))
            if not reached:
                continue  # timeout -> no clean door label
            final_pos = env._traj[-1]
            mid_row = env.height / 2.0
            door = "top" if final_pos[0] < mid_row else "bottom"
            sign = -1.0 if door == "top" else +1.0
            # APPROACH PHASE only: from passage entry (first col >= wall_col) to
            # the door. d = door_sign * approach_progress, so d=0 is AT the fork
            # (undecided) and d=+-1 is at the door -- decision-relevant, unlike
            # whole-episode progress whose d=0 is spawn.
            passage_j = None
            for j, (s_np, d_np, pos, a) in enumerate(steps):
                if pos[1] >= wall_col:
                    passage_j = j
                    break
            if passage_j is None:
                continue
            approach = steps[passage_j:]
            Ta = len(approach)
            for k, (s_np, d_np, pos, a) in enumerate(approach):
                prog = k / max(1, Ta - 1)
                feats_all.append(feat_of(s_np, d_np))
                d_all.append(sign * prog)
            # balanced fork belief = the belief AT passage entry (undecided,
            # both doors still open), the model's most ambiguous decision point.
            if cat == "balanced":
                ps, pd, _, _ = approach[0]
                balanced_fork.append((ps, pd))
        print(f"  [collect] {cat}: total {len(feats_all)} states, "
              f"{len(balanced_fork)} balanced forks ({time.time()-t0:.0f}s)", flush=True)
    return (np.stack(feats_all), np.array(d_all),
            balanced_fork, stoch_shape)


def fit_manifold(feats, d, n_bins=15, smoothness=2.0, device="cuda:0"):
    edges = np.linspace(-1, 1, n_bins + 1)
    centers, means = [], []
    for b in range(n_bins):
        hi = d <= edges[b + 1] if b == n_bins - 1 else d < edges[b + 1]
        m = (d >= edges[b]) & hi
        if m.sum() >= 5:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            means.append(feats[m].mean(0))
    manifold = SplineManifold1D(
        torch.as_tensor(np.array(centers)), torch.as_tensor(np.stack(means)),
        smoothness=smoothness).to(device)
    return manifold, np.array(centers)


@torch.no_grad()
def imagine_future(agent, device, feat_np, stoch_shape, horizon=10):
    """Roll actor+RSSM prior forward from a raw belief feat; decode each step.
    Returns grids (H,V,V), actions (H,), coherence margin (H,), door_signal (H,)
    where door_signal = mean signed row of TARGET tiles (neg=top half, pos=bottom)."""
    sdim = int(np.prod(stoch_shape))
    stoch = torch.as_tensor(feat_np[:sdim].reshape(stoch_shape), device=device, dtype=torch.float32)[None]
    deter = torch.as_tensor(feat_np[sdim:], device=device, dtype=torch.float32)[None]
    grids, actions, margins, door_sig = [], [], [], []
    up_minus_down = 0
    for _ in range(horizon):
        feat = agent.rssm.get_feat(stoch, deter)
        dec = agent.decoder(stoch, deter)["vector"].mode()[0]
        cells = dec[: VIEW * VIEW * K].reshape(VIEW, VIEW, K)
        sv, _ = cells.sort(dim=-1, descending=True)
        margins.append(float((sv[..., 0] - sv[..., 1]).mean()))
        grid = cells.argmax(-1).cpu().numpy()
        grids.append(grid.tolist())
        # target tile vertical position relative to crop centre (agent centred)
        tr, tc = np.where(grid == TARGET)
        if len(tr):
            door_sig.append(float(np.mean(tr - VIEW // 2)))
        else:
            door_sig.append(0.0)
        dist = agent.actor(feat)
        a = int(dist.mode.argmax(-1).item())
        actions.append(a)
        stoch, deter = agent.rssm.img_step(stoch, deter, dist.mode)
    return grids, actions, margins, door_sig


@torch.no_grad()
def actor_pud(agent, device, feats_t):
    dist = agent.actor(feats_t)
    logits = dist.logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, A_UP].cpu().numpy(), probs[:, A_DOWN].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="r2dreamer_model/runs/forkwall_nocommit/latest.pt")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/manifold_fork_data.json")
    ap.add_argument("--raw-cache", default="outputs/bridge_tunnel_forkwall/manifold_fork_raw.pkl")
    ap.add_argument("--decoder-cache", default="outputs/bridge_tunnel_forkwall/belief_report_raw.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-category", type=int, default=45)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--smoothness", type=float, default=2.0)
    ap.add_argument("--n-steer", type=int, default=11)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--decoder-steps", type=int, default=6000)
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = pathlib.Path(args.raw_cache)
    t0 = time.time()

    agent, config = R.load_agent(args.checkpoint, args.device)
    with open(args.decoder_cache, "rb") as f:
        decoder_buf = pickle.load(f)["decoder_buf"]
    R.train_decoder(agent, args.device, decoder_buf, steps=args.decoder_steps)

    if args.skip_collect and raw_path.exists():
        with open(raw_path, "rb") as f:
            c = pickle.load(f)
        feats, d, balanced_fork, stoch_shape = c["feats"], c["d"], c["balanced_fork"], c["stoch_shape"]
        print(f"[main] reusing cached collection ({len(feats)} states, {len(balanced_fork)} balanced forks)")
    else:
        feats, d, balanced_fork, stoch_shape = collect(agent, args.device, args.n_per_category)
        with open(raw_path, "wb") as f:
            pickle.dump(dict(feats=feats, d=d, balanced_fork=balanced_fork, stoch_shape=stoch_shape), f)
        print(f"[main] collection cached -> {raw_path}")

    manifold, centers = fit_manifold(feats, d, n_bins=args.n_bins,
                                     smoothness=args.smoothness, device=args.device)
    print(f"[main] manifold: {len(centers)} control points, d in [{centers.min():.2f},{centers.max():.2f}]")

    # Start belief b0: the MOST AMBIGUOUS balanced fork -- among balanced fork
    # beliefs, take those the actor is genuinely torn on (largest p_up+p_down,
    # i.e. actually deciding a door rather than still moving right; then
    # smallest |p_up-p_down|), and average them (class-mode stoch + mean deter).
    # Balanced maps let either door pay off, so this is the model's least
    # committed decision point -- though this agent still leans one way, which
    # we report rather than assume a perfect 50/50.
    bf_stoch = np.stack([b[0] for b in balanced_fork])
    bf_deter = np.stack([b[1] for b in balanced_fork])
    bf_feats = np.stack([feat_of(s, d_) for s, d_ in zip(bf_stoch, bf_deter)])
    bpu, bpd = actor_pud(agent, args.device, torch.as_tensor(bf_feats, device=args.device, dtype=torch.float32))
    deciding = bpu + bpd                    # how much prob is on a door at all
    torn = -np.abs(bpu - bpd)               # closeness of up vs down (higher=more torn)
    # rank: must be deciding a door (top 60%), then most torn
    thresh = np.quantile(deciding, 0.4)
    cand = np.where(deciding >= thresh)[0]
    cand = cand[np.argsort(torn[cand])[::-1]][: max(3, len(cand) // 3)]
    b0_stoch = R.class_mode_stoch(bf_stoch[cand])
    b0_deter = bf_deter[cand].mean(0)
    b0 = feat_of(b0_stoch, b0_deter)
    pu, pd = actor_pud(agent, args.device, torch.as_tensor(b0[None], device=args.device, dtype=torch.float32))
    print(f"[main] balanced fork b0 (n_torn={len(cand)}/{len(balanced_fork)}): "
          f"actor p(up)={pu[0]:.3f} p(down)={pd[0]:.3f}")

    # Steer from the fork point decode(0) (the on-manifold "undecided at the
    # passage" belief) toward each door, comparing:
    #   manifold:  move along the manifold, u: 0 -> d_target, decode  (off=0)
    #   linear:    straight line in ambient belief space decode(0)->decode(d_target)
    # Both share endpoints; only the PATH differs, so any difference in the
    # imagined future is due to the linear path leaving the manifold.
    x_start = manifold.decode(torch.tensor([[0.0]], device=args.device))[0]
    pu0, pd0 = actor_pud(agent, args.device, x_start[None])
    print(f"[main] manifold fork point decode(0): actor p(up)={pu0[0]:.3f} p(down)={pd0[0]:.3f}")
    t_values = np.linspace(0, 1, args.n_steer)

    # three steering probes:
    #   top / bottom : from the fork point decode(0) to a committed door (d=+-0.9)
    #   sweep        : the full top-door -> bottom-door traverse (d=-0.9 -> +0.9),
    #                  which crosses the neutral pinch where curvature is highest
    #                  -- linear's best chance to leave the manifold and corrupt.
    def dec(dv):
        return manifold.decode(torch.tensor([[float(dv)]], device=args.device))[0]
    PROBES = [("top", 0.0, -0.9), ("bottom", 0.0, +0.9), ("sweep", -0.9, +0.9)]
    results = {}
    for door, d_start, d_target in PROBES:
        x_s = dec(d_start)
        x_end = dec(d_target)
        lin_grids, man_grids = [], []
        lin_off, man_off, lin_pud, man_pud = [], [], [], []
        lin_doorsig, man_doorsig, lin_coh, man_coh = [], [], [], []
        for t in t_values:
            x_lin = x_s + t * (x_end - x_s)
            u = d_start + t * (d_target - d_start)
            x_man = manifold.decode(torch.tensor([[float(u)]], device=args.device))[0]
            for x, G, OFF, PUD, DS, COH in [
                (x_lin, lin_grids, lin_off, lin_pud, lin_doorsig, lin_coh),
                (x_man, man_grids, man_off, man_pud, man_doorsig, man_coh)]:
                off = float(manifold.off_manifold_distance(x[None])[0])
                fpu, fpd = actor_pud(agent, args.device, x[None])
                g, acts, margins, dsig = imagine_future(
                    agent, args.device, x.cpu().numpy(), stoch_shape, horizon=args.horizon)
                G.append(g); OFF.append(off); PUD.append(float(fpd[0] - fpu[0]))
                DS.append(float(np.mean(dsig))); COH.append(float(np.mean(margins)))
        results[door] = dict(
            linear=dict(grids=lin_grids, off_manifold=lin_off, p_down_minus_up=lin_pud,
                        door_signal=lin_doorsig, coherence=lin_coh),
            manifold=dict(grids=man_grids, off_manifold=man_off, p_down_minus_up=man_pud,
                          door_signal=man_doorsig, coherence=man_coh),
        )
        print(f"[main] door={door}: "
              f"lin off(mid)={lin_off[len(t_values)//2]:.2f} coh(mid)={lin_coh[len(t_values)//2]:.2f} | "
              f"man off(mid)={man_off[len(t_values)//2]:.2f} coh(mid)={man_coh[len(t_values)//2]:.2f}")
        print(f"          p(down)-p(up) end: lin={lin_pud[-1]:+.2f} man={man_pud[-1]:+.2f}")

    bundle = dict(
        checkpoint=args.checkpoint, n_steer=args.n_steer, horizon=args.horizon,
        t_values=t_values.tolist(), b0_p_up=float(pu[0]), b0_p_down=float(pd[0]),
        n_balanced_fork=len(balanced_fork),
        results=results,
    )
    with open(out_path, "w") as f:
        json.dump(MS.to_jsonable(bundle), f)
    print(f"[main] wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB), total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
