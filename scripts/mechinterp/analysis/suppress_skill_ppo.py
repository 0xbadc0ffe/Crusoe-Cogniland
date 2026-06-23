#!/usr/bin/env python3
"""Probe-gated projected suppression of tunnel/bridge skills — PPO (BT).

At every step:  h_t = GRU carry.  p_skill = sigmoid(w·h + b)  (lookahead probe).
If p > tau_on, apply the closed-form projected-gradient step that moves the
probe margin down to logit(tau_off) with the MINIMAL perturbation constrained
to the activation PCA subspace V (stay on-manifold):

    d  = -(logit - logit_off) / (w·Pw) · Pw ,   P = VᵀV

The suppressed h' is CARRIED (the "thought" is removed from the recurrent
state, not just hidden from the actor). When p <= tau_on nothing happens —
normal behavior is untouched.

Experiments: none | tunnel | bridge | both, on held-out bundle maps.
Writes per-episode records (paths incl.) to outputs/suppression/ppo_rollouts.npz
and a metrics CSV.

    python -m scripts.mechinterp.analysis.suppress_skill_ppo --n-maps 12 --n-traj 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechinterp.analysis.bundle import ActivationBundle  # noqa: E402

MODES = ("none", "tunnel", "bridge", "both")


class Suppressor:
    """Probe-gated minimal projected step. With kind='sub' the probe weight is
    in the top-k activation-PCA subspace BY CONSTRUCTION (trained on PCA
    coords), so the minimal step along it is the projected-gradient attack
    constrained to the model's activation subspace. kind='full' = minimal step
    along the unconstrained probe."""

    def __init__(self, npz, skills, tau_on=0.6, tau_off=0.2, kind="sub",
                 overshoot=1.0, actor_dirs=None, alpha=0.0):
        pre = "w_sub_" if kind == "sub" else "w_"
        bre = "b_sub_" if kind == "sub" else "b_"
        self.l_off = float(np.log(tau_off / (1 - tau_off)))
        self.l_on = float(np.log(tau_on / (1 - tau_on)))
        self.overshoot = overshoot
        self.alpha = alpha
        self.parts = [(sk, npz[f"{pre}{sk}"].astype(np.float64),
                       float(npz[f"{bre}{sk}"]),
                       None if actor_dirs is None else actor_dirs[sk]) for sk in skills]

    def __call__(self, h):
        """h: (D,) numpy. Returns (h', n_triggered, total_delta_norm)."""
        trig, dn = 0, 0.0
        for sk, w, b, adir in self.parts:
            logit = float(h @ w + b)
            if logit > self.l_on:
                d = -self.overshoot * (logit - self.l_off) / float(w @ w) * w
                if adir is not None:                  # hybrid: also push down the
                    d = d - self.alpha * adir         # actor's skill-action readout
                h = h + d
                trig += 1; dn += float(np.linalg.norm(d))
        return h, trig, dn


class LeafSuppressor:
    """Foliation leaf-transport. Boxer gate = LDA posterior of the skill class;
    transport = pin the 3 transverse coordinates (basis B) to the avoid-leaf
    centroid, leaving the within-leaf (tangent) coordinates untouched:
        h' = h + Bᵀ(mu_avoid − B h)
    """

    CLS = {"free": 0, "avoid": 1, "bridge": 2, "tunnel": 3}

    def __init__(self, fol, skills, tau_on=0.6, target="avoid", inlp=False):
        self.W = fol["W"]; self.bias = fol["b"]          # boxer (4, D), (4,)
        self.skills = [self.CLS[s] for s in skills]
        self.tau_on = tau_on
        if inlp:                                          # full detectable bundle
            self.parts = [(fol[f"U_{s}"].astype(np.float64),
                           fol[f"muav_{s}"].astype(np.float64)) for s in skills]
        else:                                             # 3-dim LDA transverse
            self.parts = [(fol["B"].astype(np.float64),
                           fol["mu"][self.CLS[target]].astype(np.float64))
                          for _ in skills]

    def __call__(self, h):
        logits = self.W @ h + self.bias
        e = np.exp(logits - logits.max()); p = e / e.sum()
        trig, dn = 0, 0.0
        for c, (U, mu) in zip(self.skills, self.parts):
            if p[c] > self.tau_on and len(U):
                d = U.T @ (mu - U @ h)
                h = h + d
                trig += 1; dn += float(np.linalg.norm(d))
        return h, trig, dn


def route_dirs(bundle, probes_path, H):
    """unit(mean h | <skill> coming within H) − (mean h | avoid coming within H),
    computed on the probe trainer's cached X sample (identical selection)."""
    from mechinterp.analysis.train_lookahead_probes import lookahead_labels
    name = bundle.path.name
    X = np.load(Path(probes_path).parent / f"Xcache_{name}.npy")
    lab = bundle.labels.sort_values(["map_id", "traj_id", "t"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(lab), min(200_000, len(lab)), replace=False)
    sel.sort()
    ids = lab["row_id"].to_numpy()[sel]
    sel_sorted = sel[np.argsort(ids)]
    ys = {sk: lookahead_labels(lab, sk, int(H))[sel_sorted]
          for sk in ("tunnel", "bridge", "avoid")}
    out = {}
    for sk in ("tunnel", "bridge"):
        a = X[(ys["avoid"] == 1) & (ys[sk] == 0)].mean(0)
        s = X[ys[sk] == 1].mean(0)
        d = (s - a).astype(np.float64)               # skill − avoid
        out[sk] = d / np.linalg.norm(d)
    return out


def crossings(path, terrain0, T):
    """Count obstacle crossings: maximal runs of >=2 consecutive path cells whose
    ORIGINAL tile was ROCK (tunnel crossing) / WATER (bridge crossing)."""
    runs = {"tunnel": 0, "bridge": 0}
    kinds = [("tunnel", T.ROCK), ("bridge", T.WATER)]
    for k, tile in kinds:
        run = 0
        for (r, c) in path:
            if terrain0[r, c] == tile:
                run += 1
            else:
                if run >= 2:
                    runs[k] += 1
                run = 0
        if run >= 2:
            runs[k] += 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="activation_datasets/bt_ppo")
    ap.add_argument("--checkpoint", default="released_models/bridge_tunnel/ppo_gru.pt")
    ap.add_argument("--probes", default="outputs/suppression/probes_bt_ppo.npz")
    ap.add_argument("--n-maps", type=int, default=12)
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--tau-on", type=float, default=0.6)
    ap.add_argument("--tau-off", type=float, default=0.2)
    ap.add_argument("--probe-kind", choices=("sub", "full"), default="sub")
    ap.add_argument("--overshoot", type=float, default=1.0,
                    help="scale the projected step past the probe boundary")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help=">0: hybrid — also push h along -alpha*unit(dir)")
    ap.add_argument("--dir", choices=("actor", "route"), default="actor",
                    help="hybrid direction: actor readout row | route DoM (skill−avoid)")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="after an intervention, leave the agent untouched for N steps")
    ap.add_argument("--leaf", action="store_true",
                    help="foliation leaf-transport suppression (boxer gate + transverse pin)")
    ap.add_argument("--inlp", action="store_true",
                    help="leaf transport over the FULL INLP-exhausted transverse bundle")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/suppression"))
    args = ap.parse_args()

    import torch
    import pandas as pd
    from cogniland.bridge_tunnel import tiles as T

    b = ActivationBundle(args.dataset)
    sys.path.insert(0, str(b.path))
    from env_min import MiniBridgeTunnelEnv
    from policy_min import PPOGRUPolicy

    npz = np.load(args.probes)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pol = PPOGRUPolicy.from_checkpoint(ck, b.view_size, b.manifest["n_scalars"], device="cpu")
    torch.set_grad_enabled(False)

    # held-out maps = the LAST n_maps of the bundle (probe train used a random
    # 70% of maps; to be safe we evaluate on maps whose ids were NOT in the
    # probe's train split — recompute that split identically).
    m = b.maps
    rng = np.random.default_rng(0)
    u = np.arange(len(m["map_seed"]))
    tr_maps = set(rng.choice(u, int(0.7 * len(u)), replace=False).tolist())
    ev_maps = [i for i in u if i not in tr_maps][:args.n_maps]
    print(f"eval maps: {ev_maps}", flush=True)

    # push directions for the hybrid term
    adirs = None
    if args.alpha > 0 and args.dir == "actor":
        # actor readout rows: BT action 5 = mine (tunnel), 4 = place (bridge)
        Wa = pol.actor.weight.detach().cpu().numpy().astype(np.float64)
        adirs = {"tunnel": Wa[5] / np.linalg.norm(Wa[5]),
                 "bridge": Wa[4] / np.linalg.norm(Wa[4])}
    elif args.alpha > 0 and args.dir == "route":
        # route DoM: −unit(mean h|avoid-coming − mean h|skill-coming); the
        # suppressor SUBTRACTS alpha*dir, so dir = skill − avoid ⇒ push toward avoid
        adirs = route_dirs(b, args.probes, npz["horizon"])

    if args.leaf or args.inlp:
        fol = np.load(Path(args.probes).parent / f"foliation_{b.path.name}.npz")

        def mk(skills):
            return LeafSuppressor(fol, skills, args.tau_on, inlp=args.inlp)
    else:
        def mk(skills):
            return Suppressor(npz, skills, args.tau_on, args.tau_off, args.probe_kind,
                              args.overshoot, adirs, args.alpha)
    sups = {"none": None, "tunnel": mk(["tunnel"]), "bridge": mk(["bridge"]),
            "both": mk(["tunnel", "bridge"])}

    rows, paths = [], {}
    for mode in MODES:
        sup = sups[mode]
        for mi in ev_maps:
            terr0 = m["terrain"][mi].copy()
            for tid in range(args.n_traj):
                env = MiniBridgeTunnelEnv(m["terrain"][mi], m["spawn"][mi], variant="bt",
                                          view_size=b.view_size, max_steps=args.max_steps)
                torch.manual_seed(int(m["map_seed"][mi]) * 100000 + tid)
                obs = env.reset()
                h = torch.zeros(1, 1, pol.gru_hidden)
                path = [tuple(map(int, env.pos))]
                n_trig = 0; sum_dn = 0.0; cd = 0
                reached = False
                for t in range(args.max_steps):
                    o = {k: torch.from_numpy(np.asarray(v)[None]) for k, v in obs.items()}
                    feat = pol._encode(o).reshape(1, 1, -1)
                    y, h = pol.gru(feat, h)
                    if sup is not None and cd == 0:
                        hv, trig, dn = sup(h.squeeze().numpy().astype(np.float64))
                        if trig:
                            h = torch.from_numpy(hv.astype(np.float32)).view(1, 1, -1)
                            n_trig += trig; sum_dn += dn
                            cd = args.cooldown
                    elif cd > 0:
                        cd -= 1
                    logits = pol.actor(h.squeeze(0))
                    a = int(torch.distributions.Categorical(logits=logits).sample()[0])
                    obs, reached_now, done = env.step(a)
                    path.append(tuple(map(int, env.pos)))
                    if done:
                        reached = bool(reached_now)
                        break
                mined_cells = np.argwhere((terr0 == T.ROCK) & (env.terrain != T.ROCK))
                placed_cells = np.argwhere((terr0 == T.WATER) & (env.terrain != T.WATER))
                n_mine, n_place = len(mined_cells), len(placed_cells)
                paths[f"{mode}/{mi}/{tid}/mined"] = mined_cells.astype(np.int16)
                paths[f"{mode}/{mi}/{tid}/placed"] = placed_cells.astype(np.int16)
                cr = crossings(path, terr0, T)
                rows.append(dict(mode=mode, map_id=int(mi), traj=tid, reached=reached,
                                 steps=len(path) - 1, tunnel_cross=cr["tunnel"],
                                 bridge_cross=cr["bridge"], mined_rocks=n_mine,
                                 placed_water=n_place, n_interventions=n_trig,
                                 mean_dnorm=(sum_dn / max(n_trig, 1))))
                paths[f"{mode}/{mi}/{tid}"] = np.asarray(path, np.int16)
        df = pd.DataFrame([r for r in rows if r["mode"] == mode])
        print(f"[{mode:6s}] reach={df.reached.mean():.2f} "
              f"tunnel_cross/ep={df.tunnel_cross.mean():.2f} "
              f"bridge_cross/ep={df.bridge_cross.mean():.2f} "
              f"mined={df.mined_rocks.mean():.1f} placed={df.placed_water.mean():.1f} "
              f"interventions/ep={df.n_interventions.mean():.1f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    pd.DataFrame(rows).to_csv(args.out_dir / f"ppo_metrics{tag}.csv", index=False)
    np.savez_compressed(args.out_dir / f"ppo_rollouts{tag}.npz",
                        **paths, eval_maps=np.asarray(ev_maps))
    print("wrote", args.out_dir / f"ppo_metrics{tag}.csv", "and rollouts npz")


if __name__ == "__main__":
    main()
