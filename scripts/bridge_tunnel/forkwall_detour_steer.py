#!/usr/bin/env python3
"""Steer fork_wall toward the DETOUR strategy (go around / skip the skill) instead
of mining/building — using a direction calibrated from the agent's OWN behavior:
mean hidden state in its zero-skill (detour) successes minus its skill-using
successes. Push toward detour and ask: does the agent solve maps WITHOUT mining/
building (zero-skill success up) while staying successful?

  python scripts/bridge_tunnel/forkwall_detour_steer.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_commit_map  # noqa: E402
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import (  # noqa: E402
    batched_rollout_steered, direction_class_mean_fw, direction_skill_mean_fw)
from eval_bridge_tunnel_commit_ppo_steered import A_BUILD, A_MINE  # noqa: E402

CATS = ["rocky", "lakes", "balanced"]


@torch.no_grad()
def collect_labeled(policy, rec, n, view_size, max_steps, device, commit):
    H, W = rec.terrain.shape
    envs = [BridgeTunnelCommitEnv(map_record=rec, size=H, width=W, view_size=view_size,
                                  max_steps=max_steps, commit=commit) for _ in range(n)]
    obs = [e.reset()[0] for e in envs]
    h = torch.zeros(1, n, policy.gru_hidden, device=device)
    done = torch.zeros(n, device=device)
    active = np.ones(n, bool)
    nm = np.zeros(n, int); nb = np.zeros(n, int); succ = np.zeros(n, bool)
    hid, alive = [], []
    for t in range(max_steps):
        mm = torch.from_numpy(np.stack([o["minimap"] for o in obs]))[None].to(device)
        sc = torch.from_numpy(np.stack([o["scalars"] for o in obs]))[None].to(device)
        _, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        x = h.squeeze(0)
        hid.append(x.cpu().numpy()); alive.append(active.copy())
        logits, _ = policy._heads(x)
        acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, r, term, trunc, info = e.step(int(acts[i]))
            obs[i] = o
            if info["mined"]:
                nm[i] += 1
            if info["placed"]:
                nb[i] += 1
            if term:
                succ[i] = bool(info["reached_target"]); active[i] = False
            elif trunc:
                active[i] = False
        done = torch.zeros(n, device=device)
        if not active.any():
            break
    return np.stack(hid), np.stack(alive), nm, nb, succ    # hid (T,n,H)


def detour_direction(policy, mk, view_size, device, commit, n_maps, n_traj,
                     seed0, max_steps):
    det, skl = [], []
    for cat in CATS:
        for j in range(n_maps):
            hid, alive, nm, nb, succ = collect_labeled(
                policy, mk(seed0 + j, cat), n_traj, view_size, max_steps, device, commit)
            is_det = (nm == 0) & (nb == 0) & succ
            is_skl = ((nm + nb) > 0) & succ
            for i in range(hid.shape[1]):
                if not alive[:, i].any():
                    continue
                pool = det if is_det[i] else (skl if is_skl[i] else None)
                if pool is not None:
                    pool.append(hid[alive[:, i], i])
    det = np.concatenate(det); skl = np.concatenate(skl)
    print(f"[detour] {len(det)} detour-success states / {len(skl)} skill-success states")
    dm = det.mean(0); sm = skl.mean(0)
    d = torch.from_numpy((dm - sm).astype(np.float32)).to(device)
    return (d / d.norm(),
            torch.from_numpy(dm.astype(np.float32)).to(device))


def make_push(u, rho, strength):
    def fn(h, t, info):
        x = h.squeeze(0)
        proj = x @ u
        x = x + strength * (rho - proj).unsqueeze(-1) * u.unsqueeze(0)
        return x.unsqueeze(0)
    return fn


def evaluate(policy, mk, cat, fn, view_size, device, commit, n_maps, n_traj, seed0, max_steps):
    succ, none, em, eb, zsucc = [], [], [], [], 0
    n = 0
    for j in range(n_maps):
        out = batched_rollout_steered(policy, mk(seed0 + j, cat), n_traj, view_size,
                                      max_steps, device, commit=commit, steer_fn=fn)
        s = out["success"]
        succ += s.tolist(); none += (~out["reached_any"]).tolist()
        em += out["n_mines"].tolist(); eb += out["n_builds"].tolist()
        z = (out["n_mines"] == 0) & (out["n_builds"] == 0)
        zsucc += int((z & s).sum()); n += len(s)
    return dict(success=float(np.mean(succ)), none=float(np.mean(none)),
                exec_mine=float(np.mean(em)), exec_build=float(np.mean(eb)),
                zero_skill_success=zsucc / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt"))
    ap.add_argument("--maps", type=int, default=12)
    ap.add_argument("--traj", type=int, default=12)
    ap.add_argument("--calib-maps", type=int, default=8)
    ap.add_argument("--calib-traj", type=int, default=16)
    ap.add_argument("--strengths", default="0.5,1.0,2.0")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--calib-seed", type=int, default=22_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    policy, cargs, view_size, env_size, env_width = _load_policy(a.checkpoint, device)
    commit = False if cargs.get("no_commit", False) else None
    ph = cargs.get("passage_half", 1); wm = cargs.get("wall_margin", 1)
    gh = cargs.get("goal_half", 0); gh = gh if (gh is not None and gh >= 0) else None

    def mk(seed, cat):
        return generate_commit_map(size=env_size, width=env_width, seed=seed, category=cat,
                                   tree_frac=cargs.get("tree_frac", 0.03), goal_half=gh,
                                   fork_wall=True, passage_half=ph, wall_margin=wm)

    u_detour, det_mean = detour_direction(policy, mk, view_size, device, commit,
                                          a.calib_maps, a.calib_traj, a.calib_seed, a.max_steps)
    calib = dict(map_factory=mk, view_size=view_size, device=device, commit=commit,
                 n_maps=a.calib_maps, n_traj=8, seed_start=a.calib_seed, max_steps=a.max_steps)
    u_bel, _ = direction_class_mean_fw(policy, **calib)
    u_skill, _ = direction_skill_mean_fw(policy, **calib, skill_cats=("balanced",))
    # belief-orthogonalized detour direction (isolate the 'go around' strategy)
    u_perp = u_detour - (u_detour @ u_bel) * u_bel
    u_perp = u_perp / u_perp.norm()
    rho_raw = float(det_mean @ u_detour); rho_perp = float(det_mean @ u_perp)
    print(f"[detour] cos(detour, belief)={float(u_detour@u_bel):+.2f}  "
          f"cos(detour, skill)={float(u_detour@u_skill):+.2f}  "
          f"cos(detour⊥, belief)={float(u_perp@u_bel):+.2f}")

    strengths = [float(s) for s in a.strengths.split(",")]
    axes = {"raw": (u_detour, rho_raw), "⊥belief": (u_perp, rho_perp)}
    res = {"baseline": {c: evaluate(policy, mk, c, None, view_size, device, commit,
                                    a.maps, a.traj, a.eval_seed, a.max_steps) for c in CATS}}
    print("\n[baseline]")
    for c in CATS:
        r = res["baseline"][c]
        print(f"  {c:9s} succ={r['success']:.0%} none={r['none']:.0%} "
              f"zero-skill-succ={r['zero_skill_success']:.0%} | "
              f"exec mine={r['exec_mine']:.2f} build={r['exec_build']:.2f}")
    for axname, (u, rho) in axes.items():
        for st in strengths:
            key = f"{axname} s{st}"
            res[key] = {c: evaluate(policy, mk, c, make_push(u, rho, st), view_size,
                                    device, commit, a.maps, a.traj, a.eval_seed, a.max_steps)
                        for c in CATS}
            print(f"\n[detour-{axname} strength={st}]")
            for c in CATS:
                r = res[key][c]
                print(f"  {c:9s} succ={r['success']:.0%} none={r['none']:.0%} "
                      f"zero-skill-succ={r['zero_skill_success']:.0%} | "
                      f"exec mine={r['exec_mine']:.2f} build={r['exec_build']:.2f}")

    # figure: per category, zero-skill-success & total success for
    # baseline / raw@1 / ⊥belief@1
    s1 = min(strengths, key=lambda s: abs(s - 1.0))
    conds = ["baseline", f"raw s{s1}", f"⊥belief s{s1}"]
    labs = ["baseline", "detour raw", "detour ⊥belief"]
    x = np.arange(len(conds)); col = ["#888", "#d62728", "#2ca02c"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, c in zip(axs, CATS):
        zs = [res[k][c]["zero_skill_success"] for k in conds]
        sc = [res[k][c]["success"] for k in conds]
        ax.bar(x - 0.18, sc, width=0.36, color="#2ca02c", label="total success")
        ax.bar(x + 0.18, zs, width=0.36, color="#9467bd", label="zero-skill success")
        ax.set_ylim(0, 1.03); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(c); ax.legend(fontsize=8)
    axs[0].set_ylabel("fraction of episodes")
    fig.suptitle("Steering fork_wall toward DETOUR (solve without mining/building) — "
                 "raw vs belief-orthogonalized",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path("outputs/bridge_tunnel_forkwall/detour_steer.png")
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
