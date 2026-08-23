#!/usr/bin/env python3
"""Can an activation edit inhibit BOTH skills (mine tunnels / build bridges) and
still finish fork_wall?  We push the hidden state each step (gradient descent) to
drive pi(build)+pi(mine) -> 0, then measure success and — crucially — how many
SUCCESSFUL episodes used zero skill (genuine detour successes).

  python scripts/bridge_tunnel/forkwall_disable_skills.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cogniland.bridge_tunnel import generate_commit_map  # noqa: E402
from eval_bridge_tunnel_forkwall import _load_policy  # noqa: E402
from eval_bridge_tunnel_forkwall_steered import batched_rollout_steered  # noqa: E402
from eval_bridge_tunnel_commit_ppo_steered import A_BUILD, A_MINE  # noqa: E402

CATS = ["rocky", "lakes", "balanced"]


def suppress_both(policy, alpha, iters):
    def fn(h, t, info):
        x = h.squeeze(0).detach()
        with torch.enable_grad():
            for _ in range(iters):
                x = x.clone().requires_grad_(True)
                logits, _ = policy._heads(x)
                p = torch.softmax(logits, -1)
                loss = (p[:, A_BUILD] + p[:, A_MINE]).sum()
                g, = torch.autograd.grad(loss, x)
                x = (x - alpha * g / (g.norm(dim=-1, keepdim=True) + 1e-8)).detach()
        return x.unsqueeze(0)
    return fn


def run(policy, mk, cat, fn, view_size, device, commit, n_maps, n_traj, seed0, max_steps):
    succ, wrong, none, em, eb, pm, pb = [], [], [], [], [], [], []
    zero_and_succ = 0; n = 0
    for j in range(n_maps):
        rec = mk(seed0 + j, cat)
        out = batched_rollout_steered(policy, rec, n_traj, view_size, max_steps,
                                      device, commit=commit, steer_fn=fn)
        ap = out["action_probs"]; va = np.isfinite(ap[..., 0])
        pm.append(ap[..., A_MINE][va]); pb.append(ap[..., A_BUILD][va])
        s = out["success"]; ra = out["reached_any"]
        succ += s.tolist(); wrong += (ra & ~s).tolist(); none += (~ra).tolist()
        em += out["n_mines"].tolist(); eb += out["n_builds"].tolist()
        skill0 = (out["n_mines"] == 0) & (out["n_builds"] == 0)
        zero_and_succ += int((skill0 & s).sum()); n += len(s)
    return dict(success=float(np.mean(succ)), wrong=float(np.mean(wrong)),
                none=float(np.mean(none)), exec_mine=float(np.mean(em)),
                exec_build=float(np.mean(eb)), pi_mine=float(np.concatenate(pm).mean()),
                pi_build=float(np.concatenate(pb).mean()),
                zero_skill_success=zero_and_succ / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("released_models/bridge_tunnel_commit/ppo_gru_forkwall_nocommit.pt"))
    ap.add_argument("--maps", type=int, default=12)
    ap.add_argument("--traj", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-seed", type=int, default=10_000)
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

    conds = {"baseline": None,
             f"disable-both (a={a.alpha}, it={a.iters})": suppress_both(policy, a.alpha, a.iters)}
    for name, fn in conds.items():
        print(f"\n[{name}]")
        for cat in CATS:
            r = run(policy, mk, cat, fn, view_size, device, commit, a.maps, a.traj,
                    a.eval_seed, a.max_steps)
            print(f"  {cat:9s} succ={r['success']:.0%} wrong={r['wrong']:.0%} "
                  f"none={r['none']:.0%} | exec mine={r['exec_mine']:.2f} "
                  f"build={r['exec_build']:.2f} | π mine={r['pi_mine']:.4f} "
                  f"build={r['pi_build']:.4f} | ZERO-skill success={r['zero_skill_success']:.0%}")


if __name__ == "__main__":
    main()
