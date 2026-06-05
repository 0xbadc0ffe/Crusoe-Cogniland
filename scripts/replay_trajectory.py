#!/usr/bin/env python3
"""Reproduce (and optionally steer) a single dataset trajectory.

Given (map_seed[, category], traj_seed) this re-runs the EXACT episode the
activation dataset logged (cpu + per-trajectory action seed), and can inject a
steering vector into the GRU hidden ``gru_h`` over a row range — the hook for the
activation-steering experiments.

    # plain replay
    python scripts/replay_trajectory.py --env bridge_tunnel_commit \\
        --checkpoint released_models/bridge_tunnel_commit/ppo_commit_onehot.pt \\
        --map-seed 10000 --category lakes --traj-seed 1000000000

    # steered replay: add alpha*vec to gru_h on steps [a,b)
    python scripts/replay_trajectory.py ... --inject mine_dir.npy --alpha -6 --rows 20:60
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))


def _cfg(env):
    if env == "bridge_tunnel":
        from cogniland.bridge_tunnel import generate_bridge_tunnel_map
        from cogniland.bridge_tunnel.env import BridgeTunnelEnv
        from train_ppo_bridge_tunnel import PPOGRUPolicy
        return dict(Env=BridgeTunnelEnv, Policy=PPOGRUPolicy, commit=False,
                    gen=lambda s, c, kw: generate_bridge_tunnel_map(seed=s, **kw),
                    anames=["up", "down", "left", "right", "place", "mine"])
    from cogniland.bridge_tunnel_commit import generate_commit_map
    from cogniland.bridge_tunnel_commit.env import BridgeTunnelCommitEnv
    from train_ppo_bridge_tunnel_commit import PPOGRUPolicy
    return dict(Env=BridgeTunnelCommitEnv, Policy=PPOGRUPolicy, commit=True,
                gen=lambda s, c, kw: generate_commit_map(seed=s, category=c, **kw),
                anames=["up", "down", "left", "right", "build", "mine"])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("bridge_tunnel", "bridge_tunnel_commit"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--map-seed", type=int, required=True)
    p.add_argument("--category", default=None, help="commit env: balanced/lakes/rocky")
    p.add_argument("--traj-seed", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--inject", type=Path, default=None, help=".npy vector added to gru_h")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rows", default=None, help="step range a:b for injection (default all)")
    args = p.parse_args()
    cfg = _cfg(args.env)
    if cfg["commit"] and args.category is None:
        raise SystemExit("--category is required for bridge_tunnel_commit")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ca = ckpt["args"]; es = ca.get("env_size", 32); ew = ca.get("env_width") or es
    view = ca.get("view_size", 21); gh = ca.get("goal_half", 1)
    goal_half = gh if (gh is not None and gh >= 0) else None
    if args.env == "bridge_tunnel":
        kw = dict(size=es, width=ew, orientation=ca.get("orientation", "natural"),
                  water_frac=ca.get("water_frac", 0.14), rock_frac=ca.get("rock_frac", 0.14),
                  tree_frac=ca.get("tree_frac", 0.03), goal_half=goal_half)
    else:
        kw = dict(size=es, width=ew, tree_frac=ca.get("tree_frac", 0.03), goal_half=goal_half)

    sd = ckpt["policy"]; oe = ca.get("obs_encoding", "embed")
    n_tiles = int(sd["tile_embed.weight"].shape[0]) if "tile_embed.weight" in sd \
        else int(sd["cnn.0.weight"].shape[1]) - 2
    if "tile_embed.weight" not in sd:
        oe = "onehot"
    n_act = int(sd["actor.weight"].shape[0])
    dummy = cfg["Env"](size=es, width=ew, view_size=view); dummy.reset()
    policy = cfg["Policy"](dummy.observation_space, num_actions=n_act,
                           gru_hidden=ca.get("gru_hidden", 128), embed_dim=ca.get("embed_dim", 256),
                           num_tile_classes=n_tiles, obs_encoding=oe)
    policy.load_state_dict(sd); policy.eval()

    vec = None
    if args.inject is not None:
        vec = torch.from_numpy(np.load(args.inject).astype(np.float32)).view(1, 1, -1)
    a0, b0 = 0, args.max_steps
    if args.rows:
        a0, b0 = (int(x) for x in args.rows.split(":"))

    rec = cfg["gen"](args.map_seed, args.category, kw)
    torch.manual_seed(args.traj_seed)
    env = cfg["Env"](map_record=rec, size=rec.terrain.shape[0], width=rec.terrain.shape[1],
                     view_size=view, max_steps=args.max_steps)
    obs = env.reset()[0]
    h = torch.zeros(1, 1, policy.gru_hidden); done = torch.zeros(1)
    acts, commits = [], []
    for t in range(args.max_steps):
        mm = torch.from_numpy(obs["minimap"])[None, None]; sc = torch.from_numpy(obs["scalars"])[None, None]
        gru_out, h = policy._gru_forward({"minimap": mm, "scalars": sc}, done[None], h)
        if vec is not None and a0 <= t < b0:
            gru_out = gru_out + args.alpha * vec; h = h + args.alpha * vec   # steer + persist
        logits, _ = policy._heads(gru_out.squeeze(0))
        a = int(torch.distributions.Categorical(logits=logits).sample()[0])
        acts.append(a); commits.append(int(getattr(env, "_commit", 0)))
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    reached = term
    name = {0: "none", 1: "build", 2: "mine"}
    print(f"steps={len(acts)} reached={bool(reached)} "
          f"final_commit={name[commits[-1]] if cfg['commit'] else 'n/a'} "
          f"actions={[cfg['anames'][a] for a in acts[:30]]}{' ...' if len(acts)>30 else ''}")
    if vec is not None:
        print(f"(steered: +{args.alpha}*{args.inject.name} on gru_h rows [{a0}:{b0}))")


if __name__ == "__main__":
    main()
