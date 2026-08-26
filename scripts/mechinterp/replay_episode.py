#!/usr/bin/env python3
"""Replay one episode from the belief dataset, exactly — optionally steered.

This is the piece that makes steering comparisons meaningful. Given an agent and
a map id it re-runs the episode with the seed recorded in the dataset, so with
no intervention it reproduces the stored trajectory step for step. Pass a
steering hook and the *only* difference between the two runs is the
intervention, which is what lets you attribute an effect to it.

  # verify the dataset reproduces
  python scripts/mechinterp/replay_episode.py --agent ppo --map-id 7 --verify

  # verify a whole sample of episodes
  python scripts/mechinterp/replay_episode.py --agent ppo --verify-sample 25

  # steered replay: push h along a direction, every step, alpha 1.0
  python scripts/mechinterp/replay_episode.py --agent ppo --map-id 7 \\
      --steer-npy dir.npy --alpha 1.0

Use the same environment as the dataset build (see the manifest's `reproduce`).

As a library:

    from replay_episode import replay
    base = replay("ppo", map_id=7)
    hooked = replay("ppo", map_id=7, hook=lambda h, t, info: h + alpha * v)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))
DS = REPO / "activation_datasets/cogniland_belief"

CKPT = {
    "ppo": dict(ckpt=str(REPO / "final_models/ppo/ppo_plain_noaux.pt")),
    "storm": dict(bundle=str(REPO / "final_models/storm"), step=624489),
    "dreamer": dict(ckpt=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"),
                    size="size25M"),
}


def episode_meta(agent, map_id, ds=DS):
    """The row this episode has in <agent>_episodes.csv, including its seed."""
    with open(Path(ds) / f"{agent}_episodes.csv") as fh:
        for row in csv.DictReader(fh):
            if int(row["map_id"]) == map_id:
                return row
    raise SystemExit(f"map_id {map_id} not in {agent}_episodes.csv")


def replay(agent, map_id, hook=None, ds=DS, maps=None, device="cuda", seed=None):
    """Re-run one episode. `hook(feat, t, info) -> feat` intervenes on the
    carried state each step; return the input unchanged for a plain replay."""
    import pickle
    from cogniland.bridge_tunnel.env import BridgeTunnelEnv
    from paper_rollouts import FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm

    if seed is None:
        seed = int(episode_meta(agent, map_id, ds)["seed"])
    maps = maps or str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl")
    with open(maps, "rb") as fh:
        rec = pickle.load(fh)[map_id]

    # Construct the agent BEFORE seeding, because the dataset builder does:
    # loading a checkpoint consumes RNG, so seeding first leaves the generator
    # in a different state and the replay silently diverges.
    c = CKPT[agent]
    if agent == "ppo":
        act, reset = make_ppo(c["ckpt"], sampled=True)
    elif agent == "storm":
        act, reset = make_storm(c["bundle"], c["step"], sampled=True)
    else:
        act, reset = make_dreamer(c["ckpt"], device, c["size"], sampled=True)

    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    if hook is not None:
        if not hasattr(act, "set_hook"):
            raise SystemExit(f"{agent}: adapter has no set_hook(); steering "
                             "is only wired for agents that expose it")
        act.set_hook(hook)

    if hasattr(act, "set_seed"):
        act.set_seed(seed)      # agents with their own PRNG (STORM)
    env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
    obs, _ = env.reset()
    reset()
    acts, positions, feats, ret = [], [env._pos], [], 0.0
    for t in range(FORKWALL_KWARGS["max_steps"]):
        a = act(obs, False)
        if hasattr(act, "get_features"):
            feats.append(act.get_features())
        acts.append(int(a))
        obs, r, term, trunc, _ = env.step(a)
        ret += float(r)
        positions.append(env._pos)
        if term or trunc:
            break
    fr = env._pos
    top = {p[0] for p in rec.top_goal_cells}
    bot = {p[0] for p in rec.bottom_goal_cells}
    return dict(map_id=map_id, seed=seed, category=rec.category,
                steps=len(acts), actions=acts, positions=positions,
                features=feats, ret=round(ret, 5),
                success=env._pos in (env._correct_cells or set()),
                door="top" if fr[0] in top else "bottom" if fr[0] in bot else "none")


def stored(agent, map_id, ds=DS):
    """Actions and positions as recorded in the dataset, for comparison."""
    acts, pos = [], []
    with open(Path(ds) / f"{agent}_steps.csv") as fh:
        for row in csv.DictReader(fh):
            if int(row["map_id"]) == map_id:
                acts.append(int(row["action"]))
                pos.append((int(row["row"]), int(row["col"])))
    return acts, pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    p.add_argument("--map-id", type=int)
    p.add_argument("--verify", action="store_true",
                   help="check the replay matches the stored trajectory")
    p.add_argument("--verify-sample", type=int, default=0,
                   help="verify N evenly spaced episodes")
    p.add_argument("--steer-npy", help="(D,) direction to add to the carried state")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ds", default=str(DS))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    hook = None
    if a.steer_npy:
        v = np.load(a.steer_npy).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        hook = lambda f, t, info: f + a.alpha * v          # noqa: E731

    ids = ([a.map_id] if a.map_id is not None else [])
    if a.verify_sample:
        n = sum(1 for _ in open(Path(a.ds) / f"{a.agent}_episodes.csv")) - 1
        ids = list(np.linspace(0, n - 1, a.verify_sample).astype(int))
    if not ids:
        sys.exit("give --map-id or --verify-sample")

    bad = 0
    for mid in ids:
        r = replay(a.agent, int(mid), hook=hook, ds=a.ds, device=a.device)
        if a.verify or a.verify_sample:
            sa, sp = stored(a.agent, int(mid), a.ds)
            same = (r["actions"] == sa and [tuple(x) for x in r["positions"][:len(sp)]] == sp)
            bad += not same
            print(f"map {mid:5d}  {r['category']:9s} seed={r['seed']}  "
                  f"steps={r['steps']:3d} {'MATCH' if same else 'DIFFERS'}")
        else:
            print(f"map {mid:5d}  {r['category']:9s} seed={r['seed']}  steps={r['steps']:3d} "
                  f"door={r['door']:6s} success={r['success']} ret={r['ret']:+.3f}")
    if a.verify or a.verify_sample:
        print(f"\n{len(ids) - bad}/{len(ids)} episodes reproduce exactly")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
