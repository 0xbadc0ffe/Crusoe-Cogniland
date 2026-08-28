#!/usr/bin/env python3
"""Per-step traces of the 20 rollouts behind figure 7.5, for the ghost video.

`collect_chosen_rollouts.py` keeps only the path and a route label. The video
also needs, for every step, the facing and the tool events, so that the merged
world can be rebuilt and the mine/build animations can fire on the right cell
at the right frame.

Seeds match `collect_chosen_rollouts.py` exactly (2000 + k), so these are the
same twenty episodes the figure draws.

  PYTHONPATH=src python scripts/figures/paper/collect_ghost_rollouts.py --agent ppo
  PYTHONPATH=src:r2dreamer_model ... --agent dreamer --device cuda
  (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/... --agent storm
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from paper_rollouts import (  # noqa: E402
    FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm,
)

SM = REPO / "scripts/mechinterp/steering_maps"
OUT = REPO / "outputs/ghost_videos"
A_BUILD, A_MINE = 4, 5
# facing id -> delta, mirroring env._FACE_DELTA
FACE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    chosen = json.loads((SM / "chosen_maps.json").read_text())
    pool = pickle.load(open(REPO / "data/bridge_tunnel/forkwall6k/test.pkl", "rb"))

    if a.agent == "ppo":
        act, reset = make_ppo(str(REPO / "final_models/ppo/ppo_plain_noaux.pt"), sampled=True)
    elif a.agent == "storm":
        act, reset = make_storm(str(REPO / "final_models/storm"), 624489, sampled=True)
    else:
        act, reset = make_dreamer(str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"),
                                  a.device, "size25M", sampled=True)

    out = {}
    for cat, mid in chosen.items():
        rec = pool[mid]
        rolls = []
        for k in range(a.rollouts):
            np.random.seed(2000 + k)                  # identical to the figure
            try:
                import torch
                torch.manual_seed(2000 + k)
            except Exception:
                pass
            env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
            obs, _ = env.reset()
            reset()
            if hasattr(act, "set_seed"):
                act.set_seed(2000 + k)
            steps = [dict(r=int(env._pos[0]), c=int(env._pos[1]),
                          facing=int(env._facing), ev=None)]
            for t in range(FORKWALL_KWARGS["max_steps"]):
                ac = act(obs, False)
                obs, _, term, trunc, info = env.step(ac)
                ev = None
                if ac in (A_BUILD, A_MINE) and (info.get("placed") or info.get("mined")):
                    # the tool acts on the faced cell; a tool action never turns
                    # the agent, so info["facing"] is the facing it acted with
                    dr, dc = FACE_DELTA[int(info["facing"])]
                    ev = dict(kind="build" if ac == A_BUILD else "mine",
                              r=int(env._pos[0] + dr), c=int(env._pos[1] + dc))
                steps.append(dict(r=int(env._pos[0]), c=int(env._pos[1]),
                                  facing=int(info["facing"]), ev=ev))
                if term or trunc:
                    break
            rolls.append(dict(steps=steps,
                              correct=bool(env._pos in (env._correct_cells or set()))))
        out[cat] = dict(map_id=int(mid), rollouts=rolls)
        n_ev = sum(sum(s["ev"] is not None for s in r["steps"]) for r in rolls)
        print(f"  {cat:9s} map {mid:4d}: {len(rolls)} rollouts, "
              f"lengths {min(len(r['steps']) for r in rolls)}-"
              f"{max(len(r['steps']) for r in rolls)}, {n_ev} tool events", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"ghost_{a.agent}.json"
    p.write_text(json.dumps(out))
    print("wrote", p)


if __name__ == "__main__":
    main()
