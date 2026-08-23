#!/usr/bin/env python3
"""Unified held-out evaluation of the three released agents.

One harness, one metric, one map split, so the paper's results table is
self-consistent:

  * maps      data/bridge_tunnel/forkwall6k/test.pkl (never seen in training)
  * metric    TRUE door metric -- an episode counts as a success iff the final
              cell is in the map's rewarded-door set. (The `episode return > 0`
              proxy used by the training loops scores fast wrong-door episodes
              as successes, because PBRS shaping outweighs the slack penalty.)
  * outcomes  correct door / wrong door / timeout, reported per category
  * actions   each agent in its native operating mode (PPO, STORM sample;
              Dreamer is deterministic) -- recorded in the output.

Run once per agent with the matching interpreter (see paper_rollouts.py header):

  PYTHONPATH=src python scripts/figures/paper_eval_all.py --agent ppo --episodes 900
  PYTHONPATH=src:r2dreamer_model python scripts/figures/paper_eval_all.py --agent dreamer --episodes 900
  (from STORM_model/) PYTHONPATH=.:..:../src python ../scripts/figures/paper_eval_all.py --agent storm --episodes 900

Results are merged into paper/figures/forkwall_paper/eval_all.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "figures"))

from cogniland.bridge_tunnel.env import BridgeTunnelEnv  # noqa: E402
from paper_rollouts import (  # noqa: E402
    FORKWALL_KWARGS, make_dreamer, make_ppo, make_storm,
)

CATS = ("balanced", "lakes", "rocky")
MODE = {"ppo": "sampled", "storm": "sampled", "dreamer": "deterministic"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, choices=["ppo", "dreamer", "storm"])
    p.add_argument("--episodes", type=int, default=900)
    p.add_argument("--maps", default=str(REPO / "data/bridge_tunnel/forkwall6k/test.pkl"))
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper/eval_all.json"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ppo-ckpt", default=str(REPO / "final_models/ppo/ppo_plain.pt"))
    p.add_argument("--storm-bundle", default=str(REPO / "final_models/storm"))
    p.add_argument("--storm-step", type=int, default=624489)
    p.add_argument("--dreamer-ckpt", default=str(REPO / "final_models/dreamer/dreamer_25M_bl64.pt"))
    p.add_argument("--dreamer-size", default="size25M")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.maps, "rb") as f:
        pool = pickle.load(f)
    rng = np.random.default_rng(args.seed)

    if args.agent == "ppo":
        act, reset = make_ppo(args.ppo_ckpt)
    elif args.agent == "storm":
        act, reset = make_storm(args.storm_bundle, args.storm_step)
    else:
        act, reset = make_dreamer(args.dreamer_ckpt, args.device, args.dreamer_size)

    stats = {c: defaultdict(int) for c in CATS}
    lengths = {c: [] for c in CATS}
    returns = {c: [] for c in CATS}
    t0 = time.time()
    for ep in range(args.episodes):
        rec = pool[int(rng.integers(0, len(pool)))]
        env = BridgeTunnelEnv(seed=0, map_record=rec, **FORKWALL_KWARGS)
        obs, _ = env.reset()
        reset()
        ret = 0.0
        for t in range(FORKWALL_KWARGS["max_steps"]):
            a = act(obs, False)
            obs, r, term, trunc, _ = env.step(a)
            ret += float(r)
            if term or trunc:
                break
        s = stats[rec.category]
        s["n"] += 1
        lengths[rec.category].append(t + 1)
        returns[rec.category].append(ret)
        if env._pos in (env._correct_cells or set()):
            s["correct"] += 1
        elif env._step_count < env.max_steps:
            s["wrong"] += 1
        else:
            s["timeout"] += 1
        if (ep + 1) % 100 == 0:
            done = sum(v["n"] for v in stats.values())
            ok = sum(v["correct"] for v in stats.values())
            print(f"  {done:4d}/{args.episodes}  running success {ok/done:.4f}",
                  flush=True)

    tot = {k: sum(v[k] for v in stats.values()) for k in ("n", "correct", "wrong", "timeout")}
    result = {
        "agent": args.agent,
        "mode": MODE[args.agent],
        "episodes": tot["n"],
        "success": tot["correct"] / tot["n"],
        "wrong_door": tot["wrong"] / tot["n"],
        "timeout": tot["timeout"] / tot["n"],
        "mean_length": float(np.mean(sum(lengths.values(), []))),
        "mean_return": float(np.mean(sum(returns.values(), []))),
        "wall_s": round(time.time() - t0, 1),
        "per_category": {
            c: {"n": stats[c]["n"],
                "success": stats[c]["correct"] / max(1, stats[c]["n"]),
                "wrong_door": stats[c]["wrong"] / max(1, stats[c]["n"]),
                "timeout": stats[c]["timeout"] / max(1, stats[c]["n"]),
                "mean_length": float(np.mean(lengths[c])) if lengths[c] else None}
            for c in CATS},
    }
    # decisive-door success: lakes+rocky only (a constant-door policy scores 50%)
    dec_n = stats["lakes"]["n"] + stats["rocky"]["n"]
    dec_ok = stats["lakes"]["correct"] + stats["rocky"]["correct"]
    result["decisive_success"] = dec_ok / max(1, dec_n)

    print(json.dumps({k: v for k, v in result.items() if k != "per_category"}, indent=1))
    for c in CATS:
        pc = result["per_category"][c]
        print(f"  {c:9s} n={pc['n']:4d} success={pc['success']:.4f} "
              f"wrong={pc['wrong_door']:.4f} timeout={pc['timeout']:.4f}")
    print(f"  decisive-door success (lakes+rocky): {result['decisive_success']:.4f}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(outp.read_text()) if outp.exists() else {}
    allr[args.agent] = result
    outp.write_text(json.dumps(allr, indent=1))
    print("merged ->", outp)


if __name__ == "__main__":
    main()
