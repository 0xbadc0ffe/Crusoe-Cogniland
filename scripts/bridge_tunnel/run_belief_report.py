#!/usr/bin/env python
"""Produce the data bundle for the bridge_tunnel fork_wall belief report.

    python scripts/bridge_tunnel/run_belief_report.py \
        --checkpoint r2dreamer_model/runs/forkwall_nocommit/latest.pt \
        --out outputs/bridge_tunnel_forkwall/belief_report_data.json \
        --n-per-category 65
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dreamer_belief_report_r2d as M  # noqa: E402


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
    ap.add_argument("--checkpoint", default="r2dreamer_model/runs/forkwall_nocommit/latest.pt")
    ap.add_argument("--out", default="outputs/bridge_tunnel_forkwall/belief_report_data.json")
    ap.add_argument("--raw-cache", default="outputs/bridge_tunnel_forkwall/belief_report_raw.pkl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-category", type=int, default=65)
    ap.add_argument("--decoder-steps", type=int, default=6000)
    ap.add_argument("--dream-horizon", type=int, default=16)
    ap.add_argument("--n-dream-per-category", type=int, default=25)
    ap.add_argument("--skip-collect", action="store_true", help="reuse --raw-cache episodes/buffer")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = pathlib.Path(args.raw_cache)

    t0 = time.time()
    agent, config = M.load_agent(args.checkpoint, args.device)
    print(f"[main] agent loaded ({time.time()-t0:.1f}s)")

    if args.skip_collect and raw_path.exists():
        with open(raw_path, "rb") as f:
            cache = pickle.load(f)
        episodes, decoder_buf = cache["episodes"], cache["decoder_buf"]
        print(f"[main] reusing cached rollout: {len(episodes)} episodes, {len(decoder_buf)} decoder samples")
    else:
        decoder_buf = []
        episodes = M.collect_episodes(agent, args.device, args.n_per_category, decoder_buf=decoder_buf)
        with open(raw_path, "wb") as f:
            pickle.dump({"episodes": episodes, "decoder_buf": decoder_buf}, f)
        print(f"[main] rollout cached -> {raw_path}")

    overall_success = float(np.mean([e["success"] for e in episodes]))
    per_cat_success = {c: float(np.mean([e["success"] for e in episodes if e["category"] == c]))
                        for c in M.CATEGORIES}
    print(f"[main] overall success on fresh held-out seeds: {overall_success:.3f}  per-cat: {per_cat_success}")

    decoder_acc = M.train_decoder(agent, args.device, decoder_buf, steps=args.decoder_steps)

    exp1 = M.fit_and_run_experiment1(agent, args.device, episodes)
    print(f"[main] exp1: probe_acc={exp1['probe_acc']:.3f} swap_flip_rate={exp1['swap_flip_rate']:.3f}")

    exp2 = M.run_experiment2(agent, args.device, exp1,
                              n_dream_per_category=args.n_dream_per_category,
                              horizon=args.dream_horizon)
    for c in M.CATEGORIES:
        w = np.mean(exp2["dream_stats"][c]["water"])
        r = np.mean(exp2["dream_stats"][c]["rock"])
        print(f"[main] exp2 {c}: dream water={w:.3f} rock={r:.3f}")

    # strip heavy per-episode belief vectors before dumping to JSON (keep only
    # what the report needs); minimap grids are small and kept.
    def strip_ep(e):
        return {k: v for k, v in e.items() if k not in ("stoch_pre", "deter_pre", "stoch_mid", "deter_mid")}

    bundle = dict(
        checkpoint=args.checkpoint,
        n_episodes=len(episodes),
        overall_success=overall_success,
        per_cat_success=per_cat_success,
        decoder_held_out_acc=decoder_acc,
        exp1=dict(
            probe_acc=exp1["probe_acc"],
            probe_acc_mid=exp1["probe_acc_mid"],
            probe_acc_logreg=exp1["probe_acc_logreg"],
            probe_acc_logreg_mid=exp1["probe_acc_logreg_mid"],
            logreg_C=exp1["logreg_C"],
            confusion_logreg=exp1["confusion_logreg"],
            pred_matches_door=exp1["pred_matches_door"],
            true_matches_door=exp1["true_matches_door"],
            confusion=exp1["confusion"],
            confusion_labels=exp1["confusion_labels"],
            door_matrix=exp1["door_matrix"],
            door_labels=exp1["door_labels"],
            majority_action=exp1["majority_action"],
            swap_results=exp1["swap_results"],
            swap_flip_rate=exp1["swap_flip_rate"],
            swap_change_rate=exp1["swap_change_rate"],
            swap_results_logreg=exp1["swap_results_logreg"],
            swap_flip_rate_logreg=exp1["swap_flip_rate_logreg"],
            swap_change_rate_logreg=exp1["swap_change_rate_logreg"],
            n_train=exp1["n_train"],
            n_test=exp1["n_test"],
        ),
        exp2=dict(
            dream_stats=exp2["dream_stats"],
            example_sequences=exp2["example_sequences"],
            swap_sequences=exp2["swap_sequences"],
            horizon=exp2["horizon"],
        ),
        episodes_summary=[strip_ep(e) for e in episodes],
    )

    with open(out_path, "w") as f:
        json.dump(to_jsonable(bundle), f)
    print(f"[main] wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB), total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
