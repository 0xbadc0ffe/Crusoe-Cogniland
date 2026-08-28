#!/usr/bin/env python3
"""Recover STORM per-episode training returns from the offline W&B datastores.

The storm2 trainer already logged every completed training episode to W&B
(``train/<env>/reward`` keyed by ``train_steps``), and the released seed runs
were recorded ``--offline``, so the full curves survive inside
``STORM_model/wandb/offline-run-*/run-*.wandb``. This script replays those
datastores and writes one ``metrics.jsonl`` per seed using the SAME key names
the Dreamer runs use, so the figure code reads every agent uniformly.

Reconstructing the x-axis needs care. ``train_steps`` counts steps of a SINGLE
env (the vectorized bridge_tunnel wrapper increments ``timestep`` by 1 per
``env.step``) AND it restarts at 0 every training segment, because the trainer
calls ``agent.train`` once per ``eval_interval_frames`` and the agent resets the
envs each call. The companion key ``train/<env>/frame`` holds
``total_frames_trained``, i.e. the frames finished BEFORE the current segment,
so the global frame count is

    frames = frame + train_steps * num_parallel_envs

  STORM_model/.venv/bin/python scripts/figures/paper/extract_storm_wandb_returns.py
"""
from __future__ import annotations

import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

REPO = Path(__file__).resolve().parents[3]
WANDB = REPO / "STORM_model/wandb"
OUT = REPO / "outputs/storm_train_curves"

# released fork_wall seeds -> offline run id (results/<id> holds the checkpoints)
SEEDS = {1: "8xyaj5kd", 2: "wxm1mfd4", 3: "8uf5i8tp"}
REWARD_KEY = "train/BridgeTunnel/forkwall/reward"
STEP_KEY = "train_steps"
SEG_KEY = "train/BridgeTunnel/forkwall/frame"      # frames done before this segment
NUM_ENVS = 32  # configs/envs/bridge_tunnel_storm2_seed*.yaml: num_parallel_envs


def find_run(run_id: str) -> Path:
    hits = sorted(WANDB.glob(f"offline-run-*-{run_id}/run-{run_id}.wandb"))
    if not hits:
        raise FileNotFoundError(f"no offline datastore for run {run_id}")
    return hits[-1]


def scan(path: Path) -> list[tuple[int, float]]:
    """Replay one datastore and return [(env_frames, episode return), ...]."""
    ds = DataStore()
    ds.open_for_scan(str(path))
    rows: list[tuple[int, float]] = []
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        items = {(it.key or "/".join(it.nested_key)): it.value_json
                 for it in rec.history.item}
        if REWARD_KEY not in items or STEP_KEY not in items:
            continue
        seg = int(float(json.loads(items.get(SEG_KEY, "0"))))
        step = int(float(json.loads(items[STEP_KEY])))
        rows.append((seg + step * NUM_ENVS, float(json.loads(items[REWARD_KEY]))))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for seed, run_id in SEEDS.items():
        rows = scan(find_run(run_id))
        rows.sort(key=lambda r: r[0])
        dst = OUT / f"seed{seed}.jsonl"
        with open(dst, "w") as fh:
            for step, score in rows:
                fh.write(json.dumps({"step": step, "episode/score": score,
                                     "run_id": run_id, "seed": seed}) + "\n")
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        print(f"seed{seed} ({run_id}): {len(rows)} episodes, "
              f"frames {min(xs):,}..{max(xs):,}, "
              f"return {min(ys):.2f}..{max(ys):.2f} -> {dst}")


if __name__ == "__main__":
    main()
