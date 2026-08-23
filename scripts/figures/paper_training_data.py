#!/usr/bin/env python3
"""Collect training telemetry for all three agents into one tidy JSON.

Sources (each agent logs differently; this is the single place that knows how):
  PPO      scripts/bridge_tunnel/slurm/logs/ppo_pl_*.out   (stdout iteration lines)
  Dreamer  r2dreamer_model/runs/fw_sw_*/metrics.jsonl      (jsonl, sparse keys)
  STORM    STORM_model/wandb/offline-run-*/run-*.wandb     (wandb datastore)

Output: paper/figures/forkwall_paper/training_data.json
  {agent: {run_name: {"series": {key: [[x, y], ...]}, "meta": {...}}}}

Usage:  PYTHONPATH=src python scripts/figures/paper_training_data.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PPO_LINE = re.compile(
    r"iter=\s*(?P<iter>\d+)/(?P<total>\d+)\s+step=\s*(?P<step>\d+)\s+sps=\s*(?P<sps>[\d.]+)"
    r"\s+ret=(?P<ret>[+\-][\w.+-]+)\s+succ=(?P<succ>[\w.]+)\s+build=(?P<build>[\w.]+)"
    r"\s+mine=(?P<mine>[\w.]+)\s+len=(?P<len>[\w.]+)\s+kl=(?P<kl>[\d.eE+-]+)"
    r"\s+belief_acc=(?P<belief>[\d.]+)")

# run-name -> human-readable arm (the plain-reward exploration sweep)
PPO_ARMS = {
    "ppo_pl_all":          "ent 0.15 + anneal + belief-aux",
    "ppo_pl_anneal":       "ent 0.03 + anneal",
    "ppo_pl_ent12":        "ent 0.12 (constant)",
    "ppo_pl_ent15_anneal": "ent 0.15 + anneal  ★released",
}


def f(x):
    try:
        v = float(x)
        return None if v != v else v          # drop NaN
    except Exception:
        return None


def collect_ppo(logdir: Path):
    out = {}
    for path in sorted(logdir.glob("ppo_pl_*.out")):
        name = path.stem.rsplit("_", 1)[0]                 # strip slurm id
        arm = next((v for k, v in PPO_ARMS.items()
                    if name == k or name.startswith(k + "_s")), None)
        if arm is None:
            continue
        seed = 0 if not re.search(r"_s(\d)$", name) else int(re.search(r"_s(\d)$", name).group(1))
        series = defaultdict(list)
        meta = {}
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("policy params:"):
                meta["params"] = int(line.split(":")[1].strip().replace(",", ""))
            if line.startswith("num_iterations="):
                for kv in line.split():
                    k, _, v = kv.partition("=")
                    meta[k] = int(v)
            m = PPO_LINE.match(line)
            if not m:
                continue
            step = int(m["step"])
            for key, raw in (("return", m["ret"]), ("success", m["succ"]),
                             ("ep_length", m["len"]), ("kl", m["kl"]),
                             ("belief_acc", m["belief"]), ("sps", m["sps"])):
                v = f(raw)
                if v is not None:
                    series[key].append([step, v])
        if series:
            out[f"{arm} | seed {seed}"] = {
                "series": {k: v for k, v in series.items()},
                "meta": {**meta, "arm": arm, "seed": seed, "log": path.name},
            }
    return out


DREAMER_KEYS = [
    "episode/score", "episode/eval_score", "episode/eval_success",
    "episode/length", "episode/eval_length",
    "train/loss/policy", "train/loss/value", "train/loss/dyn", "train/loss/rep",
    "train/loss/rew", "train/loss/con", "train/loss/vector",
    "train/action_entropy", "train/ret", "train/val", "fps/fps",
]


def collect_dreamer(runsdir: Path):
    out = {}
    for run in sorted(runsdir.glob("fw_sw_*")):
        mfile = run / "metrics.jsonl"
        if not mfile.exists():
            continue
        series = defaultdict(list)
        for line in mfile.read_text(errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            step = rec.get("step")
            if step is None:
                continue
            for k in DREAMER_KEYS:
                if k in rec:
                    v = f(rec[k])
                    if v is not None:
                        series[k].append([int(step), v])
        if series:
            size, bl = "25M" if "25M" in run.name else "12M", \
                       "128" if "bl128" in run.name else "64"
            out[f"{size}, batch_length {bl}"] = {
                "series": {k: v for k, v in series.items()},
                "meta": {"run": run.name, "size": size, "batch_length": int(bl)},
            }
    return out


STORM_KEYS_SUFFIX = [
    # per-episode training telemetry
    "moving_avg_reward", "moving_avg_success_rate", "moving_avg_length",
    "reward", "success", "length", "frame", "fps",
    # world-model / actor-critic losses
    "loss/rec", "loss/rew", "loss/con", "loss/dyn", "loss/rep",
    "loss/policy", "loss/value", "total_loss", "entropy",
    # periodic in-training evaluation
    "avg_success", "avg_reward", "avg_length",
]


def collect_storm(wandb_dir: Path):
    """Read the offline wandb datastore (STORM's trainer logs there)."""
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
    except Exception as e:                                  # pragma: no cover
        print("  [storm] wandb unavailable:", e)
        return {}
    out = {}
    for rundir in sorted(wandb_dir.glob("offline-run-*")):
        files = list(rundir.glob("run-*.wandb"))
        if not files:
            continue
        ds = DataStore()
        try:
            ds.open_for_scan(str(files[0]))
        except Exception:
            continue
        series = defaultdict(list)
        while True:
            try:
                data = ds.scan_data()
            except Exception:
                break
            if data is None:
                break
            rec = pb.Record()
            rec.ParseFromString(data)
            if not rec.HasField("history"):
                continue
            step = rec.history.step.num
            for item in rec.history.item:
                key = item.key or ".".join(item.nested_key)
                short = key.split("/", 2)[-1] if key.startswith("train/") else key
                if not any(key.endswith(s) or short == s for s in STORM_KEYS_SUFFIX):
                    continue
                try:
                    v = f(json.loads(item.value_json))
                except Exception:
                    v = None
                if v is not None:
                    series[key].append([int(step), v])
        if series:
            out[rundir.name.split("-")[-1]] = {
                "series": {k: v for k, v in series.items()},
                "meta": {"rundir": rundir.name},
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO / "paper/figures/forkwall_paper/training_data.json"))
    args = p.parse_args()

    data = {
        "ppo": collect_ppo(REPO / "scripts/bridge_tunnel/slurm/logs"),
        "dreamer": collect_dreamer(REPO / "r2dreamer_model/runs"),
        "storm": collect_storm(REPO / "STORM_model/wandb"),
    }
    for agent, runs in data.items():
        print(f"{agent}: {len(runs)} runs")
        for name, blob in runs.items():
            n = {k: len(v) for k, v in blob["series"].items()}
            head = ", ".join(f"{k}:{c}" for k, c in list(n.items())[:4])
            print(f"   {name:42s} {head}")
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data))
    print("wrote", outp, f"({outp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
