"""Dump training metrics from an offline W&B run directory.

Parses the run's `output.log` or live `wandb-summary.json` to surface
moving averages and loss dynamics without syncing to the cloud.

Usage:
    python scripts/dump_run_metrics.py <wandb_run_dir>
    # e.g.
    python scripts/dump_run_metrics.py wandb/offline-run-20260421_092126-hyyel1q2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: dump_run_metrics.py <wandb_run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "files" / "wandb-summary.json"
    config_path = run_dir / "files" / "config.yaml"

    if not summary_path.exists():
        print(f"[miss] {summary_path} — run may not have flushed yet")
    else:
        with summary_path.open() as f:
            try:
                s = json.load(f)
            except Exception as e:
                print(f"  failed to parse: {e}")
                return
        keys = sorted([k for k in s if k.startswith("train/") or k in ("train_steps", "train_frames")])
        for k in keys:
            v = s[k]
            if isinstance(v, (int, float)):
                print(f"  {k:40s} {v:+.4f}")
            else:
                print(f"  {k:40s} {v!r}")

    # Parse output.log to extract time-series of ma_r / ma_s if present
    log_path = run_dir / "files" / "output.log"
    if log_path.exists():
        lines = log_path.read_text(errors="replace").splitlines()
        ma_r_points = []
        for line in lines:
            if "ma_r=" in line:
                # e.g. "ep=12345, fps=7000, ma_r=-5.81"
                try:
                    chunk = line[line.rindex("ma_r="):]
                    val = float(chunk.split("=")[1].split()[0].strip(",]"))
                    ma_r_points.append(val)
                except Exception:
                    continue
        if ma_r_points:
            print(f"\nma_r samples (from tqdm stream) count={len(ma_r_points)}")
            n = len(ma_r_points)
            for i, idx in enumerate([0, n // 4, n // 2, 3 * n // 4, n - 1]):
                print(f"  @{i*25:3d}%  ma_r={ma_r_points[idx]:+.3f}")


if __name__ == "__main__":
    main()
