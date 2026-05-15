#!/usr/bin/env python3
"""Generate a Cogniland navigation map dataset as a single .npz + meta.json.

Example
-------
    python scripts/generate_cogniland_dataset.py \\
        --num-maps 1000 --size 64 \\
        --out data/cogniland_64.npz --seed 0 --lake-ratio 0.5

The .npz holds stacked arrays for every successfully-generated map, and the
sibling .meta.json records counts and generator settings. Lake-vs-rocky
assignment is deterministic given the seed and ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogniland.nav import MapGenError, generate_map  # noqa: E402
from cogniland.nav import skills as sk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-maps", type=int, required=True)
    parser.add_argument("--size", type=int, required=True, choices=(32, 64, 96, 128))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lake-ratio", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=100)
    args = parser.parse_args()

    if not 0.0 <= args.lake_ratio <= 1.0:
        raise SystemExit(f"--lake-ratio must be in [0, 1], got {args.lake_ratio}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    is_lake = rng.random(args.num_maps) < args.lake_ratio

    terrains = np.zeros((args.num_maps, args.size, args.size), dtype=np.int8)
    spawns = np.zeros((args.num_maps, 2), dtype=np.int32)
    targets = np.zeros((args.num_maps, 2), dtype=np.int32)
    map_types = np.zeros(args.num_maps, dtype=np.int8)
    correct_obj = np.zeros(args.num_maps, dtype=np.int8)
    no_skill = np.zeros(args.num_maps, dtype=np.float32)
    raft_cost = np.zeros(args.num_maps, dtype=np.float32)
    harness_cost = np.zeros(args.num_maps, dtype=np.float32)
    constraints = np.ones(args.num_maps, dtype=bool)
    seeds = np.zeros(args.num_maps, dtype=np.int64)

    t0 = time.time()
    n_failed = 0
    for i in range(args.num_maps):
        per_seed = int(rng.integers(0, 2**31))
        mt = "lake" if is_lake[i] else "rocky"
        try:
            rec = generate_map(
                size=args.size,
                map_type=mt,
                seed=per_seed,
                max_retries=args.max_retries,
            )
            terrains[i] = rec.terrain
            spawns[i] = rec.spawn
            targets[i] = rec.target
            map_types[i] = 0 if rec.map_type == "lake" else 1
            correct_obj[i] = 0 if rec.correct_object == sk.RAFT else 1
            no_skill[i] = rec.no_skill_cost
            raft_cost[i] = rec.raft_cost
            harness_cost[i] = rec.harness_cost
            seeds[i] = per_seed
        except MapGenError as exc:
            print(f"[{i:04d}] FAILED ({mt}): {exc}", file=sys.stderr)
            constraints[i] = False
            n_failed += 1
        if (i + 1) % max(1, args.num_maps // 20) == 0:
            elapsed = time.time() - t0
            print(f"  generated {i+1}/{args.num_maps}  ({elapsed:.1f}s)")

    np.savez_compressed(
        args.out,
        terrain=terrains,
        spawn=spawns,
        target=targets,
        map_type=map_types,
        correct_skill=correct_obj,
        no_skill_cost=no_skill,
        raft_cost=raft_cost,
        harness_cost=harness_cost,
        constraints_passed=constraints,
        seed=seeds,
    )

    meta = {
        "size": args.size,
        "num_maps": args.num_maps,
        "n_failed": int(n_failed),
        "n_lake": int((is_lake & constraints).sum()),
        "n_rocky": int(((~is_lake) & constraints).sum()),
        "lake_ratio": args.lake_ratio,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "elapsed_seconds": time.time() - t0,
        "version": "0.1",
    }
    meta_path = args.out.parent / (args.out.stem + ".meta.json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {args.out}  ({args.num_maps} maps, {n_failed} failed)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
