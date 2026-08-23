#!/usr/bin/env python
"""Generate a FIXED fork_wall (BT-rules) map dataset: 2000 maps per category
(balanced / lakes / rocky) = 6000 total, split 80/20 into train / test.

All models (r2dreamer DreamerV3 and PyTorch PPO) train on the SAME train split;
the test split is held out for evaluation. Map-generation params match the
fork_wall BT-rules env config (btc mapgen; commit is a MECHANICS flag that does
not affect the map): size=32, width=64, tree_frac=0.03, goal_half=0,
fork_wall=True, passage_half=1, wall_margin=1.

  python scripts/bridge_tunnel/make_forkwall_dataset.py \
      --out-dir data/bridge_tunnel/forkwall6k --n-per-category 2000 --test-frac 0.2
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cogniland.bridge_tunnel.mapgen import generate_commit_map, CATEGORIES  # noqa: E402

# distinct seed blocks per category so requested seeds never collide
SEED_BASE = {"balanced": 0, "lakes": 10_000_000, "rocky": 20_000_000}
MAP_KW = dict(size=32, width=64, tree_frac=0.03, goal_half=0,
              fork_wall=True, passage_half=1, wall_margin=1, mem_gap=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/bridge_tunnel/forkwall6k")
    ap.add_argument("--n-per-category", type=int, default=2000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=12345)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.split_seed)

    train, test = [], []
    per_cat_counts = {}
    t0 = time.time()
    for cat in CATEGORIES:
        recs = []
        base = SEED_BASE[cat]
        for i in range(args.n_per_category):
            rec = generate_commit_map(seed=base + i, category=cat, **MAP_KW)
            recs.append(rec)
            if (i + 1) % 500 == 0:
                print(f"  [{cat}] {i+1}/{args.n_per_category}  ({time.time()-t0:.0f}s)", flush=True)
        # shuffle within category, split 80/20
        idx = rng.permutation(len(recs))
        n_test = int(round(args.test_frac * len(recs)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train.extend(recs[j] for j in train_idx)
        test.extend(recs[j] for j in test_idx)
        per_cat_counts[cat] = dict(train=len(train_idx), test=len(test_idx))
        print(f"[{cat}] {len(recs)} maps -> {len(train_idx)} train / {len(test_idx)} test")

    # shuffle the pooled splits so categories are interleaved
    train = [train[j] for j in rng.permutation(len(train))]
    test = [test[j] for j in rng.permutation(len(test))]

    meta = dict(n_per_category=args.n_per_category, test_frac=args.test_frac,
                map_kw=MAP_KW, per_cat_counts=per_cat_counts,
                n_train=len(train), n_test=len(test), split_seed=args.split_seed)
    with open(out_dir / "train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(out_dir / "test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # sanity: category balance per split
    def cat_hist(recs):
        h = {c: 0 for c in CATEGORIES}
        for r in recs:
            h[r.category] += 1
        return h
    print(f"\nwrote {out_dir}/train.pkl  ({len(train)} maps, {cat_hist(train)})")
    print(f"wrote {out_dir}/test.pkl   ({len(test)} maps, {cat_hist(test)})")
    print(f"total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
