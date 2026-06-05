"""Plot trajectories of a trained PPO+GRU checkpoint on the 12 demo maps.

Loads each pickled MapRecord under ``data/demo_maps/*.pkl``, pins it as
the fixed map of a ``CognilandNavEnv``, rolls out the policy greedily,
and writes:

* ``<out_dir>/<biome>_<idx>.png``  — per-map trajectory plot
* ``<out_dir>/grid.png``           — 4×3 grid (4 maps × 3 biomes)
* ``<out_dir>/summary.json``       — per-map success / length / return

Uses ``scripts/crafter/play_ppo_gru.py``'s `_rollout` + `_plot_trajectory`
helpers via the same importlib trick the trainer uses.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cogniland.nav import CognilandNavEnv
from cogniland.nav.tiles import TILE_COLORS


def _load_helpers():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "play_ppo_gru", str(here / "play_ppo_gru.py")
    )
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def _env_from_record(record, ckpt_args: dict, seed: int) -> CognilandNavEnv:
    return CognilandNavEnv(
        size=int(record.terrain.shape[0]),
        map_type=record.map_type,
        view_size=ckpt_args.get("view_size", 21),
        tile_px=ckpt_args.get("tile_px", 8),
        obs_mode=ckpt_args.get("obs_mode", "symbolic"),
        max_steps=ckpt_args.get("max_steps", 1000),
        seed=seed,
        map_record=record,
    )


def _composite_grid(trajs: list[dict], out_path: Path, title: str) -> None:
    """4 cols (maps within biome) × 3 rows (biomes)."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    biome_order = ("balanced", "lake", "rocky")
    by_biome = {b: [] for b in biome_order}
    for t in trajs:
        by_biome[t["map_type"]].append(t)

    for r, biome in enumerate(biome_order):
        cells = by_biome.get(biome, [])
        # sort by the demo index in the filename (stored on each traj below)
        cells = sorted(cells, key=lambda x: x.get("_demo_idx", 0))
        for c in range(4):
            ax = axes[r, c]
            if c >= len(cells):
                ax.axis("off")
                continue
            traj = cells[c]
            terrain = traj["terrain"]
            ax.imshow(TILE_COLORS[terrain], origin="upper", interpolation="nearest")
            pos = np.array(traj["positions"])
            ax.plot(pos[:, 1], pos[:, 0], "-", c="white", lw=1.4, alpha=0.85)
            ax.scatter(pos[:, 1], pos[:, 0], c=np.arange(len(pos)),
                       cmap="viridis", s=4, zorder=3, edgecolors="none")
            sr, sc = traj["spawn"]; tr, tc = traj["target"]
            ax.scatter([sc], [sr], marker="o", s=80, fc="lime", ec="black", zorder=4)
            ax.scatter([tc], [tr], marker="*", s=160, fc="gold", ec="black", zorder=4)
            cs = traj.get("commit_step")
            if cs is not None:
                cr, cc = traj["positions"][cs]
                ax.scatter([cc], [cr], marker="X", s=90, fc="red", ec="black", zorder=5)
            tag = "✓" if traj["reached"] else "✗"
            commit_lbl = traj["committed_object"] or "none"
            ax.set_title(
                f"{biome} #{traj.get('_demo_idx', c)}  "
                f"{tag} L={traj['length']} R={traj['episode_return']:+.2f}\n"
                f"built: {commit_lbl}  correct: {traj['correct_object']}",
                fontsize=9,
            )
            ax.set_xticks([]); ax.set_yticks([])

    handles = [
        mpatches.Patch(color="lime", label="spawn"),
        mpatches.Patch(color="gold", label="target"),
        mpatches.Patch(color="red", label="commit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--maps-dir", type=Path, default=Path("data/demo_maps"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", default=True,
                   help="argmax move (default on); belief is always deterministic")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = args.out_dir or (Path("rollouts") / f"{args.checkpoint.stem}_on_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    helpers = _load_helpers()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    print(f"loaded {args.checkpoint}")
    print(f"  trained on env_size={ckpt_args.get('env_size')} "
          f"map_type={ckpt_args.get('map_type')}  obs_mode={ckpt_args.get('obs_mode')}")

    # Build policy once using any demo map for the obs space.
    map_files = sorted(args.maps_dir.glob("*.pkl"))
    if not map_files:
        sys.exit(f"no maps under {args.maps_dir}")
    with map_files[0].open("rb") as f:
        first_rec = pickle.load(f)
    proto_env = _env_from_record(first_rec, ckpt_args, args.seed)
    policy = helpers._build_policy(proto_env, ckpt_args, device)
    policy.load_state_dict(ckpt["policy"])
    proto_env.close()

    summary = []
    trajs = []
    for mp in map_files:
        biome, idx = mp.stem.rsplit("_", 1)
        with mp.open("rb") as f:
            record = pickle.load(f)
        env = _env_from_record(record, ckpt_args, args.seed)
        traj = helpers._rollout(policy, env, device, greedy=args.greedy)
        env.close()
        traj["_demo_idx"] = int(idx)
        trajs.append(traj)

        title = (
            f"{biome} #{idx}  R={traj['episode_return']:+.2f}  "
            f"L={traj['length']}  "
            f"{'SUCCESS' if traj['reached'] else 'FAIL'}  "
            f"built: {traj['committed_object'] or 'none'}  "
            f"correct: {traj['correct_object']}"
        )
        helpers._plot_trajectory(traj, out_dir / f"{biome}_{idx}.png", title)
        summary.append({
            "map": f"{biome}_{idx}",
            "biome": biome,
            "reached": bool(traj["reached"]),
            "length": int(traj["length"]),
            "return": float(traj["episode_return"]),
            "committed_object": traj["committed_object"],
            "correct_object": traj["correct_object"],
        })
        print(f"  {biome}_{idx}: "
              f"{'OK ' if traj['reached'] else 'FAIL'} "
              f"L={traj['length']:>3d} R={traj['episode_return']:+.2f} "
              f"built={str(traj['committed_object']):<7s} "
              f"correct={traj['correct_object']}")

    _composite_grid(
        trajs, out_dir / "grid.png",
        title=f"PPO ({args.checkpoint.name}) on demo maps  ·  "
              f"successes {sum(s['reached'] for s in summary)}/{len(summary)}",
    )
    (out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "num_maps": len(summary),
        "successes": sum(s["reached"] for s in summary),
        "results": summary,
    }, indent=2))
    print(f"\nwrote {out_dir}/grid.png + {len(summary)} per-map PNGs")


if __name__ == "__main__":
    main()
