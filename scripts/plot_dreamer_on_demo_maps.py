"""Plot trajectories of a frozen Dreamer checkpoint on the 12 demo maps.

Loads each pickled ``MapRecord`` under ``data/demo_maps/*.pkl``, runs the
Dreamer policy greedily on each in turn (one map per episode, forced via
``map_idx``), and writes:

* ``<out_dir>/<biome>_<idx>.png``  — per-map trajectory plot
* ``<out_dir>/grid.png``           — 4×3 grid (4 maps × 3 biomes)
* ``<out_dir>/summary.json``       — per-map success / length / return

Mirrors ``scripts/plot_ppo_on_demo_maps.py`` for the Dreamer side.
Reuses ``viz_dreamer_trajectory.load_frozen`` to rebuild the model
graph, and stacks all 12 demo records into a single ``EnvParams`` so
``env.step`` only JIT-compiles once.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogniland.crafter_in_cogniland import (
    CrafterInCognilandEnv, EnvParams, constants as C, build_obs,
)
from cogniland.crafter_in_cogniland.maps import _stack_records
from cogniland.crafter_in_cogniland.state import EnvState
from cogniland.nav.tiles import TILE_COLORS

OBJECT_NAMES = {C.OBJ_NONE: "none", C.OBJ_RAFT: "raft", C.OBJ_HARNESS: "harness"}


def _load_viz_helpers():
    """Import viz_dreamer_trajectory as a module (sibling script)."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "viz_dreamer_trajectory", str(here / "viz_dreamer_trajectory.py")
    )
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # register before exec so flax/dataclass `sys.modules.get(__module__)`
    # introspection during nn.Module class creation can find the module.
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _reset_to_map(params: EnvParams, map_idx: int) -> tuple[dict, EnvState]:
    """Build the initial state for a fixed ``map_idx``, bypassing reset's RNG."""
    sr = params.spawn[map_idx, 0]
    sc = params.spawn[map_idx, 1]
    state = EnvState(
        map_idx=jnp.int32(map_idx),
        agent_r=sr, agent_c=sc,
        facing=jnp.int32(1),
        active_object=jnp.int32(C.OBJ_NONE),
        step_count=jnp.int32(0),
        last_ctg=params.ctg_none[map_idx, sr, sc],
    )
    return build_obs(state, params), state


def _rollout_fixed_map(env, params, params_dict, map_idx, rng, max_steps):
    """Greedy Dreamer rollout on a single fixed map."""
    encoder = params_dict["encoder"]
    rssm = params_dict["rssm"]
    actor = params_dict["actor"]
    wm = params_dict["wm_params"]
    ac_p = params_dict["ac_params"]

    obs, state = _reset_to_map(params, map_idx)
    positions = [(int(state.agent_r), int(state.agent_c))]
    actions: list[int] = []
    rewards: list[float] = []
    committed_object: int | None = None
    commit_step: int | None = None

    rssm_state = rssm.initial_state((1,))
    is_first = jnp.ones((1,), dtype=bool)
    last_action_oh = jnp.zeros((1, C.NUM_ACTIONS))
    reached = False
    done = False

    for t in range(max_steps):
        rng, s_enc, s_step = jax.random.split(rng, 3)
        mm = obs["minimap"].astype(jnp.float32)[None] / float(C.NUM_TERRAIN_TILES)
        sc = obs["scalars"].astype(jnp.float32)[None]
        flat = jnp.concatenate([mm.reshape(1, -1), sc], axis=-1)
        embed = encoder.apply(wm["encoder"], flat)
        _, posterior = rssm.apply(
            wm["rssm"], rssm_state, last_action_oh, embed, is_first,
            rngs={"stoch": s_enc},
        )
        rssm_state = posterior
        logits = actor.apply(ac_p["actor"], posterior.features())
        action_idx = jnp.argmax(logits, axis=-1)            # deterministic
        a_oh = jax.nn.one_hot(action_idx, C.NUM_ACTIONS)
        # pass None so env.step closes over self.default_params; passing the
        # EnvParams pytree directly tries to hash it as a static arg.
        obs, state, r, done, info = env.step(s_step, state, action_idx[0], None)

        positions.append((int(state.agent_r), int(state.agent_c)))
        actions.append(int(action_idx[0]))
        rewards.append(float(r))
        if committed_object is None and int(state.active_object) != C.OBJ_NONE:
            committed_object = int(state.active_object)
            commit_step = t + 1  # positions index after this step
        if bool(info["reached_target"]):
            reached = True
        last_action_oh = a_oh
        is_first = jnp.zeros((1,), dtype=bool)
        if bool(done):
            break

    return {
        "positions": positions,
        "actions": actions,
        "rewards": rewards,
        "episode_return": float(sum(rewards)),
        "length": len(actions),
        "reached": bool(reached),
        "committed_object": (
            OBJECT_NAMES[committed_object] if committed_object is not None else None
        ),
        "commit_step": commit_step,
    }


def _plot_single(traj: dict, terrain: np.ndarray, spawn, target,
                 out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(TILE_COLORS[terrain], origin="upper", interpolation="nearest")
    pos = np.array(traj["positions"])
    ax.plot(pos[:, 1], pos[:, 0], "-", c="white", lw=1.5, alpha=0.85)
    ax.scatter(pos[:, 1], pos[:, 0], c=np.arange(len(pos)),
               cmap="viridis", s=6, zorder=3, edgecolors="none")
    sr, sc_ = spawn
    tr, tc = target
    ax.scatter([sc_], [sr], marker="o", s=90, fc="lime", ec="black", zorder=4)
    ax.scatter([tc], [tr], marker="*", s=180, fc="gold", ec="black", zorder=4)
    cs = traj.get("commit_step")
    if cs is not None and cs < len(pos):
        cr, cc = pos[cs]
        ax.scatter([cc], [cr], marker="X", s=100, fc="red", ec="black", zorder=5)
        ax.annotate(
            f"commit @ step {cs}\nbuilt: {traj['committed_object']}",
            xy=(cc, cr), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [
        mpatches.Patch(color="lime", label="spawn"),
        mpatches.Patch(color="gold", label="target"),
    ]
    if traj.get("commit_step") is not None:
        handles.append(mpatches.Patch(color="red",
                                      label=f"commit ({traj['committed_object']})"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _composite_grid(trajs: list[dict], out_path: Path, title: str) -> None:
    """4 cols (maps within biome) × 3 rows (biomes), mirroring the PPO script."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    biome_order = ("balanced", "lake", "rocky")
    by_biome = {b: [] for b in biome_order}
    for t in trajs:
        by_biome[t["map_type"]].append(t)

    for r, biome in enumerate(biome_order):
        cells = sorted(by_biome.get(biome, []), key=lambda x: x.get("_demo_idx", 0))
        for c in range(4):
            ax = axes[r, c]
            if c >= len(cells):
                ax.axis("off")
                continue
            traj = cells[c]
            ax.imshow(TILE_COLORS[traj["terrain"]], origin="upper", interpolation="nearest")
            pos = np.array(traj["positions"])
            ax.plot(pos[:, 1], pos[:, 0], "-", c="white", lw=1.4, alpha=0.85)
            ax.scatter(pos[:, 1], pos[:, 0], c=np.arange(len(pos)),
                       cmap="viridis", s=4, zorder=3, edgecolors="none")
            sr, sc_ = traj["spawn"]; tr, tc = traj["target"]
            ax.scatter([sc_], [sr], marker="o", s=80, fc="lime", ec="black", zorder=4)
            ax.scatter([tc], [tr], marker="*", s=160, fc="gold", ec="black", zorder=4)
            cs = traj.get("commit_step")
            if cs is not None and cs < len(pos):
                cr, cc = pos[cs]
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
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="either runs/<run_id>/checkpoints/step_<N> directly, or "
                        "runs/<run_id>/ (the latest step_* is picked automatically)")
    p.add_argument("--config", type=Path, default=None,
                   help="config.json from same run; defaults to the run dir's config.json")
    p.add_argument("--maps-dir", type=Path, default=Path("data/demo_maps"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Accept either a step dir or a run dir; resolve to the step dir + run dir.
    if (args.checkpoint / "checkpoints").is_dir():
        # user passed runs/<run_id>/ — find latest step_*
        steps = sorted(
            (args.checkpoint / "checkpoints").glob("step_*"),
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
        if not steps:
            sys.exit(f"no step_* under {args.checkpoint}/checkpoints/")
        run_dir = args.checkpoint
        args.checkpoint = steps[-1]
        print(f"auto-selected checkpoint: {args.checkpoint}")
    else:
        run_dir = args.checkpoint.parents[1]

    if args.config is None:
        args.config = run_dir / "config.json"
    if args.out_dir is None:
        args.out_dir = run_dir / "viz_demo_maps"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config.read_text())
    helpers = _load_viz_helpers()

    print(f"loading checkpoint from {args.checkpoint} ...")
    pd = helpers.load_frozen(args.checkpoint, cfg)

    map_files = sorted(args.maps_dir.glob("*.pkl"))
    if not map_files:
        sys.exit(f"no maps under {args.maps_dir}")

    records = []
    metadata = []  # parallel to records; carries biome + demo_idx + correct_object
    for mp in map_files:
        biome, idx = mp.stem.rsplit("_", 1)
        with mp.open("rb") as f:
            rec = pickle.load(f)
        records.append(rec)
        metadata.append({
            "biome": biome,
            "demo_idx": int(idx),
            "correct_object": OBJECT_NAMES[int(rec.correct_object)],
            "stem": mp.stem,
        })

    size = int(records[0].terrain.shape[0])
    if any(int(r.terrain.shape[0]) != size for r in records):
        sys.exit("demo maps have mismatched sizes; can't stack")

    arrays = _stack_records(records, size=size)
    env_params = EnvParams.from_map_arrays(
        **arrays, max_steps=cfg["max_steps"], view_size=cfg["view_size"],
    )
    env = CrafterInCognilandEnv(default_params=env_params)

    rng = jax.random.PRNGKey(args.seed)
    summary = []
    trajs = []
    for i, (rec, meta) in enumerate(zip(records, metadata)):
        rng, sub = jax.random.split(rng)
        traj = _rollout_fixed_map(
            env, env_params, pd, map_idx=i, rng=sub, max_steps=args.max_steps,
        )
        traj["terrain"] = np.asarray(rec.terrain)
        traj["spawn"] = tuple(int(x) for x in rec.spawn)
        traj["target"] = tuple(int(x) for x in rec.target)
        traj["map_type"] = meta["biome"]
        traj["correct_object"] = meta["correct_object"]
        traj["_demo_idx"] = meta["demo_idx"]
        trajs.append(traj)

        title = (
            f"{meta['biome']} #{meta['demo_idx']}  "
            f"R={traj['episode_return']:+.2f}  L={traj['length']}  "
            f"{'SUCCESS' if traj['reached'] else 'FAIL'}  "
            f"built: {traj['committed_object'] or 'none'}  "
            f"correct: {meta['correct_object']}"
        )
        _plot_single(
            traj, np.asarray(rec.terrain), traj["spawn"], traj["target"],
            args.out_dir / f"{meta['stem']}.png", title,
        )
        summary.append({
            "map": meta["stem"],
            "biome": meta["biome"],
            "reached": bool(traj["reached"]),
            "length": int(traj["length"]),
            "return": float(traj["episode_return"]),
            "committed_object": traj["committed_object"],
            "correct_object": meta["correct_object"],
        })
        print(f"  {meta['stem']}: "
              f"{'OK ' if traj['reached'] else 'FAIL'} "
              f"L={traj['length']:>3d} R={traj['episode_return']:+.2f} "
              f"built={str(traj['committed_object']):<7s} "
              f"correct={meta['correct_object']}")

    _composite_grid(
        trajs, args.out_dir / "grid.png",
        title=f"Dreamer ({args.checkpoint.parent.parent.name}) on demo maps  ·  "
              f"successes {sum(s['reached'] for s in summary)}/{len(summary)}",
    )
    (args.out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "num_maps": len(summary),
        "successes": sum(s["reached"] for s in summary),
        "results": summary,
    }, indent=2))
    print(f"\nwrote {args.out_dir}/grid.png + {len(summary)} per-map PNGs")


if __name__ == "__main__":
    main()
