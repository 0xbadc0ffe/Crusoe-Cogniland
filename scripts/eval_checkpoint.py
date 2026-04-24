"""Evaluate a saved checkpoint on a given split (default: test).

Loads the ``last/`` orbax checkpoint from ``<results_dir>/checkpoints/<env>/``
and runs ``agent.evaluate`` for ``--num-frames`` frames per task on the
chosen map split. Prints a per-task + aggregate success table.

Usage:
    python scripts/eval_checkpoint.py \
        --results-dir results/3m6om1zc \
        --env-config  configs/env/cogniland.yaml \
        --agent-config configs/agent/ppo_rnn.yaml \
        --split test --num-frames 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cogniland.config import setup_environment
setup_environment()

import numpy as np
import jax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from tabulate import tabulate

from cogniland.agents import load_agent
from cogniland.envs.registry import make_env


def _resolve_last_dir(results_dir: Path, env_id: str) -> Path:
    """Return the orbax checkpoint dir to restore from."""
    env_slug = env_id.replace("/", "-")
    ckpt_dir = results_dir / "checkpoints" / env_slug
    for candidate in [ckpt_dir / "last", ckpt_dir / "best"]:
        if candidate.exists():
            return candidate
    # Fall back to latest step_* directory if present.
    step_dirs = sorted(ckpt_dir.glob("step_*"))
    if step_dirs:
        return step_dirs[-1]
    raise FileNotFoundError(f"No orbax checkpoint under {ckpt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Results dir of the training run (contains checkpoints/)")
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--agent-config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-frames", type=int, default=100_000,
                    help="Eval frames per task")
    ap.add_argument("--num-parallel-envs-eval", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--biome-filter", default=None,
                    help='Comma-separated biomes (e.g. "balanced,highland")')
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task ids (default: config.tasks)")
    args = ap.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load(args.env_config),
        OmegaConf.load(args.agent_config),
    )
    cfg.offline = True
    cfg.env.num_parallel_envs_eval = args.num_parallel_envs_eval
    if args.max_steps is not None:
        cfg.env.max_steps = args.max_steps
    if args.biome_filter is not None:
        cfg.env.biome_filter = args.biome_filter.split(",")

    # Redirect val_maps to the requested split so make_env(..., train=False)
    # reads from the right .pt file.
    split_path = f"data/maps/{args.split}.pt"
    cfg.env.val_maps = split_path
    if args.tasks is not None:
        cfg.tasks = [int(x) for x in args.tasks.split(",")]

    print(f"Loading split: {split_path}   biome={cfg.env.get('biome_filter')}")
    print(f"Eval envs:     {cfg.env.num_parallel_envs_eval}")
    print(f"Tasks:         {list(cfg.tasks)}")
    print(f"Frames/task:   {args.num_frames:,}")
    print()

    env = make_env(cfg.env_id, cfg, train=False)
    agent = load_agent(cfg)

    # Init empty agent state (shapes only), then overwrite params from ckpt.
    rng = jax.random.PRNGKey(0)
    state = agent.init(rng)

    ckpt_dir = _resolve_last_dir(Path(args.results_dir), cfg.env_id)
    print(f"Checkpoint:    {ckpt_dir}")
    checkpointer = ocp.StandardCheckpointer()
    ckpt_tree = checkpointer.restore(ckpt_dir.resolve())
    # The checkpoint stores raw pytrees; the agent owns the conversion
    # back to AgentState (rebuilds the optimizer and wraps in TrainState).
    if not isinstance(ckpt_tree, dict) or "train_state" not in ckpt_tree:
        ckpt_tree = {"train_state": ckpt_tree}
    state = agent.state_from_checkpoint(ckpt_tree, state.runtime)
    print("Weights loaded.\n")

    rows = []
    for task_id in cfg.tasks:
        task_ids = np.full(env.num_envs, int(task_id), dtype=np.int32)
        env.set_tasks(task_ids)

        metrics = agent.evaluate(
            state, env, rng,
            num_eval_frames=args.num_frames,
            progress_bar=None,
            task_ids=task_ids,
        )
        ep = metrics.get("episode_info", {})
        done = np.asarray(ep.get("returned_episode", [])).reshape(-1).astype(bool)
        returns = np.asarray(ep.get("returned_episode_returns", [])).reshape(-1)
        lengths = np.asarray(ep.get("returned_episode_lengths", [])).reshape(-1)
        succ = np.asarray(ep.get("task_success", [])).reshape(-1)
        n = int(done.sum())
        mean_r = float(returns[done].mean()) if n else 0.0
        mean_l = float(lengths[done].mean()) if n else 0.0
        mean_s = float(succ[done].mean()) if n else 0.0
        rows.append([f"task_{task_id}", f"{mean_r:+.3f}", f"{mean_s:.3f}",
                     f"{mean_l:.1f}", n])

    # Aggregate over all tasks.
    if rows:
        r_avg = float(np.mean([float(r[1]) for r in rows]))
        s_avg = float(np.mean([float(r[2]) for r in rows]))
        l_avg = float(np.mean([float(r[3]) for r in rows]))
        rows.append(["AGGREGATE", f"{r_avg:+.3f}", f"{s_avg:.3f}",
                     f"{l_avg:.1f}", ""])

    print(tabulate(rows, headers=["task", "reward", "success", "length", "episodes"],
                   tablefmt="grid"))


if __name__ == "__main__":
    main()
