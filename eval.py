#!/usr/bin/env python3
"""Full test-set evaluation from a local checkpoint.

Loads a checkpoint from disk and the training config from local Hydra configs
(``configs/main.yaml``), runs one episode per test map, dumps all scalar
metrics to JSON, and renders a 4x4 grid of trajectory plots as a PNG.

Usage:
    python eval.py artifacts/ppo_1m_uw4aeis5/ckpt_last.pt
    python eval.py artifacts/good_old/ckpt_best.pt models=ppo_1m

Extra positional args are passed through as Hydra config overrides.
"""

import json
import os
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _infer_model_group(ckpt_path: str) -> str | None:
    """Guess the Hydra ``models=`` group, first from checkpoint weights then path."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        top_keys = {k.split(".", 1)[0] for k in ckpt["model_state_dict"].keys()}
        if "drc_cells" in top_keys:
            return "drc"
        if "rnn" in top_keys:
            return "recurrent_ppo"
    except Exception:
        pass

    # Fall back to parent-dir stem match against configs/models/*.yaml
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    if not parent:
        return None
    stem = parent.rsplit("_", 1)[0]
    cfg_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "models", f"{stem}.yaml"
    )
    return stem if os.path.isfile(cfg_file) else None


def _load_local_config(overrides: list[str]):
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    return cfg


def _plot_trajectory_grid(episodes, world_maps, compiled, out_path: str, run_id: str,
                          initial_targets=None):
    """Render a grid PNG with one trajectory panel per map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(episodes)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), dpi=150)
    axes = np.array(axes).reshape(-1)

    thresholds = compiled.thresholds.cpu().numpy()
    color_lut = compiled.color_lut.float().cpu().numpy() / 255.0
    num_terrains = compiled.num_terrains

    for i, ep in enumerate(episodes):
        ax = axes[i]
        wm = world_maps[ep.map_id].cpu().numpy()
        terrain_map = np.searchsorted(thresholds, wm).clip(0, num_terrains - 1)
        rgb = color_lut[terrain_map]

        if ep.observed_mask is not None:
            fog = np.where(ep.observed_mask[:, :, None], 1.0, 0.70).astype(np.float32)
            rgb = rgb * fog

        ax.imshow(rgb, origin="upper", interpolation="nearest")

        traj = ep.trajectory or []
        if traj:
            pos = np.array(traj)
            visit_counts = np.zeros(wm.shape, dtype=np.float32)
            seg_counts = []
            for r, c in traj:
                visit_counts[r, c] += 1
                seg_counts.append(visit_counts[r, c])
            max_count = 10.0
            for k in range(len(pos) - 1):
                t = min(seg_counts[k + 1], max_count) / max_count
                ax.plot(
                    pos[k:k + 2, 1], pos[k:k + 2, 0],
                    color=(1.0 - t, 0.0, 0.0),
                    linewidth=0.8, alpha=0.9, solid_capstyle="round",
                )
            ax.scatter(pos[0, 1], pos[0, 0], c="lime", s=60, marker="o",
                       edgecolors="k", linewidth=1.0, zorder=5)
            ax.scatter(pos[-1, 1], pos[-1, 0], c="red", s=60, marker="X",
                       edgecolors="k", linewidth=1.0, zorder=5, alpha=0.6)

        if initial_targets is not None:
            tgt = initial_targets[i].cpu().numpy()
            ax.scatter(tgt[1], tgt[0], c="gold", s=100, marker="*",
                       edgecolors="k", linewidth=1.0, zorder=6)

        outcome = ep.outcome.upper()
        ax.set_title(
            f"map {ep.map_id} — {outcome} ({ep.episode_length} steps)",
            fontsize=9, fontweight="bold",
        )
        ax.set_axis_off()

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"Test trajectories — {run_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Error: You must provide a checkpoint path.")
        print("Example: python eval.py artifacts/ppo_1m_uw4aeis5/ckpt_last.pt")
        return

    ckpt_path = sys.argv[1]
    if not os.path.isfile(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}")
        return

    overrides = list(sys.argv[2:])
    if not any(o.startswith("models=") for o in overrides):
        inferred = _infer_model_group(ckpt_path)
        if inferred:
            overrides.append(f"models={inferred}")
            print(f"Inferred model group from checkpoint path: {inferred}")

    run_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    parent_tag = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path))) or "run"
    run_id = f"{parent_tag}__{run_tag}"

    cfg = _load_local_config(overrides)

    print("Building model architecture from historical config...")
    from cogniland.models import build_model
    from cogniland.env.types import EnvConfig

    env_config = EnvConfig.from_hydra(cfg)
    device_str = env_config.resolved_device()
    model = build_model(cfg, env_config, device_str)

    print("Loading checkpoint weights...")
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--- Evaluation Ready ---")
    print(f"Run ID: {run_id}")
    print(f"Model Architecture: {cfg.models.name}")
    print(f"Parameters: {param_count:,}")

    # --- Test-set dataset ---
    from cogniland.env.dataset import MapDataset
    from cogniland.env.wrappers import BatchedIslandEnv
    from cogniland.eval import CognilandSummarizer, EvalRunner

    _training_cfg = cfg.get("models", {}).get("training", {})
    dataset_cfg = _training_cfg.get("dataset", {})
    dataset = MapDataset.load_from_config(dataset_cfg) if dataset_cfg else None
    if dataset is None or dataset.n_test == 0:
        raise RuntimeError("No test dataset configured — cannot run test-set evaluation.")

    test_maps = dataset.test_maps
    n_maps = int(test_maps.shape[0])
    print(f"Running full test-set eval on {n_maps} maps (one episode per map)...")

    # Deterministic seed (matches training's eval seed offset)
    eval_cfg = cfg.logging.get("eval", {}) if hasattr(cfg, "logging") else {}
    _env_cfg = cfg.env
    base_seed = (_env_cfg.get("map_generation", {}).get("seed", None)
                 or _env_cfg.get("seed", 42))
    eval_seed = base_seed + eval_cfg.get("eval_seed_offset", 1000)

    eval_env = BatchedIslandEnv(env_config, num_envs=n_maps, world_maps=test_maps)

    # Force a 1:1 env→map assignment during reset. Islands.reset samples
    # _env_map_idx via torch.randint(0, N, (B,)); with N == B == n_maps we
    # substitute arange(N). Other randint calls in reset use shape (2,) so they
    # fall through to the original implementation.
    _orig_randint = torch.randint

    def _patched_randint(low, high, size=None, *args, **kwargs):
        if (
            size is not None
            and len(size) == 1
            and int(size[0]) == n_maps
            and int(high) - int(low) == n_maps
        ):
            return torch.arange(low, high, device=kwargs.get("device", None), dtype=torch.long)
        return _orig_randint(low, high, size, *args, **kwargs)

    torch.randint = _patched_randint
    try:
        eval_env.reset(seed=eval_seed)
    finally:
        torch.randint = _orig_randint

    assigned = eval_env.env._env_map_idx.tolist()
    if sorted(assigned) != list(range(n_maps)):
        raise RuntimeError(f"Failed to assign one episode per test map: got {assigned}")

    runner = EvalRunner(eval_env, env_config, str(device))

    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "init_hidden"):
        print("Detected recurrent model — carrying hidden state across steps.")
        h = [inner_model.init_hidden(n_maps, device)]

        def policy_fn(obs):
            act, h_new = inner_model.get_deterministic_action(obs, h[0])
            h[0] = h_new
            return act
    else:
        def policy_fn(obs):
            return model.get_deterministic_action(obs)

    result = runner.run(
        policy_fn=policy_fn,
        n_episodes=n_maps,
        mode="det",
        split="test",
        global_step=0,
        max_trajectory_eps=n_maps,
    )

    summarizer = CognilandSummarizer()
    metrics = summarizer.scalar_metrics(result)
    terrain_pcts = summarizer.terrain_pcts(result)

    print("\n" + "=" * 40)
    print("TEST-SET EVALUATION RESULTS")
    print("=" * 40)
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("-" * 40)
    print("Per-terrain visit fractions:")
    for name, pct in terrain_pcts.items():
        print(f"  {name}: {pct:.4f}")
    print("=" * 40)

    # --- Persist results ---
    out_dir = os.path.join("eval_outputs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    episodes_payload = [
        {
            "map_id": ep.map_id,
            "outcome": ep.outcome,
            "return": ep.total_return,
            "episode_length": ep.episode_length,
            **ep.metrics,
        }
        for ep in result.episodes
    ]
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model": cfg.models.name,
                "n_test_maps": n_maps,
                "scalar_metrics": metrics,
                "terrain_visit_pcts": terrain_pcts,
                "episodes": episodes_payload,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")

    png_path = os.path.join(out_dir, "test_trajectories.png")
    _plot_trajectory_grid(
        result.episodes, test_maps, eval_env.compiled, png_path, run_id,
        initial_targets=result.initial_targets,
    )
    print(f"Saved trajectory grid to {png_path}")


if __name__ == "__main__":
    main()
