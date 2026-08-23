#!/usr/bin/env python
"""Evaluate trained R2-Dreamer MemoryEnv checkpoints, per cue type.

Runs in the dedicated ``r2dreamer`` conda env with the repo's ``src`` on
PYTHONPATH so both ``cogniland.memory_env`` and the r2dreamer package import::

    PYTHONPATH=src conda run -n r2dreamer python scripts/memory_env/eval_r2dreamer.py \
        --ckpt-2cue r2dreamer_model/runs/memory_2cue/latest.pt \
        --ckpt-3cue r2dreamer_model/runs/memory_3cue/latest.pt \
        --ckpt-4cue r2dreamer_model/runs/memory_4cue/latest.pt \
        --device cuda:0

All three models are evaluated on the SAME held-out 4-cue test set (via
``scripts/memory_env/datasets.eval_per_cue``); generalisation / entanglement
shows up as low reward on cues a model never trained on. The output is a
grouped bar plot of average reward per cue type for the three models:
``outputs/report/memoryenv_reward_per_cue.png``.

The act_fn resizes the env's native (56,56,3) obs up to (64,64,3) before
feeding the policy (datasets.py drives the raw MemoryEnv, which is 56x56),
maintaining the RSSM recurrent state across the episode.

Use ``--dry-run`` to render the plot from oracle/dummy numbers (no checkpoints
needed) and confirm the plotting path works.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Make both packages importable: repo src (cogniland) + r2dreamer package dir.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "external" / "r2dreamer"))

from cogniland.memory_env import MemoryEnvConfig, oracle_action  # noqa: E402

# import path mirrors scripts/memory_env/datasets.py
sys.path.insert(0, str(_REPO / "scripts" / "memory_env"))
from datasets import ALL_CUES, eval_per_cue, TRAIN_CUES  # noqa: E402

OUT_PNG = _REPO / "outputs" / "report" / "memoryenv_reward_per_cue.png"
MODELS = ["2cue", "3cue", "4cue"]


# --------------------------------------------------------------------------- #
#  obs resize (56 -> 64, nearest neighbour; identical to envs/memory.py)
# --------------------------------------------------------------------------- #
def _resize_nn(img, size=(64, 64)):
    h, w = img.shape[:2]
    h2, w2 = size
    if (h, w) == (h2, w2):
        return img
    ys = (np.arange(h2) * h // h2).clip(0, h - 1)
    xs = (np.arange(w2) * w // w2).clip(0, w - 1)
    return img[ys][:, xs]


# --------------------------------------------------------------------------- #
#  build a greedy act_fn(obs, info) -> action from a trained r2dreamer ckpt
# --------------------------------------------------------------------------- #
def build_act_fn(ckpt_path, task, device="cuda:0", model_size="size25M"):
    """Reconstruct the Dreamer agent, load the checkpoint, and return a closure
    ``act_fn(obs, info) -> int`` (greedy) that carries RSSM state per episode.

    The agent is rebuilt by composing the same Hydra config used at train time
    (``env=memory``, ``model=<size>``, ``env.task=memory_<task>``)."""
    import torch
    from hydra import compose, initialize_config_dir
    from tensordict import TensorDict

    from dreamer import Dreamer
    from envs.memory import Memory

    cfg_dir = str(_REPO / "external" / "r2dreamer" / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=memory",
                f"model={model_size}",
                f"env.task=memory_{task}",
                f"device={device}",
                "model.compile=False",  # no compile needed for inference
            ],
        )

    # obs/act spaces from the wrapper (same OneHotAction contract as training).
    probe = Memory(task, size=tuple(config.env.size), seed=0)
    import gymnasium as gym

    obs_space = probe.observation_space

    class _OneHotSpace(gym.spaces.Box):
        discrete = True

    n = probe.action_space.n
    act_space = _OneHotSpace(low=0, high=1, shape=(n,), dtype=np.float32)

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)["agent_state_dict"]
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    dev = torch.device(device)
    holder = {"state": None}

    @torch.no_grad()
    def act_fn(obs, info):
        # obs is the raw (56,56,3) MemoryEnv frame -> resize to (64,64,3).
        image = _resize_nn(np.asarray(obs, dtype=np.uint8), tuple(config.env.size))
        is_first = info.get("global_step", 0) == 0
        if is_first or holder["state"] is None:
            holder["state"] = agent.get_initial_state(1)
        # agent.act expects obs as (B, *) with NO time dim: the encoder preserves
        # leading dims, so a (1,1,H,W,3) image yields embed (1,1,E) which then
        # mismatches deter (1,D) in rssm.obs_step's cat. Use a single batch dim.
        trans = TensorDict(
            {
                "image": torch.as_tensor(image, device=dev)[None],  # (1,H,W,3)
                "is_first": torch.tensor([is_first], device=dev),    # (1,)
            },
            batch_size=(1,),
        )
        action, holder["state"] = agent.act(trans, holder["state"], eval=True)
        # one-hot (1,1,A) or (1,A) -> index
        return int(torch.argmax(action.reshape(-1)).item())

    return act_fn


# --------------------------------------------------------------------------- #
#  plotting
# --------------------------------------------------------------------------- #
def plot_reward_per_cue(results, out_png=OUT_PNG):
    """results: {model_name: {cue: avg_reward}} -> grouped bar plot.

    The train/test split is made explicit ON the figure: each model's legend
    entry lists its TRAINING cue set, and every bar whose cue was in that model's
    training set is marked with a ★ (in-distribution). Unmarked bars are held-out
    cues — all models are evaluated on the same held-out 4-cue test set.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cues = list(ALL_CUES)
    x = np.arange(len(cues))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"2cue": "#d62728", "3cue": "#ff7f0e", "4cue": "#2ca02c"}
    ymax = 0.0
    for i, model in enumerate(MODELS):
        train = list(TRAIN_CUES[model])
        vals = [results[model].get(c, np.nan) for c in cues]
        ymax = max(ymax, max([v for v in vals if v == v] or [0.0]))
        xpos = x + (i - 1) * width
        # in-distribution bars get a solid black edge; held-out bars are plain.
        ax.bar(xpos, vals, width, color=colors.get(model),
               edgecolor=["black" if c in train else "none" for c in cues],
               linewidth=[1.6 if c in train else 0.0 for c in cues],
               label=f"{model} model  (train: {', '.join(train)})")
        # ★ above every bar whose cue is in this model's training set.
        for xi, c, v in zip(xpos, cues, vals):
            if c in train and v == v:
                ax.text(xi, v, "★", ha="center", va="bottom", fontsize=9,
                        color=colors.get(model))
    ax.set_xticks(x)
    ax.set_xticklabels(cues, rotation=15)
    ax.set_ylabel("average reward")
    ax.set_xlabel("cue type (all models evaluated on the same held-out 4-cue test set)")
    ax.set_title("R2-Dreamer MemoryEnv: avg reward per cue type")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylim(top=ymax * 1.15 + 0.05)
    ax.legend(title="★ = cue in that model's training set (in-distribution)",
              fontsize=9, title_fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-2cue", type=str, default=None)
    ap.add_argument("--ckpt-3cue", type=str, default=None)
    ap.add_argument("--ckpt-4cue", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model-size", type=str, default="size25M")
    ap.add_argument("--n-per-cue", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true",
                    help="render the plot from oracle/dummy numbers; no ckpts needed")
    args = ap.parse_args()

    if args.dry_run:
        # Oracle on each cue should score ~1.0; fabricate a plausible
        # entanglement pattern (models do worse on unseen cues) to validate
        # the plotting path end to end.
        results = {
            "2cue": {"green_up": 0.95, "blue_up": 0.10, "green_down": 0.12, "blue_down": 0.93},
            "3cue": {"green_up": 0.94, "blue_up": 0.15, "green_down": 0.90, "blue_down": 0.92},
            "4cue": {"green_up": 0.93, "blue_up": 0.91, "green_down": 0.92, "blue_down": 0.94},
        }
        # also exercise the real eval harness once with the scripted oracle to
        # confirm act_fn signature + datasets plumbing work (cheap).
        rep = eval_per_cue(lambda obs, info: oracle_action(info), n_per_cue=4)
        print("oracle sanity (4 eps/cue):",
              {c: round(rep[c]["avg_reward"], 3) for c in ALL_CUES})
        plot_reward_per_cue(results)
        return

    ckpts = {"2cue": args.ckpt_2cue, "3cue": args.ckpt_3cue, "4cue": args.ckpt_4cue}
    results = {}
    for model, ckpt in ckpts.items():
        if ckpt is None:
            raise SystemExit(f"missing --ckpt-{model} (or use --dry-run)")
        print(f"== evaluating {model} model: {ckpt}")
        act_fn = build_act_fn(ckpt, model, device=args.device, model_size=args.model_size)
        rep = eval_per_cue(act_fn, n_per_cue=args.n_per_cue)
        results[model] = {c: rep[c]["avg_reward"] for c in ALL_CUES}
        print("  per-cue avg_reward:", {c: round(v, 3) for c, v in results[model].items()})
        print("  overall:", round(rep["overall"]["avg_reward"], 3))

    plot_reward_per_cue(results)


if __name__ == "__main__":
    main()
