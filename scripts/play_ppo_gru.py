#!/usr/bin/env python3
"""Headless rollout of a `train_ppo_gru.py` checkpoint with trajectory plots.

Loads a ``.pt`` checkpoint saved by ``scripts/train_ppo_gru.py``, rolls out
N episodes of ``CognilandNavEnv`` using the trained policy, and writes one
PNG per episode showing the terrain + the agent's path. The step where the
agent commits to a build (raft or harness) is highlighted with a red 'X'
and annotated with the chosen object plus the policy's belief scalar at
that step (the aux map-recognition probe).

Usage
-----
    python scripts/play_ppo_gru.py \\
        --checkpoint checkpoints/<run_name>/final.pt \\
        --num-episodes 4 \\
        --out-dir rollouts/ppo_gru_main

Add ``--greedy`` to take argmax over moves (otherwise moves are sampled,
matching training). The build is a discrete move (build_raft/build_harness);
the belief scalar is deterministic either way.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cogniland.nav import CognilandNavEnv  # noqa: E402
from cogniland.nav.tiles import TILE_COLORS  # noqa: E402

# Re-use PPOGRUPolicy from the trainer without copying the class.
_TRAIN_PATH = Path(__file__).parent / "train_ppo_gru.py"
_spec = importlib.util.spec_from_file_location("train_ppo_gru", str(_TRAIN_PATH))
_tp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tp)
PPOGRUPolicy = _tp.PPOGRUPolicy


def _build_env(ckpt_args: dict, seed: int) -> CognilandNavEnv:
    return CognilandNavEnv(
        size=ckpt_args.get("env_size", 64),
        map_type=ckpt_args.get("map_type", "random"),
        view_size=ckpt_args.get("view_size", 21),
        tile_px=ckpt_args.get("tile_px", 8),
        obs_mode=ckpt_args.get("obs_mode", "symbolic"),
        max_steps=ckpt_args.get("max_steps", 1000),
        seed=seed,
    )


def _build_policy(env: CognilandNavEnv, ckpt_args: dict, device: torch.device) -> PPOGRUPolicy:
    policy = PPOGRUPolicy(
        env.observation_space,
        num_move_actions=env.action_space.n,
        gru_hidden=ckpt_args.get("gru_hidden", 128),
        embed_dim=ckpt_args.get("embed_dim", 256),
    ).to(device)
    policy.eval()
    return policy


def _to_tensor_obs(obs: dict, device: torch.device) -> dict:
    return {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}


@torch.no_grad()
def _select_action(policy: PPOGRUPolicy, obs_t: dict, hidden: torch.Tensor,
                   done: torch.Tensor, greedy: bool):
    # Policy heads return (logits, belief, value). ``belief`` is the aux
    # map-recognition probe (deterministic tanh) — returned only for viz,
    # never used to drive the env (the build is a discrete move).
    if not greedy:
        action, belief, _, _, _, h_new = policy.get_action_and_value(obs_t, hidden, done)
        return int(action.item()), float(belief.squeeze().item()), h_new

    obs_seq = {k: v.unsqueeze(0) for k, v in obs_t.items()}
    gru_out, h_new = policy._gru_forward(obs_seq, done.unsqueeze(0), hidden)
    x = gru_out.squeeze(0)
    logits, belief, _ = policy._heads(x)
    return int(logits.argmax(-1).item()), float(belief.squeeze().item()), h_new


def _rollout(policy: PPOGRUPolicy, env: CognilandNavEnv, device: torch.device,
             greedy: bool) -> dict:
    obs, info = env.reset()
    positions = [tuple(info["position"])]
    move_actions: list[int] = []
    beliefs: list[float] = []
    commit_step: int | None = None
    committed_object: str | None = None
    correct_object = info["correct_object"]

    hidden = torch.zeros(1, 1, policy.gru_hidden, device=device)
    done_t = torch.zeros(1, device=device)
    ep_return = 0.0
    step = 0
    reached = False
    while True:
        obs_t = _to_tensor_obs(obs, device)
        move, belief, hidden = _select_action(policy, obs_t, hidden, done_t, greedy)

        obs, r, term, trunc, info = env.step(int(move))
        ep_return += float(r)
        step += 1

        positions.append(tuple(info["position"]))
        move_actions.append(move)
        beliefs.append(belief)

        if commit_step is None and info["skill_active"] == 1:
            commit_step = step
            committed_object = info["active_object"]

        done_t = torch.tensor([1.0 if (term or trunc) else 0.0], device=device)
        if term or trunc:
            reached = bool(info.get("reached_target", False))
            break

    return {
        "terrain": env._record.terrain.copy(),
        "spawn": positions[0],
        "target": tuple(info["target"]),
        "positions": positions,
        "move_actions": move_actions,
        "beliefs": beliefs,
        "commit_step": commit_step,
        "committed_object": committed_object,
        "correct_object": correct_object,
        "map_type": info["map_type"],
        "episode_return": ep_return,
        "length": step,
        "reached": reached,
    }


def _plot_trajectory(traj: dict, out_path: Path, title: str) -> None:
    terrain = traj["terrain"]
    rgb = TILE_COLORS[terrain]  # (H, W, 3) uint8

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest", origin="upper")

    pos = np.array(traj["positions"])  # (T+1, 2) in (row, col)
    rows, cols = pos[:, 0], pos[:, 1]
    ax.plot(cols, rows, color="white", linewidth=1.6, alpha=0.85, zorder=2)
    ax.scatter(cols, rows, s=6, c=np.arange(len(rows)), cmap="viridis",
               zorder=3, edgecolors="none")

    sr, sc = traj["spawn"]
    tr, tc = traj["target"]
    ax.scatter([sc], [sr], marker="o", s=140, facecolor="lime",
               edgecolor="black", linewidth=1.2, zorder=4, label="spawn")
    ax.scatter([tc], [tr], marker="*", s=260, facecolor="gold",
               edgecolor="black", linewidth=1.2, zorder=4, label="target")

    handles = [
        mpatches.Patch(color="lime", label="spawn"),
        mpatches.Patch(color="gold", label="target"),
    ]
    cs = traj["commit_step"]
    if cs is not None:
        cr, cc = traj["positions"][cs]
        belief = traj["beliefs"][cs - 1]
        ax.scatter([cc], [cr], marker="X", s=240, facecolor="red",
                   edgecolor="black", linewidth=1.4, zorder=5)
        ax.annotate(
            f"commit @ step {cs}\nbuilt: {traj['committed_object']}  (belief={belief:+.2f})",
            xy=(cc, cr),
            xytext=(8, -12), textcoords="offset points",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="red", alpha=0.75),
            arrowprops=dict(arrowstyle="-", color="red", lw=1.0),
        )
        handles.append(mpatches.Patch(color="red", label=f"commit ({traj['committed_object']})"))
    else:
        handles.append(mpatches.Patch(color="gray", label="no commit"))

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("rollouts"))
    ap.add_argument("--greedy", action="store_true",
                    help="argmax move (deterministic); belief is always deterministic")
    ap.add_argument("--map-type", default=None,
                    help="override the training map_type (lake/rocky/balanced/random)")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    if args.map_type is not None:
        ckpt_args["map_type"] = args.map_type

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded {args.checkpoint}")
    print(f"  global_step={ckpt.get('global_step', '?')} iteration={ckpt.get('iteration', '?')}")
    print(f"  env: size={ckpt_args.get('env_size')} map_type={ckpt_args.get('map_type')} "
          f"view_size={ckpt_args.get('view_size')} obs_mode={ckpt_args.get('obs_mode')}")
    print(f"  greedy={args.greedy}  episodes={args.num_episodes}  out={args.out_dir}")

    # Build env once just to instantiate the policy with the right obs space.
    base_env = _build_env(ckpt_args, seed=args.seed)
    policy = _build_policy(base_env, ckpt_args, device)
    policy.load_state_dict(ckpt["policy"])
    base_env.close()

    successes = 0
    returns = []
    for ep in range(args.num_episodes):
        env = _build_env(ckpt_args, seed=args.seed + ep)
        traj = _rollout(policy, env, device, args.greedy)
        env.close()
        returns.append(traj["episode_return"])
        successes += int(traj["reached"])

        cs = traj["commit_step"]
        commit_str = (f"step {cs} → {traj['committed_object']}"
                      if cs is not None else "never")
        print(
            f"ep {ep:02d}: map={traj['map_type']:>8s} "
            f"return={traj['episode_return']:+7.3f} length={traj['length']:4d} "
            f"reached={traj['reached']!s:5s}  commit={commit_str:>22s}  "
            f"correct={traj['correct_object']}"
        )

        title = (
            f"ep {ep}  map={traj['map_type']}  "
            f"R={traj['episode_return']:+.2f}  len={traj['length']}  "
            f"{'SUCCESS' if traj['reached'] else 'FAIL'}  "
            f"correct={traj['correct_object']}"
        )
        out_png = args.out_dir / f"ep{ep:02d}.png"
        _plot_trajectory(traj, out_png, title)
        print(f"  wrote {out_png}")

    mean_r = float(np.mean(returns)) if returns else float("nan")
    print(f"\nsummary: success {successes}/{args.num_episodes}  mean_return {mean_r:+.3f}")


if __name__ == "__main__":
    main()
