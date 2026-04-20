"""Evaluation trajectory logger.

Runs the policy deterministically on a fixed set of eval maps (one per biome)
and logs a 2x2 grid of trajectories to W&B under ``eval/trajectories``. The
same maps and spawn/target pairs are reused across every eval set, so
trajectories can be compared over training.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import wandb

from cogniland.envs.env import CognilandEnv
from cogniland.envs.multitask_wrapper import MultiTaskEnvWrapper
from cogniland.shared import setup_logger

logger = setup_logger(__name__)


class TrajectoryLogger:
    """Render & log eval-time trajectories on one fixed map per biome."""

    TRAJECTORY_SEED = 1234

    def __init__(self, config: Any, agent: Any, wandb_run: Any):
        self.config = config
        self.agent = agent
        self.wandb_run = wandb_run

        val_path = config.env.val_maps
        raw = torch.load(val_path, map_location="cpu", weights_only=False)
        biomes = list(raw["biomes"])
        seeds = list(raw["seeds"])
        rgb = raw["rgb"]
        rgb = rgb.numpy() if hasattr(rgb, "numpy") else np.asarray(rgb)

        # Preferred (biome, seed) picks — if the seed is present in the val
        # dataset for that biome, use it; otherwise fall back to the first
        # occurrence of the biome.
        preferred_seed_by_biome: dict[str, int] = {
            "balanced": 258,
            "archipelago": 262,
        }

        first_by_biome: dict[str, int] = {}
        for i, b in enumerate(biomes):
            first_by_biome.setdefault(b, i)
        self.biome_labels = list(first_by_biome.keys())

        def _pick_index(biome: str) -> int:
            want_seed = preferred_seed_by_biome.get(biome)
            if want_seed is not None:
                for i, (b, s) in enumerate(zip(biomes, seeds)):
                    if b == biome and int(s) == int(want_seed):
                        return i
            return first_by_biome[biome]

        self.map_indices = np.array(
            [_pick_index(b) for b in self.biome_labels], dtype=np.int32
        )
        self.map_seeds = [seeds[i] for i in self.map_indices]
        self.n = len(self.map_indices)
        self.rgbs = [rgb[int(mi)].copy() for mi in self.map_indices]

        # Build dedicated env with one slot per biome map, auto-reset disabled
        # so pos_r/pos_c retains the final position after termination.
        # Use a throwaway config with num_parallel_envs = n so the base env
        # allocates exactly n slots.
        self.env = CognilandEnv(config, val_path, num_envs=self.n)
        self.env._auto_reset_enabled = False
        self.env._min_manhattan = 120

        num_tasks = getattr(config, "num_tasks", 1)
        emb_dim = getattr(config, "task_embedding_dim", 7)
        self.wrapper = MultiTaskEnvWrapper(
            self.env, config, num_tasks=int(num_tasks), task_embedding_dim=int(emb_dim)
        )
        self.task_embedding_dim = int(emb_dim)
        self.max_steps = int(config.env.max_steps)

    # ------------------------------------------------------------------ #
    def log(self, agent_state: Any, rng: Any, global_train_frames: int,
            task_id: int = 0) -> None:
        """Roll out deterministically on the fixed maps and log the grid."""
        try:
            self._do_log(agent_state, rng, global_train_frames, task_id)
        except Exception as e:  # never break training on viz failure
            logger.warning("Trajectory logging failed: %s", e)

    def _do_log(self, agent_state, rng, global_train_frames, task_id):
        # Copy state so writes to ``_carry_cache`` don't alias the training
        # state's cache (which has a different batch size).
        state = copy.copy(agent_state)

        # Prime the PPO-style task embedding cache on the copy. The PPO-RNN
        # ``select_action`` reads ``state._task_emb_cache`` rather than the
        # observation's embedding.
        task_emb = jnp.asarray(
            np.eye(self.task_embedding_dim, dtype=np.float32)[
                np.full(self.n, task_id, dtype=np.int32)
            ]
        )
        object.__setattr__(state, "_task_emb_cache", task_emb)
        # Reset carry cache so it matches ``n`` envs, not the training batch.
        # ``select_action`` will zero it via is_first on the first call.
        if hasattr(state, "_carry_cache"):
            object.__delattr__(state, "_carry_cache")

        self.wrapper.set_tasks(np.full(self.n, task_id, dtype=np.int32))
        obs = self.wrapper.reset(
            seed=self.TRAJECTORY_SEED, map_indices=self.map_indices,
        )

        spawns = [(int(self.env.spawn_r[i]), int(self.env.spawn_c[i]))
                  for i in range(self.n)]
        yes_targets = [(int(self.env.yes_r[i]), int(self.env.yes_c[i]))
                       for i in range(self.n)]
        no_targets = [(int(self.env.no_r[i]), int(self.env.no_c[i]))
                      for i in range(self.n)]

        trajectories: list[list[tuple[int, int]]] = [
            [spawns[i]] for i in range(self.n)
        ]
        done_mask = np.zeros(self.n, dtype=bool)
        is_first = np.ones(self.n, dtype=bool)

        for step in range(self.max_steps):
            if done_mask.all():
                break
            actions, state = self.agent.select_action(
                state, obs, rng, is_first=is_first,
                prev_action=None, training=False,
            )
            actions_np = np.asarray(actions)
            prev_done = done_mask.copy()
            obs, _, step_dones, _ = self.wrapper.step(actions_np)

            # Record post-step positions for envs that were alive before the
            # step. Auto-reset is disabled, so pos_r/pos_c reflects the final
            # episode position for envs that died on this step.
            for i in range(self.n):
                if not prev_done[i]:
                    trajectories[i].append(
                        (int(self.env.pos_r[i]), int(self.env.pos_c[i]))
                    )
            done_mask = prev_done | np.asarray(step_dones, dtype=bool)
            is_first = np.zeros(self.n, dtype=bool)

        reached = [
            (
                (self.env.pos_r[i] == self.env.yes_r[i]
                 and self.env.pos_c[i] == self.env.yes_c[i])
                or
                (self.env.pos_r[i] == self.env.no_r[i]
                 and self.env.pos_c[i] == self.env.no_c[i])
            )
            for i in range(self.n)
        ]
        alive_end = [bool(self.env.hp[i] > 0) for i in range(self.n)]

        fig = self._render(
            trajectories, spawns, yes_targets, no_targets,
            reached, alive_end, done_mask,
        )
        self.wandb_run.log({
            "eval/trajectories": wandb.Image(fig),
            "train_frames": int(global_train_frames),
        })
        plt.close(fig)

    # ------------------------------------------------------------------ #
    def _render(self, trajectories, spawns, yes_targets, no_targets,
                reached, alive_end, done_mask) -> plt.Figure:
        rows = int(np.ceil(np.sqrt(self.n)))
        cols = int(np.ceil(self.n / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                                  squeeze=False)

        for i in range(rows * cols):
            ax = axes[i // cols][i % cols]
            if i >= self.n:
                ax.set_axis_off()
                continue

            ax.imshow(self.rgbs[i], interpolation="nearest")

            traj = np.asarray(trajectories[i], dtype=np.int32)
            if traj.shape[0] >= 2:
                # Per-cell visit count: colour each segment red→black by
                # revisit count so repeatedly-trodden cells darken.
                H, W = self.rgbs[i].shape[:2]
                visit_counts = np.zeros((H, W), dtype=np.float32)
                seg_counts = np.empty(traj.shape[0], dtype=np.float32)
                for k, (r, c) in enumerate(traj):
                    visit_counts[r, c] += 1
                    seg_counts[k] = visit_counts[r, c]
                max_count = 10.0  # 1 visit = red, >=10 = black
                for k in range(traj.shape[0] - 1):
                    t = min(seg_counts[k + 1], max_count) / max_count
                    ax.plot(
                        traj[k:k + 2, 1], traj[k:k + 2, 0],
                        color=(1.0 - t, 0.0, 0.0),
                        linewidth=0.8, alpha=0.9,
                        solid_capstyle="round", zorder=3,
                    )

            # Spawn (green circle), end (red X, faded), YES (gold ★), NO (silver ★).
            ax.scatter(spawns[i][1], spawns[i][0],
                       c="#2ecc71", s=60, marker="o",
                       edgecolors="k", linewidth=1.0, zorder=5)
            end_r, end_c = trajectories[i][-1]
            ax.scatter(end_c, end_r,
                       c="red", s=60, marker="X",
                       edgecolors="k", linewidth=1.0, alpha=0.5, zorder=5)
            ax.scatter(yes_targets[i][1], yes_targets[i][0],
                       c="gold", s=80, marker="*",
                       edgecolors="k", linewidth=1.0, zorder=5)
            ax.scatter(no_targets[i][1], no_targets[i][0],
                       c="silver", s=80, marker="*",
                       edgecolors="k", linewidth=1.0, zorder=5)

            if reached[i]:
                status = "reached"
                status_color = "#2ecc71"
            elif not alive_end[i]:
                status = "died"
                status_color = "#e74c3c"
            elif not done_mask[i]:
                status = "timeout"
                status_color = "#f39c12"
            else:
                status = "done"
                status_color = "#cccccc"

            title = (
                f"{self.biome_labels[i]}  seed={self.map_seeds[i]}  "
                f"[{status}]  steps={traj.shape[0] - 1}"
            )
            ax.set_title(title, fontsize=10, color=status_color)
            ax.set_axis_off()

        fig.tight_layout()
        return fig
