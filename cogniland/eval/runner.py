"""EvalRunner — stateless evaluation pipeline for Cogniland.

Usage:
    runner = EvalRunner(eval_env, env_config, device)
    result = runner.run(policy_fn, n_episodes=50, mode="det", split="val", global_step=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from cogniland.env.constants import TERRAIN_COSTS, TERRAIN_VISIBILITY
from cogniland.env.pathfinding import batch_astar
from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv

_TERRAIN_NAMES = [
    "ocean", "deep_water", "water", "beach", "sandy",
    "grassland", "forest", "rocky", "mountains",
]


@dataclass
class EpisodeResult:
    """Per-episode outcome and behavioral metrics."""

    outcome: str                                     # "success" | "timeout" | "death"
    total_return: float
    episode_length: int
    metrics: dict[str, float]                        # all domain-specific scalars
    map_id: int | None = None
    trajectory: list[tuple[int, int]] | None = None


@dataclass
class EvalResult:
    """Collection of episode results from one evaluation run."""

    episodes: list[EpisodeResult]
    mode: str                                        # "det" | "stoch"
    split: str                                       # "val" | "test"
    global_step: int
    initial_targets: torch.Tensor | None = None     # [n_eps, 2] — for trajectory rendering


class EvalRunner:
    """Runs evaluation episodes and returns EvalResult. No WandB calls, no aggregation."""

    def __init__(self, eval_env: BatchedIslandEnv, env_config: EnvConfig, device: str):
        self.eval_env = eval_env
        self.env_config = env_config
        self.device = device

        # Precompute survival margin conversion factors from EnvConfig
        ec = env_config
        terrain_res_drains = [
            ec.sea_resource_costs[0], ec.sea_resource_costs[1], ec.sea_resource_costs[2],
            ec.land_resource_drain, ec.land_resource_drain, ec.land_resource_drain,
            0.0,  # forest — gains resources instead of draining
            ec.mountain_resource_costs[0], ec.mountain_resource_costs[1],
        ]
        mean_drain = sum(terrain_res_drains) / len(terrain_res_drains)
        mean_cost = TERRAIN_COSTS.mean().item()
        self._k_R = mean_drain / mean_cost if mean_cost > 0 else 1.0
        self._k_HP = self._k_R * ec.no_res_hp_multiplier

        # Precompute disk offsets for each visibility range (used by exploration metric)
        self._disk_offsets: dict[int, torch.Tensor] = {}
        for vis_r in TERRAIN_VISIBILITY.unique().tolist():
            vis_r = int(vis_r)
            offsets = [
                (dr, dc)
                for dr in range(-vis_r, vis_r + 1)
                for dc in range(-vis_r, vis_r + 1)
                if dr * dr + dc * dc <= vis_r * vis_r
            ]
            self._disk_offsets[vis_r] = torch.tensor(offsets, dtype=torch.long)

    def run(
        self,
        policy_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        n_episodes: int,
        mode: str,
        split: str,
        global_step: int,
        hp_danger_threshold: float = 30.0,
        max_trajectory_eps: int = 4,
    ) -> EvalResult:
        """Run n_episodes and return per-episode results with all behavioral metrics.

        Args:
            policy_fn: maps obs dict → action tensor [n_episodes].
            n_episodes: number of parallel episodes to run.
            mode: "det" or "stoch".
            split: "val" or "test".
            global_step: current training step (stored in result).
            hp_danger_threshold: HP below this counts as a danger step.
            max_trajectory_eps: only store full trajectories for first N episodes.
        """
        eval_env = self.eval_env
        env_config = self.env_config
        device = self.device
        H = W = env_config.size

        obs = eval_env.reset()
        initial_spawns = eval_env.state.position.clone()   # [n_eps, 2]
        initial_targets = eval_env.target_pos.clone()      # [n_eps, 2]

        # A* spawn → target (for path efficiency)
        per_env_maps = eval_env.env.world_maps[eval_env.env._env_map_idx]  # [n_eps, H, W]
        astar_costs = batch_astar(
            per_env_maps, TERRAIN_COSTS, initial_spawns, initial_targets,
        ).to(device)

        # Initial distance for survival margin denominator
        initial_dist = torch.norm(
            (initial_spawns - initial_targets).float(), dim=1
        ).clamp(min=1e-6)

        # --------------- Tracking tensors ---------------
        total_rewards = torch.zeros(n_episodes, device=device)
        total_moves = torch.zeros(n_episodes, device=device)
        reached = torch.zeros(n_episodes, dtype=torch.bool, device=device)
        alive = torch.ones(n_episodes, dtype=torch.bool, device=device)

        final_hp = torch.zeros(n_episodes, device=device)
        min_hp = torch.full((n_episodes,), float(env_config.init_hp), device=device)
        danger_steps = torch.zeros(n_episodes, device=device)

        resource_sum = torch.zeros(n_episodes, device=device)
        resource_count = torch.zeros(n_episodes, device=device)
        hp_sum = torch.zeros(n_episodes, device=device)
        hp_count = torch.zeros(n_episodes, device=device)
        max_resources = torch.zeros(n_episodes, device=device)

        terrain_visits = torch.zeros(n_episodes, 9, device=device)

        # Survival margin: track step-wise minimum
        survival_margin = torch.full((n_episodes,), float("inf"), device=device)

        # Exploration: bool map of observed cells
        observed = torch.zeros(n_episodes, H, W, dtype=torch.bool, device=device)

        # Final position / cost — captured before auto-reset clears them
        final_positions = initial_spawns.clone()
        final_cost = torch.zeros(n_episodes, device=device)
        is_finalized = torch.zeros(n_episodes, dtype=torch.bool, device=device)

        # Trajectories (stored only for first max_trajectory_eps episodes)
        trajectories: list[list[tuple[int, int]] | None] = [None] * n_episodes
        for i in range(min(n_episodes, max_trajectory_eps)):
            p = eval_env.state.position[i].cpu().tolist()
            trajectories[i] = [tuple(p)]

        # --------------- Episode loop ---------------
        for _move in range(env_config.max_steps):
            still_running = alive & ~reached
            if not still_running.any():
                break

            # Snapshot before step (done episodes are auto-reset, clearing state)
            pre_step_pos = eval_env.state.position.clone()
            pre_step_cost = eval_env.state.cost.clone()
            pre_move_terrain = eval_env.state.terrain_idx.clone()

            with torch.no_grad():
                action = policy_fn(obs)
            obs, reward, done, info = eval_env.step(action)

            current_hp = eval_env.state.hp
            current_resources = eval_env.state.resources
            dist_to_target = info["dist_to_target"]   # [n_eps]
            newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
            newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
            truncated = done & ~newly_reached & ~newly_dead
            just_finished = done & still_running

            # Reward / length accumulation
            total_rewards[still_running] += reward[still_running]
            total_moves[still_running] += 1

            # HP tracking
            min_hp[still_running] = torch.minimum(
                min_hp[still_running], current_hp[still_running]
            )
            danger_mask = still_running & (current_hp < hp_danger_threshold)
            danger_steps[danger_mask] += 1
            hp_count[still_running] += 1
            hp_sum[still_running] += current_hp[still_running]

            # Resource tracking
            resource_count[still_running] += 1
            resource_sum[still_running] += current_resources[still_running]
            max_resources[still_running] = torch.maximum(
                max_resources[still_running], current_resources[still_running]
            )

            # Terrain visits (pre-step terrain, same convention as old code)
            running_idx = torch.where(still_running)[0]
            terrain_visits[running_idx, pre_move_terrain[running_idx].long()] += 1

            # Survival margin
            c_remaining = astar_costs * (dist_to_target / initial_dist)
            c_hat_hp = c_remaining * self._k_HP
            c_hat_r = c_remaining * self._k_R
            eps = 1e-6
            sm_t = torch.minimum(
                current_hp / (c_hat_hp + eps),
                current_resources / (c_hat_r + eps),
            )
            survival_margin[still_running] = torch.minimum(
                survival_margin[still_running], sm_t[still_running]
            )

            # Exploration — mark observed cells for running episodes
            for i in running_idx.tolist():
                t_lev = int(pre_move_terrain[i].item())
                vis_r = int(TERRAIN_VISIBILITY[t_lev].item())
                pos = eval_env.state.position[i]
                r, c = pos[0].item(), pos[1].item()
                offsets = self._disk_offsets[vis_r].to(device)
                rows = (r + offsets[:, 0]).clamp(0, H - 1)
                cols = (c + offsets[:, 1]).clamp(0, W - 1)
                observed[i, rows, cols] = True

            # Capture final state for just-finished episodes before auto-reset clears them
            new_finalized = just_finished & ~is_finalized
            if new_finalized.any():
                for i in torch.where(new_finalized)[0].tolist():
                    final_cost[i] = pre_step_cost[i]  # 1-step approximation (negligible error)
                    if newly_reached[i]:
                        final_positions[i] = initial_targets[i]
                    else:
                        final_positions[i] = pre_step_pos[i]
                final_hp[new_finalized] = current_hp[new_finalized]

            is_finalized = is_finalized | new_finalized
            reached = reached | (newly_reached & still_running)
            alive = alive & ~newly_dead & ~truncated

            # Trajectory recording for first max_trajectory_eps episodes
            for i in torch.where(still_running)[0].tolist():
                if i >= max_trajectory_eps or trajectories[i] is None:
                    continue
                if newly_reached[i]:
                    tgt = initial_targets[i].cpu().tolist()
                    trajectories[i].append(tuple(tgt))
                elif not (newly_dead[i] or truncated[i]):
                    p = eval_env.state.position[i].cpu().tolist()
                    trajectories[i].append(tuple(p))

        # Handle still-running episodes at loop end
        still_running = alive & ~reached & ~is_finalized
        if still_running.any():
            final_hp[still_running] = eval_env.state.hp[still_running]
            final_positions[still_running] = eval_env.state.position[still_running]
            final_cost[still_running] = eval_env.state.cost[still_running]
            total_moves[still_running] = env_config.max_steps

        # --------------- Derived metrics ---------------
        danger_fraction = danger_steps / total_moves.clamp(min=1)
        final_resources = eval_env.state.resources
        resource_mean = resource_sum / resource_count.clamp(min=1)
        hp_mean = hp_sum / hp_count.clamp(min=1)

        # Directness: D = C_agent / (C_agent - C_astar_partial), range [1, 100]
        astar_to_final = batch_astar(
            per_env_maps, TERRAIN_COSTS, initial_spawns, final_positions,
        ).to(device)
        directness = torch.where(
            final_cost > astar_to_final + 1e-6,
            (final_cost / (final_cost - astar_to_final)).clamp(max=100.0),
            torch.full_like(final_cost, 100.0),
        )

        # Exploration: fraction of map cells ever observed
        exploration = observed.sum(dim=(1, 2)).float() / (H * W)

        # Terrain visit fractions per episode
        visit_totals = terrain_visits.sum(dim=1, keepdim=True).clamp(min=1)
        terrain_visit_frac = terrain_visits / visit_totals  # [N, 9]

        # Fix inf survival margins (e.g. initial dist = 0 edge case)
        survival_margin = torch.nan_to_num(
            survival_margin, nan=0.0, posinf=100.0, neginf=0.0
        )

        # --------------- Build EpisodeResult list ---------------
        map_ids = eval_env.env._env_map_idx.tolist()
        episodes: list[EpisodeResult] = []

        for i in range(n_episodes):
            if reached[i].item():
                outcome = "success"
            elif alive[i].item():
                outcome = "timeout"
            else:
                outcome = "death"

            metrics: dict[str, float] = {
                "min_hp": min_hp[i].item(),
                "final_hp": final_hp[i].item(),
                "mean_hp": hp_mean[i].item(),
                "danger_fraction": danger_fraction[i].item(),
                "final_resources": final_resources[i].item(),
                "mean_resources": resource_mean[i].item(),
                "max_resources": max_resources[i].item(),
                "directness": directness[i].item(),
                "survival_margin": survival_margin[i].item(),
                "exploration": exploration[i].item(),
            }
            for j, name in enumerate(_TERRAIN_NAMES):
                metrics[f"terrain_visit_{name}"] = terrain_visit_frac[i, j].item()

            episodes.append(EpisodeResult(
                outcome=outcome,
                total_return=total_rewards[i].item(),
                episode_length=int(total_moves[i].item()),
                metrics=metrics,
                map_id=int(map_ids[i]),
                trajectory=trajectories[i],
            ))

        return EvalResult(
            episodes=episodes,
            mode=mode,
            split=split,
            global_step=global_step,
            initial_targets=initial_targets,
        )
