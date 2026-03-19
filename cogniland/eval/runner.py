"""EvalRunner — stateless evaluation pipeline for Cogniland.

Usage:
    runner = EvalRunner(eval_env, env_config, device)
    result = runner.run(policy_fn, n_episodes=50, mode="det", split="val", global_step=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from cogniland.env.constants import TERRAIN_COSTS
from cogniland.env.pathfinding import batch_dijkstra_from_sources
from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.eval.metrics import (
    compute_danger_fraction,
    compute_directness,
    compute_exploration,
    compute_risk_exposure,
    compute_terrain_visit_fractions,
    read_dijkstra_to_final,
)

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
    observed_mask: "np.ndarray | None" = None        # [H, W] bool — cells seen during episode


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

        # Per-terrain resource drain lookup (for risk exposure computation)
        ec = env_config
        self._terrain_res_drains = torch.tensor([
            ec.sea_resource_costs[0], ec.sea_resource_costs[1], ec.sea_resource_costs[2],
            ec.land_resource_drain, ec.land_resource_drain, ec.land_resource_drain,
            0.0,  # forest gains resources, not drains
            ec.mountain_resource_costs[0], ec.mountain_resource_costs[1],
        ])

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

        terrain_res_drains = self._terrain_res_drains.to(device)

        obs = eval_env.reset()
        initial_spawns = eval_env.state.position.clone()   # [n_eps, 2]
        initial_targets = eval_env.target_pos.clone()      # [n_eps, 2]

        # Run Dijkstra from each spawn — one call per episode, returns full distance map.
        # Distance to target AND to final position are both read from dist_maps after the loop.
        per_env_maps = eval_env.env.world_maps[eval_env.env._env_map_idx]  # [n_eps, H, W]
        dist_maps = batch_dijkstra_from_sources(
            per_env_maps.cpu(), TERRAIN_COSTS, initial_spawns.cpu()
        )  # list of n_eps arrays [H, W]

        # No dijkstra_costs needed (path_efficiency removed; directness uses spawn→final)

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

        # Risk exposure: drain_t / (resources_t + hp_t / 2), mean across episode
        risk_sum = torch.zeros(n_episodes, device=device)
        risk_count = torch.zeros(n_episodes, device=device)

        # Exploration: bool map of observed cells (updated via minimap visibility mask)
        observed = torch.zeros(n_episodes, H, W, dtype=torch.bool, device=device)

        # Minimap offset grid for mapping patch coords to world coords [1, D, D]
        max_ray = env_config.minimap_max_ray
        D = 2 * max_ray + 1
        _dy, _dx = torch.meshgrid(
            torch.arange(D, device=device) - max_ray,
            torch.arange(D, device=device) - max_ray,
            indexing="ij",
        )
        dy_grid = _dy.unsqueeze(0)  # [1, D, D]
        dx_grid = _dx.unsqueeze(0)  # [1, D, D]

        # Final position / cost — captured before auto-reset clears them
        final_positions = initial_spawns.clone()
        final_cost = torch.zeros(n_episodes, device=device)
        is_finalized = torch.zeros(n_episodes, dtype=torch.bool, device=device)

        # Trajectories (stored only for first max_trajectory_eps episodes)
        trajectories: list[list[tuple[int, int]] | None] = [None] * n_episodes
        for i in range(min(n_episodes, max_trajectory_eps)):
            p = eval_env.state.position[i].cpu().tolist()
            trajectories[i] = [tuple(p)]

        # Seed observed mask with initial visibility at spawn
        _init_vis = eval_env.state.minimap[:, 1]  # [N, D, D]
        _init_pos = eval_env.state.position        # [N, 2]
        _world_rows = (_init_pos[:, 0].view(-1, 1, 1) + dy_grid).clamp(0, H - 1)
        _world_cols = (_init_pos[:, 1].view(-1, 1, 1) + dx_grid).clamp(0, W - 1)
        _vis = _init_vis > 0.5
        _ep = torch.arange(n_episodes, device=device).view(-1, 1, 1).expand(n_episodes, D, D)
        observed[_ep[_vis], _world_rows[_vis], _world_cols[_vis]] = True

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

            # Risk exposure: drain_t / (resources_t + hp_t / 2)
            drain_t = terrain_res_drains[eval_env.state.terrain_idx.long()]  # [n_eps]
            risk_t = drain_t / (current_resources + current_hp / 2.0 + 1e-6)
            risk_sum[still_running] += risk_t[still_running]
            risk_count[still_running] += 1

            # Exploration via minimap visibility mask (accounts for occlusion).
            # Exclude just-finished episodes: their minimap has been reset by auto-reset.
            explore_idx = torch.where(still_running & ~just_finished)[0]
            if explore_idx.numel() > 0:
                G = explore_idx.shape[0]
                vis_masks = eval_env.state.minimap[explore_idx, 1]       # [G, D, D]
                pos_g = eval_env.state.position[explore_idx]              # [G, 2]
                world_rows = (pos_g[:, 0].view(G, 1, 1) + dy_grid).clamp(0, H - 1)  # [G, D, D]
                world_cols = (pos_g[:, 1].view(G, 1, 1) + dx_grid).clamp(0, W - 1)  # [G, D, D]
                visible = vis_masks > 0.5                                  # [G, D, D]
                ep_idx = explore_idx.view(G, 1, 1).expand(G, D, D)
                observed[ep_idx[visible], world_rows[visible], world_cols[visible]] = True

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
        danger_fraction = compute_danger_fraction(danger_steps, total_moves)
        final_resources = eval_env.state.resources
        resource_mean = resource_sum / resource_count.clamp(min=1)
        hp_mean = hp_sum / hp_count.clamp(min=1)

        dijkstra_to_final = read_dijkstra_to_final(dist_maps, final_positions, device)
        directness = compute_directness(final_cost, dijkstra_to_final)
        exploration = compute_exploration(observed)
        terrain_visit_frac = compute_terrain_visit_fractions(terrain_visits)
        risk_exposure = compute_risk_exposure(risk_sum, risk_count)

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
                "risk_exposure": risk_exposure[i].item(),
                "exploration": exploration[i].item(),
                "terrain_cost": final_cost[i].item(),
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
                observed_mask=observed[i].cpu().numpy(),
            ))

        return EvalResult(
            episodes=episodes,
            mode=mode,
            split=split,
            global_step=global_step,
            initial_targets=initial_targets,
        )
