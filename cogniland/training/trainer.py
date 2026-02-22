"""PPO training loop — CleanRL-style.

Entry point: `train(cfg)` called from scripts/train.py via Hydra.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.models.ppo import ActorCritic
from cogniland.models.compass_agent import CompassAgent
from cogniland.training.rollout import RolloutBuffer, collect_rollout, compute_gae
from cogniland.logging.wandb_logger import WandBLogger
from cogniland.logging.metrics import compute_behavioral_metrics
from cogniland.utils.checkpoint import save_checkpoint, load_checkpoint
from cogniland.utils.reproducibility import set_reproducibility


def _build_model(cfg, device: str) -> nn.Module:
    model_name = cfg.model.name
    if model_name == "compass":
        return CompassAgent(action_dim=cfg.model.action_dim).to(device)
    elif model_name == "ppo":
        minimap_size = 2 * cfg.env.minimap_ray + 1
        return ActorCritic(
            scalar_dim=cfg.model.scalar_dim,
            minimap_channels=cfg.model.minimap_channels,
            minimap_size=minimap_size,
            hidden_dim=cfg.model.hidden_dim,
            action_dim=cfg.model.action_dim,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_env_config(cfg) -> EnvConfig:
    env_cfg = cfg.env
    return EnvConfig(
        size=env_cfg.size,
        scale=env_cfg.scale,
        octaves=env_cfg.octaves,
        persistence=env_cfg.persistence,
        lacunarity=env_cfg.lacunarity,
        seed=env_cfg.seed,
        detailed_ocean=env_cfg.detailed_ocean,
        filtering=env_cfg.filtering,
        sink_mode=env_cfg.sink_mode,
        init_hp=env_cfg.init_hp,
        max_hp=env_cfg.max_hp,
        init_resources=env_cfg.init_resources,
        max_sea_movement_without_resources=env_cfg.max_sea_movement_without_resources,
        hard_mode=env_cfg.hard_mode,
        minimap_ray=env_cfg.minimap_ray,
        minimap_occlude=env_cfg.minimap_occlude,
        minimap_min_clear_lv=env_cfg.minimap_min_clear_lv,
        max_steps=env_cfg.max_steps,
        reward_dist_coef=env_cfg.reward_dist_coef,
        reward_reach_bonus=env_cfg.reward_reach_bonus,
        reward_death_penalty=env_cfg.reward_death_penalty,
        reward_time_penalty=env_cfg.reward_time_penalty,
        reward_hp_coef=env_cfg.reward_hp_coef,
        reward_hp_thresh=env_cfg.reward_hp_thresh,
        device=cfg.device,
    )


def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    flat_data: dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg,
) -> dict[str, float]:
    """Run PPO update epochs on the collected rollout data."""
    N = flat_data["actions"].shape[0]
    minibatch_size = cfg.training.minibatch_size
    clip_coef = cfg.training.clip_coef
    vf_coef = cfg.training.vf_coef
    ent_coef = cfg.training.ent_coef
    max_grad_norm = cfg.training.max_grad_norm

    # Normalize advantages
    adv = advantages
    if adv.std() > 0:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_clipfrac = 0.0
    total_approx_kl = 0.0
    n_updates = 0

    for _epoch in range(cfg.training.num_epochs):
        indices = torch.randperm(N, device=flat_data["actions"].device)

        for start in range(0, N, minibatch_size):
            end = start + minibatch_size
            if end > N:
                break
            mb_idx = indices[start:end]

            mb_obs = {
                "scalars": flat_data["obs_scalars"][mb_idx],
                "minimap": flat_data["obs_minimaps"][mb_idx],
            }
            mb_actions = flat_data["actions"][mb_idx]
            mb_old_logprobs = flat_data["log_probs"][mb_idx]
            mb_advantages = adv[mb_idx]
            mb_returns = returns[mb_idx]

            _, new_logprob, entropy, new_value = model.get_action_and_value(mb_obs, mb_actions)

            # Policy loss (clipped)
            log_ratio = new_logprob - mb_old_logprobs
            ratio = log_ratio.exp()
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            vf_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

            # Entropy bonus
            entropy_loss = entropy.mean()

            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Stats
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy_loss.item()
            total_clipfrac += clipfrac
            total_approx_kl += approx_kl
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "train/policy_loss": total_pg_loss / n_updates,
        "train/value_loss": total_vf_loss / n_updates,
        "train/entropy": total_entropy / n_updates,
        "train/clipfrac": total_clipfrac / n_updates,
        "train/approx_kl": total_approx_kl / n_updates,
    }


def _render_trajectory(world_map, positions, target, reached_target, env_idx):
    """Render a single agent's trajectory on the island map as a matplotlib figure.

    Returns a matplotlib Figure (caller closes it after logging).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from cogniland.env.constants import TERRAIN_LEVELS, palette

    # Build terrain-level map for colouring (same logic as core.compute_terrain_levels)
    wm = world_map.cpu().numpy()
    thresholds = np.array([TERRAIN_LEVELS[i]["threshold"] for i in range(9)])
    terrain_map = np.searchsorted(thresholds, wm).clip(0, 8)

    # Build RGB image
    color_lut = np.array([palette[TERRAIN_LEVELS[i]["color"]] for i in range(9)], dtype=np.float32) / 255.0
    rgb = color_lut[terrain_map]  # [H, W, 3]

    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Trajectory line
    pos = np.array(positions)  # [T, 2]  (row, col)
    ax.plot(pos[:, 1], pos[:, 0], "white", linewidth=3, alpha=0.6)
    ax.plot(pos[:, 1], pos[:, 0], "r-", linewidth=1.5, alpha=0.9)

    # Start / end / target markers
    ax.scatter(pos[0, 1], pos[0, 0], c="lime", s=120, marker="o", edgecolors="k", linewidth=1.5, zorder=5, label="Start")
    ax.scatter(pos[-1, 1], pos[-1, 0], c="red", s=120, marker="X", edgecolors="k", linewidth=1.5, zorder=5, label="End")
    tgt = target.cpu().numpy()
    ax.scatter(tgt[1], tgt[0], c="gold", s=160, marker="*", edgecolors="k", linewidth=1.5, zorder=5, label="Target")

    status = "SUCCESS" if reached_target else "FAILED"
    ax.set_title(f"Episode {env_idx} — {status} ({len(positions)} moves)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _run_eval(
    env_config: EnvConfig,
    model: nn.Module,
    cfg,
    device: str,
    logger=None,
    global_step: int = 0,
) -> dict[str, float]:
    """Run evaluation episodes and return metrics + log trajectory images."""
    import matplotlib.pyplot as plt

    n_eps = cfg.training.eval_episodes
    eval_env = BatchedIslandEnv(env_config, num_envs=n_eps)
    obs = eval_env.reset(seed=cfg.env.seed + 1000)

    total_rewards = torch.zeros(n_eps, device=device)
    total_moves = torch.zeros(n_eps, device=device)
    reached = torch.zeros(n_eps, dtype=torch.bool, device=device)
    alive = torch.ones(n_eps, dtype=torch.bool, device=device)
    final_hp = torch.zeros(n_eps, device=device)
    terrain_visits = torch.zeros(n_eps, 9, device=device)

    # Record positions for trajectory visualisation
    trajectories: list[list[tuple[int, int]]] = [[] for _ in range(n_eps)]
    initial_targets = eval_env.target_pos.clone()

    # Store initial positions
    for i in range(n_eps):
        p = eval_env.state.position[i].cpu().tolist()
        trajectories[i].append(tuple(p))

    # "move" = single agent action in the environment (distinct from training
    # "update" which is one PPO optimisation pass over a rollout buffer).
    for move in range(env_config.max_steps):
        still_running = alive & ~reached

        # Capture pre-move state (before auto-reset can teleport positions)
        pre_move_terrain = eval_env.state.terrain_lev.clone()
        pre_move_hp = eval_env.state.hp.clone()

        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        obs, reward, done, info = eval_env.step(action)

        # After step(), done envs are auto-reset to random positions.
        # Use info flags (computed pre-reset) to detect termination.

        total_rewards[still_running] += reward[still_running]
        total_moves[still_running] += 1

        # Update reached / alive BEFORE recording positions
        newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
        reached = reached | (newly_reached & still_running)

        newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
        alive = alive & ~newly_dead

        # Truncation: env hit max_steps internally → done but not reached/dead
        truncated = done & ~newly_reached & ~newly_dead

        # Record final HP for envs that just finished
        just_finished = (newly_reached | newly_dead | truncated) & still_running
        final_hp[just_finished] = pre_move_hp[just_finished]

        # Record positions only for envs still active.  Skip any env that
        # terminated this move (reached / dead / truncated) to avoid logging
        # the auto-reset teleport position.
        for i in torch.where(still_running)[0].tolist():
            if newly_reached[i]:
                # Snap final position to the target
                tgt = initial_targets[i].cpu().tolist()
                trajectories[i].append(tuple(tgt))
            elif newly_dead[i] or truncated[i]:
                # Trajectory ends here — don't record post-reset position
                pass
            else:
                # Normal move — position is valid (no auto-reset)
                p = eval_env.state.position[i].cpu().tolist()
                trajectories[i].append(tuple(p))

        # Mark truncated envs as no longer alive so still_running becomes False
        alive = alive & ~truncated

        # Track terrain visits (use pre-move terrain for finished envs)
        for i in torch.where(still_running)[0].tolist():
            t = int(pre_move_terrain[i].item())
            if 0 <= t <= 8:
                terrain_visits[i, t] += 1

        if (~alive | reached).all():
            break

    still_running = alive & ~reached
    final_hp[still_running] = eval_env.state.hp[still_running]
    total_moves[still_running] = env_config.max_steps

    initial_dist = torch.norm(
        (eval_env.state.position - eval_env.target_pos).float(), dim=1
    )
    path_efficiency = torch.where(
        total_moves > 0,
        initial_dist / total_moves.clamp(min=1),
        torch.zeros_like(total_moves),
    )

    eval_metrics = {
        "eval/success_rate": reached.float().mean().item(),
        "eval/mean_reward": total_rewards.mean().item(),
        "eval/mean_moves": total_moves.mean().item(),
        "eval/mean_final_hp": final_hp.mean().item(),
        "eval/path_efficiency": path_efficiency.mean().item(),
    }

    behavioral = compute_behavioral_metrics(terrain_visits, total_moves, eval_env.state)
    eval_metrics.update(behavioral)

    # Log trajectory images + behavioral profile to WandB
    if logger is not None:
        # Trajectory gallery
        max_images = cfg.logging.get("trajectory", {}).get("max_saved_per_eval", 4)
        figures, captions, outcomes, steps_counts = [], [], [], []
        for i in range(n_eps):
            if len(figures) >= max_images:
                break
            if len(trajectories[i]) < 2:
                continue
            fig = _render_trajectory(
                eval_env.env.world_map, trajectories[i],
                initial_targets[i], reached[i].item(), i,
            )
            outcome = "success" if reached[i].item() else "fail"
            n_moves = int(total_moves[i].item())
            figures.append(fig)
            captions.append(f"ep{i} {outcome} {n_moves} moves")
            outcomes.append(outcome)
            steps_counts.append(n_moves)

        if figures:
            logger.log_trajectory_images(figures, captions, outcomes, steps_counts, step=global_step)
            for fig in figures:
                plt.close(fig)

        # Behavioral profile bar chart
        logger.log_behavioral_profile(behavioral, step=global_step)

    return eval_metrics


def train(cfg) -> None:
    """Main entry point — PPO training loop or single-eval for baselines."""
    set_reproducibility(cfg.env.seed)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    env_config = _build_env_config(cfg)
    model = _build_model(cfg, device)
    trainable = cfg.model.name == "ppo"

    logger = WandBLogger(cfg)
    print(f"Model: {cfg.model.name}")

    if not trainable:
        # Deterministic baseline (e.g. compass) — single eval pass is enough.
        print("Running baseline evaluation...")
        eval_metrics = _run_eval(env_config, model, cfg, device, logger=logger, global_step=0)
        logger.log(eval_metrics, step=0)
        print(f"  eval/success_rate: {eval_metrics['eval/success_rate']:.3f}, "
              f"eval/mean_reward: {eval_metrics['eval/mean_reward']:.2f}")
        logger.finish()
        return

    # --- PPO training loop ---
    env = BatchedIslandEnv(env_config, num_envs=cfg.training.num_envs)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, eps=1e-5)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    num_envs = cfg.training.num_envs
    rollout_steps = cfg.training.rollout_steps
    total_timesteps = cfg.training.total_timesteps
    num_updates = total_timesteps // (num_envs * rollout_steps)

    print(f"Total updates: {num_updates}, Moves per update: {num_envs * rollout_steps}")

    obs = env.reset(seed=cfg.env.seed)
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # LR annealing
        if cfg.training.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr = frac * cfg.training.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Collect rollout
        buffer, obs, episode_stats = collect_rollout(env, model, obs, rollout_steps)
        global_step += num_envs * rollout_steps

        # Log episode stats
        if episode_stats:
            ep_rewards = episode_stats["episode_rewards"]
            ep_lengths = episode_stats["episode_lengths"]
            logger.log({
                "train/mean_reward": ep_rewards.mean().item(),
                "train/mean_episode_length": ep_lengths.mean().item(),
                "train/global_step": global_step,
            }, step=update)

        # Compute GAE
        with torch.no_grad():
            next_value = model.get_value(obs)
        advantages, returns = compute_gae(
            buffer, next_value,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
        )

        # PPO update
        flat_data = buffer.flatten()
        train_metrics = ppo_update(model, optimizer, flat_data, advantages, returns, cfg)

        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics["train/learning_rate"] = current_lr

        sps = int(global_step / (time.time() - start_time))
        train_metrics["train/sps"] = sps
        train_metrics["train/global_step"] = global_step

        logger.log(train_metrics, step=update)

        # Periodic eval
        if update % cfg.training.eval_interval == 0:
            print(f"[Update {update}/{num_updates}] Running evaluation...")
            model.eval()
            eval_metrics = _run_eval(env_config, model, cfg, device, logger=logger, global_step=update)
            model.train()
            logger.log(eval_metrics, step=update)
            print(f"  eval/success_rate: {eval_metrics['eval/success_rate']:.3f}, "
                  f"eval/mean_reward: {eval_metrics['eval/mean_reward']:.2f}")

        # Periodic checkpoint
        if update % cfg.training.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, global_step,
                path=f"checkpoints/ckpt_{update}.pt",
            )
            print(f"  Checkpoint saved at step {global_step}")

    logger.finish()
    print(f"Training complete. Total timesteps: {global_step}")
