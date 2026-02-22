#!/usr/bin/env python3
"""Load a saved model and run evaluation episodes.

Usage:
    python scripts/evaluate.py model_path=checkpoints/checkpoint_50000.pt
    python scripts/evaluate.py model_path=checkpoints/checkpoint_50000.pt env=hard
"""

import hydra
from omegaconf import DictConfig
import torch

from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.models.ppo import ActorCritic
from cogniland.utils.checkpoint import load_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = cfg.get("model_path", None)
    if model_path is None:
        print("Error: must specify model_path=<path_to_checkpoint>")
        return

    # Build env config
    env_cfg = cfg.env
    env_config = EnvConfig(
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
        device=device,
    )

    # Build model
    minimap_size = 2 * env_cfg.minimap_ray + 1
    model = ActorCritic(
        scalar_dim=cfg.model.get("scalar_dim", 6),
        minimap_channels=cfg.model.get("minimap_channels", 1),
        minimap_size=minimap_size,
        hidden_dim=cfg.model.get("hidden_dim", 128),
        action_dim=cfg.model.get("action_dim", 5),
    ).to(device)

    # Load checkpoint
    ckpt = load_checkpoint(model_path, model, device=device)
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    model.eval()

    # Run evaluation
    num_episodes = cfg.training.get("eval_episodes", 16)
    env = BatchedIslandEnv(env_config, num_envs=num_episodes)
    obs = env.reset(seed=env_cfg.seed + 2000)

    total_rewards = torch.zeros(num_episodes, device=device)
    total_steps = torch.zeros(num_episodes, device=device)
    reached = torch.zeros(num_episodes, dtype=torch.bool, device=device)
    alive = torch.ones(num_episodes, dtype=torch.bool, device=device)

    for step in range(env_config.max_steps):
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs)
        obs, reward, done, info = env.step(action)

        still_running = alive & ~reached
        total_rewards[still_running] += reward[still_running]
        total_steps[still_running] += 1

        newly_reached = info.get("reached", torch.zeros_like(done, dtype=torch.bool))
        reached = reached | (newly_reached & still_running)

        newly_dead = ~info.get("alive", torch.ones_like(done, dtype=torch.bool))
        alive = alive & ~newly_dead

        if (~alive | reached).all():
            break

    print(f"\n{'='*40}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*40}")
    print(f"Success rate: {reached.float().mean().item():.1%}")
    print(f"Mean reward:  {total_rewards.mean().item():.2f}")
    print(f"Mean steps:   {total_steps.mean().item():.0f}")
    print(f"Mean final HP: {env.state.hp.mean().item():.1f}")


if __name__ == "__main__":
    main()
