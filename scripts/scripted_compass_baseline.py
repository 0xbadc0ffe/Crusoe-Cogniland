"""Scripted compass-following baseline to verify env mechanics.

Follows the compass axis with larger magnitude. Forages on berry tile.
Reports success rate by spawn-distance band.
"""
from __future__ import annotations

from cogniland.config import setup_environment
setup_environment()

import numpy as np
from omegaconf import OmegaConf

from cogniland.envs.registry import make_env


def main():
    import sys
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    berry_policy = sys.argv[4] if len(sys.argv) > 4 else "hp80"  # "none", "hp80", "hp60"
    cfg = OmegaConf.load("configs/env/cogniland.yaml")
    cfg.env.num_parallel_envs = 64
    cfg.env.num_parallel_envs_eval = 64
    cfg.env.max_steps = max_steps
    cfg.env.spawn_distance_range = [lo, hi]
    cfg.env.biome_filter = ["balanced"]
    cfg.tasks = [0]
    print(f"[cfg] d=[{lo},{hi}] max_steps={max_steps} berry_policy={berry_policy}")
    berry_thresh = {"none": 0, "hp80": 80, "hp60": 60, "hp90": 90}[berry_policy]

    env = make_env("cogniland-v0", cfg, train=True)

    num_episodes_target = 2000
    returns, lengths, successes = [], [], []
    berry_forages = []  # per-episode count of successful berry forages
    min_hp_list = []

    obs = env.reset()
    ep_berries = np.zeros(env.num_envs, dtype=np.int32)
    ep_min_hp = np.full(env.num_envs, 100, dtype=np.int32)

    task_ids = np.zeros(env.num_envs, dtype=np.int32)
    env.set_tasks(task_ids)

    while len(returns) < num_episodes_target:
        scalars = obs["scalars"]  # [B, 6]
        minimap = obs["minimap"]  # [B, 45, 45]
        cx = scalars[:, 0]
        cy = scalars[:, 1]
        tile_class = (scalars[:, 2] * 9).round().astype(np.int32)  # 9 = berry
        hp_frac = scalars[:, 3]
        hp = (hp_frac * 100).round().astype(np.int32)

        # default: follow compass
        actions = np.zeros(env.num_envs, dtype=np.int32)
        use_y = np.abs(cy) >= np.abs(cx)
        # row dimension: cy > 0 -> down (action 1), cy < 0 -> up (action 0)
        actions[use_y & (cy > 0)] = 1  # down
        actions[use_y & (cy < 0)] = 0  # up
        # col dimension: cx > 0 -> right (action 3), cx < 0 -> left (action 2)
        actions[~use_y & (cx > 0)] = 3
        actions[~use_y & (cx < 0)] = 2

        # If standing on berry and HP below threshold, forage
        on_berry = (tile_class == 9)
        should_forage = on_berry & (hp < berry_thresh)
        actions[should_forage] = 4

        next_obs, rewards, dones, info = env.step(actions)

        # Track per-episode metrics
        # Berry forage = was on berry, took forage action, hp went up
        was_on_berry = on_berry
        took_forage = actions == 4
        new_hp = (next_obs["scalars"][:, 3] * 100).round().astype(np.int32)
        # Berry forage events
        forage_events = was_on_berry & took_forage & (new_hp > hp)
        ep_berries += forage_events.astype(np.int32)
        ep_min_hp = np.minimum(ep_min_hp, hp)

        # Episode ends
        if "returned_episode" in info:
            done_mask = info["returned_episode"]
            if done_mask.any():
                r = info["returned_episode_returns"][done_mask]
                l = info["returned_episode_lengths"][done_mask]
                s = info["task_success"][done_mask]
                returns.extend(r.tolist())
                lengths.extend(l.tolist())
                successes.extend(s.tolist())
                berry_forages.extend(ep_berries[done_mask].tolist())
                min_hp_list.extend(ep_min_hp[done_mask].tolist())
                # Reset per-episode trackers
                ep_berries[done_mask] = 0
                ep_min_hp[done_mask] = 100

        obs = next_obs

    returns = np.array(returns[:num_episodes_target])
    lengths = np.array(lengths[:num_episodes_target])
    successes = np.array(successes[:num_episodes_target]).astype(int)
    berry_forages = np.array(berry_forages[:num_episodes_target])
    min_hp_list = np.array(min_hp_list[:num_episodes_target])

    print(f"SCRIPTED COMPASS at d=[5,15], max_steps=200, balanced maps, {len(returns)} eps")
    print(f"  success:       {successes.mean():.3f}")
    print(f"  mean return:   {returns.mean():+.2f}")
    print(f"  mean length:   {lengths.mean():.1f}")
    print(f"  mean berries:  {berry_forages.mean():.2f}")
    print(f"  mean min hp:   {min_hp_list.mean():.1f}")
    print(f"  % died (length<200 & !success): {(1 - successes - (lengths >= 200).astype(int)).clip(0, 1).mean():.3f}")


if __name__ == "__main__":
    main()
