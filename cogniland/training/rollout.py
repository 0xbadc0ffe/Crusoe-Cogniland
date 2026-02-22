"""Rollout buffer, collection, and GAE computation for PPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutBuffer:
    """Stores a single rollout of experience for PPO training."""

    obs_scalars: list[torch.Tensor] = field(default_factory=list)
    obs_minimaps: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.obs_scalars.append(obs["scalars"])
        self.obs_minimaps.append(obs["minimap"])
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def flatten(self) -> dict[str, torch.Tensor]:
        """Stack timesteps and flatten to [T*B, ...] for minibatch training."""
        T = len(self.obs_scalars)
        B = self.obs_scalars[0].shape[0]

        return {
            "obs_scalars": torch.stack(self.obs_scalars).reshape(T * B, -1),
            "obs_minimaps": torch.stack(self.obs_minimaps).reshape(T * B, *self.obs_minimaps[0].shape[1:]),
            "actions": torch.stack(self.actions).reshape(T * B),
            "log_probs": torch.stack(self.log_probs).reshape(T * B),
            "rewards": torch.stack(self.rewards),       # [T, B]  (kept 2D for GAE)
            "dones": torch.stack(self.dones),            # [T, B]
            "values": torch.stack(self.values),          # [T, B]
        }


@torch.no_grad()
def collect_rollout(
    env,
    model: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    rollout_steps: int,
) -> tuple[RolloutBuffer, dict[str, torch.Tensor], dict]:
    """Collect `rollout_steps` steps from the batched environment.

    Returns:
        (buffer, next_obs, episode_stats)
    """
    buffer = RolloutBuffer()
    all_final_rewards = []
    all_final_lengths = []
    all_final_reached = []

    for _ in range(rollout_steps):
        value = model.get_value(obs)
        action, log_prob, _, _ = model.get_action_and_value(obs)

        next_obs, reward, done, info = env.step(action)

        buffer.add(obs, action, log_prob, reward, done, value)

        # Collect episode completion stats
        if "final_rewards" in info:
            all_final_rewards.append(info["final_rewards"])
            all_final_lengths.append(info["final_lengths"])
            all_final_reached.append(info["final_reached"])

        obs = next_obs

    episode_stats = {}
    if all_final_rewards:
        episode_stats["episode_rewards"] = torch.cat(all_final_rewards)
        episode_stats["episode_lengths"] = torch.cat(all_final_lengths)
        episode_stats["episode_reached"] = torch.cat(all_final_reached)

    return buffer, obs, episode_stats


def compute_gae(
    buffer: RolloutBuffer,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        buffer: filled rollout buffer
        next_value: V(s_{T+1}) from the critic  [B]
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        (advantages, returns) both flattened to [T*B]
    """
    rewards = torch.stack(buffer.rewards)   # [T, B]
    dones = torch.stack(buffer.dones)       # [T, B]
    values = torch.stack(buffer.values)     # [T, B]
    T, B = rewards.shape

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages.reshape(T * B), returns.reshape(T * B)
