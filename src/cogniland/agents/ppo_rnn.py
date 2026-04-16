"""PPO-RNN agent — recurrent PPO with LSTM in JAX/Flax.

Architecture:
    Minimap CNN -> flatten -> concat(scalar_mlp, task_emb) -> trunk MLP -> LSTM -> actor/critic heads

The env is numpy-based. The training loop converts obs to JAX arrays for the
forward pass and converts actions back to numpy for env.step().
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

from cogniland.agents.agent import Agent
from cogniland.agents.registry import register_agent
from cogniland.agents.state import AgentState, RuntimeState


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticRNN(nn.Module):
    """CNN + MLP + LSTM actor-critic for Cogniland maps.

    Input:
        minimap:  [B, 3, 45, 45]  (channels-first, converted to channels-last internally)
        scalars:  [B, 6]
        task_emb: [B, task_embedding_dim]
        carry:    (h, c) each [B, lstm_size]

    Output:
        logits:  [B, num_actions]
        value:   [B]
        new_carry: (h, c)
    """
    num_actions: int = 8
    lstm_size: int = 256
    hidden_size: int = 256
    task_embedding_dim: int = 7

    @nn.compact
    def __call__(self, minimap, scalars, task_emb, carry):
        # -- CNN (channels-last for Flax Conv) --
        # Input: [B, 3, 45, 45] -> transpose to [B, 45, 45, 3]
        x = jnp.transpose(minimap, (0, 2, 3, 1))

        x = nn.Conv(features=16, kernel_size=(3, 3), padding="VALID",
                     kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        # MaxPool 2x2
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID",
                     kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        # Adaptive max pool to 4x4: use global pooling with window = spatial dims
        # After conv: spatial is reduced; we just pool down to 4x4
        # Current spatial: floor((floor((45-2)/2+1) - 2)/1 + 1) = floor((22-2)+1) = 21
        # Actually: (45-3+1)=43, pool-> 21, (21-3+1)=19
        # Pool 19 -> 4: window=(4,4) stride=(4,4) gets us floor(19/4)=4
        spatial = x.shape[1]  # should be 19
        pool_size = spatial // 4
        x = nn.max_pool(x, window_shape=(pool_size, pool_size),
                        strides=(pool_size, pool_size))
        # Flatten: [B, 4, 4, 32] -> [B, 512]
        x = x.reshape((x.shape[0], -1))

        # -- Scalar MLP --
        s = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(scalars)
        s = nn.relu(s)

        # -- Concat CNN + scalars + task embedding --
        # x: [B, 512], s: [B, 64], task_emb: [B, 7] -> [B, 583]
        h = jnp.concatenate([x, s, task_emb], axis=-1)

        # -- Trunk MLP --
        h = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        h = nn.relu(h)

        # -- LSTM --
        lstm_cell = nn.OptimizedLSTMCell(features=self.lstm_size,
                                          kernel_init=nn.initializers.orthogonal(1.0),
                                          recurrent_kernel_init=nn.initializers.orthogonal(1.0))
        new_carry, h = lstm_cell(carry, h)

        # -- Actor head (small init for near-uniform initial policy) --
        logits = nn.Dense(self.num_actions,
                          kernel_init=nn.initializers.orthogonal(0.01),
                          bias_init=nn.initializers.zeros)(h)

        # -- Critic head --
        value = nn.Dense(1,
                         kernel_init=nn.initializers.orthogonal(1.0),
                         bias_init=nn.initializers.zeros)(h)
        value = value.squeeze(-1)  # [B]

        return logits, value, new_carry


# ---------------------------------------------------------------------------
# Transition storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs_minimap: jnp.ndarray    # [B, 3, 45, 45]
    obs_scalars: jnp.ndarray    # [B, 6]
    action: jnp.ndarray         # [B]
    log_prob: jnp.ndarray       # [B]
    value: jnp.ndarray          # [B]
    reward: jnp.ndarray         # [B]
    done: jnp.ndarray           # [B]
    carry_h: jnp.ndarray        # [B, lstm_size]
    carry_c: jnp.ndarray        # [B, lstm_size]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_agent("ppo_rnn")
def make_ppo_rnn(config, obs_space, act_space) -> Agent:
    """Create a PPO-RNN agent.

    Args:
        config: OmegaConf config (merged env + agent YAML).
        obs_space: Observation space dict (keys: minimap, scalars).
        act_space: Number of discrete actions (8).
    """
    agent_cfg = config.agent
    num_actions = act_space if isinstance(act_space, int) else int(act_space)
    task_embedding_dim = getattr(config, "task_embedding_dim", 7)
    num_tasks = getattr(config, "num_tasks", 1)

    lr = agent_cfg.lr
    anneal_lr = agent_cfg.anneal_lr
    num_steps = agent_cfg.num_steps       # rollout length
    gamma = agent_cfg.gamma
    gae_lambda = agent_cfg.gae_lambda
    clip_eps = agent_cfg.clip_eps
    entropy_coef = agent_cfg.entropy_coef
    value_coef = agent_cfg.value_coef
    clip_grad = agent_cfg.clip_grad
    normalize_advantages = agent_cfg.normalize_advantages
    ppo_epochs = agent_cfg.ppo_epochs
    num_minibatches = agent_cfg.num_minibatches
    hidden_size = agent_cfg.hidden_size
    lstm_size = agent_cfg.lstm_size

    num_envs = config.env.num_parallel_envs

    network = ActorCriticRNN(
        num_actions=num_actions,
        lstm_size=lstm_size,
        hidden_size=hidden_size,
        task_embedding_dim=task_embedding_dim,
    )

    # ------------------------------------------------------------------
    # JIT-compiled forward pass
    # ------------------------------------------------------------------
    @jax.jit
    def _forward(params, minimap, scalars, task_emb, carry):
        """Run network forward pass. Returns (logits, value, new_carry)."""
        return network.apply(params, minimap, scalars, task_emb, carry)

    @jax.jit
    def _sample_action(params, minimap, scalars, task_emb, carry, rng):
        """Sample action from policy, return (action, log_prob, value, new_carry)."""
        logits, value, new_carry = network.apply(params, minimap, scalars, task_emb, carry)
        dist = jax.random.categorical(rng, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), dist]
        return dist, log_prob, value, new_carry

    @jax.jit
    def _deterministic_action(params, minimap, scalars, task_emb, carry):
        """Greedy action from policy."""
        logits, value, new_carry = network.apply(params, minimap, scalars, task_emb, carry)
        return jnp.argmax(logits, axis=-1), value, new_carry

    # ------------------------------------------------------------------
    # GAE computation (jitted)
    # ------------------------------------------------------------------
    @jax.jit
    def _compute_gae(rewards, values, dones, last_value):
        """Compute GAE advantages and returns.

        Args:
            rewards: [T, B]
            values:  [T, B]
            dones:   [T, B]
            last_value: [B]

        Returns:
            advantages: [T, B]
            returns:    [T, B]
        """
        T = rewards.shape[0]

        def _scan_fn(carry, t):
            last_gae = carry
            # Read from the end backwards
            idx = T - 1 - t
            done = dones[idx]
            r = rewards[idx]
            v = values[idx]
            next_v = jnp.where(idx == T - 1, last_value, values[idx + 1])
            next_done = jnp.where(idx == T - 1, jnp.zeros_like(done), dones[idx + 1])

            delta = r + gamma * next_v * (1.0 - done) - v
            last_gae = delta + gamma * gae_lambda * (1.0 - done) * last_gae
            return last_gae, last_gae

        _, advantages_rev = jax.lax.scan(
            _scan_fn,
            jnp.zeros_like(last_value),
            jnp.arange(T),
        )
        # Reverse to get chronological order
        advantages = jnp.flip(advantages_rev, axis=0)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO loss and update (jitted)
    # ------------------------------------------------------------------
    @jax.jit
    def _ppo_update_step(train_state, batch, rng):
        """Single PPO gradient step on a minibatch.

        batch: dict with keys:
            obs_minimap  [MB, 3, 45, 45]
            obs_scalars  [MB, 6]
            task_emb     [MB, task_embedding_dim]
            action       [MB]
            old_log_prob [MB]
            advantage    [MB]
            return_      [MB]
            carry_h      [MB, lstm_size]
            carry_c      [MB, lstm_size]
        """
        def loss_fn(params):
            logits, value, _ = network.apply(
                params,
                batch["obs_minimap"],
                batch["obs_scalars"],
                batch["task_emb"],
                (batch["carry_h"], batch["carry_c"]),
            )
            # Policy loss
            log_probs = jax.nn.log_softmax(logits)
            new_log_prob = log_probs[jnp.arange(logits.shape[0]), batch["action"]]
            ratio = jnp.exp(new_log_prob - batch["old_log_prob"])
            adv = batch["advantage"]

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            value_loss = 0.5 * ((value - batch["return_"]) ** 2).mean()

            # Entropy bonus
            probs = jax.nn.softmax(logits)
            entropy = -(probs * log_probs).sum(axis=-1).mean()

            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Diagnostics
            clipfrac = (jnp.abs(ratio - 1.0) > clip_eps).mean()
            approx_kl = (0.5 * (batch["old_log_prob"] - new_log_prob) ** 2).mean()

            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "clipfrac": clipfrac,
                "approx_kl": approx_kl,
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        grads = jax.tree.map(
            lambda g: jnp.nan_to_num(g, nan=0.0), grads
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def init(rng):
        rng, init_rng = jax.random.split(rng)
        dummy_minimap = jnp.zeros((1, 3, 45, 45))
        dummy_scalars = jnp.zeros((1, 6))
        dummy_task_emb = jnp.zeros((1, task_embedding_dim))
        dummy_carry = (jnp.zeros((1, lstm_size)), jnp.zeros((1, lstm_size)))

        params = network.init(init_rng, dummy_minimap, dummy_scalars, dummy_task_emb, dummy_carry)

        # Optimizer: Adam with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.adam(lr, eps=1e-5),
        )

        ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        state = AgentState(
            train_state=ts,
            runtime=RuntimeState(
                buffer_state=None,
                wm_state=None,
                step=jnp.array(0),
                train_steps=jnp.array(0),
                rng=rng,
                current_num_actions=jnp.array(num_actions),
            ),
        )
        return state

    # ------------------------------------------------------------------
    # select_action
    # ------------------------------------------------------------------
    def select_action(state, obs, rng, is_first=None, prev_action=None, training=False):
        """Select action given observation dict.

        Returns (actions_np, updated_state).
        """
        minimap = jnp.asarray(obs["minimap"])
        scalars = jnp.asarray(obs["scalars"])

        # Task embedding from runtime (default: zeros if not set)
        task_emb = getattr(state, "_task_emb_cache", jnp.zeros((minimap.shape[0], task_embedding_dim)))

        carry = getattr(state, "_carry_cache", _zero_carry(minimap.shape[0]))

        # Reset carry for episodes that just started
        if is_first is not None:
            is_first_jax = jnp.asarray(is_first).reshape(-1, 1)
            carry = jax.tree.map(
                lambda c: jnp.where(is_first_jax, 0.0, c), carry
            )

        params = state.train_state.params

        if training:
            rng, act_rng = jax.random.split(rng)
            actions, log_prob, value, new_carry = _sample_action(
                params, minimap, scalars, task_emb, carry, act_rng
            )
        else:
            actions, value, new_carry = _deterministic_action(
                params, minimap, scalars, task_emb, carry
            )

        # Cache carry for next step
        state = state.replace(runtime=state.runtime.replace(rng=rng))
        # We store carry on the state object via a simple attribute trick
        # (AgentState is a chex dataclass, so we use object.__setattr__)
        object.__setattr__(state, "_carry_cache", new_carry)

        return np.asarray(actions), state

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def train(state, env, rng, num_train_frames, progress_bar=None,
              checkpoint_callback=None, task_ids=None):
        """Run PPO training loop for num_train_frames env steps.

        Args:
            state: AgentState
            env: Batched numpy environment with step()/reset()
            rng: JAX PRNGKey
            num_train_frames: Total env frames to collect
            progress_bar: Optional tqdm bar
            checkpoint_callback: Optional callback(state, step)
            task_ids: [num_envs] int array of task indices

        Returns:
            (new_state, metrics_dict)
        """
        train_state = state.train_state
        global_step = int(state.runtime.step)
        train_steps = int(state.runtime.train_steps)

        # Task embeddings
        if task_ids is not None:
            task_emb_np = np.eye(task_embedding_dim, dtype=np.float32)[task_ids]
        else:
            task_emb_np = np.zeros((num_envs, task_embedding_dim), dtype=np.float32)
        task_emb_jax = jnp.asarray(task_emb_np)

        # LR annealing setup
        total_updates = num_train_frames // (num_steps * num_envs)

        # Reset env — our env returns obs dict only (no info)
        obs = env.reset()
        carry = _zero_carry(num_envs)

        # Episode tracking
        all_episode_returns = []
        all_episode_lengths = []
        all_episode_flags = []

        # Aggregate loss metrics
        agg_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clipfrac": 0.0,
            "approx_kl": 0.0,
        }
        n_updates = 0

        frames_collected = 0

        while frames_collected < num_train_frames:
            # -- Collect rollout --
            storage = []
            for step_i in range(num_steps):
                minimap_jax = jnp.asarray(obs["minimap"])
                scalars_jax = jnp.asarray(obs["scalars"])

                rng, act_rng = jax.random.split(rng)
                actions_jax, log_probs, values, new_carry = _sample_action(
                    train_state.params, minimap_jax, scalars_jax, task_emb_jax, carry, act_rng
                )

                actions_np = np.asarray(actions_jax)
                next_obs, rewards, dones, info = env.step(actions_np)

                storage.append(Transition(
                    obs_minimap=minimap_jax,
                    obs_scalars=scalars_jax,
                    action=actions_jax,
                    log_prob=log_probs,
                    value=values,
                    reward=jnp.asarray(rewards, dtype=jnp.float32),
                    done=jnp.asarray(dones, dtype=jnp.float32),
                    carry_h=carry[0],
                    carry_c=carry[1],
                ))

                # Reset carry for done episodes
                done_mask = jnp.asarray(dones).reshape(-1, 1)
                carry = jax.tree.map(
                    lambda c: jnp.where(done_mask, 0.0, c), new_carry
                )

                # Track completed episodes
                if "returned_episode" in info:
                    ep_mask = info["returned_episode"]
                    all_episode_flags.append(ep_mask.copy())
                    if "returned_episode_returns" in info:
                        all_episode_returns.append(info["returned_episode_returns"].copy())
                    if "returned_episode_lengths" in info:
                        all_episode_lengths.append(info["returned_episode_lengths"].copy())
                elif np.any(dones):
                    # Fallback: use dones directly
                    all_episode_flags.append(dones.copy())
                    if "episode_return" in info:
                        all_episode_returns.append(
                            np.where(dones, info["episode_return"], 0.0)
                        )

                obs = next_obs
                global_step += num_envs
                frames_collected += num_envs

                if progress_bar is not None:
                    progress_bar.update(num_envs)

            # -- Bootstrap value for last obs --
            minimap_jax = jnp.asarray(obs["minimap"])
            scalars_jax = jnp.asarray(obs["scalars"])
            _, last_value, _ = _deterministic_action(
                train_state.params, minimap_jax, scalars_jax, task_emb_jax, carry
            )

            # -- Stack transitions: [T, B, ...] --
            rewards_batch = jnp.stack([t.reward for t in storage])   # [T, B]
            values_batch = jnp.stack([t.value for t in storage])     # [T, B]
            dones_batch = jnp.stack([t.done for t in storage])       # [T, B]

            # -- GAE --
            advantages, returns = _compute_gae(rewards_batch, values_batch, dones_batch, last_value)

            # -- Flatten for PPO update: [T*B, ...] --
            T, B = rewards_batch.shape
            flat_size = T * B

            flat_data = {
                "obs_minimap": jnp.concatenate([t.obs_minimap for t in storage], axis=0),   # [T*B, 3, 45, 45]
                "obs_scalars": jnp.concatenate([t.obs_scalars for t in storage], axis=0),   # [T*B, 6]
                "task_emb": jnp.tile(task_emb_jax, (T, 1)),                                 # [T*B, task_emb_dim]
                "action": jnp.concatenate([t.action for t in storage], axis=0),              # [T*B]
                "old_log_prob": jnp.concatenate([t.log_prob for t in storage], axis=0),      # [T*B]
                "advantage": advantages.reshape(flat_size),
                "return_": returns.reshape(flat_size),
                "carry_h": jnp.concatenate([t.carry_h for t in storage], axis=0),            # [T*B, lstm_size]
                "carry_c": jnp.concatenate([t.carry_c for t in storage], axis=0),            # [T*B, lstm_size]
            }

            # -- LR annealing --
            if anneal_lr and total_updates > 0:
                frac = 1.0 - (n_updates / total_updates)
                cur_lr = lr * max(frac, 0.0)
                # Rebuild optimizer with new LR
                tx = optax.chain(
                    optax.clip_by_global_norm(clip_grad),
                    optax.adam(cur_lr, eps=1e-5),
                )
                train_state = TrainState.create(
                    apply_fn=network.apply,
                    params=train_state.params,
                    tx=tx,
                )

            # -- PPO epochs --
            minibatch_size = flat_size // num_minibatches

            for _epoch in range(ppo_epochs):
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, flat_size)

                for mb_start in range(0, flat_size, minibatch_size):
                    mb_idx = perm[mb_start:mb_start + minibatch_size]
                    mb = {k: v[mb_idx] for k, v in flat_data.items()}

                    if normalize_advantages:
                        adv = mb["advantage"]
                        mb["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

                    train_state, step_metrics = _ppo_update_step(train_state, mb, rng)

                    for k in agg_metrics:
                        agg_metrics[k] += float(step_metrics[k])
                    train_steps += 1

            n_updates += 1

            if checkpoint_callback is not None:
                _state = AgentState(
                    train_state=train_state,
                    runtime=RuntimeState(
                        buffer_state=None,
                        wm_state=None,
                        step=jnp.array(global_step),
                        train_steps=jnp.array(train_steps),
                        rng=rng,
                        current_num_actions=jnp.array(num_actions),
                    ),
                )
                if callable(checkpoint_callback):
                    checkpoint_callback(_state, global_step)
                elif hasattr(checkpoint_callback, 'on_validation_end'):
                    checkpoint_callback.on_validation_end(
                        agent_state=_state,
                        step=global_step,
                        metrics={},
                    )

        # -- Build metrics --
        total_grad_steps = ppo_epochs * num_minibatches * max(n_updates, 1)
        metrics = {k: v / max(total_grad_steps, 1) for k, v in agg_metrics.items()}

        # Episode info
        episode_info = {}
        if all_episode_flags:
            episode_info["returned_episode"] = np.concatenate(all_episode_flags, axis=0)
        if all_episode_returns:
            episode_info["returned_episode_returns"] = np.concatenate(all_episode_returns, axis=0)
        if all_episode_lengths:
            episode_info["returned_episode_lengths"] = np.concatenate(all_episode_lengths, axis=0)
        metrics["episode_info"] = episode_info

        new_state = AgentState(
            train_state=train_state,
            runtime=RuntimeState(
                buffer_state=None,
                wm_state=None,
                step=jnp.array(global_step),
                train_steps=jnp.array(train_steps),
                rng=rng,
                current_num_actions=jnp.array(num_actions),
            ),
        )
        object.__setattr__(new_state, "_carry_cache", carry)

        return new_state, metrics

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def evaluate(state, env, rng, num_eval_frames, progress_bar=None, task_ids=None):
        """Run evaluation (no gradient updates).

        Returns metrics dict with episode_info.
        """
        params = state.train_state.params

        if task_ids is not None:
            task_emb_np = np.eye(task_embedding_dim, dtype=np.float32)[task_ids]
        else:
            task_emb_np = np.zeros((num_envs, task_embedding_dim), dtype=np.float32)
        task_emb_jax = jnp.asarray(task_emb_np)

        obs = env.reset()
        n_eval_envs = obs["minimap"].shape[0]
        carry = _zero_carry(n_eval_envs)
        # Adjust task_emb if eval uses different num envs
        if task_emb_jax.shape[0] != n_eval_envs:
            task_emb_jax = jnp.zeros((n_eval_envs, task_embedding_dim))

        all_episode_returns = []
        all_episode_lengths = []
        all_episode_flags = []
        frames = 0

        while frames < num_eval_frames:
            minimap_jax = jnp.asarray(obs["minimap"])
            scalars_jax = jnp.asarray(obs["scalars"])

            actions_jax, _, new_carry = _deterministic_action(
                params, minimap_jax, scalars_jax, task_emb_jax, carry
            )

            actions_np = np.asarray(actions_jax)
            next_obs, rewards, dones, info = env.step(actions_np)

            # Reset carry for done episodes
            done_mask = jnp.asarray(dones).reshape(-1, 1)
            carry = jax.tree.map(lambda c: jnp.where(done_mask, 0.0, c), new_carry)

            if "returned_episode" in info:
                ep_mask = info["returned_episode"]
                all_episode_flags.append(ep_mask.copy())
                if "returned_episode_returns" in info:
                    all_episode_returns.append(info["returned_episode_returns"].copy())
                if "returned_episode_lengths" in info:
                    all_episode_lengths.append(info["returned_episode_lengths"].copy())

            obs = next_obs
            frames += n_eval_envs

            if progress_bar is not None:
                progress_bar.update(n_eval_envs)

        episode_info = {}
        if all_episode_flags:
            episode_info["returned_episode"] = np.concatenate(all_episode_flags, axis=0)
        if all_episode_returns:
            episode_info["returned_episode_returns"] = np.concatenate(all_episode_returns, axis=0)
        if all_episode_lengths:
            episode_info["returned_episode_lengths"] = np.concatenate(all_episode_lengths, axis=0)

        return {"episode_info": episode_info}

    # ------------------------------------------------------------------
    # state_from_checkpoint
    # ------------------------------------------------------------------
    def state_from_checkpoint(checkpoint_data, runtime_state):
        """Restore AgentState from checkpoint.

        Args:
            checkpoint_data: dict with 'train_state' key containing params
            runtime_state: RuntimeState to use (fresh counters, new rng)

        Returns:
            AgentState with restored params and fresh runtime.
        """
        saved_ts = checkpoint_data["train_state"]

        # Rebuild optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.adam(lr, eps=1e-5),
        )

        if hasattr(saved_ts, "params"):
            params = saved_ts.params
        elif isinstance(saved_ts, dict) and "params" in saved_ts:
            params = saved_ts["params"]
        else:
            params = saved_ts

        ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        return AgentState(train_state=ts, runtime=runtime_state)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _zero_carry(batch_size):
        return (jnp.zeros((batch_size, lstm_size)), jnp.zeros((batch_size, lstm_size)))

    # ------------------------------------------------------------------
    # Return Agent
    # ------------------------------------------------------------------
    return Agent(
        init=init,
        train=train,
        evaluate=evaluate,
        select_action=select_action,
        state_from_checkpoint=state_from_checkpoint,
        obs_space=obs_space,
        action_space=num_actions,
    )
