"""PPO-RNN agent — recurrent PPO with LSTM in JAX/Flax.

Architecture:
    Minimap      [B, 45, 45] int8      -> nn.Embed(14, 8)     [B, 45, 45, 8]
    Berry mask   [B, 45, 45] float32   (visible berries)       1 channel
    Target mask  [B, 45, 45] float32   (1.0 YES / 0.5 NO)      1 channel
    CoordConv    rel (row, col) in [-1, 1]                     2 channels
      concat   -> [B, 45, 45, 12]
      -> Conv(12->24, 3x3 VALID) -> ReLU -> MaxPool(2,2)   (45 -> 43 -> 21)
      -> Conv(24->32, 3x3 VALID) -> ReLU -> MaxPool(2,2)   (21 -> 19 -> 9)
      -> Conv(32->48, 3x3 VALID) -> ReLU                   (9  -> 7)
      -> Conv(48->24, 1x1) -> ReLU  (channel bottleneck)
      -> flatten (7*7*24 = 1176)
    Scalars [B, 6] -> Dense(32) -> ReLU
    Concat (1176 + 32 + 7 task_emb)
      -> Dense(128) -> ReLU -> Dense(128) -> ReLU
      -> OptimizedLSTMCell(128)
      -> actor Dense(128->8)  critic Dense(128->1)

Compared with the previous stack (19->6->3 avg-pool chain), the new CNN keeps
a 7x7 spatial output so per-tile positions of targets and berries are no
longer averaged away inside the convolutional trunk.
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
from cogniland.envs.env import NUM_TILE_CLASSES


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ActorCriticRNN(nn.Module):
    """CNN + MLP + LSTM actor-critic consuming a tile-index minimap plus
    dedicated berry / target overlay planes.

    Input:
        minimap:     [B, 45, 45] int8   — per-cell tile class (0..NUM_TILE_CLASSES-1)
        berry_mask:  [B, 45, 45] float  — 1 where a berry is visible
        target_mask: [B, 45, 45] float  — 1.0 YES target, 0.5 NO target, 0 elsewhere
        scalars:     [B, 6]
        task_emb:    [B, task_embedding_dim]
        carry:       (h, c) each [B, lstm_size]

    The minimap is embedded via ``nn.Embed`` so the network learns a dense
    vector per tile class.  The two overlay planes are appended as extra
    channels so berry / target positions are preserved per tile (they no
    longer clobber the underlying terrain).

    The CNN keeps a 7x7 spatial output (vs the old 3x3) so fine-grained
    positions of salient entities are retained up to the flatten step.

    Output:
        logits:    [B, num_actions]
        value:     [B]
        new_carry: (h, c)
    """
    num_actions: int = 8
    lstm_size: int = 128
    hidden_size: int = 128
    embed_dim: int = 8
    num_tile_classes: int = NUM_TILE_CLASSES
    task_embedding_dim: int = 7
    use_rnn: bool = True

    @nn.compact
    def __call__(self, minimap, berry_mask, target_mask, scalars, task_emb, carry):
        # -- Tile embedding: [B, 45, 45] int -> [B, 45, 45, embed_dim] float --
        mm = minimap.astype(jnp.int32)
        x = nn.Embed(
            num_embeddings=self.num_tile_classes,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.5),
        )(mm)

        # -- Overlay planes: berry and target -> 2 extra channels --
        berry = berry_mask.astype(x.dtype)[..., None]
        target = target_mask.astype(x.dtype)[..., None]
        x = jnp.concatenate([x, berry, target], axis=-1)

        # -- CoordConv: append normalized (rel_row, rel_col) in [-1, 1] --
        # Agent sits at the patch center. Two extra channels let the CNN see
        # "direction to this pixel" without re-learning translation from
        # scratch in a non-translation-invariant egocentric obs.
        B, H, W, _ = x.shape
        rr = jnp.linspace(-1.0, 1.0, H, dtype=x.dtype)
        cc = jnp.linspace(-1.0, 1.0, W, dtype=x.dtype)
        rr = jnp.broadcast_to(rr[None, :, None, None], (B, H, W, 1))
        cc = jnp.broadcast_to(cc[None, None, :, None], (B, H, W, 1))
        x = jnp.concatenate([x, rr, cc], axis=-1)
        # x: [B, 45, 45, embed_dim + 2 overlays + 2 coords]

        x = nn.Conv(features=24, kernel_size=(3, 3), padding="VALID",
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))   # 43 -> 21

        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID",
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)                                             # 21 -> 19
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))    # 19 -> 9

        x = nn.Conv(features=48, kernel_size=(3, 3), padding="VALID",
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)                                             # 9  -> 7

        # 1x1 bottleneck to keep the flatten dimension tame while preserving
        # the 7x7 spatial resolution.
        x = nn.Conv(features=24, kernel_size=(1, 1), padding="VALID",
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))                            # [B, 1176]

        s = nn.Dense(32, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(scalars)
        s = nn.relu(s)

        h = jnp.concatenate([x, s, task_emb], axis=-1)            # [B, 471]

        h = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        h = nn.relu(h)

        if self.use_rnn:
            lstm_cell = nn.OptimizedLSTMCell(
                features=self.lstm_size,
                kernel_init=nn.initializers.orthogonal(1.0),
                recurrent_kernel_init=nn.initializers.orthogonal(1.0),
            )
            new_carry, h = lstm_cell(carry, h)
        else:
            # MLP-only path: skip the LSTM, just thread carry through for
            # a shape-stable API. The carry is returned unchanged.
            new_carry = carry

        logits = nn.Dense(self.num_actions,
                          kernel_init=nn.initializers.orthogonal(0.01),
                          bias_init=nn.initializers.zeros)(h)
        value = nn.Dense(1,
                         kernel_init=nn.initializers.orthogonal(1.0),
                         bias_init=nn.initializers.zeros)(h)
        value = value.squeeze(-1)
        return logits, value, new_carry


# ---------------------------------------------------------------------------
# Transition storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs_minimap: jnp.ndarray    # [B, 45, 45] int8
    obs_berry: jnp.ndarray      # [B, 45, 45] float32 — visible-berry plane
    obs_target: jnp.ndarray     # [B, 45, 45] float32 — YES/NO target plane
    obs_scalars: jnp.ndarray    # [B, 6] float32
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
    embed_dim = int(getattr(agent_cfg, "embed_dim", 8))
    num_tile_classes = int(getattr(agent_cfg, "num_tile_classes", NUM_TILE_CLASSES))
    use_rnn = bool(getattr(agent_cfg, "use_rnn", True))

    num_envs = config.env.num_parallel_envs

    network = ActorCriticRNN(
        num_actions=num_actions,
        lstm_size=lstm_size,
        hidden_size=hidden_size,
        embed_dim=embed_dim,
        num_tile_classes=num_tile_classes,
        task_embedding_dim=task_embedding_dim,
        use_rnn=use_rnn,
    )

    # ------------------------------------------------------------------
    # JIT-compiled forward pass
    # ------------------------------------------------------------------
    @jax.jit
    def _forward(params, minimap, berry_mask, target_mask, scalars, task_emb, carry):
        """Run network forward pass. Returns (logits, value, new_carry)."""
        return network.apply(params, minimap, berry_mask, target_mask, scalars, task_emb, carry)

    @jax.jit
    def _sample_action(params, minimap, berry_mask, target_mask, scalars, task_emb, carry, rng):
        """Sample action from policy, return (action, log_prob, value, new_carry)."""
        logits, value, new_carry = network.apply(
            params, minimap, berry_mask, target_mask, scalars, task_emb, carry
        )
        dist = jax.random.categorical(rng, logits)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), dist]
        return dist, log_prob, value, new_carry

    @jax.jit
    def _deterministic_action(params, minimap, berry_mask, target_mask, scalars, task_emb, carry):
        """Greedy action from policy."""
        logits, value, new_carry = network.apply(
            params, minimap, berry_mask, target_mask, scalars, task_emb, carry
        )
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
            obs_minimap  [MB, 6, 45, 45]
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
                batch["obs_berry"],
                batch["obs_target"],
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
    # Fused PPO update: all ppo_epochs * num_minibatches gradient steps
    # run inside a single jitted scan. This eliminates the Python loop
    # and — crucially — the per-step ``float(metric)`` device->host syncs
    # that previously serialised every gradient step.
    # ------------------------------------------------------------------
    @jax.jit
    def _run_all_updates(train_state, flat_data, rng):
        flat_size = flat_data["action"].shape[0]
        mb_size = flat_size // num_minibatches

        def _mb_body(ts, mb_idx):
            mb = jax.tree.map(lambda v: v[mb_idx], flat_data)
            if normalize_advantages:
                adv = mb["advantage"]
                mb["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

            def loss_fn(params):
                logits, value, _ = network.apply(
                    params,
                    mb["obs_minimap"],
                    mb["obs_berry"],
                    mb["obs_target"],
                    mb["obs_scalars"],
                    mb["task_emb"],
                    (mb["carry_h"], mb["carry_c"]),
                )
                log_probs = jax.nn.log_softmax(logits)
                new_log_prob = log_probs[jnp.arange(logits.shape[0]), mb["action"]]
                ratio = jnp.exp(new_log_prob - mb["old_log_prob"])
                adv = mb["advantage"]
                pg1 = -adv * ratio
                pg2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_loss = jnp.maximum(pg1, pg2).mean()
                value_loss = 0.5 * ((value - mb["return_"]) ** 2).mean()
                probs = jax.nn.softmax(logits)
                entropy = -(probs * log_probs).sum(axis=-1).mean()
                total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                clipfrac = (jnp.abs(ratio - 1.0) > clip_eps).mean()
                approx_kl = (0.5 * (mb["old_log_prob"] - new_log_prob) ** 2).mean()
                return total_loss, {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "clipfrac": clipfrac,
                    "approx_kl": approx_kl,
                }

            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
            grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0), grads)
            ts = ts.apply_gradients(grads=grads)
            return ts, metrics

        def _epoch_body(ts, epoch_rng):
            perm = jax.random.permutation(epoch_rng, flat_size)
            mb_indices = perm.reshape(num_minibatches, mb_size)
            ts, epoch_metrics = jax.lax.scan(_mb_body, ts, mb_indices)
            return ts, epoch_metrics

        epoch_rngs = jax.random.split(rng, ppo_epochs)
        train_state, all_metrics = jax.lax.scan(
            _epoch_body, train_state, epoch_rngs
        )
        # Sum across (ppo_epochs, num_minibatches) to match the old
        # agg_metrics accumulation convention (sum, divided at the end).
        sum_metrics = jax.tree.map(lambda m: m.sum(), all_metrics)
        return train_state, sum_metrics

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def init(rng):
        rng, init_rng = jax.random.split(rng)
        mm_shape = obs_space["minimap"] if isinstance(obs_space, dict) else (45, 45)
        dummy_minimap = jnp.zeros((1,) + tuple(mm_shape), dtype=jnp.int32)
        dummy_berry = jnp.zeros((1,) + tuple(mm_shape), dtype=jnp.float32)
        dummy_target = jnp.zeros((1,) + tuple(mm_shape), dtype=jnp.float32)
        dummy_scalars = jnp.zeros((1, 6))
        dummy_task_emb = jnp.zeros((1, task_embedding_dim))
        dummy_carry = (jnp.zeros((1, lstm_size)), jnp.zeros((1, lstm_size)))

        params = network.init(
            init_rng,
            dummy_minimap, dummy_berry, dummy_target,
            dummy_scalars, dummy_task_emb, dummy_carry,
        )

        # Optimizer: Adam with gradient clipping. Use ``inject_hyperparams``
        # so the learning rate is a mutable field on opt_state — lets the
        # training loop anneal LR without rebuilding the optimizer (which
        # would wipe Adam's mu / nu moving averages).
        tx = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=1e-5),
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
        berry_mask = jnp.asarray(obs["berry_mask"])
        target_mask = jnp.asarray(obs["target_mask"])
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
                params, minimap, berry_mask, target_mask, scalars, task_emb, carry, act_rng
            )
        else:
            actions, value, new_carry = _deterministic_action(
                params, minimap, berry_mask, target_mask, scalars, task_emb, carry
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

        # LR annealing setup — use the configured total training frames as the
        # horizon so the anneal is global, not per-segment. ``train()`` is
        # called once per eval interval with a slice of frames; if we used the
        # slice as the horizon, LR would oscillate between lr and ~lr*(1-seg)
        # across segments instead of annealing smoothly to 0.
        total_frames_full = int(getattr(config.trainer, "num_train_frames", num_train_frames))
        total_updates = max(1, total_frames_full // (num_steps * num_envs))

        # Global update counter lives on the runtime so the anneal advances
        # across successive ``train()`` calls in the same run.
        n_updates_global_start = int(getattr(state.runtime, "train_steps", 0))
        n_updates_global_start = n_updates_global_start // max(1, ppo_epochs * num_minibatches)

        # Reset env — our env returns obs dict only (no info)
        obs = env.reset()
        carry = _zero_carry(num_envs)

        # Episode tracking
        all_episode_returns = []
        all_episode_lengths = []
        all_episode_flags = []
        all_episode_success = []
        all_episode_biomes = []

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
                berry_jax = jnp.asarray(obs["berry_mask"])
                target_jax = jnp.asarray(obs["target_mask"])
                scalars_jax = jnp.asarray(obs["scalars"])

                rng, act_rng = jax.random.split(rng)
                actions_jax, log_probs, values, new_carry = _sample_action(
                    train_state.params,
                    minimap_jax, berry_jax, target_jax,
                    scalars_jax, task_emb_jax, carry, act_rng,
                )

                actions_np = np.asarray(actions_jax)
                next_obs, rewards, dones, info = env.step(actions_np)

                storage.append(Transition(
                    obs_minimap=minimap_jax,
                    obs_berry=berry_jax,
                    obs_target=target_jax,
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
                    if "task_success" in info:
                        all_episode_success.append(info["task_success"].copy())
                    if "biome" in info:
                        all_episode_biomes.append(np.asarray(info["biome"]).copy())
                elif np.any(dones):
                    # Fallback: use dones directly
                    all_episode_flags.append(dones.copy())
                    if "episode_return" in info:
                        all_episode_returns.append(
                            np.where(dones, info["episode_return"], 0.0)
                        )
                    if "task_success" in info:
                        all_episode_success.append(info["task_success"].copy())
                    if "biome" in info:
                        all_episode_biomes.append(np.asarray(info["biome"]).copy())

                obs = next_obs
                global_step += num_envs
                frames_collected += num_envs

                if progress_bar is not None:
                    progress_bar.update(num_envs)

            # -- Bootstrap value for last obs --
            minimap_jax = jnp.asarray(obs["minimap"])
            berry_jax = jnp.asarray(obs["berry_mask"])
            target_jax = jnp.asarray(obs["target_mask"])
            scalars_jax = jnp.asarray(obs["scalars"])
            _, last_value, _ = _deterministic_action(
                train_state.params,
                minimap_jax, berry_jax, target_jax,
                scalars_jax, task_emb_jax, carry,
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
                "obs_minimap": jnp.concatenate([t.obs_minimap for t in storage], axis=0),   # [T*B, 45, 45]
                "obs_berry":   jnp.concatenate([t.obs_berry   for t in storage], axis=0),   # [T*B, 45, 45]
                "obs_target":  jnp.concatenate([t.obs_target  for t in storage], axis=0),   # [T*B, 45, 45]
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
            # Update the injected ``learning_rate`` hyperparam in-place.
            # This preserves Adam's mu / nu moving averages (unlike
            # TrainState.create, which would re-initialize opt_state).
            # ``InjectStatefulHyperparamsState`` is a NamedTuple, so we use
            # ``_replace`` (the NamedTuple API) rather than Flax's ``replace``.
            if anneal_lr and total_updates > 0:
                global_updates = n_updates_global_start + n_updates
                frac = 1.0 - (global_updates / total_updates)
                cur_lr = lr * max(frac, 0.0)
                cur_lr_jax = jnp.asarray(cur_lr, dtype=jnp.float32)

                def _maybe_set_lr(s):
                    if hasattr(s, "hyperparams") and "learning_rate" in s.hyperparams:
                        new_hp = {**s.hyperparams, "learning_rate": cur_lr_jax}
                        return s._replace(hyperparams=new_hp)
                    return s

                new_opt_state = tuple(_maybe_set_lr(s) for s in train_state.opt_state)
                train_state = train_state.replace(opt_state=new_opt_state)

            # -- PPO epochs (fused: ppo_epochs * num_minibatches grad steps
            # run inside a single jitted scan — eliminates Python loop and
            # per-step device->host syncs).
            rng, upd_rng = jax.random.split(rng)
            train_state, sum_metrics = _run_all_updates(train_state, flat_data, upd_rng)

            # One blocking read per segment (was 32 per segment before).
            for k in agg_metrics:
                agg_metrics[k] += float(sum_metrics[k])
            train_steps += ppo_epochs * num_minibatches

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
        if all_episode_success:
            episode_info["task_success"] = np.concatenate(all_episode_success, axis=0)
        if all_episode_biomes:
            episode_info["biome"] = np.concatenate(all_episode_biomes, axis=0)
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
        all_episode_success = []
        all_episode_biomes = []
        frames = 0

        while frames < num_eval_frames:
            minimap_jax = jnp.asarray(obs["minimap"])
            berry_jax = jnp.asarray(obs["berry_mask"])
            target_jax = jnp.asarray(obs["target_mask"])
            scalars_jax = jnp.asarray(obs["scalars"])

            actions_jax, _, new_carry = _deterministic_action(
                params,
                minimap_jax, berry_jax, target_jax,
                scalars_jax, task_emb_jax, carry,
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
                if "task_success" in info:
                    all_episode_success.append(info["task_success"].copy())
                if "biome" in info:
                    all_episode_biomes.append(np.asarray(info["biome"]).copy())

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
        if all_episode_success:
            episode_info["task_success"] = np.concatenate(all_episode_success, axis=0)
        if all_episode_biomes:
            episode_info["biome"] = np.concatenate(all_episode_biomes, axis=0)

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
