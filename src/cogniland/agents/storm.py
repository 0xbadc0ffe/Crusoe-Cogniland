"""
STORM Agent - Transformer-based world model RL.

This implementation follows the same modular architecture as DreamerV3:
- WorldModel protocol (STORMWorldModel with TSSM)
- Policy protocol (MLPPolicy)
- AgentState dataclasses
- Functional actor-critic training (imagine_trajectory, imag_loss, repl_loss)

Key differences from DreamerV3:
- Uses TSSM (Transformer State-Space Model) instead of RSSM
- Uses BatchNorm in CNN layers instead of RMSNorm
- KV-cache for efficient autoregressive generation
"""

from typing import Any, Dict, Tuple, Optional
from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from cogniland.agents.agent import Agent
from cogniland.agents.registry import register_agent
from cogniland.agents.commons import ReservoirReplayBuffer
from cogniland.agents.commons.normalizers import init_normalizer
from cogniland.agents.commons.preprocessing import normalize_image
from cogniland.agents.commons.optimizer import storm_optimizer
from cogniland.agents.commons.visualization import video_predict
from cogniland.agents.utils import RatioTracker
from cogniland.agents.world_models.storm.world_model import STORMWorldModel
from cogniland.agents.world_models.storm.state import STORMParams, STORMTrainState
from cogniland.agents.state import (
    AgentState,
    WorldModelState,
    WorldModelParams,
    PolicyParams,
    RuntimeState
)
from cogniland.agents.policy.mlp import MLPPolicy
from cogniland.agents.policy.actor_critic import imagine_trajectory, imag_loss, repl_loss
from cogniland.agents.utils import sg


def temporal_contrastive_loss(logits: jnp.ndarray, is_terminal: jnp.ndarray, temp: float = 0.1) -> jnp.ndarray:
    """InfoNCE loss enforcing temporal coherence in latent space."""
    # Softmax and Flatten: [B, T, S, C] -> [B, T, F]
    x = jax.nn.softmax(logits, axis=-1)
    b, t = x.shape[:2]
    x = x.reshape(b, t, -1)
    
    # Normalize features
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

    # Pairs: (t) -> (t+1)
    anchor = x[:, :-1]
    positive = x[:, 1:]
    
    # Negative: Shift time by 4 steps
    negative = jnp.roll(positive, shift=4, axis=1)

    # Similarities
    sim_pos = jnp.sum(anchor * positive, axis=-1) / temp
    sim_neg = jnp.sum(anchor * negative, axis=-1) / temp

    # InfoNCE
    concat_sim = jnp.stack([sim_pos, sim_neg], axis=-1)
    log_prob = sim_pos - jax.nn.logsumexp(concat_sim, axis=-1)

    # Mask terminal steps
    mask = 1.0 - is_terminal[:, :-1]
    return -jnp.sum(log_prob * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@register_agent('storm')
def make_storm(config: OmegaConf, obs_space: Dict, act_space: int) -> Agent:
    """Factory function that creates a STORM agent.

    STORM agent using modular architecture with Transformer-based world model.

    Architecture:
        - World Model: STORMWorldModel (encoder, decoder, TSSM, reward, cont)
        - Policy: MLPPolicy (actor, critic)
        - Optimizer: Single laprop optimizer for all trainable params
        - Training: Combines world model + imagination + replay losses

    Note: STORM uses STORMTrainState which handles slow critic and normalizers with EMA.

    Args:
        config: Configuration object (OmegaConf)
        obs_space: Observation space dict (e.g., {'image': (64, 64, 3)})
        act_space: Number of discrete actions

    Returns:
        Agent instance with STORM policy
    """
    cfg = config.agent
    obs_modalities = list(obs_space.keys())
    action_space = act_space

    # Get number of parallel environments from config
    num_envs = config.env.get('num_parallel_envs', 16)

    # Detect observation type for symbolic handling
    navix_obs_type = config.env.get('navix_observation_type', 'rgb_first_person')

    # Initialize replay buffer with RAW observation shapes
    # (buffers store raw observations, preprocessing applied at training time)
    raw_obs_space = config.get('raw_obs_space', obs_space)
    buffer = ReservoirReplayBuffer(
        capacity=int(cfg.replay.capacity),
        obs_shapes=raw_obs_space,
        action_dim=act_space,
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        num_envs=num_envs,
    )

    # Create STORM world model with configs from OmegaConf
    encoder_config = OmegaConf.to_container(cfg.model.enc.simple, resolve=True)
    decoder_config = OmegaConf.to_container(cfg.model.dec.simple, resolve=True)

    # For symbolic observations, force MLP processing via flatten_keys
    if navix_obs_type in ['symbolic_first_person', 'symbolic']:
        encoder_config['flatten_keys'] = ('image',)
        decoder_config['flatten_keys'] = ('image',)

    world_model = STORMWorldModel(
        obs_space=obs_space,
        action_space=act_space,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        tssm_config=OmegaConf.to_container(cfg.model.dyn.tssm, resolve=True),
        reward_config=OmegaConf.to_container(cfg.model.rewhead, resolve=True),
        cont_config=OmegaConf.to_container(cfg.model.conthead, resolve=True),
    )

    # Create policy (normalizers will be set in init() with separate configs)
    policy = MLPPolicy(
        action_space=act_space,
        actor_config=OmegaConf.to_container(cfg.model.actor, resolve=True),
        critic_config=OmegaConf.to_container(cfg.model.critic, resolve=True),
    )

    # Create STORM's multi-transform optimizer with separate WM and AC configs
    wm_config = OmegaConf.to_container(cfg.model.opt.wm, resolve=True)
    ac_config = OmegaConf.to_container(cfg.model.opt.ac, resolve=True)
    optimizer = storm_optimizer(wm_config, ac_config)

    # Initialize training ratio tracker (for train_ratio logic)
    batch_steps = cfg.batch_size * cfg.batch_length
    train_ratio_tracker = RatioTracker(cfg.train_ratio / batch_steps)

    # Training hyperparameters from config
    batch_length = cfg.batch_length
    pretrain = cfg.pretrain
    imag_horizon = cfg.model.imag_length
    imag_context = cfg.model.imag_context
    slow_critic_rate = cfg.model.slowvalue.rate
    grad_checkpoint = cfg.model.grad_checkpoint
    imag_last = cfg.model.imag_last
    ac_grads = cfg.model.ac_grads
    contdisc = cfg.model.imag_loss.contdisc
    horizon = cfg.model.imag_loss.horizon

    # Experimental loss extensions
    temporal_scale = cfg.model.temp_loss.scale

    # Whether to use replay value loss (stored for JIT - determined at trace time)
    use_repval_loss = cfg.model.repval_loss

    # Imagination and replay loss configs (flattened for direct use)
    imag_config = OmegaConf.to_container(cfg.model.imag_loss, resolve=True)
    repl_config = None
    if use_repval_loss:
        repl_config = OmegaConf.to_container(cfg.model.repl_loss, resolve=True)

    # Loss scales from config
    loss_scales = OmegaConf.to_container(cfg.model.loss_scales, resolve=True)

    # Visualization config
    viz_enabled = cfg.get('visualization', {}).get('enabled', False)
    viz_interval = cfg.get('visualization', {}).get('interval', 500)
    viz_context = cfg.get('visualization', {}).get('context', 5)
    viz_log_prefix = cfg.get('visualization', {}).get('log_prefix', 'storm_reconstruction')

    def _loss_fn(
        params: STORMParams,
        slow_critic: Any,
        normalizers: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Combined loss function for STORM."""
        wm_params = params.wm
        # Reconstruct PolicyParams with slow_critic and normalizers
        policy_params = PolicyParams(
            actor=params.policy.actor,
            critic=params.policy.critic,
            slow_critic=slow_critic,
            normalizers=normalizers,
        )

        rng_wm, rng_imag = jax.random.split(rng)

        # Unpack batch - reconstruct obs dict from flat keys
        obs_dict = {}
        for key in batch.keys():
            if key.startswith('obs_'):
                modality = key[4:]  # Remove 'obs_' prefix
                obs_dict[modality] = batch[key]

        actions = batch['action']  # [B, T]
        rewards = batch['reward']  # [B, T]
        is_first = batch['is_first']  # [B, T]
        is_terminal = batch['is_terminal']  # [B, T]

        batch_size_local = actions.shape[0]
        seq_len = actions.shape[1]

        losses = {}
        metrics = {}

        # Encode observations - flatten batch and time dims, normalize images
        obs_flat = {}
        for key, value in obs_dict.items():
            obs_data = value.reshape(batch_size_local * seq_len, *value.shape[2:])
            if key in obs_space and len(obs_space[key]) == 3:
                # Normalize uint8 images to float32 (resize done by environment)
                obs_data = normalize_image(obs_data)
            obs_flat[key] = obs_data

        # Encode all observations
        embeds_flat = world_model.encode(wm_params, obs_flat)
        embeds = embeds_flat.reshape(batch_size_local, seq_len, -1)

        # Process sequence through world model using observe_sequence
        initial_state = world_model.initial_state(wm_params, batch_size_local)

        states, seq_info = world_model.observe_sequence(
            wm_params=wm_params,
            initial_state=initial_state,
            actions=actions,
            embeds=embeds,
            is_first=is_first,
            training=True,
            rng=rng_wm
        )

        # Extract losses: [B, T] -> scalar
        dyn_loss = seq_info['dyn_loss'].mean()
        rep_loss = seq_info['rep_loss'].mean()

        # New temporal loss to constrain the latent space
        temp_loss_val = temporal_contrastive_loss(states.logits, is_terminal)

        # Extract features from states
        feat = world_model.get_feat(wm_params, states)  # [B, T, F]
        feat_flat = feat.reshape(batch_size_local * seq_len, -1)

        # Flatten states for decoding
        states_flat = states.replace(
            stoch=states.stoch.reshape(batch_size_local * seq_len, *states.stoch.shape[2:]),
            hidden=states.hidden.reshape(batch_size_local * seq_len, *states.hidden.shape[2:]),
            logits=states.logits.reshape(batch_size_local * seq_len, *states.logits.shape[2:]),
        )

        # Decode observations from posterior states
        recons = world_model.decode(wm_params, states_flat)

        # Compute reconstruction loss
        rec_loss_list = []
        for key in obs_dict.keys():
            if key in recons:
                target_flat = obs_flat[key]
                rec_dist = recons[key]
                rec_loss_flat = -rec_dist.log_prob(sg(target_flat))
                # Sum over spatial dimensions if present (e.g., for discrete image observations)
                # OneHotDist returns (B*T, H, W) for categorical images, need to sum to (B*T,)
                if rec_loss_flat.ndim > 1:
                    rec_loss_flat = rec_loss_flat.sum(axis=tuple(range(1, rec_loss_flat.ndim)))
                rec_loss_list.append(rec_loss_flat.reshape(batch_size_local, seq_len))

        rec_loss = sum(rec_loss_list) if rec_loss_list else jnp.zeros((batch_size_local, seq_len))

        # Predict rewards
        rew_pred = world_model.predict_reward(wm_params, feat_flat)
        rew_target = rewards.reshape(-1)
        rew_loss_flat = -rew_pred.log_prob(sg(rew_target))
        rew_loss = rew_loss_flat.reshape(batch_size_local, seq_len)

        # Predict continuation
        cont_pred = world_model.predict_continuation(wm_params, feat_flat)
        cont_target = 1.0 - is_terminal

        if contdisc:
            cont_target = cont_target * (1 - 1 / horizon)

        cont_loss_flat = -cont_pred.log_prob(sg(cont_target.reshape(-1)))
        con_loss = cont_loss_flat.reshape(batch_size_local, seq_len)

        # Store world model losses
        losses['rec'] = rec_loss
        losses['rew'] = rew_loss
        losses['con'] = con_loss
        losses['dyn'] = jnp.ones((batch_size_local, seq_len)) * dyn_loss
        losses['rep'] = jnp.ones((batch_size_local, seq_len)) * rep_loss
        losses['temp'] = jnp.ones((batch_size_local, seq_len)) * temp_loss_val * temporal_scale

        # Imagination Losses
        K = min(imag_last or seq_len, seq_len)
        if K == 0:
            K = seq_len

        start_stoch = states.stoch[:, -K:]
        start_hidden = states.hidden[:, -K:]
        start_logits = states.logits[:, -K:]

        # Handle KV caches
        if (states.kv_cache and len(states.kv_cache) > 0 and
                states.kv_cache[0]['keys'].shape[1] > 0):
            context_caches = []
            for layer_cache in states.kv_cache:
                layer_keys = layer_cache['keys'][:, -K:]
                layer_values = layer_cache['values'][:, -K:]

                windowed_keys = world_model.tssm.extract_context_windows(
                    layer_keys, imag_context
                )
                windowed_values = world_model.tssm.extract_context_windows(
                    layer_values, imag_context
                )

                max_cache_len = world_model.tssm.max_cache_length
                pad_len = max_cache_len - imag_context
                if pad_len > 0:
                    pad_shape_keys = (windowed_keys.shape[0], pad_len) + windowed_keys.shape[2:]
                    pad_shape_values = (windowed_values.shape[0], pad_len) + windowed_values.shape[2:]
                    # FIX: Context MUST be at the BEGINNING (positions 0 to imag_context-1)
                    # so that the attention mask (valid positions 0 to step) covers the actual data.
                    # Previously, zeros were first and context was at the end, making context
                    # inaccessible since the mask only enabled the zero-filled positions.
                    windowed_keys = jnp.concatenate([
                        windowed_keys,  # Context at positions 0 to imag_context-1
                        jnp.zeros(pad_shape_keys, dtype=windowed_keys.dtype)
                    ], axis=1)
                    windowed_values = jnp.concatenate([
                        windowed_values,  # Context at positions 0 to imag_context-1
                        jnp.zeros(pad_shape_values, dtype=windowed_values.dtype)
                    ], axis=1)

                step_counter = jnp.full((batch_size_local * K,), imag_context, dtype=jnp.int32)
                context_caches.append({
                    'keys': windowed_keys,
                    'values': windowed_values,
                    'step': step_counter
                })
        else:
            max_cache_len = world_model.tssm.max_cache_length
            step_counter = jnp.zeros((batch_size_local * K,), dtype=jnp.int32)
            context_caches = [
                {'keys': jnp.zeros((batch_size_local * K, max_cache_len,
                                   world_model.tssm.hidden_dim), dtype=start_hidden.dtype),
                 'values': jnp.zeros((batch_size_local * K, max_cache_len,
                                     world_model.tssm.hidden_dim), dtype=start_hidden.dtype),
                 'step': step_counter}
                for _ in range(world_model.tssm.num_layers)
            ]

        start_cache_step = states.cache_step[:, -K:]
        start_cache_step_flat = start_cache_step.reshape(batch_size_local * K)

        start_states_flat = states.replace(
            stoch=start_stoch.reshape(batch_size_local * K, *start_stoch.shape[2:]),
            hidden=start_hidden.reshape(batch_size_local * K, *start_hidden.shape[2:]),
            logits=start_logits.reshape(batch_size_local * K, *start_logits.shape[2:]),
            kv_cache=context_caches,
            cache_step=start_cache_step_flat
        )

        def policy_fn(features, rng_action):
            rng_action, rng_sample = jax.random.split(rng_action)
            action_dist = policy.apply_actor(
                policy_params.actor, features, training=False, rng=None
            )
            action_indices = action_dist.sample(seed=rng_sample)
            actions_onehot = jax.nn.one_hot(action_indices, action_space)
            return actions_onehot, rng_action

        imag_features, imag_actions, imag_rewards, imag_conts = imagine_trajectory(
            wm=world_model,
            wm_params=wm_params,
            initial_state=start_states_flat,
            policy_fn=policy_fn,
            horizon=imag_horizon,
            rng=rng_imag,
            grad_checkpoint=grad_checkpoint,
            ac_grads=ac_grads,
        )

        actor_loss, critic_loss, imag_outputs, imag_metrics = imag_loss(
            policy=policy,
            policy_params=policy_params,
            features=imag_features,
            actions=imag_actions,
            rewards=imag_rewards,
            continuations=imag_conts,
            config=imag_config,
        )

        losses['policy'] = actor_loss
        losses['value'] = critic_loss

        # Replay Value Loss
        feat_sg = sg(feat)
        repl_feat = feat_sg[:, -K:]
        repl_rewards = rewards[:, -K:]
        repl_conts = 1.0 - is_terminal[:, -K:]
        repl_last = batch.get('is_last', jnp.zeros_like(is_terminal))[:, -K:]

        imag_returns = imag_outputs['returns']
        boot_returns = imag_returns[:, 0].reshape(batch_size_local, K)

        repl_feat_flat = repl_feat.reshape(batch_size_local * K, -1)
        repl_rewards_flat = repl_rewards.reshape(batch_size_local * K)
        repl_conts_flat = repl_conts.reshape(batch_size_local * K)
        repl_last_flat = repl_last.reshape(batch_size_local * K)
        boot_flat = boot_returns.reshape(batch_size_local * K)

        updated_policy_params = PolicyParams(
            actor=policy_params.actor,
            critic=policy_params.critic,
            slow_critic=policy_params.slow_critic,
            normalizers={
                'return': imag_outputs['retnorm_state'],
                'value': imag_outputs['valnorm_state'],
                'advantage': imag_outputs['advnorm_state'],
            },
        )

        if use_repval_loss:
            repl_critic_loss, repl_outputs, repl_metrics = repl_loss(
                policy=policy,
                policy_params=updated_policy_params,
                features=repl_feat_flat[None, :],
                rewards=repl_rewards_flat[None, :],
                continuations=repl_conts_flat[None, :],
                last=repl_last_flat[None, :],
                boot=boot_flat[None, :],
                config=repl_config,
            )
            losses['repval'] = repl_critic_loss
            metrics.update(repl_metrics)
            final_valnorm_state = repl_outputs['valnorm_state']
        else:
            final_valnorm_state = imag_outputs['valnorm_state']

        # Combine Losses
        total_loss = sum(
            loss_scales.get(name, 1.0) * loss.mean()
            for name, loss in losses.items()
        )

        for name, loss_val in losses.items():
            metrics[f'loss/{name}'] = loss_val.mean()

        metrics.update(imag_metrics)

        final_normalizers = {
            'return': imag_outputs['retnorm_state'],
            'value': final_valnorm_state,
            'advantage': imag_outputs['advnorm_state'],
        }

        aux = {
            'losses': losses,
            'normalizers': final_normalizers,
            'metrics': metrics,
        }

        return total_loss, aux

    def _train_step_core(
        train_state: STORMTrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey,
    ) -> Tuple[STORMTrainState, Dict]:
        """Core training step (JIT-compiled)."""
        (total_loss, aux), grads = jax.value_and_grad(
            _loss_fn, has_aux=True
        )(train_state.params, train_state.slow_critic, train_state.normalizers, batch, rng)

        new_train_state = train_state.apply_gradients(grads=grads)
        new_train_state = new_train_state.replace(normalizers=aux['normalizers'])

        metrics = {
            'total_loss': total_loss,
            **aux.get('metrics', {}),
        }

        return new_train_state, metrics

    # JIT compile training step
    _train_step_jit = jax.jit(_train_step_core)

    def _select_action_jit(
        params: STORMParams,
        wm_state: WorldModelState,
        obs_dict: Dict[str, jnp.ndarray],
        prev_action: jnp.ndarray,
        is_first: jnp.ndarray,
        training: bool,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, WorldModelState, jax.random.PRNGKey]:
        """JIT-compiled action selection logic."""
        batch_size_local = next(iter(obs_dict.values())).shape[0]

        init_state = world_model.initial_state(params.wm, batch_size_local)

        def reset_where_first(new, old):
            if new is None and old is None:
                return None
            elif new is None:
                return old
            elif old is None:
                return new
            ndim = old.ndim
            mask = is_first
            for _ in range(ndim - 1):
                mask = mask[..., None]
            return jnp.where(mask, new, old)

        wm_state = jax.tree.map(
            reset_where_first, init_state, wm_state, is_leaf=lambda x: x is None)

        # Normalize image modalities (resize done by environment)
        obs_processed = {}
        for key, value in obs_dict.items():
            if key in obs_space and len(obs_space[key]) == 3:
                value = normalize_image(value)
            obs_processed[key] = value

        embed = world_model.encode(params.wm, obs_processed)

        masked_prev_action = jnp.where(
            is_first[..., None],
            jnp.zeros_like(prev_action),
            prev_action,
        )

        is_first_dummy = jnp.zeros((batch_size_local,), dtype=bool)
        new_wm_state, _ = world_model.observe(
            params.wm, wm_state, masked_prev_action, embed,
            is_first_dummy, training=False, rng=None
        )

        features = world_model.get_feat(params.wm, new_wm_state)

        rng, rng_act = jax.random.split(rng)
        action_dist = policy.apply_actor(
            params.policy.actor, features, training=training, rng=None
        )

        if training:
            action_indices = action_dist.sample(seed=rng_act)
        else:
            action_indices = action_dist.mode()

        action_indices = jnp.reshape(action_indices, (batch_size_local,))

        return action_indices, new_wm_state, rng

    # Pre-compile select_action for train and eval modes
    _select_action_jit_compiled = {
        'train': jax.jit(_select_action_jit, static_argnums=(5,)),
        'eval': jax.jit(_select_action_jit, static_argnums=(5,)),
    }

    def init(rng: jax.random.PRNGKey) -> AgentState:
        """Initialize all agent parameters and state."""
        rng, rng_wm, rng_policy, rng_buffer, rng_runtime = jax.random.split(rng, 5)

        # Initialize replay buffer
        buffer_state = buffer.init(rng_buffer)

        # Initialize world model
        wm_params = world_model.init_params(rng_wm, obs_space, action_space)

        # Get initial world model state (batch size = 1 for single env inference)
        wm_state = world_model.initial_state(wm_params, batch_size=1)

        # Get feature dimension from TSSM config for policy initialization
        tssm_config = cfg.model.dyn.tssm
        hidden_dim = tssm_config.get('hidden_dim', 512)
        stoch_size = tssm_config.get('stoch', 32)
        num_classes_cfg = tssm_config.get('classes', 32)
        feat_dim = hidden_dim + stoch_size * num_classes_cfg

        # Initialize policy
        policy_params_init = policy.init_params(rng_policy, feat_dim, action_space)

        # Create normalizers with separate configs
        normalizers = {
            'return': init_normalizer(
                **OmegaConf.to_container(cfg.model.retnorm, resolve=True)),
            'value': init_normalizer(
                **OmegaConf.to_container(cfg.model.valnorm, resolve=True)),
            'advantage': init_normalizer(
                **OmegaConf.to_container(cfg.model.advnorm, resolve=True)),
        }

        # Create composite STORMParams structure
        policy_params = PolicyParams(
            actor=policy_params_init.actor,
            critic=policy_params_init.critic,
            slow_critic=None,
            normalizers={},
        )

        params = STORMParams(wm=wm_params, policy=policy_params)

        # Create STORMTrainState
        train_state = STORMTrainState.create(
            apply_fn=None,
            params=params,
            tx=optimizer,
            slow_critic=policy_params_init.slow_critic,
            normalizers=normalizers,
            slow_critic_rate=slow_critic_rate,
        )

        # Create RuntimeState
        runtime = RuntimeState(
            buffer_state=buffer_state,
            wm_state=wm_state,
            step=jnp.array(0, dtype=jnp.int32),
            train_steps=jnp.array(0, dtype=jnp.int32),
            rng=rng_runtime,
            current_num_actions=jnp.array(act_space),
        )

        return AgentState(
            train_state=train_state,
            runtime=runtime,
        )

    def train_step(
        agent_state: AgentState,
        batch: Dict[str, jnp.ndarray],
    ) -> Tuple[AgentState, Dict]:
        """Perform one training step."""
        train_state_local = agent_state.train_state
        runtime = agent_state.runtime

        rng, rng_train = jax.random.split(runtime.rng)

        new_train_state, metrics = _train_step_jit(
            train_state=train_state_local,
            batch=batch,
            rng=rng_train,
        )

        new_runtime = RuntimeState(
            buffer_state=runtime.buffer_state,
            wm_state=runtime.wm_state,
            step=runtime.step,
            train_steps=runtime.train_steps + 1,
            rng=rng,
            current_num_actions=runtime.current_num_actions,
            current_task_id=runtime.current_task_id,
        )

        new_agent_state = AgentState(
            train_state=new_train_state,
            runtime=new_runtime,
        )

        return new_agent_state, metrics

    def act(
        agent_state: AgentState,
        obs_dict: Dict[str, jnp.ndarray],
        prev_action: jnp.ndarray,
        is_first: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, AgentState]:
        """Select action given observation."""
        params = agent_state.train_state.params
        runtime = agent_state.runtime
        wm_state = runtime.wm_state

        batch_size_local = next(iter(obs_dict.values())).shape[0]

        if wm_state is None or jax.tree.leaves(wm_state)[0].shape[0] != batch_size_local:
            wm_state = world_model.initial_state(params.wm, batch_size_local)

        mode = 'train' if training else 'eval'
        action_indices, new_wm_state, new_rng = _select_action_jit_compiled[mode](
            params,
            wm_state,
            obs_dict,
            prev_action,
            is_first,
            training,
            rng,
        )

        new_runtime = RuntimeState(
            buffer_state=runtime.buffer_state,
            wm_state=new_wm_state,
            step=runtime.step,
            train_steps=runtime.train_steps,
            rng=new_rng,
            current_num_actions=runtime.current_num_actions,
            current_task_id=runtime.current_task_id,
        )

        new_agent_state = AgentState(
            train_state=agent_state.train_state,
            runtime=new_runtime,
        )

        return action_indices, new_agent_state

    def select_action(
        state: AgentState,
        obs: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey,
        is_first: Optional[jnp.ndarray] = None,
        prev_action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> Tuple[jnp.ndarray, AgentState]:
        """Select actions for given observations using the learned policy."""
        batch_size_local = next(iter(obs.values())).shape[0]

        if prev_action is None:
            prev_action = jnp.zeros((batch_size_local, action_space), dtype=jnp.float32)

        if is_first is None:
            is_first = jnp.zeros(batch_size_local, dtype=bool)
        else:
            is_first = jnp.squeeze(is_first)

        action_indices, new_state = act(
            state, obs, prev_action, is_first, rng, training=training
        )

        return action_indices, new_state

    def train(
        state: AgentState,
        env: Any,
        rng: jax.random.PRNGKey,
        num_train_frames: int,
        progress_bar=None,
        checkpoint_callback=None,
        task_ids=None,
    ) -> Tuple[AgentState, Dict[str, Any]]:
        """Train the agent for a specified number of frames."""
        metrics = defaultdict(list)
        metrics['episode_info'] = {
            'returned_episode_returns': [],
            'returned_episode_lengths': [],
            'returned_episode': [],
            'task_success': [],
            'timestep': [],
        }

        frames_collected = 0

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, env.num_envs)
        env_state = env.reset(reset_rngs)

        prev_actions = jnp.zeros((env.num_envs, action_space))

        while frames_collected < num_train_frames:
            obs = env_state.env_state.observation
            reward = env_state.env_state.reward
            done = env_state.env_state.is_done()
            terminal = env_state.env_state.is_termination()

            is_first_flags = (env_state.env_state.t == 0) & (~done)
            reward = jnp.where(is_first_flags, 0.0, reward)

            replay_transition = {}
            for modality in obs_space.keys():
                obs_data = obs[modality]
                if len(obs_space[modality]) == 3:
                    if obs_data.dtype != jnp.uint8:
                        obs_data = (obs_data * 255).astype(jnp.uint8)
                replay_transition[f'obs_{modality}'] = obs_data

            masked_prev_actions = jnp.where(
                is_first_flags[..., None],
                jnp.zeros_like(prev_actions),
                prev_actions,
            )

            replay_transition.update({
                'action': masked_prev_actions,
                'reward': reward,
                'is_first': is_first_flags,
                'is_last': done,
                'is_terminal': terminal,
            })

            if task_ids is not None:
                replay_transition['task_id'] = np.array(task_ids, dtype=np.int32)

            new_buffer_state = buffer.add_batch(state.runtime.buffer_state, replay_transition)
            state = AgentState(
                train_state=state.train_state,
                runtime=RuntimeState(
                    buffer_state=new_buffer_state,
                    wm_state=state.runtime.wm_state,
                    step=state.runtime.step + env.num_envs,
                    train_steps=state.runtime.train_steps,
                    rng=state.runtime.rng,
                    current_num_actions=state.runtime.current_num_actions,
                    current_task_id=state.runtime.current_task_id,
                ),
            )

            frames_collected += env.num_envs

            if progress_bar is not None:
                progress_bar.update(env.num_envs)

            rng, action_rng = jax.random.split(rng)
            action_indices, state = select_action(
                state, obs, action_rng,
                is_first=is_first_flags,
                prev_action=prev_actions,
                training=True
            )

            actions_onehot = jax.nn.one_hot(action_indices, action_space)
            env_state = env.step(env_state, action_indices)

            done_next = env_state.env_state.is_done()
            if jnp.any(done_next):
                for idx in jnp.where(done_next)[0]:
                    metrics['episode_info']['returned_episode_returns'].append(
                        float(env_state.returned_episode_returns[idx])
                    )
                    metrics['episode_info']['returned_episode_lengths'].append(
                        int(env_state.returned_episode_lengths[idx])
                    )
                    metrics['episode_info']['returned_episode'].append(True)
                    metrics['episode_info']['task_success'].append(
                        float(env_state.task_success[idx])
                    )
                    metrics['episode_info']['timestep'].append(
                        int(env_state.timestep[idx]))

            prev_actions = actions_onehot

            buf_stats = buffer.stats(state.runtime.buffer_state)
            if state.runtime.step >= pretrain and \
               buf_stats.get('valid_timesteps', 0) >= batch_length:
                num_updates = train_ratio_tracker(state.runtime.step)

                for _ in range(num_updates):
                    rng, sample_rng = jax.random.split(rng)
                    batch_data = buffer.sample(state.runtime.buffer_state, sample_rng)

                    state, step_metrics = train_step(state, batch_data)

                    for key, value in step_metrics.items():
                        if key != 'episode_info':
                            metrics[key].append(float(value))

                    if checkpoint_callback is not None:
                        checkpoint_callback.on_train_step_end(
                            agent_state=state,
                            step=int(state.runtime.train_steps),
                            metrics=None,
                        )

                # Visualization
                if viz_enabled:
                    if state.runtime.train_steps % viz_interval == 0:
                        rng, viz_rng = jax.random.split(rng)
                        batch_viz = buffer.sample(state.runtime.buffer_state, viz_rng)
                        video_predict(
                            state=state,
                            batch=batch_viz,
                            world_model=world_model,
                            obs_modalities=obs_modalities,
                            step_count=int(state.runtime.train_steps),
                            context=viz_context,
                            log_prefix=viz_log_prefix,
                            navix_obs_type=navix_obs_type,
                        )

        state = AgentState(
            train_state=state.train_state,
            runtime=RuntimeState(
                buffer_state=state.runtime.buffer_state,
                wm_state=state.runtime.wm_state,
                step=state.runtime.step,
                train_steps=state.runtime.train_steps,
                rng=rng,
                current_num_actions=state.runtime.current_num_actions,
                current_task_id=state.runtime.current_task_id,
            ),
        )

        buf_stats = buffer.stats(state.runtime.buffer_state)
        metrics['buffer_size'] = buf_stats['size']
        metrics['buffer_total_steps'] = buf_stats['total_steps']

        metrics_aggregated = {}
        for key, value in metrics.items():
            if key == 'episode_info':
                continue
            elif isinstance(value, list):
                if len(value) > 0:
                    metrics_aggregated[key] = float(np.mean(value))
            else:
                metrics_aggregated[key] = value

        metrics_aggregated['episode_info'] = _format_episode_info(metrics['episode_info'])

        return state, metrics_aggregated

    def evaluate(
        state: AgentState,
        env: Any,
        rng: jax.random.PRNGKey,
        num_eval_frames: int,
        progress_bar=None,
        task_ids=None,
    ) -> Dict[str, Any]:
        """Evaluate the agent."""
        metrics = {
            'episode_info': {
                'returned_episode_returns': [],
                'returned_episode_lengths': [],
                'returned_episode': [],
                'task_success': [],
                'timestep': [],
            },
            'frames': 0,
        }

        frames_evaluated = 0

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, env.num_envs)
        env_state = env.reset(reset_rngs)

        prev_actions = jnp.zeros((env.num_envs, action_space))

        step_count = 0
        progress_update_interval = 100

        while frames_evaluated < num_eval_frames:
            obs = env_state.env_state.observation
            done = env_state.env_state.is_done()
            is_first_flags = (env_state.env_state.t == 0) & (~done)

            rng, action_rng = jax.random.split(rng)
            action_indices, state = select_action(
                state, obs, action_rng,
                is_first=is_first_flags,
                prev_action=prev_actions,
                training=False
            )

            actions_onehot = jax.nn.one_hot(action_indices, action_space)

            env_state = env.step(env_state, action_indices)
            frames_evaluated += env.num_envs
            step_count += 1

            if progress_bar is not None and step_count % progress_update_interval == 0:
                progress_bar.n = frames_evaluated
                progress_bar.refresh()

            done_next = env_state.env_state.is_done()
            if jnp.any(done_next):
                for idx in jnp.where(done_next)[0]:
                    metrics['episode_info']['returned_episode_returns'].append(
                        float(env_state.returned_episode_returns[idx])
                    )
                    metrics['episode_info']['returned_episode_lengths'].append(
                        int(env_state.returned_episode_lengths[idx])
                    )
                    metrics['episode_info']['returned_episode'].append(True)
                    metrics['episode_info']['task_success'].append(
                        float(env_state.task_success[idx])
                    )
                    metrics['episode_info']['timestep'].append(
                        int(env_state.timestep[idx]))

            prev_actions = actions_onehot

        if progress_bar is not None:
            progress_bar.n = frames_evaluated
            progress_bar.refresh()

        metrics['frames'] = frames_evaluated
        metrics['episode_info'] = _format_episode_info(metrics['episode_info'])

        return metrics

    def _format_episode_info(episode_info: Dict[str, list]) -> Optional[Dict[str, np.ndarray]]:
        """Format episode_info lists into expected 3D format."""
        if len(episode_info['returned_episode_returns']) > 0:
            num_episodes = len(episode_info['returned_episode_returns'])
            return {
                'returned_episode_returns': np.array(
                    episode_info['returned_episode_returns']
                ).reshape((1, num_episodes, 1)),
                'returned_episode_lengths': np.array(
                    episode_info['returned_episode_lengths']
                ).reshape((1, num_episodes, 1)),
                'returned_episode': np.array(
                    episode_info['returned_episode']
                ).reshape((1, num_episodes, 1)),
                'task_success': np.array(
                    episode_info['task_success']
                ).reshape((1, num_episodes, 1)),
                'timestep': np.array(episode_info['timestep']).reshape((1, num_episodes, 1)),
            }
        return None

    def get_latents(state: AgentState) -> jnp.ndarray:
        """Extract latent features from the world model's current state.

        Note: Call select_action first to update wm_state with the latest observation.

        Args:
            state: Current agent state (with updated wm_state from select_action)

        Returns:
            Latent features [B, feat_dim] from the world model state
        """
        params = state.train_state.params
        return world_model.get_feat(params.wm, state.runtime.wm_state)

    def state_from_checkpoint(checkpoint_data: Dict, runtime_state: RuntimeState) -> AgentState:
        """Convert checkpoint dictionaries to proper dataclass structures."""
        train_state_dict = checkpoint_data['train_state']
        params_dict = train_state_dict['params']

        wm_params = WorldModelParams(**params_dict['wm'])
        policy_params_ckpt = PolicyParams(**params_dict['policy'])
        params = STORMParams(wm=wm_params, policy=policy_params_ckpt)

        train_state_obj = STORMTrainState(
            step=train_state_dict['step'],
            apply_fn=None,
            params=params,
            tx=optimizer,
            opt_state=train_state_dict['opt_state'],
            slow_critic=train_state_dict['slow_critic'],
            normalizers=train_state_dict['normalizers'],
            slow_critic_rate=train_state_dict.get('slow_critic_rate', 0.02),
        )

        return AgentState(
            train_state=train_state_obj,
            runtime=runtime_state,
        )

    def reset(state: AgentState, rng: jax.random.PRNGKey) -> AgentState:
        """Reset optimizer states for new task (continual learning)."""
        new_train_state = STORMTrainState.create(
            apply_fn=None,
            params=state.train_state.params,
            tx=optimizer,
            slow_critic=state.train_state.slow_critic,
            normalizers=state.train_state.normalizers,
            slow_critic_rate=slow_critic_rate,
        )

        return AgentState(
            train_state=new_train_state,
            runtime=RuntimeState(
                buffer_state=state.runtime.buffer_state,
                wm_state=None,
                step=state.runtime.step,
                train_steps=state.runtime.train_steps,
                rng=rng,
                current_num_actions=state.runtime.current_num_actions,
                current_task_id=state.runtime.current_task_id,
            ),
        )

    return Agent(
        init=init,
        train=train,
        evaluate=evaluate,
        select_action=select_action,
        state_from_checkpoint=state_from_checkpoint,
        reset=reset,
        get_latents=get_latents,
        obs_space=obs_space,
        action_space=act_space,
        # Expose for testing/debugging/analysis
        buffer=buffer,
        train_step=train_step,
        world_model=world_model,
    )


# Keep class alias for backwards compatibility during transition
STORM = make_storm


# Export public API
__all__ = [
    'make_storm',
    'STORM',
]

