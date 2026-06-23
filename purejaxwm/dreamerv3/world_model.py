"""World model: RSSM (with block-diagonal GRU) + observation/imagination losses.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from purejaxwm.dreamerv3.distributions import OneHotCategoricalSTE


def kl_categorical(
    logits_q: jnp.ndarray, logits_p: jnp.ndarray, axis: int = -1
) -> jnp.ndarray:
    """KL(q || p) where both are categoricals with `logits_*`.

    Returns the sum over `axis`. Caller is responsible for further reductions (e.g.
    summing over multiple independent categorical latents).
    """
    log_q = jax.nn.log_softmax(logits_q, axis=axis)
    log_p = jax.nn.log_softmax(logits_p, axis=axis)
    q = jax.nn.softmax(logits_q, axis=axis)
    return (q * (log_q - log_p)).sum(axis=axis)


def output_init(outscale: float):
    """Output-layer initializer.

        outscale == 0.0 → zeros (output literally starts at 0)
        outscale  > 0.0 → truncated_normal with fan_in scaling of `outscale`
    """
    if outscale == 0.0:
        return nn.initializers.zeros
    return nn.initializers.variance_scaling(
        scale=outscale, mode="fan_in", distribution="truncated_normal",
    )


class MLPBlock(nn.Module):
    hidden: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        x = nn.Dense(
            self.hidden, use_bias=False,
            dtype=self.dtype, param_dtype=self.param_dtype,
        )(x)
        x = nn.RMSNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return jax.nn.silu(x)


class MLPHead(nn.Module):
    """Stack of ``num_layers`` MLPBlocks → ``nn.Dense(out_dim)`` → fp32 logits."""
    hidden: int
    num_layers: int
    out_dim: int
    outscale: float = 1.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = MLPBlock(self.hidden, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        logits = nn.Dense(
            self.out_dim,
            kernel_init=output_init(self.outscale),
            dtype=self.dtype, param_dtype=self.param_dtype,
        )(x)
        return logits.astype(jnp.float32)


class BlockLinear(nn.Module):
    """Block-wise (grouped) linear transformation.

    Splits the input and output channels into `groups` contiguous blocks and applies
    one separate Dense per block. This is equivalent to a block-diagonal weight matrix
    of shape (groups * in_per_group, groups * out_per_group), which has `groups` times
    fewer parameters than the dense equivalent.

    Input shape:  (..., groups * input_per_group)  (or (..., input_dim) broadcast to groups)
    Output shape: (..., features)  where features % groups == 0
    """
    features: int
    groups: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.features % self.groups == 0, (
            f"features ({self.features}) must be divisible by groups ({self.groups})"
        )
        features_per_group = self.features // self.groups
        input_dim = x.shape[-1]

        x = x.astype(self.dtype)

        if input_dim % self.groups == 0:
            input_per_group = input_dim // self.groups
            x_grp = x.reshape(*x.shape[:-1], self.groups, input_per_group)
        else:
            x_grp = jnp.broadcast_to(x[..., None, :], (*x.shape[:-1], self.groups, input_dim))
            input_per_group = input_dim

        kernel = self.param(
            "kernel", self.kernel_init,
            (self.groups, input_per_group, features_per_group),
            self.param_dtype,
        )
        kernel = kernel.astype(self.dtype)
        out_grp = jnp.einsum("...gi,gif->...gf", x_grp, kernel)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init,
                (self.groups, features_per_group), self.param_dtype,
            )
            bias = bias.astype(self.dtype)
            out_grp = out_grp + bias

        return out_grp.reshape(*x.shape[:-1], self.features)


class BlockGRU(nn.Module):
    """Grouped GRU with block-diagonal gate matrix.

    The gate matrix mapping ``concat(h_prev_block, x_block) → 3*hidden_block`` is
    block-diagonal. Gates are split into (reset, candidate, update) and processed
    exactly as in Hafner's reference:

        reset = sigmoid(reset_pre)
        cand = tanh(reset * cand_pre)
        update = sigmoid(update_pre - 1)
        h = update * cand + (1 - update) * h_prev

    BlockGRU reduces parameter count for large deterministic hidden sizes (e.g.
    deter=8192 with blocks=8 uses ~9x fewer weights than a naive ``nn.GRUCell``).
    """
    hidden: int
    blocks: int = 8
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h_prev: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        assert self.hidden % self.blocks == 0
        g = self.blocks
        per_group = self.hidden // g

        h_prev = h_prev.astype(self.dtype)
        x = x.astype(self.dtype)

        gate_matrix = BlockLinear(
            3 * self.hidden, g, use_bias=True, name="gates",
            dtype=self.dtype, param_dtype=self.param_dtype,
        )
        gates = gate_matrix(jnp.concatenate([h_prev, x], axis=-1))

        gates_per_group = gates.reshape(*gates.shape[:-1], g, 3 * per_group)
        reset_g, cand_g, update_g = jnp.split(gates_per_group, 3, axis=-1)
        reset = reset_g.reshape(*gates.shape[:-1], self.hidden)
        cand_pre = cand_g.reshape(*gates.shape[:-1], self.hidden)
        update_pre = update_g.reshape(*gates.shape[:-1], self.hidden)

        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand_pre)
        update = jax.nn.sigmoid(update_pre - 1.0)
        return update * cand + (1 - update) * h_prev


class State(NamedTuple):
    deter: jnp.ndarray  # (B, D)
    stoch: jnp.ndarray  # (B, S, C) one-hot (possibly STE during training)
    logits: jnp.ndarray  # (B, S, C) the logits that produced `stoch`

    def flat_stoch(self) -> jnp.ndarray:
        return self.stoch.reshape(*self.stoch.shape[:-2], -1)

    def features(self) -> jnp.ndarray:
        return jnp.concatenate([self.deter, self.flat_stoch()], axis=-1)


def _zero_state(batch_shape, deter: int, stoch: int, classes: int) -> State:
    return State(
        deter=jnp.zeros((*batch_shape, deter)),
        stoch=jnp.zeros((*batch_shape, stoch, classes)),
        logits=jnp.zeros((*batch_shape, stoch, classes)),
    )


class RSSM(nn.Module):
    deter_dim: int = 512
    stoch_size: int = 32
    classes: int = 32
    hidden: int = 512
    prior_layers: int = 2
    post_layers: int = 1
    unimix: float = 0.01
    blocks: int = 8            # BlockGRU groups; must divide deter_dim
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def initial_state(self, batch_shape) -> State:
        return _zero_state(batch_shape, self.deter_dim, self.stoch_size, self.classes)

    @nn.compact
    def __call__(self, prev_state: State, action: jnp.ndarray, embed: jnp.ndarray = None,
                 is_first: jnp.ndarray = None, training: bool = True):
        """One-step observe or imagine.

        Args:
          prev_state: State at t-1
          action: (B, A) one-hot action taken at t-1 (already masked to zero if
                  is_first=True — caller's responsibility, see DESIGN temporal contract)
          embed: (B, E) encoded obs at t, or None to run a pure imagination step (prior only)
          is_first: (B,) bool; if True, reset prev_state to zeros before computing
          training: if True (default), sample stoch from the categorical dist
                    (with unimix + straight-through); if False, take `.mode` (argmax
                    one-hot, still straight-through so grads flow when called inside
                    a value-grad trace but output is deterministic).

        Returns:
          If embed is not None: (prior: State, posterior: State)
          If embed is None:     prior: State
        """
        if is_first is not None:
            mask = is_first[..., None]
            prev_deter = jnp.where(mask, jnp.zeros_like(prev_state.deter), prev_state.deter)
            prev_stoch = jnp.where(
                mask[..., None], jnp.zeros_like(prev_state.stoch), prev_state.stoch
            )
            prev_state = State(prev_deter, prev_stoch, jnp.zeros_like(prev_state.logits))

        dt, pdt = self.dtype, self.param_dtype

        # project (prev_stoch, action) → GRU input
        action = action / jax.lax.stop_gradient(jnp.maximum(1.0, jnp.abs(action)))
        x = jnp.concatenate([prev_state.flat_stoch(), action], axis=-1).astype(dt)
        x = nn.Dense(self.hidden, use_bias=False, dtype=dt, param_dtype=pdt)(x)
        x = nn.RMSNorm(dtype=dt, param_dtype=pdt)(x)
        x = jax.nn.silu(x)

        # BlockGRU step
        new_deter = BlockGRU(
            hidden=self.deter_dim, blocks=self.blocks, dtype=dt, param_dtype=pdt,
        )(prev_state.deter.astype(dt), x)

        new_deter = new_deter.astype(self.param_dtype)

        # prior: MLP(deter) → logits
        h = new_deter
        for _ in range(self.prior_layers):
            h = nn.Dense(self.hidden, use_bias=False, dtype=dt, param_dtype=pdt)(h)
            h = nn.RMSNorm(dtype=dt, param_dtype=pdt)(h)
            h = jax.nn.silu(h)
        prior_logits = nn.Dense(
            self.stoch_size * self.classes, dtype=dt, param_dtype=pdt,
        )(h)
        prior_logits = prior_logits.astype(jnp.float32)
        prior_logits = prior_logits.reshape(*prior_logits.shape[:-1], self.stoch_size, self.classes)

        rng = self.make_rng("stoch")
        prior_dist = OneHotCategoricalSTE(prior_logits, unimix=self.unimix)
        prior_stoch = prior_dist.sample(rng) if training else prior_dist.mode
        prior = State(new_deter, prior_stoch, prior_dist.logits)

        if embed is None:
            return prior

        # posterior: MLP(deter, embed) → logits
        h = jnp.concatenate([new_deter, embed.astype(dt)], axis=-1)
        for _ in range(self.post_layers):
            h = nn.Dense(self.hidden, use_bias=False, dtype=dt, param_dtype=pdt)(h)
            h = nn.RMSNorm(dtype=dt, param_dtype=pdt)(h)
            h = jax.nn.silu(h)
        post_logits = nn.Dense(
            self.stoch_size * self.classes, dtype=dt, param_dtype=pdt,
        )(h)
        post_logits = post_logits.astype(jnp.float32)
        post_logits = post_logits.reshape(*post_logits.shape[:-1], self.stoch_size, self.classes)

        rng = self.make_rng("stoch")
        post_dist = OneHotCategoricalSTE(post_logits, unimix=self.unimix)
        post_stoch = post_dist.sample(rng) if training else post_dist.mode
        posterior = State(new_deter, post_stoch, post_dist.logits)

        return prior, posterior


def observe_scan(
    rssm: RSSM,
    params,
    init_state: State,
    actions: jnp.ndarray,       # (T, B, A)
    embeds: jnp.ndarray,        # (T, B, E)
    is_first: jnp.ndarray,      # (T, B)
    rng: jax.Array,
    training: bool = True,
):
    """Scan observe-step over time. Returns (priors, posteriors) both (T, B, ...)."""
    def step(carry, inp):
        state, rng = carry
        a, e, f = inp
        rng, sub = jax.random.split(rng)
        prior, post = rssm.apply(
            params, state, a, e, f, training, rngs={"stoch": sub}
        )
        return (post, rng), (prior, post)

    (_, _), (priors, posts) = jax.lax.scan(
        step, (init_state, rng), (actions, embeds, is_first)
    )
    return priors, posts


def imagine_scan(
    rssm: RSSM,
    params,
    init_state: State,
    policy_fn,                   # (params_ac, state) → (action, extras)
    policy_params,
    horizon: int,
    rng: jax.Array,
    training: bool = True,
):
    """Scan imagine-step. At each step, the policy emits an action from the current
    state (state.features()), and the world model rolls one step forward.

    Returns:
      states: (T, B, ...) each field of State — states[t] is the state AFTER step t
      actions: (T, B, A)
      extras: whatever `policy_fn` returns as `extras` per step
    """
    def step(carry, _):
        state, rng = carry
        rng, sub_p, sub_rssm = jax.random.split(rng, 3)
        action, extra = policy_fn(policy_params, state, sub_p)
        prior = rssm.apply(
            params, state, action, None, None, training, rngs={"stoch": sub_rssm}
        )
        return (prior, rng), (prior, action, extra)

    (_, _), (states, actions, extras) = jax.lax.scan(
        step, (init_state, rng), None, horizon
    )
    return states, actions, extras


class WMLossAux(NamedTuple):
    total: jnp.ndarray
    rec: jnp.ndarray
    rew: jnp.ndarray
    cont: jnp.ndarray
    dyn: jnp.ndarray
    rep: jnp.ndarray
    post: State                    # (T, B, ...) posterior states, for downstream AC use
    prior: State
    belief: jnp.ndarray = jnp.float32(0.0)       # aux map-category CE loss (0 if disabled)
    belief_acc: jnp.ndarray = jnp.float32(0.0)   # aux map-category accuracy


def wm_loss(
    wm_params: Any,                # pytree of {'encoder', 'rssm', 'decoder', 'reward', 'cont'}
    *,
    encoder_apply,                 # fn(params['encoder'], obs) → embeds
    rssm: RSSM,
    decoder_apply,                 # fn(params['decoder'], features) → predicted obs tensor
    reward_apply,                  # fn(params['reward'], features) → TwoHotDist
    cont_apply,                    # fn(params['cont'], features) → logits
    obs: jnp.ndarray,              # (T, B, H, W, C)
    action: jnp.ndarray,           # (T, B, A) one-hot; must be action_{t-1} aligned with obs_t
    reward: jnp.ndarray,           # (T, B)
    is_first: jnp.ndarray,         # (T, B)
    is_terminal: jnp.ndarray,      # (T, B)
    init_rssm_state: State,        # (B, ...)
    rng: jax.Array,
    loss_scales: dict,             # {'rec', 'rew', 'cont', 'dyn', 'rep'}
    free_nats: float,
    rec_loss_fn=None,              # optional fn(rec_pred, obs_flat) → scalar rec loss;
                                   # None → default 0.5 * sum-of-squares MSE (unchanged)
    belief_apply=None,             # optional fn(params['belief'], features) → (N, C) logits
    belief_target=None,            # (T, B) int32 map-category targets
    belief_scale=0.0,              # weight of the auxiliary belief CE in the total loss
):
    """Compute WM loss over a (T, B) batch of trajectory sub-sequences.

    The temporal-alignment convention is: obs_t paired with action_{t-1}, reward_t, is_first_t, is_terminal_t
    Caller is responsible for building the replay batch in this alignment.
    """
    T = obs.shape[0]
    B = obs.shape[1]

    # encode all observations
    obs_flat = obs.reshape(T * B, *obs.shape[2:])
    embed_flat = encoder_apply(wm_params["encoder"], obs_flat)
    embed = embed_flat.reshape(T, B, -1)

    # run RSSM over time
    priors, posteriors = observe_scan(
        rssm, wm_params["rssm"], init_rssm_state, action, embed, is_first, rng
    )
    feats = posteriors.features()                   # (T, B, F)

    # decode posterior features → reconstruct observations
    feats_flat = feats.reshape(T * B, -1)
    rec_pred = decoder_apply(wm_params["decoder"], feats_flat)
    if rec_loss_fn is None:
        # default: inline 0.5 * sum-of-squares MSE
        sq = jnp.square(rec_pred - obs_flat)
        rec_loss = 0.5 * sq.sum(axis=tuple(range(1, sq.ndim))).mean()
    else:
        rec_loss = rec_loss_fn(rec_pred, obs_flat)

    # reward prediction (TwoHot cross-entropy)
    rew_dist = reward_apply(wm_params["reward"], feats_flat)
    rew_logp = rew_dist.log_prob(reward.reshape(T * B))
    rew_loss = -rew_logp.mean()

    # continuation prediction (Bernoulli BCE on (1 - is_terminal))
    cont_logits = cont_apply(wm_params["cont"], feats_flat)
    cont_target = (1.0 - is_terminal.astype(jnp.float32)).reshape(T * B)
    cont_loss = optax_sigmoid_bce(cont_logits, cont_target).mean()

    # KL losses
    kl_axis_inner = kl_categorical(posteriors.logits, jax.lax.stop_gradient(priors.logits))
    rep_kl = kl_axis_inner.sum(axis=-1)        # (T, B)
    dyn_kl = kl_categorical(
        jax.lax.stop_gradient(posteriors.logits), priors.logits
    ).sum(axis=-1)

    rep_loss = jnp.maximum(rep_kl, free_nats).mean()
    dyn_loss = jnp.maximum(dyn_kl, free_nats).mean()

    # auxiliary map-category (belief) cross-entropy on the posterior features;
    # gradients flow into the RSSM/encoder, shaping the latent to encode map type.
    if belief_apply is not None:
        belief_logits = belief_apply(wm_params["belief"], feats_flat)   # (T*B, C)
        bt = belief_target.reshape(T * B).astype(jnp.int32)
        blogp = jax.nn.log_softmax(belief_logits, axis=-1)
        belief_loss = -jnp.take_along_axis(blogp, bt[:, None], axis=-1).squeeze(-1).mean()
        belief_acc = (belief_logits.argmax(-1) == bt).mean().astype(jnp.float32)
    else:
        belief_loss = jnp.float32(0.0)
        belief_acc = jnp.float32(0.0)

    total = (
        loss_scales["rec"] * rec_loss
        + loss_scales["rew"] * rew_loss
        + loss_scales["cont"] * cont_loss
        + loss_scales["dyn"] * dyn_loss
        + loss_scales["rep"] * rep_loss
        + belief_scale * belief_loss
    )
    aux = WMLossAux(
        total=total,
        rec=rec_loss, rew=rew_loss, cont=cont_loss,
        dyn=dyn_loss, rep=rep_loss,
        post=posteriors, prior=priors,
        belief=belief_loss, belief_acc=belief_acc,
    )
    return total, aux


def optax_sigmoid_bce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Binary cross-entropy with logits (numerically stable)."""
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -(targets * log_p + (1 - targets) * log_not_p)
