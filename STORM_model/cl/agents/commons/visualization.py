"""Video prediction visualization for DreamerV3 world model."""
from typing import Dict, Any

import jax
import numpy as np
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont
import wandb

from cl.visualizations import rotate_frame


def video_predict(
    agent,
    state: Any,
    batch: Dict[str, jnp.ndarray],
    step_count: int,
    context: int = 5,
    log_prefix: str = "dreamer_reconstruction",
    upscale: int = 6,
):
    """
    Visualize world model predictions with teacher-forcing context and open-loop rollout.

    Shows:
    - Left: Ground truth frames
    - Right: Model predictions (context frames with teacher forcing, then open-loop)

    Args:
        agent: DreamerV3 agent instance
        state: Current agent state (AgentState) with trained parameters
        batch: Dictionary containing:
            - obs_image: [B, T, H, W, C] in [0, 1] float32
            - obs_direction: [B, T, 4] one-hot (if used)
            - action: [B, T, A] one-hot
            - is_first: [B, T] bool
        step_count: Current training step
        context: Number of context frames for teacher forcing (default: 5)
        log_prefix: WandB log key prefix
        upscale: Upscaling factor for visualization
    """
    # Extract first trajectory only
    batch_single = jax.tree.map(lambda x: x[0:1], batch)

    B, T = batch_single['obs_image'].shape[:2]
    assert B == 1, "video_predict only visualizes single trajectory"

    # === Context Phase: Teacher-Forcing Reconstruction ===
    # Encode context frames
    # Reshape from [B=1, T, ...] to [T, ...] for encoder
    obs_dict_ctx = {}
    for modality in agent.obs_modalities:
        obs_key = f'obs_{modality}'

        if obs_key not in batch_single:
            raise ValueError(f"Missing required observation modality '{obs_key}' in batch. "
                           f"Available keys: {list(batch_single.keys())}. "
                           f"Make sure the batch is properly sampled from the replay buffer.")

        obs_data = batch_single[obs_key][:, :context]  # [1, context, ...]
        # Flatten batch dimension: [1, context, ...] -> [context, ...]
        obs_dict_ctx[modality] = obs_data[0]

    # Extract parameters from state (AgentState with TrainState structure)
    encoder_params = state.train_state.params.wm.encoder
    rssm_params = state.train_state.params.wm.dynamics
    decoder_params = state.train_state.params.wm.decoder

    # Encode observations
    embeds_ctx = agent.encoder.apply(encoder_params, obs_dict_ctx)  # [context, E]

    # RSSM observe with context
    # Get initial state
    init_state = agent.rssm.apply(rssm_params, 1, method=agent.rssm.initial_state)

    def rssm_observe_step(carry, inputs):
        prev_state = carry
        embed, action, is_first = inputs

        # Add batch dimension (scan unpacks, so we need to add it back)
        embed = embed[None, ...]  # [E] -> [1, E]
        action = action[None, ...]  # [A] -> [1, A]
        is_first = is_first[None, ...]  # scalar -> [1]

        # RSSM forward pass
        post_state, _ = agent.rssm.apply(
            rssm_params,
            prev_state,
            action,
            embed,
            is_first,
            False,  # training=False for visualization
            None  # no rng needed
        )

        return post_state, post_state

    _, ctx_states = jax.lax.scan(
        rssm_observe_step,
        init_state,
        (
            embeds_ctx,  # Already [T, E] after removing batch dim
            batch_single['action'][0, :context],  # [T, A]
            batch_single['is_first'][0, :context],  # [T]
        )
    )

    # Scan returns [T, batch=1, ...], swap to [batch=1, T, ...]
    ctx_states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), ctx_states)

    # Decode context frames
    ctx_states_flat = jax.tree.map(lambda x: x.reshape(context, *x.shape[2:]), ctx_states)
    ctx_recons = agent.decoder.apply(decoder_params, ctx_states_flat)

    # === Rollout Phase: Open-Loop Prediction ===
    # Start from last context state
    # FIXED: Keep batch dimension by using [:, -1] instead of [0, -1]
    init_rollout = jax.tree.map(lambda x: x[:, -1], ctx_states)

    # Future actions
    actions_future = batch_single['action'][0, context:]  # [T-context, A]

    def rssm_imagine_step(carry, action):
        prev_state = carry

        # Add batch dimension (scan unpacks, so we need to add it back)
        action = action[None, ...]  # [A] -> [1, A]

        # Prior only (no observation)
        prior_state, _ = agent.rssm.apply(
            rssm_params,
            prev_state,
            action,
            None,  # No embed for imagination
            None,  # No is_first for imagination
            False,  # training=False
            None  # no rng
        )
        return prior_state, prior_state

    _, rollout_states = jax.lax.scan(
        rssm_imagine_step,
        init_rollout,
        actions_future  # Already [T-context, A] - scan axis is 0
    )

    # Decode rollout frames
    # Scan returns [T-context, batch=1, ...], squeeze batch dimension to get [T-context, ...]
    rollout_flat = jax.tree.map(lambda x: jnp.squeeze(x, axis=1), rollout_states)
    rollout_recons = agent.decoder.apply(decoder_params, rollout_flat)

    # === Assemble Predictions ===
    # Context: mean of reconstruction distributions
    ctx_imgs = ctx_recons['image'].mean  # MSEDist has mean as an attribute

    # Rollout: mean of imagination distributions
    rollout_imgs = rollout_recons['image'].mean  # MSEDist has mean as an attribute

    # Concatenate
    model_frames = jnp.concatenate([ctx_imgs, rollout_imgs], axis=0)  # [T, H, W, C]

    # Get ground truth
    gt_frames = batch_single['obs_image'][0]  # [T, H, W, C]

    # Handle direction if present
    if 'direction' in agent.obs_modalities:
        # Context: use ground truth
        dir_ctx = jnp.argmax(batch_single['obs_direction'][0, :context], axis=-1)

        # Rollout: predict from model
        dir_rollout_dist = ctx_recons.get('direction')
        if dir_rollout_dist is not None:
            # Convert one-hot to index
            dir_rollout_ctx = jnp.argmax(dir_rollout_dist.mode(), axis=-1)  
        else:
            dir_rollout_ctx = dir_ctx

        dir_rollout_future = rollout_recons.get('direction')
        if dir_rollout_future is not None:
            # Convert one-hot to index
            dir_rollout_future = jnp.argmax(dir_rollout_future.mode(), axis=-1)  
        else:
            dir_rollout_future = jnp.zeros(T - context, dtype=jnp.int32)

        dir_pred = jnp.concatenate([dir_rollout_ctx, dir_rollout_future], axis=0)
        dir_gt = jnp.argmax(batch_single['obs_direction'][0], axis=-1)
    else:
        dir_pred = None
        dir_gt = None

    # Convert to numpy and process
    gt_np = np.array(gt_frames)
    model_np = np.array(model_frames)

    # Clip and convert to uint8
    gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
    model_np = np.clip(model_np * 255, 0, 255).astype(np.uint8)

    # Upscale
    def upscale_frames(frames, factor):
        """Upscale frames using nearest neighbor. frames: [T, H, W, C]"""
        T, H, W, C = frames.shape
        # Use numpy repeat for upscaling (no scipy dependency)
        frames_up = frames.repeat(factor, axis=1).repeat(factor, axis=2)
        return frames_up

    gt_up = upscale_frames(gt_np, upscale)
    model_up = upscale_frames(model_np, upscale)

    H_up, W_up = gt_up.shape[1:3]

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 3 * upscale)
    except:
        font = ImageFont.load_default()

    # Decorate frames
    def add_border_and_label(frame, label, color, is_context):
        """Add border and label to frame."""
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)

        # Border (green for GT, red for model, yellow for context)
        border_color = (255, 255, 0) if is_context and color == (255, 0, 0) else color
        draw.rectangle([0, 0, W_up - 1, H_up - 1], outline=border_color, width=2 * upscale)

        # Label
        bbox = draw.textbbox((0, 0), label, font=font)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Create text mask
        mask = Image.new("1", (bw, bh), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text((0, 0), label, font=font, fill=1)

        # Paste white text
        txt = Image.new("RGB", (bw, bh), (255, 255, 255))
        margin = 4
        x, y = W_up - bw - margin, margin
        pil.paste(txt, (x, y), mask)

        return np.array(pil)

    # Process each frame
    frames_decorated = []
    for t in range(T):
        is_ctx = t < context

        # Rotate if needed
        if dir_gt is not None:
            frame_gt = rotate_frame(gt_up[t], int(dir_gt[t]))
            frame_model = rotate_frame(model_up[t], int(dir_pred[t]))
        else:
            frame_gt = gt_up[t]
            frame_model = model_up[t]

        # Add decorations
        label_gt = f"Ground Truth (Frame {t})"
        label_model = f"{'Context' if is_ctx else 'Prediction'} (Frame {t})"

        frame_gt = add_border_and_label(frame_gt, label_gt, (0, 255, 0), False)
        frame_model = add_border_and_label(frame_model, label_model, (255, 0, 0), is_ctx)

        # Concatenate horizontally
        frame_combined = np.concatenate([frame_gt, frame_model], axis=1)
        frames_decorated.append(frame_combined)

    # Stack to video: [T, H, W, C]
    video_np = np.stack(frames_decorated, axis=0)

    # Convert to WandB format: [T, C, H, W]
    video_wandb = video_np.transpose(0, 3, 1, 2)

    # Log to WandB if initialized
    if wandb.run is not None:
        wandb.log({
            f"{log_prefix}": wandb.Video(video_wandb, fps=5, format="mp4"),
            f"{log_prefix}_step": step_count,
        })

    # Return the video array for testing/inspection
    return video_np
