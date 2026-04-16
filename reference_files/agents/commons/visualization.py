"""Visualization utilities for rendering and world model prediction.

General-purpose rendering utilities (frame rotation, plotting, GIF creation)
and agent-specific visualization (world model predictions, buffer distribution).
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import wandb

from cl.agents.commons.preprocessing import normalize_image
from cl.agents.commons.symbolic_utils import symbolic_to_rgb, is_symbolic_observation


# ============================================================
# General-purpose rendering utilities
# ============================================================

def rotate_frame(
    frame: Union[Array, np.ndarray],
    direction: int = 0
) -> Union[Array, np.ndarray]:
    """Rotate a frame by 90° increments based on direction.

    Converts from agent-centric view to world-aligned view (north-up).
    Uses Navix direction convention.

    Args:
        frame: The frame to rotate (JAX or NumPy array)
        direction: Direction value using Navix convention:
                  0 = EAST (agent facing right)
                  1 = SOUTH (agent facing down)
                  2 = WEST (agent facing left)
                  3 = NORTH (agent facing up)

    Returns:
        Rotated frame in the same format as input (world-aligned with north pointing up)
    """
    if direction < 0 or direction > 3:
        raise ValueError(f"Invalid direction: {direction}. Must be 0, 1, 2, or 3.")

    # Determine rotation axes based on shape
    # Assume (H, W, C) format for 3D or (N, H, W, C) for 4D
    if frame.ndim == 3:  # (H, W, C)
        axes = (0, 1)
    elif frame.ndim == 4:  # (N, H, W, C)
        axes = (1, 2)
    else:
        axes = (0, 1)

    # Use appropriate library
    rot90_fn = jnp.rot90 if isinstance(frame, jnp.ndarray) else np.rot90

    # Rotation mapping from agent-centric to world-aligned (north-up) view:
    # Agent-centric view: top=forward, left=90° counter-clockwise from forward
    # World view: top=north, right=east, bottom=south, left=west
    #
    # - EAST (0): agent's left=north, forward=east
    #   → Need: agent's left→world's top, agent's forward→world's right
    #   → Rotate 90° clockwise (k=3, which is 270° counter-clockwise)
    # - SOUTH (1): agent's left=east, forward=south
    #   → Need: agent's left→world's right, agent's forward→world's bottom
    #   → Rotate 180° (k=2)
    # - WEST (2): agent's left=south, forward=west
    #   → Need: agent's left→world's bottom, agent's forward→world's left
    #   → Rotate 90° counter-clockwise (k=1)
    # - NORTH (3): agent's left=west, forward=north
    #   → Already aligned, no rotation needed (k=0)
    if direction == 0:  # EAST
        return rot90_fn(frame, k=3, axes=axes)  # 90° clockwise
    elif direction == 1:  # SOUTH
        return rot90_fn(frame, k=2, axes=axes)  # 180°
    elif direction == 2:  # WEST
        return rot90_fn(frame, k=1, axes=axes)  # 90° counter-clockwise
    else:  # direction == 3, NORTH
        return frame  # No rotation


def rotate_frame_series(
    frames: Union[Array, np.ndarray],
    directions: Union[Array, np.ndarray]
) -> Union[Array, np.ndarray]:
    """Rotate a series of frames by 90° increments based on directions.

    Args:
        frames: Frame tensors to rotate (N, H, W, C)
        directions: Direction values (N,) with values 0-3

    Returns:
        Rotated frames in the same format as input
    """
    num_frames = frames.shape[0]

    # Convert directions to numpy for indexing
    dirs_np = np.array(directions) if isinstance(directions, jnp.ndarray) else directions

    # Rotate each frame
    if isinstance(frames, jnp.ndarray):
        rotated = [rotate_frame(frames[i], int(dirs_np[i])) for i in range(num_frames)]
        return jnp.stack(rotated)
    else:
        rotated_frames = np.zeros_like(frames)
        for i in range(num_frames):
            rotated_frames[i] = rotate_frame(frames[i], int(dirs_np[i]))
        return rotated_frames


def plot_frame(
    frame: Union[Array, np.ndarray],
    direction: int = 0,
    figsize: Tuple[int, int] = (5, 5),
    title: Optional[str] = None,
    show_direction_indicator: bool = False
) -> None:
    """Plot a single frame with optional rotation and direction indicator.

    Args:
        frame: The observation to plot
        direction: Direction value (0, 1, 2, 3) to determine rotation
        figsize: Size of the figure
        title: Optional title for the plot
        show_direction_indicator: Whether to show direction arrow
    """
    frame = rotate_frame(frame, direction)

    # Convert to numpy if JAX array
    if isinstance(frame, jnp.ndarray):
        frame = np.array(frame)

    # Normalize the frame to [0, 1] for proper RGB display
    frame = frame.astype(float)
    if frame.max() > 0:
        frame /= frame.max()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame)
    ax.axis("off")

    if title:
        ax.set_title(title)

    if show_direction_indicator:
        add_direction_indicator(ax, direction)

    plt.show()


def plot_frame_series(
    frames: Union[Array, np.ndarray],
    save_path: str,
    directions: Optional[Union[Array, np.ndarray]] = None,
    titles: Optional[List[str]] = None
) -> None:
    """Save a long horizontal strip of images.

    Args:
        frames: Input frames of images (N, H, W, C)
        save_path: Path to save the final stitched plot
        directions: Optional directions for rotation (N,)
        titles: Optional list of titles for each frame
    """
    # Convert to numpy if JAX array
    if isinstance(frames, jnp.ndarray):
        frames = np.array(frames)

    # Rotate if directions provided
    if directions is not None:
        frames = rotate_frame_series(frames, directions)

    # Ensure (N, H, W, C) format
    if frames.ndim != 4 or frames.shape[-1] not in [1, 3, 4]:
        raise ValueError(f"Expected frames of shape (N, H, W, C), got {frames.shape}")

    n, h, w, c = frames.shape

    # Normalize to [0, 1]
    frames = frames.astype(float)
    if frames.max() > 0:
        frames /= frames.max()

    # Concatenate along width
    long_img = np.concatenate([frames[i] for i in range(n)], axis=1)  # (H, N*W, C)

    # Plot and save
    fig, ax = plt.subplots(1, 1, figsize=(n * (w / 50), h / 50), dpi=100)
    ax.imshow(long_img.squeeze() if c == 1 else long_img)
    ax.axis("off")

    # Add titles if provided
    if titles:
        for i, title in enumerate(titles[:n]):
            x_pos = (i + 0.5) / n
            ax.text(x_pos, 0.02, title, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=8, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def create_gif(
    frames: Union[Array, np.ndarray],
    directions: Optional[Union[Array, np.ndarray]] = None,
    filename: str = "episode.gif",
    fps: int = 10,
    normalize: bool = True
) -> None:
    """Create a GIF from a tensor of frames.

    IMPORTANT: Duplicate frames are removed in the GIF creation process.

    Args:
        frames: A tensor of shape (N, H, W, C) containing N images
        directions: Optional tensor of shape (N,) containing rotation directions
        filename: The filename of the GIF to save
        fps: Frames per second for the GIF
        normalize: Whether to normalize frames to [0, 255]
    """
    # Rotate frames if directions provided
    if directions is not None:
        frames = rotate_frame_series(frames, directions)

    # Convert to numpy if JAX array
    if isinstance(frames, jnp.ndarray):
        frames = np.array(frames)

    if normalize:
        # Normalize to [0, 255] and convert to uint8
        global_min = frames.min()
        global_max = frames.max()
        frames = (frames - global_min) / (global_max - global_min + 1e-6) * 255
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    # Convert to list for imageio
    frames_list = [frames[i] for i in range(frames.shape[0])]

    imageio.mimsave(
        filename,
        frames_list,
        format="GIF",
        fps=fps,
        palettesize=256,
        subrectangles=False,
        optimize=False,
        disposal=2,
    )


def add_direction_indicator(
    ax: plt.Axes,
    direction: int,
    position: str = 'bottom_center',
    size: float = 0.15
) -> None:
    """Add a direction indicator arrow to the plot.

    Args:
        ax: Matplotlib axes to add indicator to
        direction: Agent direction using Navix convention:
                  0 = EAST (agent facing right)
                  1 = SOUTH (agent facing down)
                  2 = WEST (agent facing left)
                  3 = NORTH (agent facing up)
        position: Where to place indicator ('bottom_center', 'top_left', 'top_right', 'bottom_left')
        size: Size of the arrow as fraction of axis
    """
    # Navix convention: 0=EAST, 1=SOUTH, 2=WEST, 3=NORTH
    direction_names = ['→ E', '↓ S', '← W', '↑ N']
    colors = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']  # Blue, Orange, Red, Green

    # Arrow angles (in degrees, 0° points right)
    angles = [0, 270, 180, 90]  # East, South, West, North

    # Position mapping
    positions = {
        'bottom_center': (0.5, 0.08),
        'bottom_left': (0.15, 0.08),
        'top_left': (0.15, 0.92),
        'top_right': (0.85, 0.92),
    }

    x, y = positions.get(position, (0.5, 0.08))

    # Add arrow
    arrow_len = size * 0.8
    dx = arrow_len * np.cos(np.radians(angles[direction]))
    dy = arrow_len * np.sin(np.radians(angles[direction]))

    arrow = FancyArrowPatch(
        (x - dx/2, y - dy/2),
        (x + dx/2, y + dy/2),
        transform=ax.transAxes,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=3,
        color=colors[direction],
        zorder=10
    )
    ax.add_patch(arrow)

    # Add direction label
    ax.text(
        x, y - 0.05,
        direction_names[direction],
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=10,
        fontweight='bold',
        color=colors[direction],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=colors[direction]),
        zorder=11
    )


def create_episode_grid(
    frames: Union[Array, np.ndarray],
    directions: Optional[Union[Array, np.ndarray]] = None,
    actions: Optional[Union[Array, np.ndarray, List]] = None,
    rewards: Optional[Union[Array, np.ndarray, List]] = None,
    action_names: Optional[List[str]] = None,
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    rotate: bool = False,
    show_direction: bool = True,
    save_path: Optional[str] = None,
    suptitle: str = "Episode Trajectory"
) -> plt.Figure:
    """Create a grid visualization of episode frames.

    Args:
        frames: Episode frames (N, H, W, C)
        directions: Optional agent directions (N,)
        actions: Optional actions taken (N,)
        rewards: Optional rewards received (N,)
        action_names: Optional list of action names for display
        ncols: Number of columns in grid
        figsize: Figure size (auto-calculated if None)
        rotate: Whether to rotate frames to world coordinates
        show_direction: Whether to show direction indicators
        save_path: Optional path to save figure
        suptitle: Title for the entire figure

    Returns:
        Matplotlib figure
    """
    # Convert to numpy if JAX array
    if isinstance(frames, jnp.ndarray):
        frames = np.array(frames)

    n_frames = frames.shape[0]
    nrows = (n_frames + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 3, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Rotate frames if requested
    if rotate and directions is not None:
        frames = rotate_frame_series(frames, directions)

    for i in range(n_frames):
        frame = frames[i]

        # Normalize
        frame = frame.astype(float)
        if frame.max() > 0:
            frame /= frame.max()

        axes[i].imshow(frame)
        axes[i].axis('off')

        # Build title
        title_parts = [f"Step {i}"]

        if actions is not None:
            action_idx = int(actions[i])
            action_str = action_names[action_idx] if action_names else str(action_idx)
            title_parts.append(f"Act: {action_str}")

        if rewards is not None:
            title_parts.append(f"R: {float(rewards[i]):.2f}")

        if directions is not None and not rotate:
            title_parts.append(f"Dir: {int(directions[i])}")

        axes[i].set_title("\n".join(title_parts), fontsize=9)

        # Add direction indicator
        if show_direction and directions is not None:
            dir_idx = int(directions[i])
            add_direction_indicator(axes[i], dir_idx, position='bottom_center', size=0.12)

    # Hide extra subplots
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode grid to {save_path}")

    return fig


# ============================================================
# Agent-specific visualization (world model predictions, buffer)
# ============================================================

def video_predict(
    state: Any,
    batch: Dict[str, jnp.ndarray],
    world_model: Any,
    obs_modalities: List[str],
    step_count: int,
    context: int = 5,
    log_prefix: str = "world_model_reconstruction",
    upscale: int = 6,
    navix_obs_type: str = None,
) -> np.ndarray:
    """Visualize world model predictions with teacher-forcing context and open-loop rollout.

    This function works with any world model that implements the standard interface:
    - encode(wm_params, obs_dict) -> embeddings
    - initial_state(wm_params, batch_size) -> state
    - observe(wm_params, state, action, embed, is_first, training, rng) -> (post, prior)
    - imagine(wm_params, state, action, training, rng) -> (prior, info)
    - decode(wm_params, state) -> Dict[modality, distribution]

    Shows:
    - Left: Ground truth frames
    - Right: Model predictions (context frames with teacher forcing, then open-loop)

    Args:
        state: AgentState with trained parameters (state.train_state.params.wm)
        batch: Dictionary containing:
            - obs_image: [B, T, H, W, C] in uint8 [0, 255] or float32 [0, 1]
            - obs_direction: [B, T, 4] one-hot (optional, if used)
            - action: [B, T, A] one-hot
            - is_first: [B, T] bool
        world_model: World model instance (DreamerV3WorldModel or STORMWorldModel)
        obs_modalities: List of observation modality names (e.g., ['image', 'direction'])
        step_count: Current training step (for logging)
        context: Number of context frames for teacher forcing (default: 5)
        log_prefix: WandB log key prefix
        upscale: Upscaling factor for visualization
        navix_obs_type: Observation type string (e.g., 'rgb_first_person', 'symbolic_first_person').
                       Used to detect if symbolic-to-RGB conversion is needed.

    Returns:
        np.ndarray: Video frames [T, H, W, C] in uint8
    """
    # Extract first trajectory only
    batch_single = jax.tree.map(lambda x: x[0:1], batch)

    B, T = batch_single['obs_image'].shape[:2]
    assert B == 1, "video_predict only visualizes single trajectory"

    # Get world model params from state
    wm_params = state.train_state.params.wm

    # === Context Phase: Teacher-Forcing Reconstruction ===
    # Prepare context observations
    obs_dict_ctx = {}
    for modality in obs_modalities:
        obs_key = f'obs_{modality}'
        if obs_key not in batch_single:
            raise ValueError(
                f"Missing required observation modality '{obs_key}' in batch. "
                f"Available keys: {list(batch_single.keys())}. "
                f"Make sure the batch is properly sampled from the replay buffer."
            )
        obs_data = batch_single[obs_key][:, :context]  # [1, context, ...]
        # Flatten batch dimension: [1, context, ...] -> [context, ...]
        obs_dict_ctx[modality] = obs_data[0]

    # Normalize image data (resize already done by environment)
    obs_dict_ctx_norm = {}
    for modality, data in obs_dict_ctx.items():
        if modality == 'image':
            obs_dict_ctx_norm[modality] = normalize_image(data)
        else:
            obs_dict_ctx_norm[modality] = data

    # Encode context observations
    embeds_ctx = world_model.encode(wm_params, obs_dict_ctx_norm)  # [context, E]

    # Get initial state with batch_size=1
    init_state = world_model.initial_state(wm_params, batch_size=1)

    # Run through context with teacher forcing
    def observe_step(carry, inputs):
        prev_state = carry
        embed, action, is_first_t = inputs

        # Add batch dimension (scan unpacks along axis 0)
        embed = embed[None, ...]  # [E] -> [1, E]
        action = action[None, ...]  # [A] -> [1, A]
        is_first_t = is_first_t[None, ...]  # scalar -> [1]

        # World model observe step (posterior inference)
        post_state, _ = world_model.observe(
            wm_params,
            prev_state,
            action,
            embed,
            is_first_t,
            training=False,
            rng=None
        )
        return post_state, post_state

    _, ctx_states = jax.lax.scan(
        observe_step,
        init_state,
        (
            embeds_ctx,  # [context, E]
            batch_single['action'][0, :context],  # [context, A]
            batch_single['is_first'][0, :context],  # [context]
        )
    )

    # Scan returns [T, batch=1, ...], swap to [batch=1, T, ...]
    ctx_states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), ctx_states)

    # Decode context frames
    # Flatten: [1, context, ...] -> [context, ...]
    ctx_states_flat = jax.tree.map(lambda x: x.reshape(context, *x.shape[2:]), ctx_states)
    ctx_recons = world_model.decode(wm_params, ctx_states_flat)

    # === Rollout Phase: Open-Loop Prediction ===
    # Start from last context state (keep batch dim)
    init_rollout = jax.tree.map(lambda x: x[:, -1], ctx_states)

    # Future actions
    actions_future = batch_single['action'][0, context:]  # [T-context, A]

    def imagine_step(carry, action):
        prev_state = carry
        # Add batch dimension
        action = action[None, ...]  # [A] -> [1, A]

        # Prior prediction (no observation)
        prior_state, _ = world_model.imagine(
            wm_params,
            prev_state,
            action,
            training=False,
            rng=None
        )
        return prior_state, prior_state

    _, rollout_states = jax.lax.scan(
        imagine_step,
        init_rollout,
        actions_future  # [T-context, A]
    )

    # Decode rollout frames
    # Scan returns [T-context, batch=1, ...], squeeze batch dim
    rollout_flat = jax.tree.map(lambda x: jnp.squeeze(x, axis=1), rollout_states)
    rollout_recons = world_model.decode(wm_params, rollout_flat)

    # === Assemble Predictions ===
    # Get predictions from reconstruction distributions
    # OneHotDist (categorical) has .mode(), MSEDist (continuous) has .mean
    ctx_img_dist = ctx_recons['image']
    rollout_img_dist = rollout_recons['image']

    if hasattr(ctx_img_dist, 'mean'):
        # Continuous (MSEDist)
        ctx_imgs = ctx_img_dist.mean
        rollout_imgs = rollout_img_dist.mean
    else:
        # Discrete (OneHotDist) - mode() returns one-hot, argmax to get entity IDs
        ctx_imgs = jnp.argmax(ctx_img_dist.mode(), axis=-1)
        rollout_imgs = jnp.argmax(rollout_img_dist.mode(), axis=-1)

    # Concatenate context and rollout predictions
    model_frames = jnp.concatenate([ctx_imgs, rollout_imgs], axis=0)  # [T, H, W, C] or [T, H, W]

    # Get ground truth
    gt_frames = batch_single['obs_image'][0]  # [T, H, W, C]

    # For categorical observations, ground truth is one-hot - convert to entity IDs
    is_categorical = navix_obs_type in ['categorical', 'categorical_first_person'] if navix_obs_type else False
    if is_categorical and gt_frames.ndim == 4 and gt_frames.shape[-1] > 3:
        # One-hot encoded categorical: (T, H, W, num_classes) -> (T, H, W)
        gt_frames = jnp.argmax(gt_frames, axis=-1)
    elif gt_frames.dtype == jnp.uint8:
        gt_frames = gt_frames.astype(jnp.float32) / 255.0

    # Resize ground truth to match model output size if they differ
    # (buffer stores raw observations, model outputs at network resolution)
    if gt_frames.shape[1:3] != model_frames.shape[1:3]:
        target_h, target_w = model_frames.shape[1:3]
        gt_frames = jax.image.resize(
            gt_frames,
            shape=(gt_frames.shape[0], target_h, target_w) + gt_frames.shape[3:],
            method='nearest'
        )

    # Handle direction modality if present
    dir_pred = None
    dir_gt = None
    if 'direction' in obs_modalities and 'obs_direction' in batch_single:
        # Context: decode from context states
        dir_ctx_dist = ctx_recons.get('direction')
        if dir_ctx_dist is not None:
            dir_ctx = jnp.argmax(dir_ctx_dist.mode(), axis=-1)
        else:
            dir_ctx = jnp.argmax(batch_single['obs_direction'][0, :context], axis=-1)

        # Rollout: decode from rollout states (or use ground truth if model doesn't predict direction)
        dir_rollout_dist = rollout_recons.get('direction')
        if dir_rollout_dist is not None:
            dir_rollout = jnp.argmax(dir_rollout_dist.mode(), axis=-1)
        else:
            # Model doesn't predict direction (e.g., STORM) - use ground truth for visualization
            dir_rollout = jnp.argmax(batch_single['obs_direction'][0, context:], axis=-1)

        dir_pred = jnp.concatenate([dir_ctx, dir_rollout], axis=0)
        dir_gt = jnp.argmax(batch_single['obs_direction'][0], axis=-1)

    # Convert to numpy and process
    gt_np = np.array(gt_frames)
    model_np = np.array(model_frames)

    # Check if symbolic observations (need entity-to-color conversion)
    is_symbolic = is_symbolic_observation(navix_obs_type) if navix_obs_type else False

    if is_symbolic:
        # For symbolic/categorical observations, convert entity IDs to RGB colors
        # Symbolic: [T, H, W, 3] where channel 0 is entity ID
        # Categorical: [T, H, W] where values are entity IDs directly
        gt_np = symbolic_to_rgb(gt_np)
        model_np = symbolic_to_rgb(model_np)

        # Log temporal coherence metrics for categorical observations
        # Measures how much predictions change between consecutive frames
        if wandb.run is not None:
            # Use pre-RGB-converted frames (entity IDs) for accurate comparison
            model_frames_np = np.array(model_frames)
            gt_frames_np = np.array(gt_frames)

            # Temporal difference: how much adjacent frames differ
            pred_diff = np.abs(np.diff(model_frames_np.astype(np.float32), axis=0)).mean()
            gt_diff = np.abs(np.diff(gt_frames_np.astype(np.float32), axis=0)).mean()

            # Ratio > 1 means model predictions are more temporally unstable than ground truth
            temporal_ratio = pred_diff / (gt_diff + 1e-8)

            wandb.log({
                f'{log_prefix}_pred_temporal_diff': pred_diff,
                f'{log_prefix}_gt_temporal_diff': gt_diff,
                f'{log_prefix}_temporal_coherence_ratio': temporal_ratio,
            }, step=step_count)
    else:
        # Standard RGB processing: clip and convert to uint8
        gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
        model_np = np.clip(model_np * 255, 0, 255).astype(np.uint8)

    # Upscale frames
    def upscale_frames(frames: np.ndarray, factor: int) -> np.ndarray:
        """Upscale frames using nearest neighbor. frames: [T, H, W, C]"""
        return frames.repeat(factor, axis=1).repeat(factor, axis=2)

    # Auto-adjust upscale factor for small observations (e.g., 7x7 categorical)
    # Target 384 pixels for good visibility
    obs_size = min(gt_np.shape[1], gt_np.shape[2])
    target_size = 384
    auto_upscale = max(upscale, target_size // obs_size)

    gt_up = upscale_frames(gt_np, auto_upscale)
    model_up = upscale_frames(model_np, auto_upscale)

    H_up, W_up = gt_up.shape[1:3]

    # Load font (scale with upscale factor)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, 3 * auto_upscale // 2))
    except Exception:
        font = ImageFont.load_default()

    def add_border_and_label(
        frame: np.ndarray,
        label: str,
        color: tuple,
        is_context: bool
    ) -> np.ndarray:
        """Add border and label to frame."""
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)

        # Border (yellow for context model frames, otherwise use specified color)
        border_color = (255, 255, 0) if is_context and color == (255, 0, 0) else color
        draw.rectangle([0, 0, W_up - 1, H_up - 1], outline=border_color, width=max(2, auto_upscale // 4))

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

        # Rotate if direction is available
        if dir_gt is not None and dir_pred is not None:
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

    return video_np


def visualize_buffer_task_distribution(
    buffer_state,
    valid_size: int,
    env_names: list,
    step_count: int,
    log_prefix: str = "buffer/task_distribution",
) -> None:
    """Render a bar chart of task ID proportions in the replay buffer and log to WandB.

    Args:
        buffer_state: ReplayBufferState with data['task_id'] array
        valid_size: Number of valid entries in buffer
        env_names: List of environment names (indexed by task_id)
        step_count: Current training step for WandB logging
        log_prefix: WandB log key
    """
    if wandb.run is None or valid_size == 0:
        return

    import matplotlib
    matplotlib.use('Agg')

    task_ids = buffer_state.data['task_id'][:valid_size]
    num_tasks = len(env_names)

    counts = np.zeros(num_tasks, dtype=np.int64)
    for t in range(num_tasks):
        counts[t] = np.sum(task_ids == t)

    total = counts.sum()
    if total == 0:
        return

    proportions = counts / total

    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
              '#1abc9c', '#e67e22', '#34495e']
    bar_colors = [colors[i % len(colors)] for i in range(num_tasks)]

    # Shorten env names for display (e.g. "Navix/Navix-DoorKey-8x8-v0" -> "DoorKey-8x8")
    short_names = []
    for name in env_names:
        short = name.split('/')[-1]
        for prefix in ('Navix-', 'ALE-'):
            short = short.replace(prefix, '')
        short = short.rsplit('-v', 1)[0]
        short_names.append(short)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(num_tasks), proportions, color=bar_colors)

    ax.set_xticks(range(num_tasks))
    ax.set_xticklabels([f"Task {i}" for i in range(num_tasks)], fontsize=9)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Buffer Task Distribution (step {step_count})", fontsize=11)

    for bar, prop in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{prop:.2f}", ha='center', va='bottom', fontsize=9)

    # Uniform reference line
    ax.axhline(y=1.0 / num_tasks, color='gray', linestyle='--', alpha=0.5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_colors[i])
        for i in range(num_tasks)
    ]
    ax.legend(legend_handles, short_names, loc='upper right', fontsize=8)

    fig.tight_layout()

    wandb.log({
        log_prefix: wandb.Image(fig),
    })

    plt.close(fig)
