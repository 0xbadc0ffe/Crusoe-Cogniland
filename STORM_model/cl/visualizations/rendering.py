"""Rendering utilities for visualizing environment observations (JAX-first)."""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import Array
from typing import Union, Optional, Tuple, List
from matplotlib.patches import FancyArrowPatch


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

    NEW FEATURE: Visual indicator for agent direction.

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

    NEW FEATURE: Grid layout with metadata (actions, rewards, directions).

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
