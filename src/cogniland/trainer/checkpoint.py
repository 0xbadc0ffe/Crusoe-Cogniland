"""Checkpoint management for JAX-based agents using orbax.

This module provides functionality to save and load agent checkpoints using
orbax.checkpoint, which is the recommended checkpointing library for JAX/Flax.

Key Features:
- Save/load AgentState (train_state containing params + optimizer)
- Automatic cleanup of old checkpoints
- Versioning for backward compatibility
- Integration with WandB for checkpoint uploads
- Support for best checkpoint tracking

Design Philosophy:
- Only trainable state (TrainState) is saved - params, optimizer, agent-specific state
- RuntimeState (buffer, wm_state, counters) is NOT saved - it's ephemeral
- Checkpoints include version info and config for reproducibility
- Agent-agnostic: works with any JAX agent (DreamerV3, STORM, IRIS, etc.)
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil

import orbax.checkpoint as ocp
from omegaconf import OmegaConf, DictConfig
import wandb

from cogniland.agents.state import AgentState, RuntimeState
from cogniland.shared import setup_logger

logger = setup_logger(__name__)


# Current checkpoint version for JAX agents
# v2: TrainState-based architecture (train_state instead of wm_params/policy_params/opt)
CHECKPOINT_VERSION = "jax_agent_v2"


class CheckpointManager:
    """Manages checkpoint saving/loading with orbax.

    This class handles:
    - Saving agent state to disk using orbax
    - Loading agent state from disk
    - Cleaning up old checkpoints based on keep_last policy
    - Best checkpoint tracking
    - WandB artifact uploads

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_last: Number of checkpoints to keep (0 = keep all)
        save_best: Whether to save best checkpoint separately

    Example:
        manager = CheckpointManager(
            checkpoint_dir='results/run_001/checkpoints',
            keep_last=3,
            save_best=True
        )

        # Save checkpoint
        manager.save(
            agent_state=state,
            step=1000,
            config=cfg,
            metrics={'eval_return': 100.0}
        )

        # Load checkpoint
        state_dict, config, metadata = manager.load(step=1000)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last: int = 3,
        save_best: bool = True,
    ):
        # CRITICAL: orbax requires absolute paths!
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.keep_last = keep_last
        self.save_best = save_best

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        # Track best checkpoint
        self.best_metric = None
        self.best_step = None

    def save(
        self,
        agent_state: AgentState,
        step: int,
        config: DictConfig,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        best_only: bool = False,
        save_last: bool = False,
    ) -> Optional[Path]:
        """Save agent checkpoint.

        Saves only trainable parameters and optimizer state. RuntimeState
        (replay buffer, world model state) is NOT saved as it's ephemeral.

        Args:
            agent_state: Complete agent state
            step: Training step number
            config: Agent configuration (OmegaConf)
            metrics: Optional metrics to save with checkpoint
            is_best: Whether this is the best checkpoint so far
            best_only: If True, only save to 'best/' if this is the best checkpoint,
                      skip creating intermediate step_* checkpoints

        Returns:
            Path to saved checkpoint directory, or None if best_only and not best
        """
        # Extract trainable state (only JAX PyTree - orbax compatible)
        # All agents now use unified AgentState with train_state + runtime
        # train_state is a PyTree (TrainState or subclass like DreamerV3TrainState)
        state_pytree = {
            'train_state': agent_state.train_state,
        }

        # Prepare metadata (non-PyTree data)
        custom_metadata = {
            'version': CHECKPOINT_VERSION,
            'step': step,
            'metrics': metrics or {},
            'config': OmegaConf.to_container(config, resolve=True),
        }

        # Check if this is the best checkpoint
        is_new_best = is_best or (self.save_best and self._is_best(metrics))

        # If best_only mode and not the best, skip saving entirely
        if best_only and not is_new_best:
            return None

        checkpoint_path = None

        # Save regular step checkpoint (skip if best_only mode)
        if not best_only:
            checkpoint_path = self.checkpoint_dir / f"step_{step:08d}"

            # Check if checkpoint already exists (can happen during cross-task evaluation)
            if checkpoint_path.exists():
                logger.debug(f"Checkpoint at step {step} already exists, skipping save")
            else:
                self.checkpointer.save(
                    checkpoint_path,
                    state_pytree,
                )

                # Wait for async save to complete before writing metadata
                self.checkpointer.wait_until_finished()

                # Save metadata separately as JSON
                import json
                metadata_path = checkpoint_path / "custom_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(custom_metadata, f, indent=2)

                logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")

        # Save as best if this is a new best
        if is_new_best:
            best_path = self.checkpoint_dir / "best"

            # Remove existing best checkpoint if it exists (orbax doesn't allow overwrite)
            if best_path.exists():
                import shutil
                shutil.rmtree(best_path)

            self.checkpointer.save(
                best_path,
                state_pytree,
            )

            # Wait for async save to complete
            self.checkpointer.wait_until_finished()

            # Save metadata for best checkpoint
            import json
            best_metadata_path = best_path / "custom_metadata.json"
            with open(best_metadata_path, 'w') as f:
                json.dump(custom_metadata, f, indent=2)

            if metrics:
                self.best_metric = metrics.get('eval_success', metrics.get('eval_return'))
                self.best_step = step
                logger.info(
                    f"New best checkpoint at step {step} "
                    f"(eval_success={self.best_metric:.4f})"
                )

        # Save / overwrite "last" checkpoint
        if save_last:
            last_path = self.checkpoint_dir / "last"
            if last_path.exists():
                shutil.rmtree(last_path)
            self.checkpointer.save(last_path, state_pytree)
            self.checkpointer.wait_until_finished()
            import json
            last_metadata_path = last_path / "custom_metadata.json"
            with open(last_metadata_path, 'w') as f:
                json.dump(custom_metadata, f, indent=2)
            logger.debug(f"Saved last checkpoint at step {step} to {last_path}")

        if best_only and is_new_best:
            return self.checkpoint_dir / "best"

        # Cleanup old checkpoints (only relevant if not best_only mode)
        if not best_only and self.keep_last > 0:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        step: Optional[int] = None,
        load_best: bool = False,
    ) -> tuple[Dict[str, Any], DictConfig, Dict[str, Any]]:
        """Load agent checkpoint.

        Args:
            step: Training step to load (None = latest)
            load_best: Load best checkpoint instead of step-based

        Returns:
            Tuple of (state_dict, config, metadata)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best"
        elif step is not None:
            checkpoint_path = self.checkpoint_dir / f"step_{step:08d}"
        else:
            # Load latest
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint PyTree
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = self.checkpointer.restore(checkpoint_path)

        # Load custom metadata from JSON file
        import json
        metadata_path = checkpoint_path / "custom_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                custom_metadata = json.load(f)
        else:
            custom_metadata = {}

        # Extract config and metadata
        config = OmegaConf.create(custom_metadata.get('config', {}))
        metadata = {
            'step': custom_metadata.get('step', 0),
            'metrics': custom_metadata.get('metrics', {}),
            'version': custom_metadata.get('version', CHECKPOINT_VERSION),
        }

        return state_dict, config, metadata

    def restore_agent_state(
        self,
        checkpoint_data: Dict[str, Any],
        runtime_state: RuntimeState,
    ) -> AgentState:
        """Restore AgentState from checkpoint data.

        Combines saved train_state (params + optimizer + agent-specific state)
        with current runtime state (replay buffer, world model state, etc.).

        Args:
            checkpoint_data: Loaded checkpoint data (contains 'train_state')
            runtime_state: Current runtime state (buffer, wm_state, counters, rng)

        Returns:
            Complete AgentState
        """
        return AgentState(
            train_state=checkpoint_data['train_state'],
            runtime=runtime_state,
        )

    def _is_best(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if current metrics are the best so far.

        Tracks the highest average eval success rate. Falls back to
        ``eval_return`` for back-compat with older callers.
        """
        if not metrics:
            return False
        current = metrics.get('eval_success', metrics.get('eval_return'))
        if current is None:
            return False
        if self.best_metric is None:
            return True
        return current > self.best_metric

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        checkpoint_dirs = sorted(
            [p for p in self.checkpoint_dir.glob("step_*") if p.is_dir()]
        )
        return checkpoint_dirs[-1] if checkpoint_dirs else None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_dirs = sorted(
            [p for p in self.checkpoint_dir.glob("step_*") if p.is_dir()]
        )

        # Keep last N checkpoints
        if len(checkpoint_dirs) > self.keep_last:
            for old_dir in checkpoint_dirs[:-self.keep_last]:
                logger.debug(f"Removing old checkpoint: {old_dir}")
                shutil.rmtree(old_dir)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata.

        Returns:
            List of checkpoint info dicts with 'step', 'path', 'size'
        """
        checkpoints = []

        for checkpoint_path in sorted(self.checkpoint_dir.glob("step_*")):
            if checkpoint_path.is_dir():
                # Extract step from directory name
                step = int(checkpoint_path.name.split("_")[1])

                # Calculate checkpoint size
                size_bytes = sum(
                    f.stat().st_size
                    for f in checkpoint_path.rglob("*")
                    if f.is_file()
                )

                checkpoints.append({
                    'step': step,
                    'path': checkpoint_path,
                    'size_mb': size_bytes / (1024 * 1024),
                })

        return checkpoints


class CheckpointCallback:
    """Checkpoint callback for integration with Trainer.

    This callback provides a similar interface to the old PyTorch-based
    CheckpointCallback, but uses orbax for JAX agents.

    Args:
        agent: JAX agent instance
        config: Full configuration (must have checkpoint section with optional upload_to_wandb flag)
        results_dir: Results directory path
        wandb_run: Optional WandB run for artifact uploads

    Config options (in model.checkpoint):
        - enabled: Whether to enable checkpointing (default: True)
        - interval: Save checkpoint every N steps (default: 1000)
        - keep_last: Number of checkpoints to keep (default: 3)
        - save_best: Whether to track and save best checkpoint (default: True)
        - save_only_best: Only save best checkpoint, skip intermediate step_* checkpoints (default: False)
        - upload_to_wandb: Whether to upload best checkpoint to WandB (default: False)

    Usage:
        callback = CheckpointCallback(
            agent=agent,
            config=cfg,
            results_dir='results/run_001',
            wandb_run=wandb.run
        )

        # During training
        callback.initialize(env_name='CartPole-v1')
        callback.on_train_step_end(agent_state, step=1000)
        callback.on_validation_end(agent_state, metrics)
    """

    def __init__(
        self,
        agent: Any,
        config: DictConfig,
        results_dir: str,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        self.agent = agent
        self.full_config = config
        self.checkpoint_config = config.agent.get('checkpoint', {})
        self.wandb_run = wandb_run

        # Setup checkpoint directory
        checkpoint_dir_name = self.checkpoint_config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = Path(results_dir) / checkpoint_dir_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config once
        OmegaConf.save(config=config, f=self.checkpoint_dir / "run_config.yaml")

        # Checkpoint settings
        self.enabled = self.checkpoint_config.get('enabled', True)
        self.interval = self.checkpoint_config.get('interval', 1000)
        self.keep_last = self.checkpoint_config.get('keep_last', 3)
        self.save_best = self.checkpoint_config.get('save_best', True)
        self.save_last = self.checkpoint_config.get('save_last', True)
        self.save_only_best = self.checkpoint_config.get('save_only_best', False)
        self.upload_to_wandb = self.checkpoint_config.get('upload_to_wandb', False)

        # Create checkpoint manager (will be initialized per environment)
        self.manager = None
        self.env_name = None

    def initialize(self, env_name: str) -> None:
        """Initialize for a new environment.

        Args:
            env_name: Name of the environment
        """
        self.env_name = env_name
        env_checkpoint_dir = self.checkpoint_dir / env_name
        env_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create manager for this environment
        self.manager = CheckpointManager(
            checkpoint_dir=str(env_checkpoint_dir),
            keep_last=self.keep_last,
            save_best=self.save_best,
        )

        logger.info(f"Initialized checkpointing for environment: {env_name}")

    def on_train_step_end(
        self,
        agent_state: AgentState,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called at the end of each training step.

        Args:
            agent_state: Current agent state
            step: Training step number
            metrics: Optional training metrics
        """
        if not self.enabled or self.manager is None:
            return

        # Skip periodic saves if save_only_best is enabled
        if self.save_only_best:
            return

        # Save checkpoint at interval
        if self.interval > 0 and step % self.interval == 0:
            self.manager.save(
                agent_state=agent_state,
                step=step,
                config=self.full_config,
                metrics=metrics,
            )

    def on_validation_end(
        self,
        agent_state: AgentState,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called after validation.

        Args:
            agent_state: Current agent state
            step: Training step number
            metrics: Validation metrics
        """
        if not self.enabled or self.manager is None:
            return
        if not (self.save_best or self.save_last):
            return

        # Save checkpoint with metrics (manager will determine if it's best).
        # If save_only_best is enabled, only save to 'best/' when this is actually the best.
        # If save_last is enabled, always refresh 'last/'.
        checkpoint_path = self.manager.save(
            agent_state=agent_state,
            step=step,
            config=self.full_config,
            metrics=metrics,
            best_only=self.save_only_best,
            save_last=self.save_last,
        )

        # Upload best checkpoint to WandB if this was a new best (and upload is enabled)
        if self.upload_to_wandb and self.wandb_run is not None and self.manager.best_step == step:
            self._upload_to_wandb(checkpoint_path)

    def restore_best_from(
        self,
        env_name: str,
        runtime_state: RuntimeState,
    ) -> Optional[AgentState]:
        """Restore best checkpoint from a previous environment.

        Args:
            env_name: Name of environment to restore from
            runtime_state: Current runtime state (buffer, wm_state, etc.)

        Returns:
            Restored AgentState if checkpoint exists, None otherwise
        """
        env_checkpoint_dir = self.checkpoint_dir / env_name
        if not env_checkpoint_dir.exists():
            logger.info(f"No checkpoints found for {env_name}")
            return None

        manager = CheckpointManager(
            checkpoint_dir=str(env_checkpoint_dir),
            keep_last=0,
            save_best=False,
        )

        try:
            checkpoint_data, config, metadata = manager.load(load_best=True)

            # Use agent's method to convert checkpoint dicts to dataclasses
            restored_state = self.agent.state_from_checkpoint(
                checkpoint_data=checkpoint_data,
                runtime_state=runtime_state,
            )

            logger.info(
                f"Restored best checkpoint from {env_name} "
                f"(step={metadata['step']})"
            )
            return restored_state
        except FileNotFoundError:
            logger.info(f"No best checkpoint found for {env_name}")
            return None

    def _upload_to_wandb(self, checkpoint_path: Path) -> None:
        """Upload checkpoint to WandB as artifact.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        if self.wandb_run is None:
            return

        try:
            # Sanitize environment name for WandB
            # Extract just the env name after namespace (e.g., "Navix/Navix-DoorKey" -> "Navix-DoorKey")
            if '/' in self.env_name:
                safe_env_name = self.env_name.split('/', 1)[1].replace('/', '-')
            else:
                safe_env_name = self.env_name.replace('/', '-')

            artifact_name = f"ckpt-{safe_env_name}-best"
            art = wandb.Artifact(artifact_name, type="model")

            # Add checkpoint directory (avoids duplicate file issues with orbax metadata)
            art.add_dir(str(checkpoint_path))

            self.wandb_run.log_artifact(art, aliases=[f"{safe_env_name}-latest"])
            logger.info(f"Uploaded checkpoint to WandB: {artifact_name}")
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint to WandB: {e}")


# Convenience functions for standalone use

def save_checkpoint(
    agent_state: AgentState,
    checkpoint_dir: str,
    step: int,
    config: DictConfig,
    metrics: Optional[Dict[str, float]] = None,
    keep_last: int = 3,
    save_best: bool = True,
) -> Path:
    """Convenience function to save a checkpoint.

    Args:
        agent_state: Complete agent state
        checkpoint_dir: Directory to save checkpoints
        step: Training step number
        config: Agent configuration
        metrics: Optional metrics to save
        keep_last: Number of checkpoints to keep
        save_best: Whether to save best checkpoint

    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last=keep_last,
        save_best=save_best,
    )

    return manager.save(
        agent_state=agent_state,
        step=step,
        config=config,
        metrics=metrics,
    )


def load_checkpoint(
    checkpoint_dir: str,
    step: Optional[int] = None,
    load_best: bool = False,
) -> tuple[Dict[str, Any], DictConfig, Dict[str, Any]]:
    """Convenience function to load a checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        step: Training step to load (None = latest)
        load_best: Load best checkpoint instead

    Returns:
        Tuple of (checkpoint_data, config, metadata)
    """
    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    return manager.load(step=step, load_best=load_best)


# Export public API
__all__ = [
    'CheckpointManager',
    'CheckpointCallback',
    'save_checkpoint',
    'load_checkpoint',
    'CHECKPOINT_VERSION',
]
