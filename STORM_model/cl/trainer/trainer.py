"""
Minimal Trainer for Continual RL

Responsibilities:
1. Environment lifecycle management (task switching)
2. Metrics tracking and logging
3. Evaluation orchestration

Agent owns its own training loop (sync or async).
"""
import time
import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm

from cl.agents.base import ContinualAgent
from cl.agents.utils import get_agent_parameter_count
from cl.environments.manager import EnvironmentManager
from cl.metrics.metrics_trackers import TaskMetricsTracker, GlobalMetricsTracker
from cl.shared import log_table, setup_logger
from cl.trainer.run_logger import RunLogger
from cl.trainer.utils import RNGManager
from cl.trainer.checkpoint import CheckpointCallback

logger = setup_logger(__name__)


class Trainer:
    def __init__(self, config: OmegaConf, agent: ContinualAgent):
        """
        Initialize minimal trainer for continual RL.

        Args:
            config: OmegaConf configuration
            agent: ContinualAgent instance (owns its training loop)
        """
        # Configuration
        self.config = config
        self.agent = agent
        self.num_train_frames = self.config.trainer.num_train_frames
        self.num_eval_frames = self.config.trainer.num_eval_frames
        self.eval_interval_frames = self.config.trainer.get("eval_interval_frames", None)

        # Experiment logger
        self.run_logger = RunLogger(config=self.config)
        self.config.results_dir = self.run_logger.results_dir
        logger.info(
            "Starting run: %s, run ID: %s, results directory: %s",
            self.run_logger.run_name,
            self.run_logger.run_id,
            self.run_logger.results_dir
        )

        # RNG management (JAX-style)
        self.rng_manager = RNGManager(seed=self.config.seed)
        logger.info("Initialized RNG with seed: %s", self.config.seed)

        # Load environments
        self.env_manager = EnvironmentManager(self.config)
        self.config.max_actions = int(self.env_manager.max_actions)

        logger.info("Experiment environments:")
        for idx, env_name in self.env_manager.envs_dict.items():
            logger.info(" - %s: %s", idx, env_name)
        logger.info("Max actions: %s", self.config.max_actions)
        logger.info("Observation space: %s", self.env_manager.obs_space)

        # Initialize agent state
        agent_rng = self.rng_manager.get_key()
        self.agent_state = self.agent.init(agent_rng)
        logger.info("Agent initialized")

        # Log parameter counts for agents that support it
        self._log_parameter_counts()

        # Initialize metrics trackers
        self.global_metrics_tracker = GlobalMetricsTracker(config=self.config)
        self.train_metrics_tracker = None  # Initialized per-task
        self.eval_metrics_tracker = None  # Initialized per-task

        # Global frame counter across all tasks
        self.global_frame_counter = 0

        self.evaluation_set = 0

        # Initialize checkpoint callback if checkpointing is enabled
        checkpoint_enabled = self.config.agent.get('checkpoint', {}).get('enabled', False)
        if checkpoint_enabled:
            self.checkpoint_callback = CheckpointCallback(
                agent=self.agent,
                config=self.config,
                results_dir=self.run_logger.results_dir,
                wandb_run=self.run_logger.wandb_run if hasattr(self.run_logger, 'wandb_run') else None,
            )
            logger.info("Checkpoint saving enabled")
        else:
            self.checkpoint_callback = None
            logger.info("Checkpoint saving disabled")

    def _log_parameter_counts(self) -> None:
        """Log agent parameter counts if the agent supports it."""
        trainable, non_trainable, breakdown = get_agent_parameter_count(
            self.agent_state,
            show_breakdown=False
        )

        logger.info("=" * 70)
        logger.info("AGENT PARAMETER COUNT")
        logger.info("=" * 70)
        logger.info("Trainable parameters:     %s (%.2fM)", f"{trainable:,}", trainable/1e6)
        logger.info("Non-trainable parameters: %s (%.2fM)", f"{non_trainable:,}", non_trainable/1e6)
        logger.info("Total parameters:         %s (%.2fM)", f"{trainable + non_trainable:,}", (trainable + non_trainable)/1e6)

        # Log breakdown by component
        logger.info("")
        logger.info("Component breakdown:")
        for name, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            display_name = name.replace('_params', '')
            logger.info("  %-20s: %s (%7.2fM)", display_name, f"{count:12,}", count/1e6)
        logger.info("=" * 70)

        # Log gradient checkpointing status if agent has the config
        use_grad_checkpoint = self.config.agent.get('grad_checkpoint', False)
        if use_grad_checkpoint:
            logger.info("Gradient checkpointing: ENABLED (lower memory, ~10-30% slower)")
        else:
            logger.info("Gradient checkpointing: DISABLED (higher memory, faster)")
        logger.info("")

    def _train_on_task(self, env, env_name: str) -> None:
        """
        Train agent on a single environment with interleaved evaluations.

        Args:
            env: Environment instance
            env_name: Environment name for logging
        """
        logger.info(
            "Training on environment: %s for %s frames",
            env_name,
            f"{self.num_train_frames:,}"
        )

        # Initialize training metrics tracker for this task
        num_parallel_envs = self.config.env.get(
            "num_parallel_envs", self.config.env.num_parallel_envs
        )
        self.train_metrics_tracker = TaskMetricsTracker(
            config=self.config,
            num_parallel_envs=num_parallel_envs,
            mode="train",
        )
        self.train_metrics_tracker.initialize(env_name)

        # Register training metrics with WandB (defines step metric for x-axis)
        self.run_logger.register_metrics(self.train_metrics_tracker, env_name)

        # Initialize checkpoint callback for this environment
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.initialize(env_name)

        # Track the starting global frame for this task (used for offset)
        task_start_frame = self.global_frame_counter

        # Reset agent for new environment (preserves parameters/buffers for continual learning)
        reset_rng = self.rng_manager.get_key()
        self.agent_state = self.agent.reset(self.agent_state, reset_rng)

        # Train with periodic evaluations
        total_frames_trained = 0
        pbar = tqdm(total=self.num_train_frames, desc=f"Training on {env_name}")

        # Initialize timing for FPS calculation
        last_time = time.time()

        while total_frames_trained < self.num_train_frames:
            # Calculate frames for this training segment
            frames_remaining = self.num_train_frames - total_frames_trained
            frames_this_segment = min(self.eval_interval_frames, frames_remaining) if self.eval_interval_frames else frames_remaining

            # Train for this segment
            train_start_time = time.time()
            train_rng = self.rng_manager.get_key()
            self.agent_state, metrics = self.agent.train(
                self.agent_state,
                env,
                train_rng,
                frames_this_segment,
                progress_bar=pbar,
                checkpoint_callback=self.checkpoint_callback,
            )
            train_end_time = time.time()

            # Calculate FPS for this training segment
            elapsed_time = train_end_time - train_start_time
            fps = frames_this_segment / elapsed_time if elapsed_time > 0 else 0.0

            # Process and log metrics from the training segment
            # Pass task start frame as offset for global counter, and total_frames_trained for local counter
            self._log_training_metrics(metrics, env_name, task_start_frame, total_frames_trained, pbar, fps)

            total_frames_trained += frames_this_segment
            # Progress bar is already updated incrementally by agent.train(), no need to update here

            # Run evaluation if we haven't finished training yet and eval_interval is set
            if self.eval_interval_frames and total_frames_trained < self.num_train_frames:
                self.rng_manager.checkpoint()
                self._run_evaluation(
                    current_training_env=env_name,
                    current_set=self.evaluation_set
                )
                self.evaluation_set += 1
                self.rng_manager.restore()

        pbar.close()
        logger.info("Training completed on %s", env_name)

        # Update global frame counter (total frames trained on this task)
        self.global_frame_counter += total_frames_trained
        logger.info(
            "Global frame counter: %s / %s",
            f"{self.global_frame_counter:,}",
            f"{len(self.env_manager.envs_dict) * self.num_train_frames:,}"
        )

        # Save agent state for environment
        self.agent_state = self.agent.on_environment_end(
            self.agent_state, env_name=env_name
        )

    def _log_training_metrics(self, metrics: dict, env_name: str, task_frame_offset: int, total_frames_trained: int, pbar: tqdm, fps: float) -> None:
        """
        Process and log per-episode training metrics from agent using metrics tracker.
        Also logs agent-specific metrics (losses, value estimates, etc.).

        Args:
            metrics: Metrics dict from agent.train() containing episode_info and agent-specific metrics
            env_name: Current environment name
            task_frame_offset: Global frame offset at the start of this task
            total_frames_trained: Total frames trained on this task so far
            pbar: Progress bar to update
            fps: Frames per second for this training segment
        """
        # Extract episode_info from metrics
        # episode_info has shape (num_updates, num_steps, num_envs) for each field
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            # Still log agent-specific metrics even if no episodes completed
            self._log_agent_metrics(metrics, env_name, task_frame_offset + total_frames_trained)
            return

        # Extract the arrays
        returned_episode_returns = jnp.array(episode_info["returned_episode_returns"])  # (num_updates, num_steps, num_envs)
        returned_episode_lengths = jnp.array(episode_info["returned_episode_lengths"])  # (num_updates, num_steps, num_envs)
        returned_episode = jnp.array(episode_info["returned_episode"])                  # (num_updates, num_steps, num_envs) - boolean mask
        timesteps = jnp.array(episode_info["timestep"])                                 # (num_updates, num_steps, num_envs) - LOCAL to this env

        # Flatten to (total_steps * num_envs,) to process all episodes
        returns_flat = returned_episode_returns.reshape(-1)
        lengths_flat = returned_episode_lengths.reshape(-1)
        done_flat = returned_episode.reshape(-1)
        timesteps_flat = timesteps.reshape(-1)

        # VECTORIZED: Extract completed episodes and filter by frame budget
        completed_mask = done_flat & (timesteps_flat <= self.num_train_frames)
        completed_returns = returns_flat[completed_mask]
        completed_lengths = lengths_flat[completed_mask]
        completed_timesteps = timesteps_flat[completed_mask]
        completed_successes = (completed_returns > 0).astype(jnp.int32)

        # Convert episode count to Python (single scalar - fast)
        episode_count = int(completed_mask.sum())
        if episode_count == 0:
            return

        # Batch convert all data from GPU to CPU in one shot (single sync point)
        # This is much faster than converting one value at a time in the loop
        returns_np = np.array(completed_returns)
        lengths_np = np.array(completed_lengths)
        timesteps_np = np.array(completed_timesteps)
        successes_np = np.array(completed_successes)

        # Log each episode to WandB with incrementally updated moving averages
        for i in range(episode_count):
            # Extract values from numpy arrays (no device transfer, already on CPU)
            reward_val = float(returns_np[i])
            success_val = int(successes_np[i])
            length_val = int(lengths_np[i])
            timestep_val = int(timesteps_np[i])

            # Add this episode to history first
            self.train_metrics_tracker.episode_reward_history.append(reward_val)
            self.train_metrics_tracker.episode_success_history.append(success_val)
            self.train_metrics_tracker.episode_length_history.append(length_val)

            self.train_metrics_tracker.env_total_episodes += 1
            self.train_metrics_tracker.env_total_frames = total_frames_trained

            # Calculate global timestep
            global_timestep = task_frame_offset + timestep_val

            # Compute moving averages after adding this episode
            # This gives us high-resolution moving average curves that update per episode
            moving_avg_reward = float(jnp.mean(jnp.array(self.train_metrics_tracker.episode_reward_history)))
            moving_avg_success = float(jnp.mean(jnp.array(self.train_metrics_tracker.episode_success_history)))
            moving_avg_length = float(jnp.mean(jnp.array(self.train_metrics_tracker.episode_length_history)))

            # Log individual episode metrics AND moving averages
            wandb_metrics = {
                f"train/{env_name}/reward": reward_val,
                f"train/{env_name}/success": success_val,
                f"train/{env_name}/length": length_val,
                f"train/{env_name}/moving_avg_reward": moving_avg_reward,
                f"train/{env_name}/moving_avg_success_rate": moving_avg_success,
                f"train/{env_name}/moving_avg_length": moving_avg_length,
                f"train/{env_name}/fps": fps,
                f"train/{env_name}/frame": total_frames_trained,
                f"train/{env_name}/episode": self.train_metrics_tracker.env_total_episodes,
                "train_steps": global_timestep,
                # Add the step metric that WandB expects (without env prefix)
                "train_episode": self.train_metrics_tracker.env_total_episodes,
            }
            self.run_logger.wandb_run.log(wandb_metrics)

        # Update tqdm with latest moving average and FPS
        pbar.set_postfix({
            "episodes": self.train_metrics_tracker.env_total_episodes,
            "moving_avg_reward": f"{moving_avg_reward:.2f}",
            "fps": f"{fps:.0f}"
        })

        # Log agent-specific metrics (losses, advantages, etc.)
        self._log_agent_metrics(metrics, env_name, global_timestep)

    def _log_agent_metrics(self, metrics: dict, env_name: str, global_timestep: int) -> None:
        """
        Log agent-specific training metrics (losses, value estimates, etc.).

        This is a generic method that works for all agents (PPO, DreamerV3, IMPALA, etc.).
        It logs any scalar metrics in the metrics dict, excluding episode_info.

        Args:
            metrics: Metrics dict from agent.train()
            env_name: Current environment name
            global_timestep: Global timestep for x-axis
        """
        agent_metrics = {}

        # Extract all scalar metrics (excluding episode_info which is already handled)
        for key, value in metrics.items():
            if key == 'episode_info':
                continue

            # Log scalar values (int or float)
            if isinstance(value, (int, float)):
                # Add with train/{env_name}/ prefix to match episode metrics
                agent_metrics[f"train/{env_name}/{key}"] = value

        # Log to wandb if there are any agent-specific metrics
        if agent_metrics:
            agent_metrics["train_steps"] = global_timestep  # Use same x-axis as episode metrics
            self.run_logger.wandb_run.log(agent_metrics)

    def _evaluate_on_task(
        self,
        env,
        env_name: str,
        current_set: int,
    ) -> dict:
        """
        Evaluate agent on environment.

        Args:
            env: Environment instance
            env_name: Environment name
            current_set: Evaluation set number

        Returns:
            metrics: Evaluation metrics dict with aggregated statistics
        """
        # Initialize evaluation metrics tracker if not already created
        if self.eval_metrics_tracker is None:
            num_parallel_envs = self.config.env.get(
                "num_parallel_envs_eval",
                self.config.trainer.get("num_parallel_envs_eval", self.config.env.num_parallel_envs)
            )
            self.eval_metrics_tracker = TaskMetricsTracker(
                config=self.config,
                num_parallel_envs=num_parallel_envs,
                mode="eval",
            )

        self.eval_metrics_tracker.initialize(env_name)

        # Create progress bar for evaluation
        pbar = tqdm(total=self.num_eval_frames, desc=f"Eval Set {current_set} on {env_name}", leave=False)

        # Agent owns the evaluation loop
        eval_rng = self.rng_manager.get_key()
        agent_metrics = self.agent.evaluate(
            self.agent_state,
            env,
            eval_rng,
            self.num_eval_frames,
            progress_bar=pbar,  # Pass progress bar for updates
        )

        pbar.n = self.num_eval_frames
        pbar.close()

        # Extract episode info and update metrics tracker
        episode_info = agent_metrics.get("episode_info")
        if episode_info is not None:
            # Process completed episodes - VECTORIZED (no Python loop!)
            returned_episode_returns = jnp.array(episode_info["returned_episode_returns"])
            returned_episode_lengths = jnp.array(episode_info["returned_episode_lengths"])
            returned_episode = jnp.array(episode_info["returned_episode"])

            # Flatten and find completed episodes
            returns_flat = returned_episode_returns.reshape(-1)
            lengths_flat = returned_episode_lengths.reshape(-1)
            done_flat = returned_episode.reshape(-1)

            # Extract completed episodes (vectorized - no loop!)
            completed_returns = returns_flat[done_flat]
            completed_lengths = lengths_flat[done_flat]
            completed_successes = (completed_returns > 0).astype(jnp.int32)

            # Update metrics tracker (bulk append)
            num_completed = int(done_flat.sum())
            self.eval_metrics_tracker.env_total_episodes += num_completed

            # Convert to Python lists once at the end
            if num_completed > 0:
                self.eval_metrics_tracker.episode_reward_history.extend(
                    completed_returns.tolist()
                )
                self.eval_metrics_tracker.episode_success_history.extend(
                    completed_successes.tolist()
                )
                self.eval_metrics_tracker.episode_length_history.extend(
                    completed_lengths.tolist()
                )

        # Get aggregated metrics from tracker
        aggregated_metrics = {
            "avg_reward": float(jnp.mean(jnp.array(list(self.eval_metrics_tracker.episode_reward_history)))),
            "avg_success": float(jnp.mean(jnp.array(list(self.eval_metrics_tracker.episode_success_history)))),
            "avg_length": float(jnp.mean(jnp.array(list(self.eval_metrics_tracker.episode_length_history)))),
            "episodes": self.eval_metrics_tracker.env_total_episodes,
            "frames": agent_metrics.get('frames', self.num_eval_frames),
        }

        # Log evaluation metrics in table format
        log_table(aggregated_metrics, logger, title=f"Eval Set {current_set} on {env_name}")

        # Log to WandB
        wandb_metrics = {
            f"eval/{env_name}/avg_reward": aggregated_metrics["avg_reward"],
            f"eval/{env_name}/avg_success": aggregated_metrics["avg_success"],
            f"eval/{env_name}/avg_length": aggregated_metrics["avg_length"],
            f"eval/{env_name}/episodes": aggregated_metrics["episodes"],
            f"eval/{env_name}/frames": aggregated_metrics["frames"],
            "eval_set": current_set,
        }
        self.run_logger.wandb_run.log(wandb_metrics)

        return aggregated_metrics

    def _run_evaluation(self, current_training_env: str, current_set: int) -> None:
        """
        Run evaluation across all environments and compute continual learning metrics.

        Args:
            current_training_env: Current training environment name
            current_set: Evaluation set number
        """
        logger.info("═══ Evaluation Set %s ═══", current_set)
        logger.info(
            "Evaluating on %s different environments for %s frames each.",
            len(self.env_manager.envs_dict),
            self.num_eval_frames
        )

        all_env_metrics = {}

        for env_idx, env_name in self.env_manager.envs_dict.items():
            env = self.env_manager.get_environment(env_idx=env_idx, train=False)

            # Set environment (for task-specific parameters if needed)
            self.agent_state = self.agent.set_environment(
                self.agent_state, env_name=env_name
            )

            metrics = self._evaluate_on_task(
                env=env,
                env_name=env_name,
                current_set=current_set,
            )
            all_env_metrics[env_name] = metrics

            # Add task metrics to global tracker
            self.global_metrics_tracker.add_task_metrics(
                aggregated_metrics=metrics,
                set_id=current_set,
                env_name=env_name,
            )

        # Restore training environment
        self.agent_state = self.agent.set_environment(
            self.agent_state, env_name=current_training_env
        )

        # Compute and log continual learning metrics (forward transfer, backward transfer, forgetting)
        current_env_idx = self.env_manager._get_env_id(current_training_env)
        if current_env_idx is None:
            current_env_idx = 0

        global_metrics = self.global_metrics_tracker.get_aggregated_metrics(
            current_set_id=current_set,
            current_env_idx=current_env_idx,
            current_training_env=current_training_env,
        )

        # Log global metrics to WandB
        wandb_global_metrics = {
            f"eval/aggregated/{key}": value
            for key, value in global_metrics.items()
        }
        wandb_global_metrics["eval_set"] = current_set
        self.run_logger.wandb_run.log(wandb_global_metrics)

        # Log cross-task summary
        logger.info("─── Cross-Task Performance Summary ───")
        for env_name, metrics in all_env_metrics.items():
            logger.info(
                "%s: Reward=%.3f, Success=%.3f, Episodes=%s",
                env_name,
                metrics.get('avg_reward', 0),
                metrics.get('avg_success', 0),
                metrics.get('episodes', 0)
            )

        # Log continual learning metrics
        logger.info("─── Continual Learning Metrics ───")
        logger.info("Average Reward: %.3f", global_metrics.get('avg_reward', 0))
        logger.info("Average Success: %.3f", global_metrics.get('avg_success', 0))
        if 'bt_avg_reward' in global_metrics:
            logger.info("Backward Transfer (Reward): %.3f", global_metrics['bt_avg_reward'])
        if 'ft_avg_reward' in global_metrics:
            logger.info("Forward Transfer (Reward): %.3f", global_metrics['ft_avg_reward'])
        if 'fg_avg_reward' in global_metrics:
            logger.info("Forgetting (Reward): %.3f", global_metrics['fg_avg_reward'])

        # Save checkpoint after evaluation with validation metrics
        if self.checkpoint_callback is not None:
            # Use current training environment's metrics for best checkpoint tracking
            train_env_metrics = all_env_metrics.get(current_training_env, {})
            checkpoint_metrics = {
                'eval_return': train_env_metrics.get('avg_reward', 0.0),
                'eval_success': train_env_metrics.get('avg_success', 0.0),
            }

            # Get step count from runtime state (unified across all agents)
            step = int(self.agent_state.runtime.train_steps)

            self.checkpoint_callback.on_validation_end(
                agent_state=self.agent_state,
                step=step,
                metrics=checkpoint_metrics,
            )

    def run(self) -> None:
        """Execute complete continual learning pipeline."""
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║           CONTINUAL RL TRAINING PIPELINE                       ║")
        logger.info("╚════════════════════════════════════════════════════════════════╝")
        logger.info(
            "Starting sequential training on %s different environment types, with %s training frames each.",
            len(self.env_manager.envs_dict),
            f"{self.num_train_frames:,}"
        )
        if self.eval_interval_frames is not None:
            logger.info(
                "Evaluation interval: %s frames (%s eval frames per checkpoint)",
                f"{self.eval_interval_frames:,}",
                f"{self.num_eval_frames:,}"
            )

        for env_idx, env_name in self.env_manager.envs_dict.items():
            logger.info("")
            logger.info("╔════════════════════════════════════════════════════════════════╗")
            logger.info("║  Task %s/%s: %-52s ║", env_idx + 1, len(self.env_manager.envs_dict), env_name)
            logger.info("╚════════════════════════════════════════════════════════════════╝")

            # Create training environment
            env = self.env_manager.get_environment(env_idx=env_idx, train=True)

            # Run initial evaluation before training
            if self.eval_interval_frames is not None:
                self.rng_manager.checkpoint()
                self._run_evaluation(
                    current_training_env=env_name,
                    current_set=self.evaluation_set
                )
                self.evaluation_set += 1
                self.rng_manager.restore()

            # Train on environment (includes periodic evaluations)
            self._train_on_task(env, env_name)

            # Run final evaluation after training
            if self.eval_interval_frames is not None:
                self.rng_manager.checkpoint()
                self._run_evaluation(
                    current_training_env=env_name,
                    current_set=self.evaluation_set
                )
                self.evaluation_set += 1
                self.rng_manager.restore()

        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║         TRAINING COMPLETE - ALL TASKS FINISHED                 ║")
        logger.info("╚════════════════════════════════════════════════════════════════╝")
