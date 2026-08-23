"""
Multi-Seed Trainer for Continual RL

Trains K agents in parallel across multiple tasks using jax.vmap.
Provides 3-6× speedup vs K separate processes on a single GPU.
"""

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from cl.agents.experimental.vmapped_agent import VmappedAgent
from cl.environments.manager import EnvironmentManager
from cl.shared import log_table, setup_logger
from cl.trainer.run_logger import RunLogger
from cl.trainer.utils import RNGManager

logger = setup_logger(__name__)


class MultiSeedTrainer:
    """
    Trainer for multi-seed continual RL experiments.

    Uses a VmappedAgent to train K seeds in parallel on a single GPU.
    Handles metrics aggregation and per-seed logging to W&B.

    Args:
        config: OmegaConf configuration
        agent: VmappedAgent instance (wraps K agents)
        num_seeds: Number of seeds being trained
        base_seed: Base random seed
    """

    def __init__(
        self,
        config: OmegaConf,
        agent: VmappedAgent,
        num_seeds: int,
        base_seed: int = 42,
    ):
        self.config = config
        self.agent = agent
        self.num_seeds = num_seeds
        self.base_seed = base_seed

        # Training configuration
        self.num_train_frames = self.config.trainer.num_train_frames
        self.num_eval_frames = self.config.trainer.num_eval_frames
        self.eval_interval_frames = self.config.trainer.get("eval_interval_frames", None)

        # Setup RunLogger WITHOUT creating a W&B run (we'll create K seed runs later)
        self.run_logger = RunLogger(config=self.config, disable_wandb=True)
        self.wandb_group = f"{self.run_logger.run_name}-multiseed"

        # Check if W&B is enabled (not offline mode)
        self.wandb_enabled = not config.get("offline", False)

        # Base run name for seed-specific runs
        base_run_name = self.run_logger.run_name

        # Store W&B config for batch logging at the end
        # (W&B doesn't support multiple concurrent runs in same process)
        self.seed_configs = []
        for seed_idx in range(num_seeds):
            self.seed_configs.append({
                "project": self.run_logger.wandb_project,
                "entity": self.run_logger.wandb_entity,
                "group": self.wandb_group,
                "name": f"{base_run_name}_seed-{seed_idx}",
                "config": {
                    "seed": base_seed + seed_idx,
                    "seed_idx": seed_idx,
                    **OmegaConf.to_container(config, resolve=True),
                },
            })

        self.config.results_dir = self.run_logger.results_dir
        logger.info(
            f"Starting multi-seed run: {self.run_logger.run_name}, "
            f"num_seeds: {num_seeds}, "
            f"W&B group: {self.wandb_group}"
        )

        # RNG management
        self.rng_manager = RNGManager(seed=self.config.seed)
        logger.info(f"Initialized RNG with base seed: {self.config.seed}")

        # Load environments
        self.env_manager = EnvironmentManager(self.config)
        self.config.max_actions = int(self.env_manager.max_actions)

        logger.info("Experiment environments:")
        for idx, env_name in self.env_manager.envs_dict.items():
            logger.info(f" - {idx}: {env_name}")
        logger.info(f"Max actions: {self.config.max_actions}")
        logger.info(f"Observation space: {self.env_manager.obs_space}")

        # Initialize vmapped agent state
        agent_rng = self.rng_manager.get_key()
        self.agent_state = self.agent.init(agent_rng)
        logger.info(f"Initialized {num_seeds} agents in parallel")

        # Global frame counter
        self.global_frame_counter = 0
        self.evaluation_set = 0

        # Accumulate metrics for batch logging at end
        # (W&B doesn't support multiple concurrent runs)
        self.seed_train_metrics = [[] for _ in range(num_seeds)]
        self.seed_eval_metrics = [[] for _ in range(num_seeds)]

    def _accumulate_episode_metrics(self, seed_idx: int, metrics: dict, mode: str = "train"):
        """
        Accumulate episode-level metrics for later batch logging.

        Args:
            seed_idx: Seed index
            metrics: Metrics dict to accumulate
            mode: "train" or "eval"
        """
        if not self.wandb_enabled:
            return

        if mode == "train":
            self.seed_train_metrics[seed_idx].append(metrics)
        else:
            self.seed_eval_metrics[seed_idx].append(metrics)

    def _finish_wandb_logging(self):
        """Create W&B runs and log all accumulated metrics."""
        if not self.wandb_enabled:
            return

        import wandb

        logger.info(f"Logging {self.num_seeds} seed runs to W&B...")

        for seed_idx in range(self.num_seeds):
            # Create run for this seed
            run = wandb.init(**self.seed_configs[seed_idx], reinit=True)

            # Log all accumulated metrics for this seed
            for metrics in self.seed_train_metrics[seed_idx]:
                run.log(metrics)

            for metrics in self.seed_eval_metrics[seed_idx]:
                run.log(metrics)

            # Finish this seed's run
            wandb.finish()

        logger.info(f"✓ Logged {self.num_seeds} W&B runs (group: {self.wandb_group})")

    def _log_training_metrics_multiseed(
        self,
        metrics: dict,
        env_name: str,
        total_frames_trained: int,
        pbar: tqdm,
        fps: float,
    ) -> None:
        """
        Process and log per-seed training metrics.

        Args:
            metrics: Metrics dict with shape (num_seeds, num_updates, num_steps, num_envs)
            env_name: Current environment name
            total_frames_trained: Total frames trained on this task
            pbar: Progress bar to update
            fps: Frames per second
        """
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            return

        # Extract arrays: shape (num_seeds, num_updates, num_steps, num_envs)
        returned_episode_returns = jnp.array(episode_info["returned_episode_returns"])
        returned_episode_lengths = jnp.array(episode_info["returned_episode_lengths"])
        returned_episode = jnp.array(episode_info["returned_episode"])

        # Process each seed separately
        per_seed_metrics = []

        for seed_idx in range(self.num_seeds):
            # Extract this seed's data: (num_updates, num_steps, num_envs)
            seed_returns = returned_episode_returns[seed_idx]
            seed_lengths = returned_episode_lengths[seed_idx]
            seed_done = returned_episode[seed_idx]

            # Flatten to find completed episodes
            returns_flat = seed_returns.reshape(-1)
            lengths_flat = seed_lengths.reshape(-1)
            done_flat = seed_done.reshape(-1)

            # Filter completed episodes
            completed_mask = done_flat > 0
            completed_returns = returns_flat[completed_mask]
            completed_lengths = lengths_flat[completed_mask]

            if len(completed_returns) > 0:
                mean_return = float(jnp.mean(completed_returns))
                mean_length = float(jnp.mean(completed_lengths))
                num_episodes = int(jnp.sum(completed_mask))
            else:
                mean_return = 0.0
                mean_length = 0.0
                num_episodes = 0

            per_seed_metrics.append({
                "mean_return": mean_return,
                "mean_length": mean_length,
                "num_episodes": num_episodes,
            })

            # Extract and log INDIVIDUAL episodes (not just aggregates)
            # This gives us 25k+ data points like single-agent case
            if len(completed_returns) > 0:
                # Convert to numpy once
                import numpy as np
                returns_np = np.array(completed_returns)
                lengths_np = np.array(completed_lengths)

                # Log each episode individually
                for ep_idx in range(len(returns_np)):
                    self._accumulate_episode_metrics(seed_idx, {
                        f"train/{env_name}/reward": float(returns_np[ep_idx]),
                        f"train/{env_name}/length": int(lengths_np[ep_idx]),
                        f"train/{env_name}/fps": fps,
                        f"train/{env_name}/frame": total_frames_trained,
                    }, mode="train")

        # Compute aggregate statistics across seeds
        all_returns = [m["mean_return"] for m in per_seed_metrics]
        all_lengths = [m["mean_length"] for m in per_seed_metrics]

        mean_return = float(np.mean(all_returns))
        std_return = float(np.std(all_returns))
        mean_length = float(np.mean(all_lengths))

        # Update progress bar with aggregate stats
        pbar.set_postfix({
            "return": f"{mean_return:.2f}±{std_return:.2f}",
            "length": f"{mean_length:.1f}",
            "fps": f"{fps:.0f}",
        })

    def _train_on_task(self, env, env_name: str) -> None:
        """
        Train K agents in parallel on a single environment.

        Args:
            env: Environment instance
            env_name: Environment name for logging
        """
        logger.info(
            f"Training {self.num_seeds} agents on {env_name} for {self.num_train_frames:,} frames each"
        )

        # Reset agents for new environment
        reset_rng = self.rng_manager.get_key()
        self.agent_state = self.agent.reset(reset_rng)

        # Train with periodic evaluations
        total_frames_trained = 0
        pbar = tqdm(total=self.num_train_frames, desc=f"Training on {env_name}")

        while total_frames_trained < self.num_train_frames:
            # Calculate frames for this training segment
            frames_remaining = self.num_train_frames - total_frames_trained
            frames_this_segment = (
                min(self.eval_interval_frames, frames_remaining)
                if self.eval_interval_frames
                else frames_remaining
            )

            # Train for this segment (vmapped across all seeds)
            train_start_time = time.time()
            train_rng = self.rng_manager.get_key()
            self.agent_state, metrics = self.agent.train(
                self.agent_state,
                env,
                train_rng,
                frames_this_segment,
            )
            train_end_time = time.time()

            # Calculate FPS
            elapsed_time = train_end_time - train_start_time
            fps = frames_this_segment / elapsed_time if elapsed_time > 0 else 0.0

            # Log metrics for all seeds
            self._log_training_metrics_multiseed(
                metrics, env_name, total_frames_trained, pbar, fps
            )

            total_frames_trained += frames_this_segment
            pbar.n = total_frames_trained
            pbar.refresh()

            # Run evaluation if needed
            if self.eval_interval_frames and total_frames_trained < self.num_train_frames:
                self.rng_manager.checkpoint()
                self._run_evaluation(current_training_env=env_name)
                self.evaluation_set += 1
                self.rng_manager.restore()

        pbar.close()
        logger.info(f"Training completed on {env_name}")

        # Update global frame counter
        self.global_frame_counter += total_frames_trained

        # Save agent state for environment
        self.agent_state = self.agent.on_environment_end(
            self.agent_state, env_name=env_name
        )

    def _evaluate_on_task(self, env, env_name: str) -> dict:
        """
        Evaluate K agents in parallel on an environment.

        Args:
            env: Environment instance
            env_name: Environment name

        Returns:
            metrics: Aggregated evaluation metrics across seeds
        """
        pbar = tqdm(
            total=self.num_eval_frames,
            desc=f"Eval Set {self.evaluation_set} on {env_name}",
            leave=False,
        )

        # Evaluate all seeds in parallel
        eval_rng = self.rng_manager.get_key()
        metrics = self.agent.evaluate(
            self.agent_state,
            env,
            eval_rng,
            self.num_eval_frames,
        )

        pbar.n = self.num_eval_frames
        pbar.close()

        # Process evaluation metrics per seed
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            return {}

        # Extract arrays: shape (num_seeds, ...)
        returned_episode_returns = jnp.array(episode_info["returned_episode_returns"])
        returned_episode_lengths = jnp.array(episode_info["returned_episode_lengths"])
        returned_episode = jnp.array(episode_info["returned_episode"])

        per_seed_stats = []

        for seed_idx in range(self.num_seeds):
            # Extract this seed's data
            seed_returns = returned_episode_returns[seed_idx]
            seed_lengths = returned_episode_lengths[seed_idx]
            seed_done = returned_episode[seed_idx]

            # Flatten and filter completed episodes
            returns_flat = seed_returns.reshape(-1)
            lengths_flat = seed_lengths.reshape(-1)
            done_flat = seed_done.reshape(-1)

            completed_returns = returns_flat[done_flat > 0]
            completed_lengths = lengths_flat[done_flat > 0]

            if len(completed_returns) > 0:
                mean_return = float(jnp.mean(completed_returns))
                mean_length = float(jnp.mean(completed_lengths))
                num_episodes = len(completed_returns)
            else:
                mean_return = 0.0
                mean_length = 0.0
                num_episodes = 0

            per_seed_stats.append({
                "avg_reward": mean_return,
                "avg_length": mean_length,
                "episodes": num_episodes,
            })

            # Accumulate eval metrics for batch logging
            self._accumulate_episode_metrics(seed_idx, {
                f"eval/{env_name}/avg_reward": mean_return,
                f"eval/{env_name}/avg_length": mean_length,
                f"eval/{env_name}/episodes": num_episodes,
                "eval_set": self.evaluation_set,
            }, mode="eval")

        # Compute aggregate statistics
        all_rewards = [s["avg_reward"] for s in per_seed_stats]
        all_lengths = [s["avg_length"] for s in per_seed_stats]

        aggregated_metrics = {
            "avg_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "avg_length": float(np.mean(all_lengths)),
            "std_length": float(np.std(all_lengths)),
            "episodes": sum(s["episodes"] for s in per_seed_stats),
        }

        # Log aggregated metrics
        log_table(
            aggregated_metrics,
            logger,
            title=f"Eval Set {self.evaluation_set} on {env_name} (mean±std across {self.num_seeds} seeds)",
        )

        return aggregated_metrics

    def _run_evaluation(self, current_training_env: str) -> None:
        """
        Run evaluation across all environments for all seeds.

        Args:
            current_training_env: Current training environment name
        """
        logger.info(f"═══ Evaluation Set {self.evaluation_set} ═══")
        logger.info(
            f"Evaluating {self.num_seeds} agents on {len(self.env_manager.envs_dict)} "
            f"environments for {self.num_eval_frames} frames each"
        )

        for env_idx, env_name in self.env_manager.envs_dict.items():
            env = self.env_manager.get_environment(env_idx=env_idx, train=False)

            # Set environment for all seeds
            self.agent_state = self.agent.set_environment(
                self.agent_state, env_name=env_name
            )

            # Evaluate
            self._evaluate_on_task(env=env, env_name=env_name)

        # Restore training environment
        self.agent_state = self.agent.set_environment(
            self.agent_state, env_name=current_training_env
        )

    def run(self) -> None:
        """Execute complete continual learning pipeline for all seeds."""
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║      MULTI-SEED CONTINUAL RL TRAINING PIPELINE                ║")
        logger.info("╚════════════════════════════════════════════════════════════════╝")
        logger.info(
            f"Training {self.num_seeds} agents in parallel on {len(self.env_manager.envs_dict)} "
            f"tasks sequentially, {self.num_train_frames:,} frames each"
        )

        for env_idx, env_name in self.env_manager.envs_dict.items():
            logger.info("")
            logger.info("╔════════════════════════════════════════════════════════════════╗")
            logger.info(f"║  Task {env_idx + 1}/{len(self.env_manager.envs_dict)}: {env_name:<52} ║")
            logger.info("╚════════════════════════════════════════════════════════════════╝")

            # Create training environment
            env = self.env_manager.get_environment(env_idx=env_idx, train=True)

            # Initial evaluation
            if self.eval_interval_frames is not None:
                self.rng_manager.checkpoint()
                self._run_evaluation(current_training_env=env_name)
                self.evaluation_set += 1
                self.rng_manager.restore()

            # Train on environment
            self._train_on_task(env, env_name)

            # Final evaluation
            if self.eval_interval_frames is not None:
                self.rng_manager.checkpoint()
                self._run_evaluation(current_training_env=env_name)
                self.evaluation_set += 1
                self.rng_manager.restore()

        logger.info("")
        logger.info("╔════════════════════════════════════════════════════════════════╗")
        logger.info("║         TRAINING COMPLETE - ALL TASKS FINISHED                 ║")
        logger.info("╚════════════════════════════════════════════════════════════════╝")

        # Log all accumulated metrics to W&B
        self._finish_wandb_logging()
