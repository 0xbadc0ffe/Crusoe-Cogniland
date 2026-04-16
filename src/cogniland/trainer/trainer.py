import time
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm
from tabulate import tabulate

from cogniland.agents.agent import Agent
from cogniland.envs.registry import make_env
from cogniland.envs.task_sampler import TaskSampler
from cogniland.metrics.tracker import MetricsTracker
from cogniland.trainer.run_logger import RunLogger
from cogniland.trainer.utils import RNGManager
from cogniland.shared import setup_logger

logger = setup_logger(__name__)

# Import CheckpointCallback; it may be a stub -- that's fine,
# the Trainer guards usage behind config.agent.checkpoint.enabled.
try:
    from cogniland.trainer.checkpoint import CheckpointCallback
except ImportError:
    CheckpointCallback = None


class Trainer:
    def __init__(self, config: OmegaConf, agent: Agent):
        self.config = config
        self.agent = agent
        self.num_tasks = config.num_tasks

        self.num_train_frames = config.trainer.num_train_frames
        self.num_eval_frames = config.trainer.num_eval_frames
        self.eval_interval_frames = config.trainer.get("eval_interval_frames", None)

        # W&B
        self.run_logger = RunLogger(config)
        config.results_dir = self.run_logger.results_dir
        self.run_logger.wandb_run.define_metric("eval/*", step_metric="train_frames")

        # RNG
        self.rng_manager = RNGManager(seed=config.seed)

        # Environments
        self.train_env = make_env(config.env_id, config, train=True)
        self.eval_env = make_env(config.env_id, config, train=False)

        # Task sampler
        self.task_sampler = TaskSampler(
            num_tasks=self.num_tasks,
            num_envs=config.env.num_parallel_envs,
            mode=config.get("task_sampling", "round_robin"),
        )

        # Agent
        self.agent_state = self.agent.init(self.rng_manager.get_key())

        # Metrics: one train tracker (aggregate), N eval trackers (per-task)
        self.train_metrics = MetricsTracker(config, config.env.num_parallel_envs, "train")
        self.train_metrics.initialize()
        self.run_logger.register_metrics(self.train_metrics)

        num_eval_envs = config.env.get("num_parallel_envs_eval", config.env.num_parallel_envs)
        self.eval_trackers = {}
        for task_id in range(self.num_tasks):
            t = MetricsTracker(config, num_eval_envs, "eval")
            self.eval_trackers[task_id] = t
            self.run_logger.register_metrics(t, prefix_override=f"eval/task_{task_id}")

        self.eval_set = 0

        # Checkpoint
        if config.agent.get("checkpoint", {}).get("enabled", False) and CheckpointCallback is not None:
            self.checkpoint_callback = CheckpointCallback(
                agent=self.agent, config=config,
                results_dir=self.run_logger.results_dir,
                wandb_run=self.run_logger.wandb_run,
            )
        else:
            self.checkpoint_callback = None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        logger.info("=== Multi-task training start (%d tasks) ===", self.num_tasks)
        total_trained = 0
        pbar = tqdm(total=self.num_train_frames, desc="train")

        if self.eval_interval_frames is not None:
            self._run_evaluation(global_train_frames=0)

        while total_trained < self.num_train_frames:
            remaining = self.num_train_frames - total_trained
            seg = min(self.eval_interval_frames or remaining, remaining)

            # Sample task assignments for this training segment
            rng = self.rng_manager.get_key()
            task_rng, train_rng = jax.random.split(rng)
            task_ids = self.task_sampler.sample(rng=task_rng)

            t0 = time.time()
            self.agent_state, metrics = self.agent.train(
                self.agent_state, self.train_env, train_rng, seg,
                progress_bar=pbar,
                checkpoint_callback=self.checkpoint_callback,
                task_ids=task_ids,
            )
            fps = seg / max(time.time() - t0, 1e-9)

            self._log_training_metrics(metrics, total_trained, pbar, fps)
            total_trained += seg

            if self.eval_interval_frames and total_trained < self.num_train_frames:
                self.rng_manager.checkpoint()
                self._run_evaluation(global_train_frames=total_trained)
                self.rng_manager.restore()

        if self.eval_interval_frames is not None:
            self._run_evaluation(global_train_frames=total_trained)

        pbar.close()
        logger.info("=== Training done ===")

    # ------------------------------------------------------------------ #
    # Training metrics (aggregate across tasks)
    # ------------------------------------------------------------------ #
    def _log_training_metrics(self, metrics: dict, total_trained: int, pbar, fps: float):
        """Log aggregate training metrics. Task identity is not tracked here."""
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            self._log_agent_metrics(metrics, total_trained)
            return

        returns = jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
        lengths = jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
        done = jnp.array(episode_info["returned_episode"]).reshape(-1)

        if not bool(done.any()):
            self._log_agent_metrics(metrics, total_trained)
            return

        returns_np = np.array(returns[done])
        lengths_np = np.array(lengths[done])
        successes_np = (returns_np > 0).astype(np.int32)

        for i in range(len(returns_np)):
            r, l, s = float(returns_np[i]), int(lengths_np[i]), int(successes_np[i])
            self.train_metrics.episode_reward_history.append(r)
            self.train_metrics.episode_length_history.append(l)
            self.train_metrics.episode_success_history.append(s)
            self.train_metrics.env_total_episodes += 1

            ma_r = float(np.mean(self.train_metrics.episode_reward_history))
            ma_s = float(np.mean(self.train_metrics.episode_success_history))
            ma_l = float(np.mean(self.train_metrics.episode_length_history))

            self.run_logger.wandb_run.log({
                "train/reward": r,
                "train/success": s,
                "train/length": l,
                "train/moving_avg_reward":       ma_r,
                "train/moving_avg_success_rate": ma_s,
                "train/moving_avg_length":       ma_l,
                "train/fps":     fps,
                "train/frame":   total_trained,
                "train/episode": self.train_metrics.env_total_episodes,
                "train_steps":   total_trained,
                "train_episode": self.train_metrics.env_total_episodes,
            })

        pbar.set_postfix(ep=self.train_metrics.env_total_episodes,
                         ma_r=f"{ma_r:.2f}", fps=f"{fps:.0f}")
        self._log_agent_metrics(metrics, total_trained)

    def _log_agent_metrics(self, metrics: dict, train_steps: int):
        extras = {f"train/{k}": v for k, v in metrics.items()
                  if k != "episode_info" and isinstance(v, (int, float))}
        if extras:
            extras["train_steps"] = train_steps
            self.run_logger.wandb_run.log(extras)

    # ------------------------------------------------------------------ #
    # Evaluation -- runs all N tasks separately
    # ------------------------------------------------------------------ #
    def _run_evaluation(self, global_train_frames: int):
        logger.info("=== Eval set %d (all %d tasks) ===", self.eval_set, self.num_tasks)

        all_task_metrics = {}

        for task_id in range(self.num_tasks):
            tracker = self.eval_trackers[task_id]
            tracker.initialize()

            # All eval envs run the same task
            task_ids = self.task_sampler.fixed(task_id)

            pbar = tqdm(total=self.num_eval_frames,
                        desc=f"eval task {task_id}", leave=False)

            rng = self.rng_manager.get_key()
            agent_metrics = self.agent.evaluate(
                self.agent_state, self.eval_env, rng,
                self.num_eval_frames, progress_bar=pbar,
                task_ids=task_ids,
            )
            pbar.close()

            # Process episode info
            episode_info = agent_metrics.get("episode_info")
            if episode_info is not None:
                returns = jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
                lengths = jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
                done = jnp.array(episode_info["returned_episode"]).reshape(-1)
                r = returns[done]; l = lengths[done]
                tracker.episode_reward_history.extend(r.tolist())
                tracker.episode_length_history.extend(l.tolist())
                tracker.episode_success_history.extend(
                    (r > 0).astype(jnp.int32).tolist()
                )
                tracker.env_total_episodes += int(done.sum())

            agg = {
                "avg_reward":  float(np.mean(tracker.episode_reward_history)),
                "avg_success": float(np.mean(tracker.episode_success_history)),
                "avg_length":  float(np.mean(tracker.episode_length_history)),
                "episodes":    tracker.env_total_episodes,
            }
            all_task_metrics[task_id] = agg

            # Log per-task eval
            self.run_logger.wandb_run.log({
                f"eval/task_{task_id}/avg_reward":  agg["avg_reward"],
                f"eval/task_{task_id}/avg_success": agg["avg_success"],
                f"eval/task_{task_id}/avg_length":  agg["avg_length"],
                f"eval/task_{task_id}/episodes":    agg["episodes"],
                "train_frames": global_train_frames,
            })

        # Log aggregate across all tasks
        avg_reward = np.mean([m["avg_reward"] for m in all_task_metrics.values()])
        avg_success = np.mean([m["avg_success"] for m in all_task_metrics.values()])
        avg_length = np.mean([m["avg_length"] for m in all_task_metrics.values()])

        self.run_logger.wandb_run.log({
            "eval/aggregate/avg_reward":  avg_reward,
            "eval/aggregate/avg_success": avg_success,
            "eval/aggregate/avg_length":  avg_length,
            "train_frames": global_train_frames,
        })

        # Console table
        rows = []
        for tid, m in all_task_metrics.items():
            rows.append([f"task_{tid}", f"{m['avg_reward']:.3f}",
                         f"{m['avg_success']:.3f}", m['episodes']])
        rows.append(["AGGREGATE", f"{avg_reward:.3f}", f"{avg_success:.3f}", ""])
        logger.info("\nEval set %d\n%s", self.eval_set,
                    tabulate(rows, headers=["task", "reward", "success", "episodes"],
                             tablefmt="grid"))

        # Checkpoint (use aggregate reward as the tracking metric)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.on_validation_end(
                agent_state=self.agent_state,
                step=int(self.agent_state.runtime.train_steps),
                metrics={"eval_return": avg_reward, "eval_success": avg_success},
            )
        self.eval_set += 1
