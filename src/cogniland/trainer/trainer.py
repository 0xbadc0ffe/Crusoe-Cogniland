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
from cogniland.trainer.trajectory_viz import TrajectoryLogger
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
        # When false, skip the expensive per-task ``agent.evaluate`` loop.
        # Trajectory viz still runs according to its own interval.
        self.full_eval_enabled = bool(
            config.trainer.get("full_eval_enabled", True)
        )
        # Optional override for trajectory-viz cadence. Falls back to
        # ``eval_interval_frames`` when unset.
        self.trajectory_viz_interval_frames = config.trainer.get(
            "trajectory_viz_interval_frames", None
        )
        # Internal counter so trajectory-viz cadence stays independent of
        # full-eval cadence when they differ.
        self._next_trajectory_viz_frame: int = 0

        # W&B
        self.run_logger = RunLogger(config)
        config.results_dir = self.run_logger.results_dir
        self.run_logger.wandb_run.define_metric("eval/*", step_metric="train_frames")

        # RNG
        self.rng_manager = RNGManager(seed=config.seed)

        # Environments
        self.train_env = make_env(config.env_id, config, train=True)
        self.eval_env = make_env(config.env_id, config, train=False)

        # Curriculum (forage bonus). Eval is pinned to frac=1.0 for the whole
        # run so reported eval/* metrics always reflect the unshaped task
        # reward, making curves comparable across the anneal.
        fb = None
        cur_cfg = config.get("curriculum", None) if hasattr(config, "get") else getattr(config, "curriculum", None)
        if cur_cfg is not None:
            fb = cur_cfg.get("forage_bonus", None) if hasattr(cur_cfg, "get") else getattr(cur_cfg, "forage_bonus", None)
        if fb is not None:
            self._forage_initial_coef = float(fb.get("initial_coef", 0.0) if hasattr(fb, "get") else getattr(fb, "initial_coef", 0.0))
            self._forage_anneal_frames = float(fb.get("anneal_frames", 0) if hasattr(fb, "get") else getattr(fb, "anneal_frames", 0))
        else:
            self._forage_initial_coef = 0.0
            self._forage_anneal_frames = 0.0
        if hasattr(self.eval_env, "set_curriculum_progress"):
            self.eval_env.set_curriculum_progress(1.0)

        # Task sampler
        self.task_sampler = TaskSampler(
            num_tasks=self.num_tasks,
            num_envs=config.env.num_parallel_envs,
            mode=config.get("task_sampling", "round_robin"),
        )

        # Agent
        self.agent_state = self.agent.init(self.rng_manager.get_key())

        # Metrics: one train tracker (aggregate), N eval trackers (per-task)
        self.train_metrics = MetricsTracker(
            config, config.env.num_parallel_envs, "train", num_tasks=self.num_tasks,
        )
        self.train_metrics.initialize()
        self.run_logger.register_metrics(self.train_metrics)

        num_eval_envs = config.env.get("num_parallel_envs_eval", config.env.num_parallel_envs)
        self.eval_trackers = {}
        for task_id in range(self.num_tasks):
            t = MetricsTracker(config, num_eval_envs, "eval", num_tasks=self.num_tasks)
            self.eval_trackers[task_id] = t
            self.run_logger.register_metrics(t, prefix_override=f"eval/task_{task_id}")

        # Last segment's task assignment per env (set at the start of each
        # training segment). Used to map finished episodes back to their task
        # for per-task metric logging.
        self._train_task_ids: np.ndarray | None = None

        self.eval_set = 0

        # Trajectory visualization (4 fixed eval maps, one per biome)
        try:
            self.trajectory_logger = TrajectoryLogger(
                config, self.agent, self.run_logger.wandb_run,
            )
        except Exception as e:
            logger.warning("Trajectory logger init failed: %s", e)
            self.trajectory_logger = None

        # Checkpoint
        if config.agent.get("checkpoint", {}).get("enabled", False) and CheckpointCallback is not None:
            self.checkpoint_callback = CheckpointCallback(
                agent=self.agent, config=config,
                results_dir=self.run_logger.results_dir,
                wandb_run=self.run_logger.wandb_run,
            )
            self.checkpoint_callback.initialize(env_name=config.env_id)
        else:
            self.checkpoint_callback = None

    # ------------------------------------------------------------------ #
    # Curriculum helpers
    # ------------------------------------------------------------------ #
    def _curriculum_frac(self, total_trained: int) -> float:
        """Linear anneal: 0 at start, 1 at ``anneal_frames``, clamped."""
        if self._forage_anneal_frames <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, total_trained / self._forage_anneal_frames)))

    def _curriculum_forage_coef(self, total_trained: int) -> float:
        """Current scalar value of the annealed forage coefficient."""
        return self._forage_initial_coef * max(0.0, 1.0 - self._curriculum_frac(total_trained))

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
            self.train_env.set_tasks(task_ids)
            self._train_task_ids = np.asarray(task_ids, dtype=np.int32)

            # Advance the curriculum schedule for this segment.
            frac = self._curriculum_frac(total_trained)
            if hasattr(self.train_env, "set_curriculum_progress"):
                self.train_env.set_curriculum_progress(frac)

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
        """Log aggregate + per-task training metrics.

        Per-task identity is recovered from ``self._train_task_ids`` (the task
        assignment fixed for this segment) by mapping each flat episode index
        ``i`` back to its env via ``i % num_envs``.
        """
        episode_info = metrics.get("episode_info")
        if episode_info is None:
            self._log_agent_metrics(metrics, total_trained)
            return

        returns_flat = np.asarray(
            jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
        )
        lengths_flat = np.asarray(
            jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
        )
        done_flat = np.asarray(
            jnp.array(episode_info["returned_episode"]).reshape(-1)
        ).astype(bool)
        success_flat = np.asarray(
            jnp.array(episode_info["task_success"]).reshape(-1)
        ).astype(np.int32)

        if not bool(done_flat.any()):
            self._log_agent_metrics(metrics, total_trained)
            return

        # Map each finished episode to its env, then to its task. Agent
        # rollouts concatenate per-step [B] arrays, so flat index i -> env idx
        # (i % num_envs). Task assignment is fixed for the whole segment via
        # ``set_tasks(task_ids)``.
        num_envs = self.train_env.num_envs
        done_indices = np.where(done_flat)[0]
        env_idx = done_indices % num_envs
        if self._train_task_ids is not None and len(self._train_task_ids) == num_envs:
            task_of_ep = self._train_task_ids[env_idx]
        else:
            task_of_ep = np.zeros(len(done_indices), dtype=np.int32)

        returns_np = returns_flat[done_flat]
        lengths_np = lengths_flat[done_flat]
        successes_np = success_flat[done_flat]

        ma_r = ma_s = ma_l = 0.0
        for i in range(len(returns_np)):
            r = float(returns_np[i])
            l = int(lengths_np[i])
            s = int(successes_np[i])
            t_id = int(task_of_ep[i])

            self.train_metrics.episode_reward_history.append(r)
            self.train_metrics.episode_length_history.append(l)
            self.train_metrics.episode_success_history.append(s)
            self.train_metrics.env_total_episodes += 1

            if t_id in self.train_metrics.per_task_reward_history:
                self.train_metrics.per_task_reward_history[t_id].append(r)
                self.train_metrics.per_task_success_history[t_id].append(s)
                self.train_metrics.per_task_length_history[t_id].append(l)
                self.train_metrics.per_task_total_episodes[t_id] += 1

            ma_r = float(np.mean(self.train_metrics.episode_reward_history))
            ma_s = float(np.mean(self.train_metrics.episode_success_history))
            ma_l = float(np.mean(self.train_metrics.episode_length_history))

            log_dict = {
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
            }
            # Per-task rolling averages. Only emit for tasks with at least
            # one observed episode to avoid flatlining at 0 pre-data.
            for t in range(self.num_tasks):
                hist_s = self.train_metrics.per_task_success_history[t]
                hist_r = self.train_metrics.per_task_reward_history[t]
                hist_l = self.train_metrics.per_task_length_history[t]
                if len(hist_s) == 0:
                    continue
                log_dict[f"train/task_{t}/avg_success_rate"] = float(np.mean(hist_s))
                log_dict[f"train/task_{t}/avg_reward"] = float(np.mean(hist_r))
                log_dict[f"train/task_{t}/avg_length"] = float(np.mean(hist_l))
                log_dict[f"train/task_{t}/episodes"] = \
                    self.train_metrics.per_task_total_episodes[t]

            self.run_logger.wandb_run.log(log_dict)

        pbar.set_postfix(ep=self.train_metrics.env_total_episodes,
                         ma_r=f"{ma_r:.2f}", fps=f"{fps:.0f}")
        self._log_agent_metrics(metrics, total_trained)

    def _log_agent_metrics(self, metrics: dict, train_steps: int):
        extras = {f"train/{k}": v for k, v in metrics.items()
                  if k != "episode_info" and isinstance(v, (int, float))}
        extras["train/curriculum/forage_coef"] = self._curriculum_forage_coef(train_steps)
        extras["train_steps"] = train_steps
        self.run_logger.wandb_run.log(extras)

    # ------------------------------------------------------------------ #
    # Evaluation -- dispatches to full eval and/or trajectory viz
    # ------------------------------------------------------------------ #
    def _run_evaluation(self, global_train_frames: int):
        """Dispatch evaluation: full multi-task eval (if enabled) and
        trajectory viz (if its interval has been reached)."""
        ran_full_eval = False
        eval_metrics_for_ckpt: dict | None = None

        if self.full_eval_enabled:
            eval_metrics_for_ckpt = self._run_full_eval(global_train_frames)
            ran_full_eval = True

        # Trajectory viz on its own cadence. ``trajectory_viz_interval_frames``
        # overrides ``eval_interval_frames`` when set.
        traj_interval = self.trajectory_viz_interval_frames or self.eval_interval_frames
        should_render_traj = (
            self.trajectory_logger is not None
            and traj_interval is not None
            and global_train_frames >= self._next_trajectory_viz_frame
        )
        if should_render_traj:
            self._run_trajectory_viz(global_train_frames)
            self._next_trajectory_viz_frame = global_train_frames + int(traj_interval)

        # Checkpoint: prefer real eval metrics; otherwise fall back to the
        # train tracker rolling averages so best-tracking still works.
        if self.checkpoint_callback is not None:
            if eval_metrics_for_ckpt is None:
                eval_metrics_for_ckpt = self._checkpoint_metrics_from_train()
            self.checkpoint_callback.on_validation_end(
                agent_state=self.agent_state,
                step=int(self.agent_state.runtime.train_steps),
                metrics=eval_metrics_for_ckpt,
            )

        if ran_full_eval or should_render_traj:
            self.eval_set += 1

    def _checkpoint_metrics_from_train(self) -> dict:
        """Fallback 'eval' metrics for checkpoint best-tracking when full
        evaluation is disabled. Uses the train tracker's rolling averages."""
        hist_r = self.train_metrics.episode_reward_history
        hist_s = self.train_metrics.episode_success_history
        avg_r = float(np.mean(hist_r)) if len(hist_r) else 0.0
        avg_s = float(np.mean(hist_s)) if len(hist_s) else 0.0
        return {"eval_return": avg_r, "eval_success": avg_s}

    def _run_trajectory_viz(self, global_train_frames: int):
        """Render the 4-biome fixed-map trajectory plots and log to W&B."""
        logger.info("=== Trajectory viz @ %d frames ===", global_train_frames)
        traj_rng = self.rng_manager.get_key()
        self.trajectory_logger.log(
            agent_state=self.agent_state,
            rng=traj_rng,
            global_train_frames=global_train_frames,
        )

    def _run_full_eval(self, global_train_frames: int) -> dict:
        """Full multi-task evaluation: runs ``agent.evaluate`` for every task,
        logs per-task and aggregate ``eval/*`` metrics, and returns the
        aggregate dict suitable for checkpoint best-tracking."""
        logger.info("=== Eval set %d (all %d tasks) ===", self.eval_set, self.num_tasks)

        all_task_metrics = {}

        for task_id in range(self.num_tasks):
            tracker = self.eval_trackers[task_id]
            tracker.initialize()

            # All eval envs run the same task. Size the task_ids array to
            # the eval env (num_parallel_envs_eval), not the train sampler's
            # num_envs, so the agent's task embedding lookup matches.
            task_ids = np.full(self.eval_env.num_envs, task_id, dtype=np.int32)
            self.eval_env.set_tasks(task_ids)

            pbar = tqdm(total=self.num_eval_frames,
                        desc=f"eval task {task_id}", leave=False)

            rng = self.rng_manager.get_key()
            agent_metrics = self.agent.evaluate(
                self.agent_state, self.eval_env, rng,
                self.num_eval_frames, progress_bar=pbar,
                task_ids=task_ids,
            )
            pbar.close()

            episode_info = agent_metrics.get("episode_info")
            if episode_info is not None:
                returns = jnp.array(episode_info["returned_episode_returns"]).reshape(-1)
                lengths = jnp.array(episode_info["returned_episode_lengths"]).reshape(-1)
                done = jnp.array(episode_info["returned_episode"]).reshape(-1)
                r = returns[done]; l = lengths[done]
                tracker.episode_reward_history.extend(r.tolist())
                tracker.episode_length_history.extend(l.tolist())
                task_success = jnp.array(episode_info["task_success"]).reshape(-1)
                s = task_success[done].astype(jnp.int32)
                tracker.episode_success_history.extend(s.tolist())
                tracker.env_total_episodes += int(done.sum())

            agg = {
                "avg_reward":  float(np.mean(tracker.episode_reward_history)),
                "avg_success": float(np.mean(tracker.episode_success_history)),
                "avg_length":  float(np.mean(tracker.episode_length_history)),
                "episodes":    tracker.env_total_episodes,
            }
            all_task_metrics[task_id] = agg

            self.run_logger.wandb_run.log({
                f"eval/task_{task_id}/avg_reward":  agg["avg_reward"],
                f"eval/task_{task_id}/avg_success": agg["avg_success"],
                f"eval/task_{task_id}/avg_length":  agg["avg_length"],
                f"eval/task_{task_id}/episodes":    agg["episodes"],
                "train_frames": global_train_frames,
            })

        avg_reward = float(np.mean([m["avg_reward"] for m in all_task_metrics.values()]))
        avg_success = float(np.mean([m["avg_success"] for m in all_task_metrics.values()]))
        avg_length = float(np.mean([m["avg_length"] for m in all_task_metrics.values()]))

        self.run_logger.wandb_run.log({
            "eval/aggregate/avg_reward":  avg_reward,
            "eval/aggregate/avg_success": avg_success,
            "eval/aggregate/avg_length":  avg_length,
            "train_frames": global_train_frames,
        })

        rows = []
        for tid, m in all_task_metrics.items():
            rows.append([f"task_{tid}", f"{m['avg_reward']:.3f}",
                         f"{m['avg_success']:.3f}", m['episodes']])
        rows.append(["AGGREGATE", f"{avg_reward:.3f}", f"{avg_success:.3f}", ""])
        logger.info("\nEval set %d\n%s", self.eval_set,
                    tabulate(rows, headers=["task", "reward", "success", "episodes"],
                             tablefmt="grid"))

        return {"eval_return": avg_reward, "eval_success": avg_success}
