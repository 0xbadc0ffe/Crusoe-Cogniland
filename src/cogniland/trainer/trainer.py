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
        # ``config.tasks`` is the list of task ids included in training. The
        # sampler draws from this list; eval loops iterate over it. Coerce to
        # a plain Python list so downstream indexing works with OmegaConf or
        # native lists alike.
        self.tasks: list[int] = [int(t) for t in config.tasks]
        self.num_tasks: int = len(self.tasks)

        self.num_train_frames = config.trainer.num_train_frames
        self.num_eval_frames = config.trainer.num_eval_frames
        self.eval_interval_frames = config.trainer.get("eval_interval_frames", None)
        # When false, skip the expensive per-task ``agent.evaluate`` loop.
        # Trajectory viz still runs according to its own interval.
        self.full_eval_enabled = bool(
            config.trainer.get("full_eval_enabled", True)
        )
        # Optional override for trajectory-viz cadence. Falls back to
        # ``eval_interval_frames`` when unset. Coerce to int eagerly so a
        # stray string (e.g. "None" from a W&B sweep dotlist) fails loudly
        # at construction time rather than mid-run.
        _traj_raw = config.trainer.get("trajectory_viz_interval_frames", None)
        if isinstance(_traj_raw, str) and _traj_raw.strip().lower() in ("none", "null", ""):
            _traj_raw = None
        self.trajectory_viz_interval_frames = (
            int(_traj_raw) if _traj_raw is not None else None
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

        # Task sampler
        self.task_sampler = TaskSampler(
            task_ids=self.tasks,
            num_envs=config.env.num_parallel_envs,
            mode=config.get("task_sampling", "round_robin"),
        )

        # Agent
        self.agent_state = self.agent.init(self.rng_manager.get_key())

        # Metrics: one train tracker (aggregate), one eval tracker per task id
        self.train_metrics = MetricsTracker(
            config, config.env.num_parallel_envs, "train",
        )
        self.train_metrics.initialize()
        self.run_logger.register_metrics(self.train_metrics)

        num_eval_envs = config.env.get("num_parallel_envs_eval", config.env.num_parallel_envs)
        self.eval_trackers = {}
        for task_id in self.tasks:
            t = MetricsTracker(config, num_eval_envs, "eval")
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
    # Spawn-distance curriculum
    # ------------------------------------------------------------------ #
    def _apply_spawn_distance_schedule(self, total_trained: int) -> None:
        """Interpolate the env's spawn-distance band at this training point.

        Reads ``env.spawn_distance_schedule`` from the train env; the eval
        env is intentionally NOT advanced, so eval stays on whatever
        distribution the user configured (typically the full band).
        """
        sched = getattr(self.train_env, "spawn_distance_schedule", None)
        if not sched:
            return
        start = sched["start"]
        end = sched["end"]
        anneal = sched["anneal_frames"]
        frac = min(1.0, max(0.0, total_trained / float(anneal)))
        lo = int(round(start[0] + frac * (end[0] - start[0])))
        hi = int(round(start[1] + frac * (end[1] - start[1])))
        self.train_env.set_spawn_distance_range(lo, hi)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        logger.info(
            "=== Multi-task training start (tasks=%s) ===", self.tasks,
        )
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

            # Advance spawn-distance curriculum if the env exposes one. Eval
            # env is left pinned at its configured range (typically the full
            # evaluation distribution).
            self._apply_spawn_distance_schedule(total_trained)

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
        """Log per-episode training metrics.

        One W&B log call per finished episode. Raw {0,1} success is too noisy
        to read in the UI, so ``train/success`` is a moving average over the
        last ``TRAIN_SUCCESS_MA_WINDOW`` finished episodes (global, across
        tasks and biomes); per-task and per-biome success remain raw.

        ``train/frame`` / ``train/episode`` are intentionally not logged to
        W&B as user-visible metrics — the ``train_steps`` / ``train_episode``
        step-metric anchors are kept so ``define_metric`` can align the x-axis.

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
        biome_flat = episode_info.get("biome")
        if biome_flat is not None:
            biome_flat = np.asarray(biome_flat).reshape(-1)
        discounted_flat = episode_info.get("returned_episode_returns_discounted")
        if discounted_flat is not None:
            discounted_flat = np.asarray(
                jnp.array(discounted_flat).reshape(-1)
            )

        if not bool(done_flat.any()):
            self._log_agent_metrics(metrics, total_trained)
            return

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
        biomes_np = biome_flat[done_flat] if biome_flat is not None else None
        discounted_np = (
            discounted_flat[done_flat] if discounted_flat is not None else None
        )

        success_window = self.train_metrics.train_success_window

        last_r = last_s_ma = 0.0
        for i in range(len(returns_np)):
            r = float(returns_np[i])
            l = int(lengths_np[i])
            s = int(successes_np[i])
            r_disc = float(discounted_np[i]) if discounted_np is not None else None
            t_id = int(task_of_ep[i])
            biome = str(biomes_np[i]) if biomes_np is not None else None

            self.train_metrics.env_total_episodes += 1
            success_window.append(s)
            success_ma = float(np.mean(success_window))
            last_r, last_s_ma = r, success_ma

            log_dict = {
                "train/reward": r,
                "train/success": success_ma,
                "train/length": l,
                "train/fps":     fps,
                "train_steps":   total_trained,
                "train_episode": self.train_metrics.env_total_episodes,
                f"train/task_{t_id}/reward": r,
                f"train/task_{t_id}/success": s,
                f"train/task_{t_id}/length": l,
            }
            if r_disc is not None:
                log_dict["train/reward_discounted"] = r_disc
                log_dict[f"train/task_{t_id}/reward_discounted"] = r_disc
            if biome is not None:
                log_dict[f"train/biome_{biome}/reward"] = r
                log_dict[f"train/biome_{biome}/success"] = s
                log_dict[f"train/biome_{biome}/length"] = l
                if r_disc is not None:
                    log_dict[f"train/biome_{biome}/reward_discounted"] = r_disc

            self.run_logger.wandb_run.log(log_dict)

        pbar.set_postfix(ep=self.train_metrics.env_total_episodes,
                         r=f"{last_r:.2f}",
                         s=f"{last_s_ma:.2f}",
                         fps=f"{fps:.0f}")
        # Print loss dynamics to stdout every segment so offline runs have a
        # human-readable trace of entropy / value_loss / policy_loss.
        loss_bits = []
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                loss_bits.append(f"{k}={v:+.3f}")
        if loss_bits:
            logger.info("[frame %d] r=%+.2f s_ma=%+.2f  %s",
                        total_trained, last_r, last_s_ma, "  ".join(loss_bits))
        self._log_agent_metrics(metrics, total_trained)

    def _log_agent_metrics(self, metrics: dict, train_steps: int):
        extras = {f"train/{k}": v for k, v in metrics.items()
                  if k != "episode_info" and isinstance(v, (int, float))}
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
        # overrides ``eval_interval_frames`` when set. Use ``is not None`` so
        # a legitimate interval of 0 (fire every eval) isn't swallowed by
        # truthiness.
        traj_interval = (
            self.trajectory_viz_interval_frames
            if self.trajectory_viz_interval_frames is not None
            else self.eval_interval_frames
        )
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
        evaluation is disabled. Returns zeros — with raw (no rolling-average)
        logging the trainer no longer has a smoothed train signal to expose,
        and checkpoint best-tracking should be driven by real eval when
        it matters.
        """
        return {"eval_return": 0.0, "eval_success": 0.0}

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
        """Full multi-task evaluation: runs ``agent.evaluate`` for every
        configured task, logs per-task and aggregate ``eval/*`` metrics, and
        returns the aggregate dict for checkpoint best-tracking."""
        logger.info(
            "=== Eval set %d (tasks=%s) ===", self.eval_set, self.tasks,
        )

        all_task_metrics = {}

        for task_id in self.tasks:
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
                "reward":   float(np.mean(tracker.episode_reward_history))
                            if tracker.episode_reward_history else 0.0,
                "success":  float(np.mean(tracker.episode_success_history))
                            if tracker.episode_success_history else 0.0,
                "length":   float(np.mean(tracker.episode_length_history))
                            if tracker.episode_length_history else 0.0,
                "episodes": tracker.env_total_episodes,
            }
            all_task_metrics[task_id] = agg

            self.run_logger.wandb_run.log({
                f"eval/task_{task_id}/reward":   agg["reward"],
                f"eval/task_{task_id}/success":  agg["success"],
                f"eval/task_{task_id}/length":   agg["length"],
                f"eval/task_{task_id}/episodes": agg["episodes"],
                "train_frames": global_train_frames,
            })

        reward = float(np.mean([m["reward"] for m in all_task_metrics.values()]))
        success = float(np.mean([m["success"] for m in all_task_metrics.values()]))
        length = float(np.mean([m["length"] for m in all_task_metrics.values()]))

        self.run_logger.wandb_run.log({
            "eval/aggregate/reward":  reward,
            "eval/aggregate/success": success,
            "eval/aggregate/length":  length,
            "train_frames": global_train_frames,
        })

        rows = []
        for tid, m in all_task_metrics.items():
            rows.append([f"task_{tid}", f"{m['reward']:.3f}",
                         f"{m['success']:.3f}", m['episodes']])
        rows.append(["AGGREGATE", f"{reward:.3f}", f"{success:.3f}", ""])
        logger.info("\nEval set %d\n%s", self.eval_set,
                    tabulate(rows, headers=["task", "reward", "success", "episodes"],
                             tablefmt="grid"))

        return {"eval_return": reward, "eval_success": success}
