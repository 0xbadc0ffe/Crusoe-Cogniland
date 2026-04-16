import os
from omegaconf import OmegaConf
import wandb
from cogniland.metrics.tracker import MetricsTracker
from cogniland.shared import setup_logger

logger = setup_logger(__name__)


class RunLogger:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.wandb_run = self._init_wandb_run(config)
        self.run_name = self.wandb_run.name
        self.run_id = self.wandb_run.id
        self.results_dir = os.path.join(config.results_path, self.run_id)
        os.makedirs(self.results_dir, exist_ok=True)

    @staticmethod
    def _init_wandb_run(config):
        # When running under a sweep, wandb.init() was already called in
        # train.py::get_config(). wandb.init() is idempotent in the same
        # process -- calling it again returns the existing run.
        # BUT: if you're NOT in a sweep, this is the first wandb.init() call.
        run = wandb.init(
            entity=config.entity,
            project=config.project,
            config=OmegaConf.to_container(config, resolve=True),
            mode="offline" if config.offline else "online",
        )
        # Only set name if we own the run (not a sweep-managed run)
        if run.sweep_id is None:
            run.name = "_".join([
                config.name, config.agent.name, config.experiment_name, run.id
            ])
        artifact = wandb.Artifact(name="config", type="config")
        path = "config.yaml"
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        artifact.add_file(path)
        run.log_artifact(artifact)
        try:
            os.remove(path)
        except OSError:
            pass
        return run

    def register_metrics(self, tracker: MetricsTracker, prefix_override: str = None):
        prefix = prefix_override or tracker.metric_prefix
        for name in tracker.get_metric_names():
            full = f"{prefix}/{name}"
            self.wandb_run.define_metric(full, step_metric=tracker.step_metric)
