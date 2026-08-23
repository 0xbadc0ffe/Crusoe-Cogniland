import numpy as np

from cl.shared import setup_logger
import wandb

logger = setup_logger(__name__)


class WandbMetricsClient:
    """
    Dedicated wandb client for continual learning experiments.
    """

    def __init__(self, entity: str, project: str, run_id: str):
        self.entity = entity
        self.project = project
        self.run_id = run_id

        self.api = wandb.Api()
        self.run = self.api.run(f"{self.entity}/{self.project}/{self.run_id}")
        self.config = self.run.config
        # Incomplete history of the run, only default step_metrics are logged
        # But all the custom metrics names are available as columns
        self.metrics_df = self.run.history()

    def delete(self) -> None:
        """
        Deletes the W&B run associated with this client.
        """
        self.run.delete()

    def get_metric_names(self, include_patterns: list = []) -> list[str]:
        """
        Return a list of metric names that are logged to wandb.

        Args:
            include_patterns (list, optional): List of strings that need to be present
                in the metrics names. If provided, only metrics containing
                all of these strings will be returned.

        Returns:
            list: Filtered list of metric names.
        """
        # Filter out gif files
        metric_names = [m for m in self.metrics_df.keys() if ".gif" not in m]
        # Filter out other non-metric columns e.g. "_step"
        metric_names = [m for m in metric_names if not m.startswith("_")]
        # # Filter out step_metrics
        # metric_names = [m for m in metric_names if "/" in m]
        # # Filter out metrics that are not in the include_patterns
        metric_names = [
            m for m in metric_names if all(p in m for p in include_patterns)
        ]
        return metric_names

    def get_metric_values(self, metric_name: str, n_samples: int = 10000) -> np.ndarray:
        """
        Retrieves the complete history of values for a specific metric from the W&B run.

        This method uses two different W&B APIs depending on the n_samples parameter:
        - For n_samples <= 10000: Uses the faster run.history API
        - For n_samples > 10000 or None: Uses the slower but more complete scan_history API

        Args:
            metric_name (str): The name of the metric to retrieve values for.
            n_samples (int, optional): Number of samples to retrieve. Defaults to 10000.
                For best performance, keep this value at or below 10000. The run.history API
                does not support retrieving more than 10000 samples. Values above 10000
                or setting this to None will switch to the scan_history API, which is
                significantly slower but retrieves all available data points.

        Returns:
            np.ndarray: An array containing all recorded values for the specified metric.
        """
        if n_samples is not None and n_samples <= 10000:
            metric_values = self.run.history(keys=[metric_name], samples=n_samples)
            return metric_values[metric_name].to_numpy()

        logger.warning(
            f"Using scan_history for metric {metric_name} with n_samples={n_samples}. "
            "This may take up to one minute for each metric to complete."
        )
        metric_values = self.run.scan_history(keys=[metric_name])
        return np.array([m[metric_name] for m in metric_values])

    def get_run_summary(
        self, include_patterns: list = [], n_samples: int = 10000
    ) -> dict[str, np.ndarray]:
        """
        Retrieves a comprehensive summary of metrics for the current W&B run.

        This method collects all metric names that match the specified filter patterns and
        retrieves their complete value history. The result is a dictionary mapping each
        metric name to its corresponding array of values.

        Args:
            include_patterns (list, optional): List of substring patterns used to filter metric names.
                Only metrics containing all specified patterns will be included in the summary.
                Defaults to an empty list, which includes all metrics.
            n_samples (int, optional): Number of samples to retrieve for each metric.

        Returns:
            dict[str, np.ndarray]: A dictionary where keys are metric names and values are numpy
                arrays containing the complete history of values for each metric.
        """
        run_summary = {
            metric_name: self.get_metric_values(metric_name, n_samples)
            for metric_name in self.get_metric_names(include_patterns)
        }
        return run_summary
