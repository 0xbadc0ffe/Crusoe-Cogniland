import logging

from omegaconf import OmegaConf
from tabulate import tabulate


def log_config(config: OmegaConf, logger: logging.Logger) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(config))


def log_table(
    data: dict,
    logger: logging.Logger,
    title: str = "",
    sort_data: bool = True,
    float_precision: int = 4,
) -> None:
    """
    Log a table of data to the console.

    Args:
        data (dict): The data to log, where keys are column names and values are lists of values.
        logger (logging.Logger): The logger to use for logging.
        title (str): The title of the table.
        sort_data (bool): Whether to sort the data by the column names (keys in the dict).
        float_precision (int): The number of decimal places to display for float values.
    """
    # Sort the data by the column name if specified
    if sort_data:
        data = dict(sorted(data.items()))

    # Convert single values to lists
    if not isinstance(next(iter(data.values())), list):
        data = {k: [v] for k, v in data.items()}

    # Print the table
    table = tabulate(
        data, headers="keys", tablefmt="grid", floatfmt=f".{float_precision}f"
    )
    logger.info("\n%s\n%s", title, table)
