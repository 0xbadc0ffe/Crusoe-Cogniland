from argparse import ArgumentParser
import os

# IMPORTANT: setup_environment() must be called BEFORE importing any JAX-dependent modules
# This sets XLA environment variables that control JAX memory management
from cl.config import setup_environment
setup_environment()

# Now safe to import JAX-dependent modules
from omegaconf import OmegaConf

from cl.agents import load_agent
from cl.config import configure_sweep_config, load_config
from cl.shared import setup_logger
from cl.trainer import Trainer
import wandb

logger = setup_logger(__name__)


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="path to the environment YAML config file",
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        required=True,
        help="path to the agent YAML config file",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="if set, the experiment will run in offline mode (W&B won't sync to the cloud)",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="override the device ID from the config file (e.g., 0 for GPU 0, 1 for GPU 1, etc.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="if set, the script will run in sweep mode",
    )

    args = parser.parse_args()

    return args


def get_config(args) -> OmegaConf:
    """
    Load the base OmegaConf, and if we're in a sweep,
    kick off W&B and merge sweep overrides back in.
    """
    # 1) load the base OmegaConf from YAML exactly once
    base_config = load_config(
        agent_config_path=args.agent_config,
        env_config_path=args.env_config,
    )

    # Override the offline setting from the config if the --offline flag is set
    if args.offline:
        base_config.offline = True

    # Override the device from config if the --device flag is set
    if args.device is not None:
        base_config.device_id = args.device

    if not args.sweep:
        return base_config

    # 2) initialize W&B with a flat dict of the base config
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", base_config.project),
        entity=os.getenv("WANDB_ENTITY", base_config.entity),
        config=OmegaConf.to_container(base_config, resolve=True),
    )

    # 3) merge the sweep’s overrides back into an OmegaConf
    return configure_sweep_config(base_config=base_config, sweep_config_dict=run.config)


def main():
    args = get_args()

    config = get_config(args)

    # Initialize agent
    agent = load_agent(config)

    # Initialize trainer with agent
    trainer = Trainer(config, agent)
    trainer.run()


if __name__ == "__main__":
    main()
