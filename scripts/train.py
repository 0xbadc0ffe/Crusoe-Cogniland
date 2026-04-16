from argparse import ArgumentParser

from cogniland.config import setup_environment
setup_environment()

from omegaconf import OmegaConf
import wandb
from cogniland.config import load_config, configure_sweep_config
from cogniland.agents import load_agent
from cogniland.trainer import Trainer
from cogniland.shared import setup_logger

logger = setup_logger(__name__)


def get_args():
    p = ArgumentParser()
    p.add_argument("--env-config",   required=True)
    p.add_argument("--agent-config", required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device",  type=int)
    p.add_argument("--sweep",   action="store_true")
    args, unknown = p.parse_known_args()
    args.set = [u.lstrip("-") for u in unknown if "=" in u]
    return args


def get_config(args):
    cfg = load_config(args.agent_config, args.env_config)
    if args.offline:             cfg.offline = True
    if args.device is not None:  cfg.device_id = args.device
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    if not args.sweep:
        return cfg
    run = wandb.init(project=cfg.project, entity=cfg.entity,
                     config=OmegaConf.to_container(cfg, resolve=True))
    return configure_sweep_config(base_config=cfg, sweep_config_dict=run.config)


def main():
    args   = get_args()
    config = get_config(args)
    agent  = load_agent(config)
    Trainer(config, agent).run()

if __name__ == "__main__":
    main()
