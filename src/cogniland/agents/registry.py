"""Agent registry — decorator-based auto-discovery."""

import importlib
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from cogniland.agents.agent import Agent


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, Callable] = {}

    def register(self, name: str, factory: Callable):
        self.agents[name] = factory

    def discover(self, paths: list[tuple[str, str]]):
        infrastructure = ("__init__", "registry", "agent", "state", "utils")
        for path, package in paths:
            for item in Path(path).glob("*.py"):
                if item.stem not in infrastructure:
                    importlib.import_module(f"{package}.{item.stem}")

    def load(self, config: OmegaConf) -> Agent:
        from cogniland.envs.registry import make_env

        env = make_env(config.env_id, config)
        obs_space = env.observation_space()
        act_space = env.action_space()

        if hasattr(env, "raw_observation_space"):
            config.raw_obs_space = env.raw_observation_space()
        else:
            config.raw_obs_space = obs_space

        return self.agents[config.agent.name](config, obs_space, act_space)


AGENT_REGISTRY = AgentRegistry()


def register_agent(name: str):
    """Decorator to register an agent factory function."""

    def decorator(fn):
        if name not in AGENT_REGISTRY.agents:
            AGENT_REGISTRY.register(name, fn)
        return fn

    return decorator


def load_agent(config: OmegaConf) -> Agent:
    """Load and instantiate an agent from config."""
    return AGENT_REGISTRY.load(config)
