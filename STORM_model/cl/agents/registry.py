"""Agent registry for dynamic discovery and loading"""

import importlib
from pathlib import Path
import pkgutil
from typing import Type

from omegaconf import OmegaConf

from cl.agents.base import ContinualAgent
from cl.shared import setup_logger

logger = setup_logger(__name__)


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, Type[ContinualAgent]] = {}
        self.discovered_paths = set()

    def register(self, name: str, agent: Type[ContinualAgent]) -> None:
        self.agents[name] = agent

    @staticmethod
    def _safe_import(module_name: str) -> None:
        """
        Import an agent module so its @register_agent decorator runs.

        Handling:
        - ImportError (missing optional dependency): log a warning and skip (agent not registered).
        - Any other exception: re-raise immediately to stop the runtime and surface real errors.
        """
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
        except Exception:
            raise  # Propagate other exceptions

    def discover(self, paths: list[tuple[str, str]]) -> None:
        """
        Discover and import all agent modules in the specified paths.

        Args:
            paths: List of tuples (path, package_name) where:
                - path is the directory path to search
                - package_name is the base package name for imports
        """
        for path, package_name in paths:
            if (path, package_name) in self.discovered_paths:
                continue

            # Handle standalone files in the directory
            for item in Path(path).glob("*.py"):
                if item.stem not in ("__init__", "registry", "base"):
                    self._safe_import(module_name=f"{package_name}.{item.stem}")

            # Handle subdirectories with modules
            for _, module_name, _ in pkgutil.iter_modules([path]):
                if module_name not in ("registry", "base"):
                    self._safe_import(module_name=f"{package_name}.{module_name}")

            self.discovered_paths.add((path, package_name))

    def load(self, config: OmegaConf) -> ContinualAgent:
        name = config.agent.name
        if name not in self.agents:
            raise ValueError(f"Agent {name} not registered.")

        # Get observation and action spaces from environment manager
        from cl.environments.manager import EnvironmentManager
        env_manager = EnvironmentManager(config)
        obs_space = env_manager.obs_space
        act_space = env_manager.max_actions

        return self.agents[name](config, obs_space, act_space)


AGENT_REGISTRY = AgentRegistry()


def register_agent(name: str):
    def decorator(agent_class):
        AGENT_REGISTRY.register(name, agent_class)
        return agent_class

    return decorator


def load_agent(config: OmegaConf) -> ContinualAgent:
    """Load an agent from the registry."""
    return AGENT_REGISTRY.load(config)
