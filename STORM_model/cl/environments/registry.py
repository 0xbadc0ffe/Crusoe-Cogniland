import importlib
from pathlib import Path
import pkgutil
from typing import Callable

from cl.shared import setup_logger

logger = setup_logger(__name__)


class EnvironmentRegistry:
    """Registry for environment factory functions"""

    def __init__(self):
        self.environment_factories: dict[str, Callable] = {}
        self.discovered_paths = set()

    def register(self, env_suite: str, factory_fn: Callable) -> None:
        """Register an environment factory function

        Args:
            env_suite: Name of the environment suite (e.g., "Craftax", "Envpool", "Navix")
            factory_fn: Factory function that takes (env_name, config) and returns environment
        """
        self.environment_factories[env_suite] = factory_fn
        logger.debug(f"Registered environment suite: {env_suite}")

    @staticmethod
    def _safe_import(module_name: str) -> None:
        """Import a module to trigger @register_environment decorators

        Handling:
        - ImportError (missing optional dependency): log warning and skip
        - Other exceptions: re-raise to surface real errors
        """
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import {module_name}: {e}")
        except Exception:
            raise

    def discover(self, paths: list[tuple[str, str]]) -> None:
        """Discover and import all environment modules

        Args:
            paths: List of tuples (path, package_name) where:
                - path is the directory path to search
                - package_name is the base package name for imports
        """
        for path, package_name in paths:
            if (path, package_name) in self.discovered_paths:
                continue

            # Import standalone files
            for item in Path(path).glob("*.py"):
                if item.stem not in ("__init__", "registry"):
                    self._safe_import(module_name=f"{package_name}.{item.stem}")

            # Import subdirectories
            for _, module_name, _ in pkgutil.iter_modules([str(path)]):
                self._safe_import(module_name=f"{package_name}.{module_name}")

            self.discovered_paths.add((path, package_name))

    def make(self, env_name: str, config) -> object:
        """Create an environment using its factory function

        Args:
            env_name: Full environment name (e.g., "Craftax/Craftax-Classic-1M-v1")
            config: OmegaConf configuration object

        Returns:
            Environment instance (type depends on the environment suite)
        """
        # Extract suite name (e.g., "Craftax" from "Craftax/Craftax-Classic-1M-v1")
        env_suite = env_name.split("/")[0]

        if env_suite not in self.environment_factories:
            available = list(self.environment_factories.keys())
            raise ValueError(
                f"Environment suite '{env_suite}' is not registered. "
                f"Available suites: {available}"
            )

        factory_fn = self.environment_factories[env_suite]
        return factory_fn(env_name, config)


# Global registry instance
ENVIRONMENT_REGISTRY = EnvironmentRegistry()


def register_environment(name: str):
    """Decorator to register an environment factory function

    Args:
        name: Environment suite name (e.g., "Craftax", "Envpool", "Navix")

    Example:
        @register_environment("Craftax")
        def make_craftax_env(env_name: str, config: OmegaConf):
            return CraftaxVectorized(...)
    """

    def decorator(factory_fn):
        ENVIRONMENT_REGISTRY.register(name, factory_fn)
        return factory_fn

    return decorator


def make_environment(env_name: str, config):
    """Create an environment from the registry

    Args:
        env_name: Full environment name (e.g., "Craftax/Craftax-Classic-1M-v1")
        config: OmegaConf configuration object

    Returns:
        Environment instance
    """
    return ENVIRONMENT_REGISTRY.make(env_name, config)
