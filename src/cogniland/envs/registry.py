"""Environment factory. Stub -- will be implemented by the env-layer agent."""
from omegaconf import OmegaConf


def make_env(env_id: str, config: OmegaConf, train: bool = True):
    """Create and return an environment instance.

    This is a stub that will be replaced by the env-layer implementation.
    """
    raise NotImplementedError(
        f"make_env('{env_id}') is not yet implemented. "
        "The env-layer branch must be merged first."
    )
