import random

from omegaconf import OmegaConf

from cl.environments import make_environment
from cl.shared import setup_logger

logger = setup_logger(__name__)


class EnvironmentManager:
    """Manages environment lifecycle for continual RL training"""

    def __init__(self, config: OmegaConf):
        self.config = config
        self.env_names = self.config.envs
        self.envs_dict = {
            env_idx: env_name for env_idx, env_name in enumerate(self.env_names)
        }
        # Get num_parallel_envs from env config
        self.num_parallel_envs_train = self.config.env.num_parallel_envs
        self.num_parallel_envs_eval = self.config.env.get(
            "num_parallel_envs_eval", self.num_parallel_envs_train
        )
        self.cache_environments = self.config.env.get("cache_environments", False)

        # Current training environment
        self.current_env_idx: int = None
        self.current_env: str = None
        self.trained_env_names = []

        # Determine environment properties using placeholder envs
        placeholder_config = OmegaConf.create({
            "seed": 0,
            "env": OmegaConf.to_container(self.config.env, resolve=True)
        })
        placeholder_config.env["num_parallel_envs"] = 1  # Use single env for probing

        placeholder_envs = [
            make_environment(env_name, placeholder_config)
            for env_name in self.env_names
        ]
        self.max_actions = max([env.action_space() for env in placeholder_envs])

        # Get observation space from first environment (assumes all have same obs space)
        first_env = placeholder_envs[0]
        self.obs_space = first_env.observation_space()

        # Note: max_length needs to be defined per environment
        # For now, we'll handle this when needed

        # Store cached eval environments
        self.cached_envs = {}

    def _get_environment_cache_key(self, env_name: str, train: bool = True) -> str:
        """Generate cache key for environment based on name and mode"""
        return f"{env_name}_train" if train else f"{env_name}_eval"

    def _get_env_id(self, env_name: str) -> int:
        """Get environment index by name"""
        return next(
            (env_idx for env_idx, name in self.envs_dict.items() if name == env_name),
            None,
        )

    def _update_environment_history(self, env_name: str, train: bool = True) -> None:
        """Update environment history"""
        if not train:
            return
        # Move previous env to trained envs
        if self.current_env is not None:
            self.trained_env_names.append(self.current_env)
        # Update current env
        self.current_env_idx = self._get_env_id(env_name)
        self.current_env = env_name

    def get_environment(self, env_idx: int, train: bool = True):
        """
        Return an environment for the given index and mode (train/eval).
        Creates a new environment unless an evaluation environment is cached.

        Args:
            env_idx: The index of the environment to create
            train: Whether to create a training or evaluation environment

        Returns:
            Environment instance
        """
        env_name = self.envs_dict[env_idx]
        cache_key = self._get_environment_cache_key(env_name, train)

        # Only cache evaluation environments
        caching_enabled = self.cache_environments and not train

        # Update the environment history
        self._update_environment_history(env_name, train)

        # Use cached environment if available
        if caching_enabled and cache_key in self.cached_envs:
            logger.info(f"Using cached environment for {cache_key}")
            return self.cached_envs[cache_key]

        # Load the environment
        num_parallel_envs = (
            self.num_parallel_envs_train if train else self.num_parallel_envs_eval
        )

        logger.info(
            f"Creating {num_parallel_envs} parallel environments for {env_name}"
            f" ({'train' if train else 'eval'} mode)"
        )

        # Create config for environment (inherit all env settings from main config)
        # Use different seed ranges for train vs eval to prevent memorization
        # Seed ranges are defined as [LOW, HIGH] and we randomly sample from that range
        if train:
            seed_range = self.config.get("train_env_seeds", [self.config.seed, self.config.seed])
        else:
            seed_range = self.config.get("eval_env_seeds", [self.config.seed, self.config.seed])

        # Sample a random seed from the range
        env_seed = random.randint(seed_range[0], seed_range[1])

        env_config = OmegaConf.create({
            "seed": env_seed,
            "env": OmegaConf.to_container(self.config.env, resolve=True)
        })
        env_config.env["num_parallel_envs"] = num_parallel_envs

        env = make_environment(env_name=env_name, config=env_config)

        if caching_enabled:
            self.cached_envs[cache_key] = env

        return env
