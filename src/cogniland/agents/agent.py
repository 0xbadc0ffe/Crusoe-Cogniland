from dataclasses import dataclass
from typing import Any, Callable, Optional
import jax

PRNGKey = jax.random.PRNGKey


@dataclass
class Agent:
    """Container for agent functions. NOT a PyTree -- just a namespace.

    Kept structurally compatible with the reference repo so that
    dreamer.py and storm.py can be copied in with minor edits.
    """
    # Required
    init: Callable[[PRNGKey], Any]
    train: Callable        # (state, env, rng, num_frames, **kw) -> (state, metrics)
    evaluate: Callable     # (state, env, rng, num_frames, **kw) -> metrics

    # Optional with defaults (kept for reference-repo compat)
    reset: Callable = lambda s, rng: s
    set_environment: Callable = lambda s, **kw: s
    on_environment_end: Callable = lambda s, **kw: s

    # Optional agent-specific extras
    select_action: Optional[Callable] = None
    state_from_checkpoint: Optional[Callable] = None
    get_latents: Optional[Callable] = None

    # Metadata
    obs_space: Any = None
    action_space: Any = None
