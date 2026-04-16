"""Agent module — auto-discovers agent implementations on import."""

from cogniland.agents.registry import AGENT_REGISTRY, load_agent

# Auto-discover agent implementations in this package
AGENT_REGISTRY.discover([
    (__path__[0], __name__),
])

__all__ = ["load_agent"]
