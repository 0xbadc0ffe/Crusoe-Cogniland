"""Agent module with automatic discovery and registration"""

from pathlib import Path

from cl.agents.base import AgentState, ContinualAgent
from cl.agents.registry import AGENT_REGISTRY, load_agent, register_agent

# Discover all agents in this package
agents_path = Path(__file__).parent
AGENT_REGISTRY.discover([(str(agents_path), "cl.agents")])


__all__ = [
    "AgentState",
    "ContinualAgent",
    "AGENT_REGISTRY",
    "load_agent",
    "register_agent",
]
