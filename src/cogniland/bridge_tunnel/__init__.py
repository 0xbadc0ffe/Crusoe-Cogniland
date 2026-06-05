"""bridge_tunnel — natural-terrain POMDP navigation, two variants.

One env, two variants (select with ``variant=``):

* ``bt``  — cross water (place a wood bridge) / rock (mine to grass) or detour;
  both tools always available.
* ``btc`` — the agent must irreversibly **commit** to one tool (implicitly, via
  its first successful build/mine); maps come in 3 categories
  (balanced/lakes/rocky).

See ``BridgeTunnelEnv(variant=...)`` and ``generate_map(variant=...)``.
"""
from .env import BridgeTunnelEnv, BridgeTunnelCommitEnv, VARIANTS
from .mapgen import (
    CATEGORIES, MapRecord, generate_bridge_tunnel_map, generate_commit_map,
    generate_map, is_reachable, is_winnable, make_split,
)
from . import tiles

__all__ = [
    "BridgeTunnelEnv", "BridgeTunnelCommitEnv", "VARIANTS",
    "MapRecord", "CATEGORIES",
    "generate_map", "generate_bridge_tunnel_map", "generate_commit_map",
    "is_reachable", "is_winnable", "make_split", "tiles",
]
