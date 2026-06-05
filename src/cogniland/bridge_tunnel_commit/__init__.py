"""bridge_tunnel_commit — DEPRECATED shim.

The commit variant now lives in the unified ``cogniland.bridge_tunnel`` package
(``variant='btc'``). This module re-exports it for backward compatibility; new
code should use ``cogniland.bridge_tunnel``.
"""
from cogniland.bridge_tunnel import tiles
from cogniland.bridge_tunnel.env import BridgeTunnelCommitEnv
from cogniland.bridge_tunnel.mapgen import (
    CATEGORIES, MapRecord, generate_commit_map, is_winnable, make_split,
)

__all__ = ["BridgeTunnelCommitEnv", "MapRecord", "CATEGORIES",
           "generate_commit_map", "is_winnable", "make_split", "tiles"]
