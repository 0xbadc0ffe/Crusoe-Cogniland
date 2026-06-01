"""bridge_tunnel_commit — bridge_tunnel with a one-shot build/mine commitment.

Same natural terrain as ``cogniland.bridge_tunnel`` but with two extra actions
(``COMMIT_BUILD`` / ``COMMIT_MINE``) that irreversibly unlock exactly one
crossing tool. Maps come in three labelled categories — ``balanced`` (14/14
water/rock), ``lakes`` (water-dominated), ``rocky`` (rock-dominated) — so a
class-balanced train / val / test split exercises the "read the terrain, commit
to the right tool" decision.

See ``BridgeTunnelCommitEnv`` / ``generate_commit_map`` / ``make_split``.
"""
from .env import BridgeTunnelCommitEnv
from .mapgen import (
    CATEGORIES, MapRecord, generate_commit_map, is_winnable, make_split,
)
from . import tiles

__all__ = [
    "BridgeTunnelCommitEnv", "MapRecord", "CATEGORIES",
    "generate_commit_map", "is_winnable", "make_split", "tiles",
]
