"""DEPRECATED shim → cogniland.bridge_tunnel.mapgen."""
from cogniland.bridge_tunnel.mapgen import (  # noqa: F401
    CATEGORIES, MapRecord, _CATEGORY_FRACS, _can_reach_goal, category_fracs,
    generate_commit_map, is_winnable, make_split,
)

__all__ = ["MapRecord", "CATEGORIES", "generate_commit_map", "is_winnable",
           "make_split", "category_fracs"]
