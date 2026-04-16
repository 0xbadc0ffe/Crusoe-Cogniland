"""Terrain drain/heal parameters for the strategy game.

Single source of truth — copied from scripts/tune_tile_effects.py.
"""

from dataclasses import dataclass, field


@dataclass
class TileEffects:
    hp_drain: dict[str, int] = field(
        default_factory=lambda: {
            "ocean": 16,
            "deep_water": 10,
            "water": 6,
            "beach": 1,
            "sandy": 1,
            "grassland": 1,
            "forest": 3,
            "rocky": 6,
            "mountains": 12,
        }
    )
    raft_drain: dict[str, int] = field(
        default_factory=lambda: {
            "water": 1,
            "deep_water": 3,
            "ocean": 8,
        }
    )
    rope_drain: dict[str, int] = field(
        default_factory=lambda: {
            "rocky": 1,
            "mountains": 3,
        }
    )
    shoes_drain_grassland: float = 0.5
    shoes_k: int = 10

    berry_heal: int = 10
    forest_wood: int = 10
    wood_max: int = 100
    craft_cost: int = 100
    hp_max: int = 100
    init_hp: int = 100


def drain_for(
    terrain: str, tools: frozenset[str], consec_grass: int, fx: TileEffects
) -> float:
    """Compute HP drain for stepping onto *terrain* with the given tools."""
    if "raft" in tools and terrain in fx.raft_drain:
        return fx.raft_drain[terrain]
    if "rope" in tools and terrain in fx.rope_drain:
        return fx.rope_drain[terrain]
    if "shoes" in tools and terrain == "grassland" and consec_grass >= fx.shoes_k:
        return fx.shoes_drain_grassland
    return fx.hp_drain.get(terrain, 1)
