"""Natural-terrain POMDP navigation env (bridge_tunnel).

An open procedural grid where the agent spawns at the centre of the left edge
and must reach the goal on the right wall (a central door by default). The
terrain mixes water (cross by PLACE → wood bridge), rock (cross by MINE → grass)
and impassable TREE patches (walk around). Trees are biased toward the top &
bottom walls so naive wall-hugging to the centre door is blocked by forest. The
agent chooses per obstacle whether to cross or detour. POMDP via an egocentric
crop. (The earlier diagonal/vertical stripe orientations + obsidian/cue tiles
have been retired.)

See ``BridgeTunnelEnv`` / ``generate_bridge_tunnel_map``.
"""
from .env import BridgeTunnelEnv
from .mapgen import MapRecord, generate_bridge_tunnel_map
from . import tiles

__all__ = ["BridgeTunnelEnv", "MapRecord", "generate_bridge_tunnel_map", "tiles"]
