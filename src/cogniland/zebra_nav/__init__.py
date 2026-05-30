"""Zebra-stripe POMDP navigation env.

A 32×32 grid where the agent must navigate BL→TR across a stack of diagonal
*zebra stripes* of water and rock. Within each stripe, water and rock segments
are separated by impassable obsidian; one of the two is ~1 cell thinner. On the
grass between stripes sits a **cue tile** revealing which side is thinner — the
agent must remember it and choose to *mine* (rock route) or *place* (water
route) when it reaches the stripe. POMDP via an egocentric crop.

See ``ZebraNavEnv`` / ``generate_zebra_map``.
"""
from .env import ZebraNavEnv
from .mapgen import MapRecord, generate_zebra_map
from . import tiles

__all__ = ["ZebraNavEnv", "MapRecord", "generate_zebra_map", "tiles"]
