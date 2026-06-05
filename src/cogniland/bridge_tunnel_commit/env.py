"""DEPRECATED shim → cogniland.bridge_tunnel.env (variant='btc')."""
from cogniland.bridge_tunnel.env import (  # noqa: F401
    A_UP, A_DOWN, A_LEFT, A_RIGHT, A_BUILD, A_MINE, A_PLACE, NUM_ACTIONS,
    COMMIT_NONE, COMMIT_BUILD, COMMIT_MINE, F_UP, F_DOWN, F_LEFT, F_RIGHT,
    BridgeTunnelCommitEnv, BridgeTunnelEnv,
)

N_SCALARS = 7   # btc observation has 7 scalars (face one-hot + step + 2 commit flags)

__all__ = ["BridgeTunnelCommitEnv", "BridgeTunnelEnv", "NUM_ACTIONS", "N_SCALARS",
           "A_UP", "A_DOWN", "A_LEFT", "A_RIGHT", "A_BUILD", "A_MINE", "A_PLACE",
           "COMMIT_NONE", "COMMIT_BUILD", "COMMIT_MINE",
           "F_UP", "F_DOWN", "F_LEFT", "F_RIGHT"]
