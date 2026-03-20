"""Environment constants — action definitions only.

Terrain data is now fully driven by the config YAML via EnvConfig.terrains.
Only action-related constants remain here as they are true engine invariants.
"""

import torch

# Action mapping
ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "stay": 4,
}

NUM_ACTIONS = len(ACTIONS)

# Movement deltas indexed by action id: [dy, dx]
ACTION_DELTAS = torch.tensor(
    [
        [-1, 0],  # up
        [1, 0],   # down
        [0, 1],   # right
        [0, -1],  # left
        [0, 0],   # stay
    ],
    dtype=torch.long,
)
