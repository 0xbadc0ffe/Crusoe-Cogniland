"""Environment constants extracted from the original environment.py."""

import torch

# RGB color palette for terrain visualization
palette = {
    "deepocean": [5, 35, 225],
    "ocean": [25, 65, 225],
    "blue": [65, 105, 225],
    "green": [34, 139, 34],
    "darkgreen": [0, 100, 0],
    "sandy": [210, 180, 140],
    "beach": [238, 214, 175],
    "snow": [255, 250, 250],
    "mountain": [139, 137, 137],
}

# Terrain levels: 0=ocean .. 8=mountains
# Each entry: name, height threshold, movement cost, palette key
TERRAIN_LEVELS = {
    0: {"name": "ocean", "threshold": 0.007, "cost": 1.0/5, "color": "deepocean"},
    1: {"name": "deep_water", "threshold": 0.03, "cost": 1.25/5, "color": "ocean"},
    2: {"name": "water", "threshold": 0.07, "cost": 1.5/5, "color": "blue"},
    3: {"name": "beach", "threshold": 0.09, "cost": 1.75/5, "color": "beach"},
    4: {"name": "sandy", "threshold": 0.13, "cost": 2.0/5, "color": "sandy"},
    5: {"name": "grassland", "threshold": 0.35, "cost": 2.25/5, "color": "green"},
    6: {"name": "forest", "threshold": 0.6, "cost": 3.0/5, "color": "darkgreen"},
    7: {"name": "rocky", "threshold": 0.75, "cost": 3.5/5, "color": "mountain"},
    8: {"name": "mountains", "threshold": 1.0, "cost": 4.0/5, "color": "snow"},
}

# Visibility ranges for minimap based on terrain
VISIBILITY_RANGES = {
    0: 16, 1: 12, 2: 8,  # water levels
    3: 5, 4: 5,          # beach, sandy
    5: 5,                # grassland
    6: 3,                # forest — limited canopy visibility
    7: 10,               # rocky — high ground advantage
    8: 22,               # mountains — full strategic view
}

# Action mapping
ACTIONS = {
    "up": 0,
    "down": 1,
    "right": 2,
    "left": 3,
    "stay": 4,
}

NUM_ACTIONS = len(ACTIONS)

# Pre-computed tensor versions for fast lookups in batched operations
TERRAIN_THRESHOLDS = torch.tensor(
    [TERRAIN_LEVELS[i]["threshold"] for i in range(9)], dtype=torch.float32
)

TERRAIN_COSTS = torch.tensor([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 3.0, 3.5, 4.0], dtype=torch.float32)

TERRAIN_VISIBILITY = torch.tensor([10, 8, 6, 5, 5, 5, 3, 10, 22], dtype=torch.long)

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
