# Island Navigation RL Environment

A reinforcement learning environment for training agents on navigation and decision-making tasks in a 2D procedurally generated island world.

## Overview

This project implements a complex navigation game where an agent must navigate from a randomly spawned position to a target location while managing resources, health, and movement costs across different terrain types.

## Features

- **Procedural Island Generation**: Uses Perlin noise to generate realistic island landscapes
- **Complex Terrain System**: 9 different terrain types with unique movement costs and effects
- **Resource Management**: Wood/resource gathering and consumption mechanics
- **Health System**: HP management with passive healing and environmental damage
- **Dynamic Visibility**: Terrain-based visibility ranges for exploration
- **Interactive Demo**: Pygame-based visual interface for testing and gameplay
- **RL Environment**: OpenAI Gym-compatible interface for training RL agents

## Terrain Types

| Level | Name       | Cost | Special Effects |
|-------|------------|------|-----------------|
| 0     | Ocean      | 0.5  | Requires resources after 7 moves |
| 1     | Deep Water | 0.75 | Requires resources after 7 moves |
| 2     | Water      | 1.0  | Requires resources after 7 moves |
| 3     | Beach      | 2.5  | - |
| 4     | Sandy      | 2.5  | - |
| 5     | Grassland  | 1.8  | - |
| 6     | Forest     | 3.0  | Gain 1 resource + 4 HP per turn |
| 7     | Rocky      | 4.0  | Consumes 0.25 resources per turn |
| 8     | Mountains  | 8.0  | Consumes 0.75 resources per turn |

## Game Rules

### Navigation
- 5 actions: Up, Down, Left, Right, Stay
- Movement costs vary by terrain type
- Land-to-water transition costs additional 3 time units

### Sea Navigation
- Max 7 sea movements without resources
- After limit exceeded:
  - Ocean (0): 0.75 resources per move, 25 HP damage if no resources
  - Deep Water (1): 0.5 resources per move, 25 HP damage if no resources  
  - Water (2): 0.25 resources per move, 10 HP damage if no resources

### Resource System
- Start with 75 HP and 0 resources
- Passive healing: +1 HP per turn (max 100)
- Forest gathering: +1 resource and +4 HP per turn
- Mountain crossing consumes resources (0.25 for rocky, 0.75 for mountains)
- HP loss if resources unavailable in mountains (5 HP for rocky, 20 HP for mountains)

### Visibility System
- Terrain-based visibility ranges:
  - Water levels: 6 tiles
  - Beach/Sandy: 4 tiles  
  - Grassland: 5 tiles
  - Forest: 2 tiles
  - Rocky: 6 tiles
  - Mountains: 10 tiles

## Project Structure

```
Crusoe-Cogniland/
├── src/                    # Python source code
│   ├── __init__.py        # Package initialization
│   ├── environment.py     # Core environment implementation  
│   ├── game_demo.py       # Interactive pygame demo
│   ├── map.py            # Map generation utilities
│   ├── trajectory_visualizer.py  # Visualization tools
│   └── utils.py          # Utility functions
├── assets/                # Project assets
│   └── images/           # Generated island visualizations
├── lib/                   # External dependencies
│   └── simplexnoise/     # Perlin noise generation library
├── logs/                  # Training/execution logs
├── models/               # Saved models and checkpoints
├── run_demo.py           # Easy launcher script
├── run_demo.bat          # Windows batch launcher
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Crusoe-Cogniland
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. For the interactive demo, ensure pygame is installed:
```bash
pip install pygame>=2.0.0
```

## Usage

### Interactive Demo

**Easy way (recommended):**
```bash
# From project root
python run_demo.py          # Interactive difficulty selection
python run_demo.py easy     # Easy mode directly
python run_demo.py hard     # Hard mode directly
```

**Or on Windows:**
```batch
run_demo.bat                # Double-click or run from command line
```

**Direct way:**
```bash
# From src/ directory
cd src
python game_demo.py
```

**Game Modes:**
- **Easy Mode**: Standard gameplay with passive healing
- **Hard Mode**: Resource management required - lose 0.5 HP/turn without resources!

**Controls:**
- Arrow keys or WASD: Move
- Space: Stay (useful for resource gathering in forests)
- R: Reset game
- ESC: Quit

### RL Environment
Use the Gym-compatible environment for training:
```python
from rl_env import IslandNavigationEnv

env = IslandNavigationEnv()
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

### Basic Environment Usage
```python
import sys
import os

# Add src directory to path
sys.path.append('src')

from environment import Islands, ACTIONS
import torch

# Create environment
env = Islands(batch_dim=1)

# Take actions
action = torch.tensor([ACTIONS["up"]])
state, alive, reached_target = env.act(action)

print(f"Position: {state['position']}")
print(f"HP: {state['hp']}")
print(f"Resources: {state['resources']}")
print(f"Cost: {state['cost']}")
```

## Environment State

The environment provides the following observation space:

- **position**: Current (x, y) coordinates
- **compass**: Distance vector (dx, dy) to target
- **terrain_lev**: Current terrain level (0-8)
- **terrain_clock**: Turns spent in current terrain type
- **hp**: Current health points (0-100)
- **resources**: Current resources available
- **cost**: Accumulated time cost
- **minimap**: Visibility-limited terrain map around player

## Objective

The goal is to navigate from the spawn position to the target location while:
- Minimizing total time cost
- Maximizing remaining health points
- Managing resources efficiently
- Surviving environmental hazards

## Training RL Agents

The environment is designed to challenge RL agents with:
- **Long-term planning**: Resource management across different terrains
- **Risk assessment**: Choosing between safe but slow vs. fast but dangerous routes
- **Exploration vs. exploitation**: Balancing efficient movement with resource gathering
- **Multi-objective optimization**: Time cost vs. health preservation

## Customization

Environment parameters can be customized:

```python
island_options = {
    "size": 250,           # Map size
    "scale": 0.33,         # Terrain scale
    "octaves": 6,          # Noise complexity
    "seed": 42,            # Random seed
    "filtering": "square"  # Island shape
}

agent_opts = {
    "init_hp": 75,                              # Starting HP
    "max_hp": 100,                             # Maximum HP
    "init_resources": 0,                       # Starting resources
    "max_sea_movement_without_resources": 7    # Free sea moves
}
```

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the environment.

## License

This project is available for academic and research purposes.