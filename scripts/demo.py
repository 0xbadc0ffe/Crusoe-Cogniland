#!/usr/bin/env python3
"""Launch the interactive PyGame demo.

Usage:
    python scripts/demo.py              # Interactive difficulty selection
    python scripts/demo.py easy         # Easy mode
    python scripts/demo.py hard         # Hard mode
"""

import sys

from game_demo import IslandGameDemo, show_difficulty_selection


def main():
    if len(sys.argv) > 1:
        hard_mode = sys.argv[1].lower() in ("hard", "h", "2")
    else:
        hard_mode = show_difficulty_selection()

    game = IslandGameDemo(hard_mode=hard_mode)
    game.run()


if __name__ == "__main__":
    main()
