#!/usr/bin/env python3
"""
Launcher script for the Island Navigation Game Demo.
Run this from the project root directory.
"""

import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the demo
from game_demo import IslandGameDemo, show_difficulty_selection

if __name__ == "__main__":
    import sys
    
    print("🏝️  Island Navigation Game Demo")
    print("=" * 40)
    
    # Check if difficulty is specified as command line argument
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['hard', 'h', '2']:
            hard_mode = True
            print("Starting in HARD MODE...")
        else:
            hard_mode = False
            print("Starting in EASY MODE...")
    else:
        # Show difficulty selection screen
        print("Select difficulty mode...")
        hard_mode = show_difficulty_selection()
    
    # Start game with selected difficulty
    try:
        game = IslandGameDemo(hard_mode=hard_mode)
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError running game: {e}")
        import traceback
        traceback.print_exc()