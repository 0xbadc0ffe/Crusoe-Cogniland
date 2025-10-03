"""
Trajectory Visualization System for Island Navigation

This module provides functionality to visualize agent trajectories on island maps,
saving images that show the path taken from spawn to target.
"""

import os
import numpy as np
import torch

# Set environment variable to fix OpenMP issues on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import logging

# Import environment constants
from environment import TERRAIN_LEVELS, palette


class TrajectoryVisualizer:
    """
    Visualizes agent trajectories on island maps with detailed terrain rendering
    """
    
    def __init__(self, save_dir: str = "trajectory_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create color mapping for terrain types
        self.setup_colormap()
        
        # Trajectory storage
        self.current_trajectory = []
        self.current_episode_info = None
        
    def setup_colormap(self):
        """Setup colormap for terrain visualization"""
        # Convert RGB values from 0-255 to 0-1 range
        terrain_colors = []
        terrain_names = []
        
        for level in range(9):
            terrain_info = TERRAIN_LEVELS[level]
            color_name = terrain_info["color"]
            if color_name in palette:
                rgb = np.array(palette[color_name]) / 255.0
                terrain_colors.append(rgb)
                terrain_names.append(terrain_info["name"])
        
        self.terrain_colors = terrain_colors
        self.terrain_names = terrain_names
        self.terrain_cmap = ListedColormap(terrain_colors)
        
    def start_episode(self, episode_info: Dict):
        """Start tracking a new episode"""
        self.current_trajectory = []
        self.current_episode_info = episode_info.copy()
        
        # Add initial position
        initial_pos = episode_info.get('initial_position')
        if initial_pos is not None:
            self.current_trajectory.append({
                'position': tuple(initial_pos),
                'step': 0,
                'action': None,
                'reward': 0,
                'hp': episode_info.get('initial_hp', 75),
                'resources': episode_info.get('initial_resources', 0),
                'terrain': episode_info.get('initial_terrain', 'unknown')
            })
    
    def add_step(self, position: np.ndarray, step: int, action: int, reward: float, 
                 hp: float, resources: float, terrain: str):
        """Add a step to the current trajectory"""
        self.current_trajectory.append({
            'position': tuple(position),
            'step': step,
            'action': action,
            'reward': reward,
            'hp': hp,
            'resources': resources,
            'terrain': terrain
        })
    
    def save_trajectory_image(self, world_map: torch.Tensor, target_position: np.ndarray,
                            success: bool, episode_num: int, 
                            additional_info: Optional[Dict] = None) -> str:
        """
        Save trajectory visualization as image
        
        Args:
            world_map: The island heightmap
            target_position: Target coordinates
            success: Whether episode was successful
            episode_num: Episode number for filename
            additional_info: Additional info to display
            
            Returns:
            Path to saved image
        """
        print(f"[DEBUG] Attempting to save trajectory for episode {episode_num}")
        print(f"[DEBUG] Trajectory length: {len(self.current_trajectory)}")
        print(f"[DEBUG] Save directory: {self.save_dir}")
        
        if not self.current_trajectory:
            print("[WARNING] No trajectory data to visualize")
            return None
            
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "SUCCESS" if success else "FAILED"
        filename = f"trajectory_ep{episode_num:04d}_{status}_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Terrain map with trajectory
        self._plot_terrain_trajectory(ax1, world_map, target_position)
        
        # Right plot: Episode statistics
        self._plot_episode_stats(ax2, additional_info)
        
        # Overall title
        total_reward = sum(step['reward'] for step in self.current_trajectory)
        final_step = len(self.current_trajectory) - 1
        
        fig.suptitle(
            f"Episode {episode_num} - {status}\\n"
            f"Steps: {final_step}, Total Reward: {total_reward:.2f}\\n"
            f"Island Seed: {self.current_episode_info.get('island_seed', 'Unknown')}", 
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_terrain_trajectory(self, ax, world_map: torch.Tensor, target_position: np.ndarray):
        """Plot the terrain map with agent trajectory"""
        # Convert heightmap to terrain levels for visualization
        terrain_map = self._heightmap_to_terrain(world_map)
        
        # Plot terrain
        im = ax.imshow(terrain_map, cmap=self.terrain_cmap, vmin=0, vmax=8, 
                      origin='upper', alpha=0.8)
        
        # Extract trajectory positions
        positions = [step['position'] for step in self.current_trajectory]
        if positions:
            x_coords = [pos[1] for pos in positions]  # Note: matplotlib uses (x,y) = (col,row)
            y_coords = [pos[0] for pos in positions]
            
            # Plot trajectory path
            ax.plot(x_coords, y_coords, 'white', linewidth=3, alpha=0.8, linestyle='-')
            ax.plot(x_coords, y_coords, 'red', linewidth=2, linestyle='-', label='Agent Path')
            
            # Mark start position
            ax.scatter(x_coords[0], y_coords[0], c='lime', s=150, marker='o', 
                      edgecolors='black', linewidth=2, label='Start', zorder=10)
            
            # Mark end position  
            if len(positions) > 1:
                ax.scatter(x_coords[-1], y_coords[-1], c='red', s=150, marker='X', 
                          edgecolors='black', linewidth=2, label='End', zorder=10)
        
        # Mark target position
        ax.scatter(target_position[1], target_position[0], c='gold', s=200, marker='*', 
                  edgecolors='black', linewidth=2, label='Target', zorder=10)
        
        ax.set_title('Agent Trajectory on Island', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add colorbar for terrain types
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_ticks(range(9))
        cbar.set_ticklabels([TERRAIN_LEVELS[i]['name'].title() for i in range(9)])
        cbar.set_label('Terrain Type', rotation=270, labelpad=20)
    
    def _plot_episode_stats(self, ax, additional_info: Optional[Dict]):
        """Plot episode statistics and information"""
        ax.axis('off')  # Turn off axis for text display
        
        if not self.current_trajectory:
            ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', fontsize=14)
            return
        
        # Calculate statistics
        stats = self._calculate_trajectory_stats()
        
        # Create info text
        info_lines = [
            f"Episode Summary",
            f"="*29,
            f"Total Steps: {stats['total_steps']}",
            f"Total Reward: {stats['total_reward']:.2f}",
            f"Average Reward/Step: {stats['avg_reward_per_step']:.3f}",
            f"",
            f"Navigation Stats",
            f"="*29,
            f"Initial Distance: {stats['initial_distance']:.1f}",
            f"Final Distance: {stats['final_distance']:.1f}",
            f"Distance Covered: {stats['distance_covered']:.1f}",
            f"Efficiency: {stats['efficiency']:.3f}",
            f"",
            f"Agent Condition",
            f"="*29,
            f"Initial HP: {stats['initial_hp']:.1f}",
            f"Final HP: {stats['final_hp']:.1f}",
            f"HP Change: {stats['hp_change']:.1f}",
            f"Final Resources: {stats['final_resources']:.1f}",
            f"",
            f"Terrain Explored",
            f"="*29
        ]
        
        # Add terrain distribution
        terrain_counts = stats['terrain_distribution']
        for terrain, count in terrain_counts.items():
            if count > 0:
                percentage = (count / stats['total_steps']) * 100
                info_lines.append(f"{terrain}: {count} steps ({percentage:.1f}%)")
        
        # Add additional info if provided
        if additional_info:
            info_lines.extend([
                "",
                "Additional Info", 
                "="*29
            ])
            for key, value in additional_info.items():
                if isinstance(value, float):
                    info_lines.append(f"{key}: {value:.3f}")
                else:
                    info_lines.append(f"{key}: {value}")
        
        # Display text
        text = "\\n".join(info_lines)
        ax.text(0.05, 0.95, text, ha='left', va='top', fontsize=10, 
               fontfamily='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def _calculate_trajectory_stats(self) -> Dict:
        """Calculate statistics from current trajectory"""
        if not self.current_trajectory:
            return {}
        
        first_step = self.current_trajectory[0]
        last_step = self.current_trajectory[-1]
        
        # Basic stats
        total_steps = len(self.current_trajectory) - 1  # Exclude initial position
        total_reward = sum(step['reward'] for step in self.current_trajectory)
        avg_reward_per_step = total_reward / max(1, total_steps)
        
        # Navigation stats
        target_pos = self.current_episode_info.get('target_position', [0, 0])
        initial_pos = first_step['position']
        final_pos = last_step['position']
        
        initial_distance = np.linalg.norm(np.array(initial_pos) - np.array(target_pos))
        final_distance = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
        distance_covered = initial_distance - final_distance
        efficiency = distance_covered / max(1, total_steps)
        
        # Agent condition
        initial_hp = first_step['hp']
        final_hp = last_step['hp']
        hp_change = final_hp - initial_hp
        final_resources = last_step['resources']
        
        # Terrain distribution
        terrain_counts = {}
        for step in self.current_trajectory[1:]:  # Skip initial position
            terrain = step['terrain']
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        
        return {
            'total_steps': total_steps,
            'total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'distance_covered': distance_covered,
            'efficiency': efficiency,
            'initial_hp': initial_hp,
            'final_hp': final_hp,
            'hp_change': hp_change,
            'final_resources': final_resources,
            'terrain_distribution': terrain_counts
        }
    
    def _heightmap_to_terrain(self, world_map: torch.Tensor) -> np.ndarray:
        """Convert heightmap to terrain level map for visualization"""
        terrain_map = np.zeros_like(world_map.numpy(), dtype=int)
        
        for i in range(world_map.shape[0]):
            for j in range(world_map.shape[1]):
                height = world_map[i, j].item()
                
                # Determine terrain level based on height thresholds
                terrain_level = 8  # Default to highest level
                for level in range(9):
                    if height <= TERRAIN_LEVELS[level]["threshold"]:
                        terrain_level = level
                        break
                
                terrain_map[i, j] = terrain_level
        
        return terrain_map
    
    def save_summary_image(self, episode_trajectories: List[Dict], filename: str = None):
        """
        Create a summary image showing multiple episode trajectories
        
        Args:
            episode_trajectories: List of trajectory data for multiple episodes
            filename: Optional custom filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_summary_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Create grid of trajectory plots
        n_episodes = len(episode_trajectories)
        cols = min(4, n_episodes)
        rows = (n_episodes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_episodes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, traj_data in enumerate(episode_trajectories):
            if i >= len(axes):
                break
                
            ax = axes[i]
            # Plot individual trajectory (simplified version)
            # This would need the world_map and other data for each episode
            ax.set_title(f"Episode {traj_data.get('episode_num', i+1)}")
            ax.text(0.5, 0.5, f"Success: {traj_data.get('success', False)}\\n"
                              f"Steps: {traj_data.get('steps', 'N/A')}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for i in range(n_episodes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath


# Utility functions for integration with training

def create_trajectory_logger(log_dir: str, save_every_n_episodes: int = 10, 
                           save_successful_only: bool = False) -> TrajectoryVisualizer:
    """
    Create a trajectory logger with common settings
    
    Args:
        log_dir: Directory to save trajectory images
        save_every_n_episodes: Save trajectory image every N episodes
        save_successful_only: Only save images for successful episodes
        
    Returns:
        Configured TrajectoryVisualizer instance
    """
    trajectory_dir = os.path.join(log_dir, "trajectories")
    visualizer = TrajectoryVisualizer(trajectory_dir)
    visualizer.save_every_n_episodes = save_every_n_episodes
    visualizer.save_successful_only = save_successful_only
    
    return visualizer


# Example usage and testing
if __name__ == "__main__":
    # Test the visualizer with dummy data
    print("Testing Trajectory Visualizer...")
    
    # Create dummy world map
    world_map = torch.rand(100, 100) * 0.8  # Random heightmap
    
    # Create visualizer
    visualizer = TrajectoryVisualizer("test_trajectories")
    
    # Simulate episode
    episode_info = {
        'initial_position': np.array([10, 10]),
        'target_position': np.array([80, 80]),
        'initial_hp': 75,
        'initial_resources': 0,
        'initial_terrain': 'grassland',
        'island_seed': 12345
    }
    
    visualizer.start_episode(episode_info)
    
    # Simulate trajectory (random walk towards target)
    current_pos = np.array([10, 10])
    target_pos = np.array([80, 80])
    
    for step in range(50):
        # Simple movement towards target with some randomness
        direction = target_pos - current_pos
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        noise = np.random.randn(2) * 0.3
        movement = direction + noise
        current_pos = current_pos + movement
        current_pos = np.clip(current_pos, 0, 99).astype(int)
        
        # Add step
        visualizer.add_step(
            position=current_pos,
            step=step + 1,
            action=np.random.randint(5),
            reward=np.random.randn() * 0.1,
            hp=75 - step * 0.5,
            resources=max(0, np.random.randn()),
            terrain=np.random.choice(['ocean', 'grassland', 'forest', 'mountain'])
        )
        
        # Stop if close to target
        if np.linalg.norm(current_pos - target_pos) < 5:
            break
    
    # Save trajectory image
    success = np.linalg.norm(current_pos - target_pos) < 10
    filepath = visualizer.save_trajectory_image(
        world_map=world_map,
        target_position=target_pos,
        success=success,
        episode_num=1,
        additional_info={'test_mode': True, 'random_seed': 42}
    )
    
    print(f"✅ Test trajectory saved to: {filepath}")
    print("Trajectory visualizer is working correctly!")