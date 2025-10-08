"""
Simple Heightfield Generator for CoppeliaSim
=============================================
This script creates a heightfield shape from a PNG map in the currently open CoppeliaSim scene.

USAGE:
1. Open CoppeliaSim
2. Create or open a scene
3. Run this script with a map file
4. The heightfield will be added to your current scene
5. Save the scene manually in CoppeliaSim (File > Save Scene)
conda activate tp2; python scripts/simple_heightfield_generator.py --map mapas/square_maze.png --width 10 --height 10
Run with a smaller --max-dim parameter:


python scripts/simple_heightfield_generator.py --map mapas/cave.png --width 10 --height 4 --max-dim 100

Author: Daniel Terra Gomes
Organization: Synergia UFMG
Date: 2025
"""

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not found. Install with: pip install pillow")
    sys.exit(1)

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("Error: CoppeliaSim API not found. Install with: pip install coppeliasim-zmqremoteapi-client")
    sys.exit(1)


def load_and_downsample_map(map_path, invert=False, threshold=0.5, max_dim=200):
    """Load PNG map and convert to downsampled occupancy grid."""

    # Load image
    img = Image.open(map_path)
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy array [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    if invert:
        img_array = 1.0 - img_array

    # Binarize: 1 = free, 0 = obstacle
    occupancy_grid = (img_array >= threshold).astype(np.float32)

    print(f"Loaded map: {map_path}")
    print(f"  Original size: {occupancy_grid.shape}")
    print(f"  Free space: {np.sum(occupancy_grid)/occupancy_grid.size*100:.1f}%")

    # Downsample if too large
    if occupancy_grid.shape[0] > max_dim or occupancy_grid.shape[1] > max_dim:
        from scipy import ndimage

        factor = max(occupancy_grid.shape[0] // max_dim, occupancy_grid.shape[1] // max_dim)
        new_shape = (occupancy_grid.shape[0] // factor, occupancy_grid.shape[1] // factor)
        occupancy_grid = ndimage.zoom(
            occupancy_grid,
            (new_shape[0] / occupancy_grid.shape[0], new_shape[1] / occupancy_grid.shape[1]),
            order=1
        )
        print(f"  Downsampled by {factor}x to: {occupancy_grid.shape}")

    return occupancy_grid


def create_heightfield(occupancy_grid, world_width=10.0, world_height=10.0, obstacle_height=1.0):
    """Create heightfield in currently open CoppeliaSim scene."""

    print("\nConnecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.require('sim')
    print("✓ Connected!")

    # Create height array: obstacles raised, free space at ground level
    height_array = (1.0 - occupancy_grid) * obstacle_height
    height_array = np.flipud(height_array)  # Flip for CoppeliaSim coordinates

    # Get dimensions
    y_count, x_count = occupancy_grid.shape
    heights = height_array.flatten().tolist()

    print(f"\nCreating heightfield:")
    print(f"  Grid: {x_count} x {y_count} points")
    print(f"  World size: {world_width}m x {world_height}m")
    print(f"  Obstacle height: {obstacle_height}m")

    # Create heightfield shape
    try:
        handle = sim.createHeightfieldShape(
            0,           # options (0 = default, full physics)
            45.0,        # shading angle
            x_count,     # x points
            y_count,     # y points
            world_width, # x size in meters
            heights      # height values
        )

        # Configure the heightfield
        sim.setObjectAlias(handle, "Terrain_Heightfield")
        sim.setObjectPosition(handle, -1, [0, 0, 0])
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5])

        print(f"✓ Heightfield created successfully (handle: {handle})")
        print("\n" + "="*70)
        print("SUCCESS! Heightfield added to your scene.")
        print("Now save your scene in CoppeliaSim: File > Save Scene")
        print("="*70)

        return handle

    except Exception as e:
        print(f"\n✗ Error creating heightfield: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Create heightfield from PNG map in currently open CoppeliaSim scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  1. Start CoppeliaSim
  2. Open or create a scene
  3. Run: python scripts/simple_heightfield_generator.py --map mapas/paredes.png
  4. Save the scene in CoppeliaSim (File > Save Scene)

EXAMPLES:
  # Simple usage
  python scripts/simple_heightfield_generator.py --map mapas/paredes.png

  # Custom dimensions
  python scripts/simple_heightfield_generator.py --map mapas/circular_maze.png --width 15 --height 15

  # Taller obstacles
  python scripts/simple_heightfield_generator.py --map mapas/cave.png --obstacle-height 2.0

  # Inverted map (black=free, white=obstacle)
  python scripts/simple_heightfield_generator.py --map mapas/map.png --invert
        """
    )

    parser.add_argument('--map', '-m', required=True, help='Path to PNG map file')
    parser.add_argument('--width', '-w', type=float, default=10.0, help='World width (m, default: 10.0)')
    parser.add_argument('--height', '-ht', type=float, default=10.0, help='World height (m, default: 10.0)')
    parser.add_argument('--obstacle-height', '-oh', type=float, default=1.0, help='Obstacle height (m, default: 1.0)')
    parser.add_argument('--invert', '-i', action='store_true', help='Invert map colors')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Binarization threshold (default: 0.5)')
    parser.add_argument('--max-dim', type=int, default=200, help='Max dimension before downsampling (default: 200)')

    args = parser.parse_args()

    print("="*70)
    print("Simple Heightfield Generator for CoppeliaSim")
    print("="*70)

    try:
        # Load map
        occupancy_grid = load_and_downsample_map(
            args.map,
            invert=args.invert,
            threshold=args.threshold,
            max_dim=args.max_dim
        )

        # Create heightfield in current scene
        create_heightfield(
            occupancy_grid,
            world_width=args.width,
            world_height=args.height,
            obstacle_height=args.obstacle_height
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
