import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not found. Install with: pip install pillow")
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
        plt.imsave("mapa_convertido.png", occupancy_grid, cmap='gray')
        plt.imsave("mapa_convertido_invertido.png", 1 - occupancy_grid, cmap='gray')
        print("✓ Imagem convertida salva em 'mapa_convertido.png'")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
