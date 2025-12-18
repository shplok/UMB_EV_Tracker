import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Tuple, Dict, Any


# THIS IS WHERE THE FIRST SHORTCOMING ARISES
def create_large_ev_filter(radius: int = 10, size: int = 41, sigma: float = 2.0) -> np.ndarray:

    filter_img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)

    # Create filter for larger EVs
    # Bright center (positive response to particle center)
    filter_img[distance <= radius] = 1.0
    # Dark ring (negative response to particle edges)
    filter_img[(distance > radius) & (distance <= radius*1.5)] = -0.8
    # Faint outer ring (slight positive response to background contrast)
    filter_img[(distance > radius*1.5) & (distance <= radius*2.0)] = 0.3 # specify max values in readme...

    # Soften the filter with Gaussian blur
    filter_img = ndimage.gaussian_filter(filter_img, sigma=sigma)

    # Normalize the filter (zero mean, unit norm)
    filter_img -= np.mean(filter_img)
    filter_img /= np.linalg.norm(filter_img)

    return filter_img


# Create Visualization of the Filter
def visualize_filter(ev_filter: np.ndarray, output_dir: str, 
                    filter_params: Dict[str, Any] = None) -> str:
 
    fig = plt.figure(figsize=(18, 6))
    
    # 2D heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(ev_filter, cmap='RdBu_r', interpolation='bilinear')
    ax1.set_title('EV Filter (2D Heatmap)\nRed=Positive, Blue=Negative', fontsize=12)
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    plt.colorbar(im1, ax=ax1)
    
    # Add circle overlays to show the different regions
    center = ev_filter.shape[0] // 2
    if filter_params:
        radius = filter_params.get('radius', 10)
        circle1 = plt.Circle((center, center), radius, color='yellow', fill=False, linewidth=2, alpha=0.7)
        circle2 = plt.Circle((center, center), radius*1.5, color='orange', fill=False, linewidth=2, alpha=0.7)
        circle3 = plt.Circle((center, center), radius*2.0, color='red', fill=False, linewidth=2, alpha=0.7)
        ax1.add_patch(circle1)
        ax1.add_patch(circle2)
        ax1.add_patch(circle3)
    
    # 3D surface plot
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    x = np.arange(0, ev_filter.shape[0])
    y = np.arange(0, ev_filter.shape[1])
    X, Y = np.meshgrid(x, y)
    surf = ax2.plot_surface(X, Y, ev_filter, cmap='RdBu_r', alpha=0.9)
    ax2.set_title('EV Filter (3D Surface)', fontsize=12)
    ax2.set_xlabel('X Pixels')
    ax2.set_ylabel('Y Pixels')
    ax2.set_zlabel('Filter Value')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    # Cross-section plot
    ax3 = fig.add_subplot(1, 3, 3)
    center = ev_filter.shape[0] // 2
    ax3.plot(ev_filter[center, :], 'r-', linewidth=2, label='Horizontal Cross-section')
    ax3.plot(ev_filter[:, center], 'b-', linewidth=2, label='Vertical Cross-section')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Filter Cross-Sections', fontsize=12)
    ax3.set_xlabel('Pixel Position')
    ax3.set_ylabel('Filter Value')
    ax3.legend()
    
    # Add parameter information if available
    if filter_params:
        param_text = f"Parameters: radius={filter_params.get('radius', 'N/A')}, " \
                    f"size={filter_params.get('size', 'N/A')}, " \
                    f"sigma={filter_params.get('sigma', 'N/A')}"
        fig.suptitle(f'EV Detection Filter Visualization\n{param_text}', fontsize=14)
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, "ev_filter_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Filter visualization saved to: {viz_path}")
    return viz_path


def save_filter_data(ev_filter: np.ndarray, filter_params: Dict[str, Any], 
                    output_dir: str) -> str:
    filter_path = os.path.join(output_dir, "ev_filter.npy")
    np.save(filter_path, ev_filter)
    
    print(f"Filter data saved to: {filter_path}")
    return filter_path
