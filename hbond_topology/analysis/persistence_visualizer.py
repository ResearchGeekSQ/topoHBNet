"""
Persistence Visualization Module

This module provides functions for visualizing persistent homology results,
including persistence barcodes and persistence diagrams.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple


def plot_persistence_barcode(
    barcodes: Dict[str, List[List[float]]],
    title: str = "Persistence Barcode",
    max_epsilon: float = 5.0,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot persistence barcode for multiple dimensions.

    Args:
        barcodes: Dictionary mapping dimension names (e.g., 'H0', 'H1') to lists of [birth, death] pairs.
        title: Title of the plot.
        max_epsilon: Maximum value for the x-axis (epsilon).
        figsize: Size of the figure.
        save_path: Path to save the plot. If None, the plot is displayed.
        dpi: DPI for the saved plot.
    """
    plt.figure(figsize=figsize)
    
    # Sort dimensions for consistent plotting
    dims = sorted(barcodes.keys())
    
    current_y = 0
    try:
        colors = plt.cm.get_cmap("tab10")
    except Exception:
        # Compatibility for newer matplotlib
        import matplotlib as mpl
        colors = mpl.colormaps["tab10"]
    
    for i, dim in enumerate(dims):
        intervals = barcodes[dim]
        color = colors(i)
        
        # Sort intervals by birth time, then by death time
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        
        for birth, death in intervals:
            # Handle infinite death time
            if death == float('inf'):
                death = max_epsilon
            
            plt.hlines(current_y, birth, death, colors=color, linewidth=2, label=dim if birth == intervals[0][0] and death == intervals[0][1] else "")
            current_y += 1
            
        # Add a small gap between dimensions
        current_y += 1
        
    plt.xlabel("Distance Threshold ($\\epsilon$)", fontsize=12)
    plt.ylabel("Bars", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks([])  # Hide y-axis ticks as they are just bar indices
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.xlim(0, max_epsilon)
    
    # Create a custom legend to show only dimensions, not every bar
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors(i), lw=2, label=dim) for i, dim in enumerate(dims)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_persistence_diagram(
    barcodes: Dict[str, List[List[float]]],
    title: str = "Persistence Diagram",
    max_epsilon: float = 5.0,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot persistence diagram (Birth vs Death) for multiple dimensions.

    Args:
        barcodes: Dictionary mapping dimension names (e.g., 'H0', 'H1') to lists of [birth, death] pairs.
        title: Title of the plot.
        max_epsilon: Maximum value for the x-axis and y-axis.
        figsize: Size of the figure.
        save_path: Path to save the plot. If None, the plot is displayed.
        dpi: DPI for the saved plot.
    """
    plt.figure(figsize=figsize)
    
    dims = sorted(barcodes.keys())
    try:
        colors = plt.cm.get_cmap("tab10")
    except Exception:
        import matplotlib as mpl
        colors = mpl.colormaps["tab10"]
    
    for i, dim in enumerate(dims):
        intervals = np.array(barcodes[dim])
        if len(intervals) == 0:
            continue
            
        births = intervals[:, 0]
        deaths = intervals[:, 1]
        
        # Replace infinity with max_epsilon for visualization
        deaths = np.where(np.isinf(deaths), max_epsilon, deaths)
        
        plt.scatter(births, deaths, color=colors(i), label=dim, alpha=0.7, edgecolors='white', s=50)
        
    # Plot diagonal line (birth = death)
    plt.plot([0, max_epsilon], [0, max_epsilon], 'k--', alpha=0.5)
    
    plt.xlabel("Birth ($\\epsilon$)", fontsize=12)
    plt.ylabel("Death ($\\epsilon$)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, max_epsilon)
    plt.ylim(0, max_epsilon)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
