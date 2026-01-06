"""
Persistent Homology Module

Provides persistent homology analysis for hydrogen bond networks
using the gudhi library.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    gudhi = None

from ..io.trajectory_parser import Frame


class PersistenceAnalyzer:
    """
    Persistent homology analyzer for hydrogen bond networks.
    
    Uses gudhi to compute Rips complexes and persistence diagrams
    from water molecule positions and hydrogen bond networks.
    
    Parameters
    ----------
    max_edge_length : float
        Maximum edge length for Rips complex (default: 5.0 Å)
    max_dimension : int
        Maximum homology dimension to compute (default: 2)
    """
    
    def __init__(
        self,
        max_edge_length: float = 5.0,
        max_dimension: int = 2
    ):
        if not HAS_GUDHI:
            raise ImportError(
                "gudhi is required for persistence homology. "
                "Install with: pip install gudhi"
            )
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
    
    def compute_rips_persistence(
        self, 
        positions: np.ndarray
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Compute persistence diagram from Rips complex on point cloud.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (n_points, 3) containing 3D coordinates
            
        Returns
        -------
        list
            List of (dimension, (birth, death)) tuples
        """
        rips = gudhi.RipsComplex(
            points=positions,
            max_edge_length=self.max_edge_length
        )
        
        simplex_tree = rips.create_simplex_tree(
            max_dimension=self.max_dimension + 1
        )
        
        simplex_tree.compute_persistence()
        persistence = simplex_tree.persistence()
        
        return persistence
    
    def compute_alpha_persistence(
        self, 
        positions: np.ndarray
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Compute persistence diagram from Alpha complex.
        
        Alpha complex is more efficient than Rips for 3D data.
        
        Parameters
        ----------
        positions : np.ndarray
            Array of shape (n_points, 3) containing 3D coordinates
            
        Returns
        -------
        list
            List of (dimension, (birth, death)) tuples
        """
        alpha = gudhi.AlphaComplex(points=positions)
        simplex_tree = alpha.create_simplex_tree()
        simplex_tree.compute_persistence()
        
        return simplex_tree.persistence()
    
    def persistence_to_array(
        self, 
        persistence: List[Tuple[int, Tuple[float, float]]],
        dimension: Optional[int] = None
    ) -> np.ndarray:
        """
        Convert persistence list to numpy array.
        
        Parameters
        ----------
        persistence : list
            Output from compute_*_persistence
        dimension : int, optional
            Filter by dimension (None = all dimensions)
            
        Returns
        -------
        np.ndarray
            Array of shape (n, 3) with columns [dimension, birth, death]
        """
        if dimension is not None:
            filtered = [
                (d, b, death if death != float('inf') else -1)
                for d, (b, death) in persistence
                if d == dimension
            ]
        else:
            filtered = [
                (d, b, death if death != float('inf') else -1)
                for d, (b, death) in persistence
            ]
        
        if not filtered:
            return np.array([]).reshape(0, 3)
        
        return np.array(filtered)
    
    def compute_betti_curve(
        self,
        persistence: List[Tuple[int, Tuple[float, float]]],
        dimension: int,
        n_points: int = 100,
        max_scale: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Betti curve (Betti number as function of scale).
        
        Parameters
        ----------
        persistence : list
            Persistence diagram
        dimension : int
            Homology dimension
        n_points : int
            Number of scale points
        max_scale : float, optional
            Maximum scale value
            
        Returns
        -------
        scales : np.ndarray
            Scale parameter values
        betti : np.ndarray
            Betti numbers at each scale
        """
        dgm = self.persistence_to_array(persistence, dimension)
        
        if len(dgm) == 0:
            scales = np.linspace(0, max_scale or self.max_edge_length, n_points)
            return scales, np.zeros(n_points)
        
        births = dgm[:, 1]
        deaths = dgm[:, 2]
        deaths[deaths < 0] = max_scale or self.max_edge_length
        
        if max_scale is None:
            max_scale = np.max(deaths[deaths > 0]) * 1.1
        
        scales = np.linspace(0, max_scale, n_points)
        betti = np.zeros(n_points)
        
        for i, s in enumerate(scales):
            # Count features alive at scale s
            betti[i] = np.sum((births <= s) & (deaths > s))
        
        return scales, betti
    
    def compute_persistence_landscape(
        self,
        persistence: List[Tuple[int, Tuple[float, float]]],
        dimension: int,
        n_layers: int = 5,
        n_points: int = 100,
        max_scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute persistence landscape for machine learning.
        
        Parameters
        ----------
        persistence : list
            Persistence diagram
        dimension : int
            Homology dimension
        n_layers : int
            Number of landscape layers
        n_points : int
            Number of discretization points
        max_scale : float, optional
            Maximum scale value
            
        Returns
        -------
        np.ndarray
            Landscape of shape (n_layers, n_points)
        """
        dgm = self.persistence_to_array(persistence, dimension)
        
        if len(dgm) == 0:
            return np.zeros((n_layers, n_points))
        
        births = dgm[:, 1]
        deaths = dgm[:, 2]
        deaths[deaths < 0] = max_scale or self.max_edge_length
        
        if max_scale is None:
            max_scale = np.max(deaths[deaths > 0]) * 1.1
        
        scales = np.linspace(0, max_scale, n_points)
        
        # Compute tent functions for each feature
        def tent(b, d, s):
            mid = (b + d) / 2
            if s < b or s > d:
                return 0
            elif s <= mid:
                return s - b
            else:
                return d - s
        
        # Compute landscape values at each scale
        landscapes = np.zeros((n_layers, n_points))
        
        for i, s in enumerate(scales):
            # Get tent function values for all features at this scale
            values = [tent(births[j], deaths[j], s) for j in range(len(births))]
            values = sorted(values, reverse=True)
            
            # Take top n_layers values
            for layer in range(min(n_layers, len(values))):
                landscapes[layer, i] = values[layer]
        
        return landscapes
    
    def bottleneck_distance(
        self,
        dgm1: List[Tuple[int, Tuple[float, float]]],
        dgm2: List[Tuple[int, Tuple[float, float]]],
        dimension: int = 1
    ) -> float:
        """
        Compute bottleneck distance between two persistence diagrams.
        
        Parameters
        ----------
        dgm1, dgm2 : list
            Persistence diagrams
        dimension : int
            Homology dimension to compare
            
        Returns
        -------
        float
            Bottleneck distance
        """
        arr1 = self.persistence_to_array(dgm1, dimension)
        arr2 = self.persistence_to_array(dgm2, dimension)
        
        if len(arr1) == 0 and len(arr2) == 0:
            return 0.0
        
        # Convert to gudhi format
        dgm1_gudhi = arr1[:, 1:3] if len(arr1) > 0 else np.array([]).reshape(0, 2)
        dgm2_gudhi = arr2[:, 1:3] if len(arr2) > 0 else np.array([]).reshape(0, 2)
        
        # Replace -1 (infinity) with large value
        dgm1_gudhi[dgm1_gudhi < 0] = 1e10
        dgm2_gudhi[dgm2_gudhi < 0] = 1e10
        
        return gudhi.bottleneck_distance(dgm1_gudhi, dgm2_gudhi)
    
    def analyze_frame(
        self,
        positions: np.ndarray,
        method: str = 'rips'
    ) -> Dict[str, Any]:
        """
        Perform full persistence analysis on a set of positions.
        
        Parameters
        ----------
        positions : np.ndarray
            Point cloud coordinates
        method : str
            'rips' or 'alpha'
            
        Returns
        -------
        dict
            Dictionary containing persistence diagram and derived features
        """
        if method == 'rips':
            persistence = self.compute_rips_persistence(positions)
        else:
            persistence = self.compute_alpha_persistence(positions)
        
        result = {
            'persistence': persistence,
            'n_features': {
                dim: len(self.persistence_to_array(persistence, dim))
                for dim in range(self.max_dimension + 1)
            }
        }
        
        # Compute Betti curves
        for dim in range(self.max_dimension + 1):
            scales, betti = self.compute_betti_curve(persistence, dim)
            result[f'betti_curve_{dim}'] = (scales, betti)
        
        return result


def plot_persistence_diagram(
    persistence: List[Tuple[int, Tuple[float, float]]],
    title: str = "Persistence Diagram",
    save_path: Optional[str | Path] = None,
    show: bool = True
):
    """
    Plot persistence diagram.
    
    Parameters
    ----------
    persistence : list
        Persistence diagram from gudhi
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['blue', 'orange', 'green', 'red']
    
    max_val = 0
    for dim, (birth, death) in persistence:
        if death == float('inf'):
            death = birth + 1
        max_val = max(max_val, birth, death)
        
        color = colors[dim % len(colors)]
        ax.scatter(birth, death, c=color, alpha=0.7, s=50, 
                   label=f'H{dim}' if dim not in [d for d, _ in persistence[:persistence.index((dim, (birth, death)))]] else '')
    
    # Diagonal line
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
