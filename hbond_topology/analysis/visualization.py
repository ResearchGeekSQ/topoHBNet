"""
Visualization Module

Provides visualization tools for hydrogen bond network topology analysis.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .dynamics import TrajectoryAnalysisResult


class TopologyVisualizer:
    """
    Visualization tools for H-bond network topology analysis.
    
    Parameters
    ----------
    figsize : tuple
        Default figure size
    dpi : int
        Resolution for saved figures
    style : str
        Matplotlib style to use
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for visualization")
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except Exception:
            pass
    
    def plot_dynamics(
        self,
        results: TrajectoryAnalysisResult,
        save_path: Optional[str | Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot time evolution of topological properties.
        
        Creates a 2x2 subplot with:
        - Betti numbers over time
        - H-bond count over time  
        - Triangle count over time
        - Euler characteristic over time
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        timesteps = results.timesteps
        
        # Betti numbers
        ax = axes[0, 0]
        ax.plot(timesteps, results.betti_0, label='β₀ (components)', 
                color='#1f77b4', linewidth=1.5)
        ax.plot(timesteps, results.betti_1, label='β₁ (loops)',
                color='#ff7f0e', linewidth=1.5)
        ax.plot(timesteps, results.betti_2, label='β₂ (voids)',
                color='#2ca02c', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Betti Number')
        ax.set_title('Topological Invariants')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # H-bond count
        ax = axes[0, 1]
        ax.plot(timesteps, results.n_hbonds, color='#d62728', linewidth=1.5)
        ax.fill_between(timesteps, results.n_hbonds, alpha=0.3, color='#d62728')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of H-bonds')
        ax.set_title('Hydrogen Bond Count')
        ax.grid(True, alpha=0.3)
        
        # Triangle count
        ax = axes[1, 0]
        ax.plot(timesteps, results.n_triangles, color='#9467bd', linewidth=1.5)
        ax.fill_between(timesteps, results.n_triangles, alpha=0.3, color='#9467bd')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Triangles')
        ax.set_title('3-Water H-bond Rings')
        ax.grid(True, alpha=0.3)
        
        # Euler characteristic
        ax = axes[1, 1]
        ax.plot(timesteps, results.euler_characteristic, 
                color='#8c564b', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Euler Characteristic (χ)')
        ax.set_title('Euler Characteristic')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_hbond_statistics(
        self,
        results: TrajectoryAnalysisResult,
        save_path: Optional[str | Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot H-bond distance and angle distributions.
        """
        data = results.to_numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Mean H-bond distance over time
        ax = axes[0]
        ax.plot(data['timesteps'], data['mean_hbond_distance'], 
                color='#1f77b4', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean D-A Distance (Å)')
        ax.set_title('H-bond Distance Evolution')
        ax.grid(True, alpha=0.3)
        
        # Mean H-bond angle over time
        ax = axes[1]
        ax.plot(data['timesteps'], data['mean_hbond_angle'],
                color='#ff7f0e', linewidth=1.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean D-H-A Angle (°)')
        ax.set_title('H-bond Angle Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_network_snapshot(
        self,
        edges: List[Tuple[int, int]],
        positions: Optional[Dict[int, np.ndarray]] = None,
        node_colors: Optional[Dict[int, str]] = None,
        title: str = "H-bond Network",
        save_path: Optional[str | Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot a snapshot of the H-bond network.
        
        Parameters
        ----------
        edges : list
            List of (node1, node2) edges
        positions : dict, optional
            Node positions for layout
        node_colors : dict, optional
            Color mapping for nodes
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required for network visualization")
        
        G = nx.Graph()
        G.add_edges_from(edges)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Compute layout
        if positions is None:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            # Use 2D projection of 3D positions
            pos = {k: v[:2] for k, v in positions.items() if k in G.nodes()}
            # Add any missing nodes with spring layout
            missing = set(G.nodes()) - set(pos.keys())
            if missing:
                pos_missing = nx.spring_layout(G.subgraph(missing))
                pos.update(pos_missing)
        
        # Node colors
        if node_colors is None:
            colors = ['#1f77b4'] * len(G.nodes())
        else:
            colors = [node_colors.get(n, '#1f77b4') for n in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=colors, 
                               node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='#888888',
                               alpha=0.5, width=1.0, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_statistics_summary(
        self,
        results: TrajectoryAnalysisResult,
        save_path: Optional[str | Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create summary box plots for all properties.
        """
        data = results.to_numpy()
        
        properties = ['n_hbonds', 'betti_0', 'betti_1', 'n_triangles']
        labels = ['H-bonds', 'β₀', 'β₁', 'Triangles']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_data = [data[prop] for prop in properties]
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Topological Properties')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_autocorrelation(
        self,
        autocorr: np.ndarray,
        property_name: str,
        save_path: Optional[str | Path] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot autocorrelation function.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        lags = np.arange(len(autocorr))
        ax.plot(lags, autocorr, color='#1f77b4', linewidth=1.5)
        ax.fill_between(lags, autocorr, alpha=0.3, color='#1f77b4')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Lag (frames)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Autocorrelation: {property_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
