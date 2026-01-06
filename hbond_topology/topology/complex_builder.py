"""
Simplicial Complex Builder for Hydrogen Bond Networks

Constructs TopoNetX simplicial complexes from hydrogen bond networks.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict, Any
from ..io.trajectory_parser import Frame
from ..detection.hbond_detector import HBond, HBondDetector

try:
    import toponetx as tnx
    HAS_TOPONETX = True
except ImportError:
    HAS_TOPONETX = False
    tnx = None


class HBondComplexBuilder:
    """
    Build simplicial complexes from hydrogen bond networks.
    
    Creates topological structures where:
    - 0-simplices (nodes): Water molecules (represented by oxygen atoms)
    - 1-simplices (edges): Hydrogen bonds between water molecules
    - 2-simplices (triangles): Three water molecules mutually connected by H-bonds
    
    Parameters
    ----------
    include_triangles : bool
        Whether to detect and include 2-simplices (default: True)
    max_simplex_dim : int
        Maximum dimension of simplices to detect (default: 2)
    """
    
    def __init__(
        self,
        include_triangles: bool = True,
        max_simplex_dim: int = 2
    ):
        if not HAS_TOPONETX:
            raise ImportError(
                "TopoNetX is required for this module. "
                "Install it with: pip install toponetx"
            )
        self.include_triangles = include_triangles
        self.max_simplex_dim = max_simplex_dim
    
    def build_networkx_graph(
        self, 
        edges: List[Tuple[int, int]],
        node_positions: Optional[Dict[int, np.ndarray]] = None
    ) -> nx.Graph:
        """
        Build a NetworkX graph from H-bond edges.
        
        Parameters
        ----------
        edges : List[Tuple[int, int]]
            List of (donor_O_idx, acceptor_O_idx) tuples
        node_positions : dict, optional
            Mapping from node index to 3D position
            
        Returns
        -------
        nx.Graph
            NetworkX graph representing the H-bond network
        """
        G = nx.Graph()
        G.add_edges_from(edges)
        
        if node_positions:
            nx.set_node_attributes(G, node_positions, 'position')
        
        return G
    
    def find_triangles(self, G: nx.Graph) -> List[Tuple[int, int, int]]:
        """
        Find all triangles (3-cliques) in the graph.
        
        These represent three water molecules mutually connected by H-bonds.
        """
        triangles = []
        for clique in nx.enumerate_all_cliques(G):
            if len(clique) == 3:
                triangles.append(tuple(sorted(clique)))
            elif len(clique) > 3:
                # For larger cliques, we get triangles as subsets
                continue
        return list(set(triangles))
    
    def find_higher_cliques(self, G: nx.Graph, max_size: int = 4) -> Dict[int, List[Tuple]]:
        """
        Find cliques of various sizes in the graph.
        
        Returns dict mapping clique size to list of cliques.
        """
        cliques_by_size = {k: [] for k in range(3, max_size + 1)}
        
        for clique in nx.enumerate_all_cliques(G):
            size = len(clique)
            if 3 <= size <= max_size:
                cliques_by_size[size].append(tuple(sorted(clique)))
        
        return cliques_by_size
    
    def build_simplicial_complex(
        self,
        edges: List[Tuple[int, int]],
        node_positions: Optional[Dict[int, np.ndarray]] = None,
        node_attributes: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> "tnx.SimplicialComplex":
        """
        Build a SimplicialComplex from H-bond edges.
        
        Parameters
        ----------
        edges : List[Tuple[int, int]]
            Undirected edges representing H-bonds
        node_positions : dict, optional
            Mapping from node index to 3D position
        node_attributes : dict, optional
            Additional attributes for each node
            
        Returns
        -------
        tnx.SimplicialComplex
            Simplicial complex representing the H-bond network
        """
        # Build NetworkX graph first
        G = self.build_networkx_graph(edges, node_positions)
        
        # Create simplicial complex
        sc = tnx.SimplicialComplex()
        
        # Add nodes (0-simplices)
        for node in G.nodes():
            attrs = {}
            if node_positions and node in node_positions:
                attrs['position'] = node_positions[node]
            if node_attributes and node in node_attributes:
                attrs.update(node_attributes[node])
            sc.add_simplex([node], **attrs)
        
        # Add edges (1-simplices)
        for edge in G.edges():
            sc.add_simplex(list(edge))
        
        # Add triangles (2-simplices)
        if self.include_triangles:
            triangles = self.find_triangles(G)
            for triangle in triangles:
                sc.add_simplex(list(triangle))
        
        # Add higher-order simplices if requested
        if self.max_simplex_dim > 2:
            cliques_by_size = self.find_higher_cliques(G, self.max_simplex_dim + 1)
            for size in range(3, self.max_simplex_dim + 2):
                if size in cliques_by_size:
                    for clique in cliques_by_size[size]:
                        try:
                            sc.add_simplex(list(clique))
                        except Exception:
                            pass  # Skip if simplex cannot be added
        
        return sc
    
    def build_from_frame(
        self,
        frame: Frame,
        hbonds: List[HBond],
        directed: bool = False
    ) -> "tnx.SimplicialComplex":
        """
        Build simplicial complex directly from a frame and detected H-bonds.
        
        Parameters
        ----------
        frame : Frame
            Trajectory frame
        hbonds : List[HBond]
            Detected hydrogen bonds
        directed : bool
            If True, use directed edges; if False, use undirected
            
        Returns
        -------
        tnx.SimplicialComplex
            Simplicial complex of the H-bond network
        """
        # Identify all water molecules from the frame for node representation
        detector = HBondDetector()
        water_mols = detector.identify_water_molecules(frame)
        o_indices = [w.o_idx for w in water_mols]
        
        if not o_indices:
            # Fallback to oxygens involved in H-bonds if no water molecules identified
            o_indices = set()
            for hb in hbonds:
                o_indices.add(hb.donor_o_idx)
                o_indices.add(hb.acceptor_o_idx)
            o_indices = list(o_indices)
        
        # Get node positions
        node_positions = {
            idx: frame.positions[idx] for idx in o_indices
        }
        
        # Get edges
        if directed:
            edges = [(hb.donor_o_idx, hb.acceptor_o_idx) for hb in hbonds]
        else:
            edges = list(set(
                tuple(sorted([hb.donor_o_idx, hb.acceptor_o_idx])) 
                for hb in hbonds
            ))
        
        return self.build_simplicial_complex(edges, node_positions)
    
    def build_cell_complex(
        self,
        edges: List[Tuple[int, int]],
        node_positions: Optional[Dict[int, np.ndarray]] = None
    ) -> "tnx.CellComplex":
        """
        Build a CellComplex from H-bond edges.
        
        Cell complexes allow non-triangular cycles, which may be useful
        for detecting ring structures in the H-bond network.
        """
        G = self.build_networkx_graph(edges, node_positions)
        
        cc = tnx.CellComplex()
        
        # Add nodes
        for node in G.nodes():
            cc.add_node(node)
        
        # Add edges
        for edge in G.edges():
            cc.add_edge(*edge)
        
        # Find cycles of length >= 3 up to specified size
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) >= 3:
                try:
                    cc.add_cell(cycle, rank=2)
                except Exception:
                    pass
        
        return cc
