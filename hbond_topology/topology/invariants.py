"""
Topological Invariants Calculator

Computes various topological invariants from simplicial complexes,
including Betti numbers, Hodge Laplacians, and adjacency matrices.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

try:
    import toponetx as tnx
    HAS_TOPONETX = True
except ImportError:
    HAS_TOPONETX = False
    tnx = None


class TopologicalInvariants:
    """
    Compute topological invariants from simplicial complexes.
    
    Provides methods to calculate:
    - Betti numbers (counts of topological holes)
    - Hodge Laplacian matrices
    - Incidence/boundary matrices
    - Adjacency matrices
    - Euler characteristic
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Parameters
        ----------
        tolerance : float
            Numerical tolerance for detecting zero eigenvalues
        """
        if not HAS_TOPONETX:
            raise ImportError(
                "TopoNetX is required for this module. "
                "Install it with: pip install toponetx"
            )
        self.tolerance = tolerance
    
    def compute_betti_numbers(
        self, 
        sc: "tnx.SimplicialComplex"
    ) -> Dict[int, int]:
        """
        Compute Betti numbers of the simplicial complex.
        
        Betti numbers count topological features:
        - β₀: Number of connected components
        - β₁: Number of 1-dimensional holes (loops)
        - β₂: Number of 2-dimensional voids (cavities)
        
        Parameters
        ----------
        sc : tnx.SimplicialComplex
            Input simplicial complex
            
        Returns
        -------
        dict
            Mapping from dimension to Betti number
        """
        betti = {}
        
        for rank in range(sc.dim + 1):
            try:
                L = sc.hodge_laplacian_matrix(rank)
                # Convert to dense if needed
                if hasattr(L, 'toarray'):
                    L_dense = L.toarray()
                else:
                    L_dense = np.array(L)
                
                # Count eigenvalues close to zero
                eigenvalues = np.linalg.eigvalsh(L_dense)
                betti[rank] = np.sum(np.abs(eigenvalues) < self.tolerance)
            except Exception:
                betti[rank] = 0
        
        return betti
    
    def compute_euler_characteristic(
        self, 
        sc: "tnx.SimplicialComplex"
    ) -> int:
        """
        Compute the Euler characteristic χ = Σ (-1)^k * f_k
        
        where f_k is the number of k-simplices.
        """
        shape = sc.shape
        chi = sum((-1)**k * n for k, n in enumerate(shape))
        return chi
    
    def get_hodge_laplacian(
        self, 
        sc: "tnx.SimplicialComplex", 
        rank: int
    ) -> csr_matrix:
        """
        Get the Hodge Laplacian matrix of specified rank.
        
        The Hodge Laplacian L_k = B_{k+1} @ B_{k+1}^T + B_k^T @ B_k
        captures the connectivity of k-simplices.
        """
        return sc.hodge_laplacian_matrix(rank)
    
    def get_up_laplacian(
        self, 
        sc: "tnx.SimplicialComplex", 
        rank: int
    ) -> csr_matrix:
        """Get the up Laplacian matrix L_up = B_{k+1} @ B_{k+1}^T."""
        return sc.up_laplacian_matrix(rank)
    
    def get_down_laplacian(
        self, 
        sc: "tnx.SimplicialComplex", 
        rank: int
    ) -> csr_matrix:
        """Get the down Laplacian matrix L_down = B_k^T @ B_k."""
        return sc.down_laplacian_matrix(rank)
    
    def get_adjacency_matrix(
        self, 
        sc: "tnx.SimplicialComplex", 
        rank: int = 0
    ) -> csr_matrix:
        """
        Get the adjacency matrix for simplices of given rank.
        
        Two k-simplices are adjacent if they share a (k+1)-simplex.
        """
        return sc.adjacency_matrix(rank)
    
    def get_incidence_matrix(
        self, 
        sc: "tnx.SimplicialComplex", 
        rank: int
    ) -> csr_matrix:
        """Get the incidence (boundary) matrix B_k."""
        return sc.incidence_matrix(rank)
    
    def compute_all_invariants(
        self, 
        sc: "tnx.SimplicialComplex"
    ) -> Dict[str, Any]:
        """
        Compute all standard topological invariants.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'shape': Tuple of simplex counts by dimension
            - 'dim': Maximum dimension
            - 'euler_characteristic': Euler characteristic
            - 'betti_numbers': Dict of Betti numbers by dimension
            - 'n_components': Number of connected components (β₀)
            - 'n_loops': Number of loops (β₁)
            - 'n_voids': Number of voids (β₂, if applicable)
        """
        betti = self.compute_betti_numbers(sc)
        
        result = {
            'shape': sc.shape,
            'dim': sc.dim,
            'euler_characteristic': self.compute_euler_characteristic(sc),
            'betti_numbers': betti,
            'n_components': betti.get(0, 0),
            'n_loops': betti.get(1, 0),
        }
        
        if sc.dim >= 2:
            result['n_voids'] = betti.get(2, 0)
        
        return result
    
    def compute_spectral_features(
        self, 
        sc: "tnx.SimplicialComplex",
        rank: int = 0,
        n_eigenvalues: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute spectral features from the Hodge Laplacian.
        
        Parameters
        ----------
        sc : tnx.SimplicialComplex
            Input simplicial complex
        rank : int
            Rank of the Laplacian to analyze
        n_eigenvalues : int
            Number of smallest eigenvalues to compute
            
        Returns
        -------
        dict
            Dictionary containing eigenvalues and spectral gap
        """
        L = self.get_hodge_laplacian(sc, rank)
        
        # Handle small matrices
        if L.shape[0] <= n_eigenvalues:
            if hasattr(L, 'toarray'):
                L_dense = L.toarray()
            else:
                L_dense = np.array(L)
            eigenvalues = np.sort(np.linalg.eigvalsh(L_dense))
        else:
            # Use sparse eigenvalue solver for large matrices
            eigenvalues, _ = eigsh(L.astype(float), k=n_eigenvalues, which='SM')
            eigenvalues = np.sort(eigenvalues)
        
        # Compute spectral gap (difference between first non-zero and zero eigenvalues)
        non_zero_mask = np.abs(eigenvalues) > self.tolerance
        if np.any(non_zero_mask):
            spectral_gap = eigenvalues[non_zero_mask][0]
        else:
            spectral_gap = 0.0
        
        return {
            'eigenvalues': eigenvalues,
            'spectral_gap': spectral_gap,
            'algebraic_connectivity': spectral_gap  # For rank 0, this is the Fiedler value
        }
    
    def compute_persistence_diagram(
        self, 
        sc: "tnx.SimplicialComplex"
    ) -> Optional[np.ndarray]:
        """
        Compute persistence diagram using gudhi if available.
        
        Returns
        -------
        np.ndarray or None
            Persistence diagram as (dimension, birth, death) triples
        """
        try:
            import gudhi
            
            # Convert simplicial complex to gudhi SimplexTree
            st = gudhi.SimplexTree()
            
            for simplex in sc:
                st.insert(list(simplex))
            
            st.compute_persistence()
            persistence = st.persistence()
            
            # Convert to numpy array
            if persistence:
                diagram = np.array([
                    (dim, birth, death if death != float('inf') else -1)
                    for dim, (birth, death) in persistence
                ])
                return diagram
            return None
            
        except ImportError:
            return None
