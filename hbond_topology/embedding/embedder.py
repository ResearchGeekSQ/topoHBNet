"""
Hydrogen Bond Network Embedder

Generates low-dimensional embeddings of H-bond networks
using TopoEmbedX algorithms like Cell2Vec and HOPE.
"""

import numpy as np
from typing import Optional, Dict, Any, Literal

try:
    import toponetx as tnx
    HAS_TOPONETX = True
except ImportError:
    HAS_TOPONETX = False
    tnx = None

try:
    import topoembedx as tex
    HAS_TOPOEMBEDX = True
except ImportError:
    HAS_TOPOEMBEDX = False
    tex = None


class HBondEmbedder:
    """
    Generate embeddings for hydrogen bond network simplicial complexes.
    
    Supports multiple embedding algorithms from TopoEmbedX:
    - Cell2Vec: Random walk-based embedding
    - HOPE: Higher-Order Preserving Embedding
    - DeepCell: Deep learning-based embedding
    - HOGLEE: Higher Order Geometric Laplacian EigenMaps
    
    Parameters
    ----------
    method : str
        Embedding method: 'cell2vec', 'hope', 'deepcell', 'hoglee'
    dimensions : int
        Embedding dimension (default: 32)
    rank : int
        Rank of simplices to embed (0=nodes, 1=edges, 2=triangles)
    """
    
    SUPPORTED_METHODS = ['cell2vec', 'hope', 'deepcell', 'hoglee']
    
    def __init__(
        self,
        method: Literal['cell2vec', 'hope', 'deepcell', 'hoglee'] = 'cell2vec',
        dimensions: int = 32,
        rank: int = 0,
        **kwargs
    ):
        if not HAS_TOPOEMBEDX:
            raise ImportError(
                "TopoEmbedX is required for embedding. Install with:\n"
                "pip install topoembedx\n"
                "pip install 'pygsp @ git+https://github.com/epfl-lts2/pygsp'\n"
                "pip install 'karateclub @ git+https://github.com/benedekrozemberczki/karateclub'"
            )
        
        self.method = method.lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        
        self.dimensions = dimensions
        self.rank = rank
        self.kwargs = kwargs
        self._model = None
        self._embedding = None
    
    def _create_model(self):
        """Create the embedding model based on method."""
        if self.method == 'cell2vec':
            self._model = tex.Cell2Vec(
                dimensions=self.dimensions,
                walk_length=self.kwargs.get('walk_length', 10),
                num_walks=self.kwargs.get('num_walks', 80),
                workers=self.kwargs.get('workers', 1)
            )
        elif self.method == 'hope':
            self._model = tex.HOPE(
                dimensions=self.dimensions
            )
        elif self.method == 'deepcell':
            self._model = tex.DeepCell(
                dimensions=self.dimensions,
                walk_length=self.kwargs.get('walk_length', 10),
                num_walks=self.kwargs.get('num_walks', 80)
            )
        elif self.method == 'hoglee':
            self._model = tex.HOGLEE(
                dimensions=self.dimensions
            )
    
    def fit(
        self, 
        sc: "tnx.SimplicialComplex",
        neighborhood_type: str = "adj"
    ) -> "HBondEmbedder":
        """
        Fit the embedding model to a simplicial complex.
        
        Parameters
        ----------
        sc : tnx.SimplicialComplex
            Input simplicial complex
        neighborhood_type : str
            Type of neighborhood: 'adj' (adjacency) or 'coadj' (coadjacency)
            
        Returns
        -------
        self
        """
        self._create_model()
        
        # Configure neighborhood dimension
        # For nodes (rank 0), we use edges (rank 1) as via_rank
        # For other ranks, we use rank-1
        if self.rank == 0:
            v_rank = 1
        else:
            v_rank = self.rank - 1
            
        neighborhood_dim = {"rank": self.rank, "via_rank": v_rank}
        
        try:
            self._model.fit(
                sc, 
                neighborhood_type=neighborhood_type,
                neighborhood_dim=neighborhood_dim
            )
            self._embedding = self._model.get_embedding()
        except Exception as e:
            # Fallback for very small complexes: try basic adjacency if default fails
            try:
                self._model.fit(sc, neighborhood_type=neighborhood_type)
                self._embedding = self._model.get_embedding()
            except Exception:
                print(f"Warning: Embedding failed - {e}")
                self._embedding = None
        
        return self
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Get the computed embedding vectors.
        
        Returns
        -------
        np.ndarray or None
            Embedding matrix of shape (n_simplices, dimensions)
        """
        return self._embedding
    
    def fit_transform(
        self, 
        sc: "tnx.SimplicialComplex",
        neighborhood_type: str = "adj"
    ) -> Optional[np.ndarray]:
        """
        Fit and return embedding in one step.
        
        Parameters
        ----------
        sc : tnx.SimplicialComplex
            Input simplicial complex
        neighborhood_type : str
            Type of neighborhood
            
        Returns
        -------
        np.ndarray or None
            Embedding matrix
        """
        self.fit(sc, neighborhood_type)
        return self.get_embedding()
    
    def compute_similarity(
        self, 
        idx1: int, 
        idx2: int,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embedded simplices.
        
        Parameters
        ----------
        idx1, idx2 : int
            Indices of simplices in the embedding
        metric : str
            Similarity metric: 'cosine', 'euclidean', 'dot'
            
        Returns
        -------
        float
            Similarity score
        """
        if self._embedding is None:
            raise ValueError("Must fit model before computing similarity")
        
        v1 = self._embedding[idx1]
        v2 = self._embedding[idx2]
        
        if metric == 'cosine':
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
        elif metric == 'euclidean':
            return -np.linalg.norm(v1 - v2)  # Negative for similarity
        elif metric == 'dot':
            return np.dot(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def pairwise_similarities(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute pairwise similarity matrix for all embeddings.
        
        Returns
        -------
        np.ndarray
            Similarity matrix of shape (n_simplices, n_simplices)
        """
        if self._embedding is None:
            raise ValueError("Must fit model before computing similarities")
        
        n = len(self._embedding)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity(i, j, metric)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix


def embed_trajectory_frames(
    complexes: list,
    method: str = 'cell2vec',
    dimensions: int = 32,
    aggregate: str = 'mean'
) -> np.ndarray:
    """
    Embed multiple frames and aggregate embeddings.
    
    Parameters
    ----------
    complexes : list
        List of SimplicialComplex objects (one per frame)
    method : str
        Embedding method
    dimensions : int
        Embedding dimension
    aggregate : str
        How to aggregate node embeddings: 'mean', 'sum', 'max'
        
    Returns
    -------
    np.ndarray
        Frame-level embeddings of shape (n_frames, dimensions)
    """
    embedder = HBondEmbedder(method=method, dimensions=dimensions)
    frame_embeddings = []
    
    for sc in complexes:
        embedding = embedder.fit_transform(sc)
        
        if embedding is not None and len(embedding) > 0:
            if aggregate == 'mean':
                frame_emb = np.mean(embedding, axis=0)
            elif aggregate == 'sum':
                frame_emb = np.sum(embedding, axis=0)
            elif aggregate == 'max':
                frame_emb = np.max(embedding, axis=0)
            else:
                frame_emb = np.mean(embedding, axis=0)
        else:
            frame_emb = np.zeros(dimensions)
        
        frame_embeddings.append(frame_emb)
    
    return np.array(frame_embeddings)
