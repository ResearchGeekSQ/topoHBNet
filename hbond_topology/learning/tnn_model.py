"""
Topological Neural Network Model

Implements topological deep learning models for hydrogen bond networks
using TopoModelX (Simplicial Attention Network).
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import toponetx as tnx
    HAS_TOPONETX = True
except ImportError:
    HAS_TOPONETX = False
    tnx = None

try:
    from topomodelx.nn.simplicial.san import SAN
    from topomodelx.utils.sparse import from_sparse
    HAS_TOPOMODELX = True
except ImportError:
    HAS_TOPOMODELX = False
    SAN = None


def check_dependencies():
    """Check if all required dependencies are available."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    if not HAS_TOPONETX:
        raise ImportError("TopoNetX is required. Install with: pip install toponetx")
    if not HAS_TOPOMODELX:
        raise ImportError("TopoModelX is required. Install with: pip install topomodelx")


class HBondTNN(nn.Module):
    """
    Topological Neural Network for hydrogen bond network analysis.
    
    Uses Simplicial Attention Network (SAN) to learn representations
    of H-bond networks on simplicial complexes.
    
    Parameters
    ----------
    in_channels : int
        Number of input features per node
    hidden_channels : int
        Hidden layer dimension
    out_channels : int
        Output dimension
    n_layers : int
        Number of SAN layers
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 16,
        n_layers: int = 2
    ):
        check_dependencies()
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # SAN layers
        self.san = SAN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # For graph-level prediction
        self.graph_proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        laplacian_up: torch.Tensor,
        laplacian_down: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (n_edges, in_channels)
        laplacian_up : torch.Tensor
            Up Laplacian matrix
        laplacian_down : torch.Tensor
            Down Laplacian matrix
            
        Returns
        -------
        edge_features : torch.Tensor
            Learned edge representations
        graph_pred : torch.Tensor
            Graph-level prediction
        """
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)
        
        # SAN message passing
        x = self.san(x, laplacian_up, laplacian_down)
        
        # Output projection
        edge_features = self.output_proj(x)
        
        # Graph-level readout (mean pooling)
        graph_emb = edge_features.mean(dim=0, keepdim=True)
        graph_pred = self.graph_proj(graph_emb)
        
        return edge_features, graph_pred


def prepare_tnn_data(
    sc: "tnx.SimplicialComplex",
    node_features: Optional[np.ndarray] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare data for TNN from a simplicial complex.
    
    Parameters
    ----------
    sc : tnx.SimplicialComplex
        Input simplicial complex
    node_features : np.ndarray, optional
        Node features of shape (n_nodes, n_features)
        If None, uses constant features
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'edge_features': Edge feature tensor
        - 'laplacian_up': Up Laplacian
        - 'laplacian_down': Down Laplacian
        - 'n_edges': Number of edges
    """
    check_dependencies()
    
    # Get number of edges
    n_edges = sc.shape[1] if len(sc.shape) > 1 else 0
    
    if n_edges == 0:
        return {
            'edge_features': torch.zeros(0, 1),
            'laplacian_up': torch.zeros(0, 0),
            'laplacian_down': torch.zeros(0, 0),
            'n_edges': 0
        }
    
    # Get Laplacian matrices
    try:
        laplacian_up = sc.up_laplacian_matrix(rank=1)
        laplacian_down = sc.down_laplacian_matrix(rank=1)
    except Exception:
        # Fallback if Laplacians can't be computed
        laplacian_up = np.zeros((n_edges, n_edges))
        laplacian_down = np.zeros((n_edges, n_edges))
    
    # Convert to torch tensors
    laplacian_up = from_sparse(laplacian_up)
    laplacian_down = from_sparse(laplacian_down)
    
    # Create edge features (default: constant 1)
    if node_features is None:
        edge_features = torch.ones(n_edges, 1)
    else:
        # Aggregate node features to edges
        # Simple averaging of endpoint features
        edges = list(sc.skeleton(1))
        edge_feats = []
        for edge in edges:
            nodes = list(edge)
            if len(nodes) == 2:
                feat = (node_features[nodes[0]] + node_features[nodes[1]]) / 2
                edge_feats.append(feat)
        edge_features = torch.tensor(np.array(edge_feats), dtype=torch.float32)
    
    return {
        'edge_features': edge_features,
        'laplacian_up': laplacian_up,
        'laplacian_down': laplacian_down,
        'n_edges': n_edges
    }


def train_tnn(
    model: HBondTNN,
    train_data: List[Tuple[Dict[str, torch.Tensor], float]],
    n_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train the TNN model.
    
    Parameters
    ----------
    model : HBondTNN
        Model to train
    train_data : list
        List of (data_dict, target) tuples
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    verbose : bool
        Print training progress
        
    Returns
    -------
    dict
        Training history with loss values
    """
    check_dependencies()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        model.train()
        
        for data, target in train_data:
            if data['n_edges'] == 0:
                continue
            
            optimizer.zero_grad()
            
            _, pred = model(
                data['edge_features'],
                data['laplacian_up'],
                data['laplacian_down']
            )
            
            target_tensor = torch.tensor([[target]], dtype=torch.float32)
            loss = criterion(pred, target_tensor)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(len(train_data), 1)
        history['loss'].append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return history


class HBondGNN(nn.Module):
    """
    Simple Graph Neural Network for H-bond networks.
    
    A lighter alternative to SAN when TopoModelX is not available.
    Uses standard message passing on the graph structure.
    
    Parameters
    ----------
    in_channels : int
        Input feature dimension
    hidden_channels : int
        Hidden layer dimension
    out_channels : int
        Output dimension
    n_layers : int
        Number of GNN layers
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 16,
        n_layers: int = 2
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Output layer
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
        # Graph prediction
        self.graph_pred = nn.Linear(out_channels, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features
        adj : torch.Tensor, optional
            Adjacency matrix (sparse or dense)
            
        Returns
        -------
        node_features : torch.Tensor
            Learned node representations
        graph_pred : torch.Tensor
            Graph-level prediction
        """
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            
            # Message passing with adjacency if provided
            if adj is not None:
                if adj.is_sparse:
                    x = torch.sparse.mm(adj, x) + x
                else:
                    x = torch.mm(adj, x) + x
        
        node_features = self.out_layer(x)
        
        # Graph-level readout
        graph_emb = node_features.mean(dim=0, keepdim=True)
        graph_pred = self.graph_pred(graph_emb)
        
        return node_features, graph_pred
