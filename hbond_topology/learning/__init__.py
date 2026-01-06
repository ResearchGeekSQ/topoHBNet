"""Learning module for Topological Neural Networks."""

from .tnn_model import HBondTNN, HBondGNN, train_tnn, prepare_tnn_data

# Experimental: GNN-Enhanced TNN
try:
    from .gnn_enhanced_tnn import (
        GNNEnhancedTNN,
        prepare_gnn_tnn_data,
        train_gnn_enhanced_tnn
    )
    __all__ = [
        "HBondTNN", "HBondGNN", "train_tnn", "prepare_tnn_data",
        "GNNEnhancedTNN", "prepare_gnn_tnn_data", "train_gnn_enhanced_tnn"
    ]
except ImportError:
    __all__ = ["HBondTNN", "HBondGNN", "train_tnn", "prepare_tnn_data"]
