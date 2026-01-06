"""
Tests for the GNN-Enhanced TNN module (experimental).
"""

import pytest
import numpy as np

# Skip all tests if dependencies are not installed
torch = pytest.importorskip("torch")
toponetx = pytest.importorskip("toponetx")

from hbond_topology.learning.gnn_enhanced_tnn import (
    GNNLayer,
    GNNEncoder,
    GNNEnhancedTNN,
    prepare_gnn_tnn_data
)


class TestGNNLayer:
    """Tests for GNN layers."""
    
    def test_gnn_layer_forward(self):
        """Test single GNN layer forward pass."""
        layer = GNNLayer(in_channels=4, out_channels=8)
        
        x = torch.randn(5, 4)
        adj = torch.randn(5, 5)
        
        out = layer(x, adj)
        
        assert out.shape == (5, 8)
    
    def test_gnn_encoder(self):
        """Test multi-layer GNN encoder."""
        encoder = GNNEncoder(
            in_channels=4, 
            hidden_channels=8, 
            out_channels=16,
            n_layers=3
        )
        
        x = torch.randn(5, 4)
        adj = torch.randn(5, 5)
        
        out = encoder(x, adj)
        
        assert out.shape == (5, 16)


class TestGNNEnhancedTNN:
    """Tests for GNN-Enhanced TNN model."""
    
    @pytest.fixture
    def sample_complex(self):
        """Create sample simplicial complex."""
        import toponetx as tnx
        
        sc = tnx.SimplicialComplex()
        sc.add_simplex([0, 1])
        sc.add_simplex([1, 2])
        sc.add_simplex([0, 2])
        sc.add_simplex([2, 3])
        sc.add_simplex([0, 1, 2])
        
        return sc
    
    def test_model_creation_parallel(self):
        """Test creating model with parallel fusion."""
        model = GNNEnhancedTNN(
            node_in_channels=1,
            edge_in_channels=1,
            hidden_channels=16,
            out_channels=8,
            fusion='parallel'
        )
        
        assert model.fusion == 'parallel'
    
    def test_model_creation_residual(self):
        """Test creating model with residual fusion."""
        model = GNNEnhancedTNN(
            node_in_channels=1,
            edge_in_channels=1,
            hidden_channels=16,
            out_channels=8,
            fusion='residual'
        )
        
        assert model.fusion == 'residual'
    
    def test_prepare_data(self, sample_complex):
        """Test data preparation."""
        data = prepare_gnn_tnn_data(sample_complex)
        
        assert 'node_features' in data
        assert 'edge_features' in data
        assert 'adj' in data
        assert data['n_nodes'] == 4
        assert data['n_edges'] == 4
    
    def test_forward_pass(self, sample_complex):
        """Test forward pass through model."""
        model = GNNEnhancedTNN(
            node_in_channels=1,
            edge_in_channels=1,
            hidden_channels=16,
            out_channels=8,
            fusion='parallel'
        )
        
        data = prepare_gnn_tnn_data(sample_complex)
        
        node_emb, edge_emb, pred = model(
            data['node_features'],
            data['edge_features'],
            data['adj'],
            data.get('laplacian_up'),
            data.get('laplacian_down')
        )
        
        assert node_emb.shape[0] == 4  # n_nodes
        assert edge_emb.shape[0] == 4  # n_edges
        assert pred.shape == (1, 1)
    
    def test_all_fusion_strategies(self, sample_complex):
        """Test all fusion strategies work."""
        for fusion in ['parallel', 'hierarchical', 'residual']:
            model = GNNEnhancedTNN(
                node_in_channels=1,
                edge_in_channels=1,
                hidden_channels=8,
                out_channels=4,
                fusion=fusion
            )
            
            data = prepare_gnn_tnn_data(sample_complex)
            
            # Should not raise
            _, _, pred = model(
                data['node_features'],
                data['edge_features'],
                data['adj'],
                data.get('laplacian_up'),
                data.get('laplacian_down')
            )
            
            assert pred.shape == (1, 1)
