"""
Tests for the learning module (TNN model).
"""

import pytest
import numpy as np

# Skip all tests if dependencies are not installed
torch = pytest.importorskip("torch")
toponetx = pytest.importorskip("toponetx")


class TestHBondTNN:
    """Tests for the HBondTNN model."""
    
    @pytest.fixture
    def sample_complex(self):
        """Create a sample simplicial complex for testing."""
        import toponetx as tnx
        
        sc = tnx.SimplicialComplex()
        sc.add_simplex([0, 1])
        sc.add_simplex([1, 2])
        sc.add_simplex([0, 2])
        sc.add_simplex([0, 1, 2])
        
        return sc
    
    def test_prepare_tnn_data(self, sample_complex):
        """Test preparing TNN data from simplicial complex."""
        from hbond_topology.learning.tnn_model import prepare_tnn_data
        
        data = prepare_tnn_data(sample_complex)
        
        assert 'edge_features' in data
        assert 'laplacian_up' in data
        assert 'laplacian_down' in data
        assert 'n_edges' in data
        assert data['n_edges'] == 3
    
    def test_hbond_tnn_creation(self):
        """Test creating HBondTNN model."""
        topomodelx = pytest.importorskip("topomodelx")
        from hbond_topology.learning.tnn_model import HBondTNN
        
        model = HBondTNN(
            in_channels=1,
            hidden_channels=16,
            out_channels=8,
            n_layers=2
        )
        
        assert model is not None
        assert model.in_channels == 1
        assert model.hidden_channels == 16
        assert model.out_channels == 8
    
    def test_hbond_gnn_creation(self):
        """Test creating HBondGNN model (fallback)."""
        from hbond_topology.learning.tnn_model import HBondGNN
        
        model = HBondGNN(
            in_channels=1,
            hidden_channels=16,
            out_channels=8,
            n_layers=2
        )
        
        assert model is not None
    
    def test_hbond_gnn_forward(self):
        """Test HBondGNN forward pass."""
        from hbond_topology.learning.tnn_model import HBondGNN
        
        model = HBondGNN(in_channels=1, hidden_channels=8, out_channels=4)
        
        x = torch.randn(5, 1)
        node_feat, graph_pred = model(x)
        
        assert node_feat.shape == (5, 4)
        assert graph_pred.shape == (1, 1)
