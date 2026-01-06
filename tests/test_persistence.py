"""
Tests for the persistence homology module.
"""

import pytest
import numpy as np

# Skip all tests if gudhi is not installed
gudhi = pytest.importorskip("gudhi")

from hbond_topology.topology.persistence import PersistenceAnalyzer


class TestPersistenceAnalyzer:
    """Tests for the PersistenceAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a default analyzer."""
        return PersistenceAnalyzer(max_edge_length=5.0, max_dimension=2)
    
    @pytest.fixture
    def triangle_points(self):
        """Create points forming an equilateral triangle."""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0]
        ])
    
    @pytest.fixture
    def circle_points(self):
        """Create points arranged in a circle (should have H1 feature)."""
        n = 20
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([
            np.cos(angles),
            np.sin(angles),
            np.zeros(n)
        ])
    
    def test_compute_rips_persistence(self, analyzer, triangle_points):
        """Test Rips persistence computation."""
        persistence = analyzer.compute_rips_persistence(triangle_points)
        
        assert len(persistence) > 0
        assert all(isinstance(p, tuple) for p in persistence)
    
    def test_compute_alpha_persistence(self, analyzer, triangle_points):
        """Test Alpha persistence computation."""
        persistence = analyzer.compute_alpha_persistence(triangle_points)
        
        assert len(persistence) > 0
    
    def test_persistence_to_array(self, analyzer, triangle_points):
        """Test converting persistence to numpy array."""
        persistence = analyzer.compute_rips_persistence(triangle_points)
        arr = analyzer.persistence_to_array(persistence)
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        if len(arr) > 0:
            assert arr.shape[1] == 3  # [dimension, birth, death]
    
    def test_persistence_filter_by_dimension(self, analyzer, triangle_points):
        """Test filtering persistence by dimension."""
        persistence = analyzer.compute_rips_persistence(triangle_points)
        
        arr_h0 = analyzer.persistence_to_array(persistence, dimension=0)
        arr_h1 = analyzer.persistence_to_array(persistence, dimension=1)
        
        # All entries should have the specified dimension
        if len(arr_h0) > 0:
            assert all(arr_h0[:, 0] == 0)
        if len(arr_h1) > 0:
            assert all(arr_h1[:, 0] == 1)
    
    def test_compute_betti_curve(self, analyzer, triangle_points):
        """Test Betti curve computation."""
        persistence = analyzer.compute_rips_persistence(triangle_points)
        scales, betti = analyzer.compute_betti_curve(persistence, dimension=0)
        
        assert len(scales) == len(betti)
        assert len(scales) == 100  # Default n_points
        assert betti[0] >= 0  # Betti numbers are non-negative
    
    def test_compute_persistence_landscape(self, analyzer, circle_points):
        """Test persistence landscape computation."""
        persistence = analyzer.compute_rips_persistence(circle_points)
        landscape = analyzer.compute_persistence_landscape(
            persistence, 
            dimension=1, 
            n_layers=3, 
            n_points=50
        )
        
        assert landscape.shape == (3, 50)
        assert np.all(landscape >= 0)  # Landscape values are non-negative
    
    def test_bottleneck_distance_same_diagram(self, analyzer, triangle_points):
        """Test bottleneck distance between identical diagrams."""
        persistence = analyzer.compute_rips_persistence(triangle_points)
        distance = analyzer.bottleneck_distance(persistence, persistence, dimension=0)
        
        assert distance == pytest.approx(0.0)
    
    def test_bottleneck_distance_different_diagrams(self, analyzer, triangle_points, circle_points):
        """Test bottleneck distance between different diagrams."""
        pers1 = analyzer.compute_rips_persistence(triangle_points)
        pers2 = analyzer.compute_rips_persistence(circle_points)
        
        distance = analyzer.bottleneck_distance(pers1, pers2, dimension=0)
        
        # Distance should be non-negative
        assert distance >= 0
    
    def test_analyze_frame(self, analyzer, triangle_points):
        """Test full frame analysis."""
        result = analyzer.analyze_frame(triangle_points, method='rips')
        
        assert 'persistence' in result
        assert 'n_features' in result
        assert 'betti_curve_0' in result
        assert 'betti_curve_1' in result
    
    def test_circle_has_h1_feature(self, analyzer, circle_points):
        """Test that a circle has H1 (loop) features."""
        persistence = analyzer.compute_rips_persistence(circle_points)
        h1_features = analyzer.persistence_to_array(persistence, dimension=1)
        
        # Circle should have at least one H1 feature
        assert len(h1_features) >= 1
    
    def test_empty_points(self, analyzer):
        """Test behavior with empty point set."""
        empty = np.array([]).reshape(0, 3)
        
        # Should handle empty input gracefully
        persistence = analyzer.compute_rips_persistence(empty)
        assert persistence == []
