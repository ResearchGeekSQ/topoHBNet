"""
Tests for the topology module (complex builder and invariants).
"""

import pytest
import numpy as np

# Skip tests if toponetx is not installed
toponetx = pytest.importorskip("toponetx")

from hbond_topology.io.trajectory_parser import Frame
from hbond_topology.detection.hbond_detector import HBond, HBondDetector
from hbond_topology.topology.complex_builder import HBondComplexBuilder
from hbond_topology.topology.invariants import TopologicalInvariants


class TestHBondComplexBuilder:
    """Tests for the HBondComplexBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create a default builder."""
        return HBondComplexBuilder()
    
    @pytest.fixture
    def sample_edges(self):
        """Create sample H-bond edges forming a triangle."""
        return [(0, 1), (1, 2), (0, 2)]
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample node positions."""
        return {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([3.0, 0.0, 0.0]),
            2: np.array([1.5, 2.6, 0.0])
        }
    
    def test_build_networkx_graph(self, builder, sample_edges):
        """Test NetworkX graph construction."""
        G = builder.build_networkx_graph(sample_edges)
        
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3
    
    def test_find_triangles(self, builder, sample_edges):
        """Test triangle detection."""
        G = builder.build_networkx_graph(sample_edges)
        triangles = builder.find_triangles(G)
        
        assert len(triangles) == 1
        assert set(triangles[0]) == {0, 1, 2}
    
    def test_build_simplicial_complex(self, builder, sample_edges, sample_positions):
        """Test simplicial complex construction."""
        sc = builder.build_simplicial_complex(sample_edges, sample_positions)
        
        # Check shape: (nodes, edges, triangles)
        assert sc.shape[0] == 3  # 3 nodes
        assert sc.shape[1] == 3  # 3 edges
        assert sc.shape[2] == 1  # 1 triangle
    
    def test_build_without_triangles(self, sample_edges, sample_positions):
        """Test building complex without triangles."""
        builder = HBondComplexBuilder(include_triangles=False)
        sc = builder.build_simplicial_complex(sample_edges, sample_positions)
        
        assert sc.shape[0] == 3
        assert sc.shape[1] == 3
        assert len(sc.shape) == 2 or sc.shape[2] == 0
    
    def test_empty_complex(self, builder):
        """Test building from empty edge list."""
        sc = builder.build_simplicial_complex([])
        
        # Safe shape check
        n_nodes = sc.shape[0] if len(sc.shape) > 0 else 0
        assert n_nodes == 0


class TestTopologicalInvariants:
    """Tests for the TopologicalInvariants class."""
    
    @pytest.fixture
    def invariants(self):
        """Create TopologicalInvariants calculator."""
        return TopologicalInvariants()
    
    @pytest.fixture
    def triangle_complex(self):
        """Create a simplicial complex with one triangle."""
        builder = HBondComplexBuilder()
        edges = [(0, 1), (1, 2), (0, 2)]
        return builder.build_simplicial_complex(edges)
    
    @pytest.fixture
    def path_complex(self):
        """Create a simplicial complex that is just a path (no loops)."""
        builder = HBondComplexBuilder()
        edges = [(0, 1), (1, 2), (2, 3)]
        return builder.build_simplicial_complex(edges)
    
    def test_compute_betti_numbers_triangle(self, invariants, triangle_complex):
        """Test Betti numbers for a filled triangle."""
        betti = invariants.compute_betti_numbers(triangle_complex)
        
        # Filled triangle: 1 component, 0 loops
        assert betti[0] == 1  # One connected component
        assert betti[1] == 0  # No loops (triangle is filled)
    
    def test_compute_euler_characteristic(self, invariants, triangle_complex):
        """Test Euler characteristic computation."""
        chi = invariants.compute_euler_characteristic(triangle_complex)
        
        # For filled triangle: V - E + F = 3 - 3 + 1 = 1
        assert chi == 1
    
    def test_compute_all_invariants(self, invariants, triangle_complex):
        """Test computing all invariants at once."""
        result = invariants.compute_all_invariants(triangle_complex)
        
        assert 'shape' in result
        assert 'dim' in result
        assert 'euler_characteristic' in result
        assert 'betti_numbers' in result
        assert 'n_components' in result
        assert 'n_loops' in result
    
    def test_get_hodge_laplacian(self, invariants, triangle_complex):
        """Test Hodge Laplacian matrix computation."""
        L0 = invariants.get_hodge_laplacian(triangle_complex, rank=0)
        
        # L0 should be a 3x3 matrix (3 nodes)
        assert L0.shape == (3, 3)
    
    def test_path_has_no_loops(self, invariants, path_complex):
        """Test that a path graph has no loops."""
        betti = invariants.compute_betti_numbers(path_complex)
        
        assert betti[0] == 1  # One connected component
        assert betti[1] == 0  # No loops
