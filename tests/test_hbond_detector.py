"""
Tests for the hydrogen bond detector module.
"""

import pytest
import numpy as np

from hbond_topology.io.trajectory_parser import Frame
from hbond_topology.detection.hbond_detector import (
    HBondDetector, 
    HBond, 
    WaterMolecule
)


class TestHBondDetector:
    """Tests for the HBondDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a default HBondDetector."""
        return HBondDetector()
    
    @pytest.fixture
    def water_frame(self):
        """Create a frame with two water molecules forming an H-bond."""
        # Water 1: O at origin, H atoms nearby
        # Water 2: O at ~3.0 Angstrom, oriented to accept H-bond
        positions = np.array([
            [0.0, 0.0, 0.0],      # O1 (index 0)
            [0.96, 0.0, 0.0],     # H1a (index 1)
            [-0.24, 0.93, 0.0],   # H1b (index 2)
            [2.8, 0.0, 0.0],      # O2 (index 3) - acceptor
            [3.76, 0.0, 0.0],     # H2a (index 4)
            [2.56, 0.93, 0.0],    # H2b (index 5)
        ])
        
        return Frame(
            timestep=0,
            n_atoms=6,
            box_bounds=np.array([[0, 20], [0, 20], [0, 20]]),
            atom_ids=np.arange(1, 7),
            atom_types=np.array([1, 2, 2, 1, 2, 2]),
            positions=positions
        )
    
    def test_minimum_image_distance(self, detector):
        """Test minimum image distance calculation."""
        box_lengths = np.array([10.0, 10.0, 10.0])
        
        # Normal distance
        pos1 = np.array([1.0, 1.0, 1.0])
        pos2 = np.array([2.0, 2.0, 2.0])
        dist, delta = detector.minimum_image_distance(pos1, pos2, box_lengths)
        
        assert np.isclose(dist, np.sqrt(3))
        
        # Distance across PBC
        pos1 = np.array([0.5, 0.5, 0.5])
        pos2 = np.array([9.5, 9.5, 9.5])
        dist, delta = detector.minimum_image_distance(pos1, pos2, box_lengths)
        
        assert np.isclose(dist, np.sqrt(3))  # Should be ~1.73, not ~15.6
    
    def test_identify_water_molecules(self, detector, water_frame):
        """Test water molecule identification."""
        waters = detector.identify_water_molecules(water_frame)
        
        assert len(waters) == 2
        assert all(isinstance(w, WaterMolecule) for w in waters)
        
        # Check that O atoms are correctly identified
        o_indices = {w.o_idx for w in waters}
        assert o_indices == {0, 3}
    
    def test_detect_hbonds(self, detector, water_frame):
        """Test hydrogen bond detection."""
        hbonds = detector.detect_hbonds(water_frame)
        
        # Should detect at least one H-bond
        assert len(hbonds) >= 1
        assert all(isinstance(hb, HBond) for hb in hbonds)
        
        # Check H-bond parameters are within expected ranges
        for hb in hbonds:
            assert hb.distance_da < 3.5
            assert hb.distance_ha < 2.5
            assert hb.angle_dha > 120
    
    def test_get_undirected_edges(self, detector, water_frame):
        """Test getting undirected edges from H-bonds."""
        hbonds = detector.detect_hbonds(water_frame)
        edges = detector.get_undirected_edges(hbonds)
        
        # Edges should be tuples with smaller index first
        for edge in edges:
            assert edge[0] < edge[1]
    
    def test_custom_parameters(self):
        """Test detector with custom H-bond criteria."""
        detector = HBondDetector(
            r_da_max=3.0,
            r_ha_max=2.0,
            angle_min=150.0
        )
        
        assert detector.r_da_max == 3.0
        assert detector.r_ha_max == 2.0
        assert detector.angle_min == 150.0
    
    def test_no_hbonds_when_too_far(self, detector):
        """Test that no H-bonds are detected when molecules are far apart."""
        # Two water molecules 10 Angstrom apart
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [10.0, 0.0, 0.0],
            [10.96, 0.0, 0.0],
            [9.76, 0.93, 0.0],
        ])
        
        frame = Frame(
            timestep=0,
            n_atoms=6,
            box_bounds=np.array([[0, 50], [0, 50], [0, 50]]),
            atom_ids=np.arange(1, 7),
            atom_types=np.array([1, 2, 2, 1, 2, 2]),
            positions=positions
        )
        
        hbonds = detector.detect_hbonds(frame)
        assert len(hbonds) == 0


class TestHBond:
    """Tests for the HBond dataclass."""
    
    def test_hbond_creation(self):
        """Test creating an HBond object."""
        hbond = HBond(
            donor_o_idx=0,
            donor_h_idx=1,
            acceptor_o_idx=3,
            distance_da=2.8,
            distance_ha=1.9,
            angle_dha=165.0
        )
        
        assert hbond.donor_o_idx == 0
        assert hbond.acceptor_o_idx == 3
        assert hbond.distance_da == 2.8
