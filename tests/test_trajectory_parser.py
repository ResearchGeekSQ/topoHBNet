"""
Tests for the trajectory parser module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from hbond_topology.io.trajectory_parser import TrajectoryParser, Frame


class TestFrame:
    """Tests for the Frame dataclass."""
    
    def test_frame_creation(self):
        """Test creating a Frame object."""
        frame = Frame(
            timestep=100,
            n_atoms=3,
            box_bounds=np.array([[0, 10], [0, 10], [0, 10]]),
            symbols=np.array(['O', 'H', 'H']),
            atom_ids=np.array([1, 2, 3]),
            atom_types=np.array([1, 2, 2]),
            positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        )
        
        assert frame.timestep == 100
        assert frame.n_atoms == 3
        assert len(frame.atom_ids) == 3
    
    def test_box_lengths(self):
        """Test box_lengths property."""
        frame = Frame(
            timestep=0,
            n_atoms=1,
            box_bounds=np.array([[0, 10], [0, 20], [0, 30]]),
            symbols=np.array(['O']),
            atom_ids=np.array([1]),
            atom_types=np.array([1]),
            positions=np.array([[0, 0, 0]])
        )
        
        np.testing.assert_array_equal(frame.box_lengths, [10, 20, 30])
    
    def test_get_atoms_by_type(self):
        """Test filtering atoms by type."""
        frame = Frame(
            timestep=0,
            n_atoms=5,
            box_bounds=np.array([[0, 10], [0, 10], [0, 10]]),
            symbols=np.array(['O', 'H', 'H', 'O', 'He']),
            atom_ids=np.array([1, 2, 3, 4, 5]),
            atom_types=np.array([1, 2, 2, 1, 3]),
            positions=np.zeros((5, 3))
        )
        
        o_atoms = frame.get_atoms_by_type(1)
        h_atoms = frame.get_atoms_by_type(2)
        
        assert len(o_atoms) == 2
        assert len(h_atoms) == 2
        np.testing.assert_array_equal(o_atoms, [0, 3])
        np.testing.assert_array_equal(h_atoms, [1, 2])


class TestTrajectoryParser:
    """Tests for the TrajectoryParser class."""
    
    @pytest.fixture
    def sample_trajectory_file(self):
        """Create a temporary LAMMPS trajectory file for testing."""
        content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 2 0.96 0.0 0.0
3 2 -0.24 0.93 0.0
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 0.1 0.1 0.1
2 2 1.06 0.1 0.1
3 2 -0.14 1.03 0.1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dump', 
                                          delete=False) as f:
            f.write(content)
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        os.unlink(filepath)
    
    def test_parse_trajectory(self, sample_trajectory_file):
        """Test parsing a LAMMPS trajectory file."""
        parser = TrajectoryParser(sample_trajectory_file)
        frames = parser.parse()
        
        assert len(frames) == 2
        assert frames[0].timestep == 0
        assert frames[1].timestep == 100
    
    def test_parser_len(self, sample_trajectory_file):
        """Test __len__ method."""
        parser = TrajectoryParser(sample_trajectory_file)
        assert len(parser) == 2
    
    def test_parser_getitem(self, sample_trajectory_file):
        """Test __getitem__ method."""
        parser = TrajectoryParser(sample_trajectory_file)
        frame = parser[0]
        
        assert frame.timestep == 0
        assert frame.n_atoms == 3
    
    def test_parser_iteration(self, sample_trajectory_file):
        """Test iterating over parser."""
        parser = TrajectoryParser(sample_trajectory_file)
        timesteps = [frame.timestep for frame in parser]
        
        assert timesteps == [0, 100]
    
    def test_atom_type_map(self, sample_trajectory_file):
        """Test atom type mapping."""
        parser = TrajectoryParser(
            sample_trajectory_file,
            element_to_type={'O': 1, 'H': 2, 'Si': 3}
        )
        
        assert parser.get_element_symbol(1) == 'O'
        assert parser.get_element_symbol(2) == 'H'
        assert parser.get_element_symbol(3) == 'Si'
        assert parser.get_element_symbol(99) == 'X'  # Unknown type
