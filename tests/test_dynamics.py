"""
Tests for the dynamics analyzer module.
"""

import pytest
import numpy as np
import tempfile
import os

from hbond_topology.io.trajectory_parser import TrajectoryParser
from hbond_topology.analysis.dynamics import (
    DynamicsAnalyzer,
    FrameAnalysisResult,
    TrajectoryAnalysisResult
)


class TestFrameAnalysisResult:
    """Tests for the FrameAnalysisResult dataclass."""
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = FrameAnalysisResult(
            timestep=100,
            n_water_molecules=10,
            n_hbonds=15,
            n_nodes=10,
            n_edges=15,
            n_triangles=3,
            betti_0=2,
            betti_1=1,
            betti_2=0,
            euler_characteristic=-2,
            mean_hbond_distance=2.8,
            mean_hbond_angle=165.0
        )
        
        d = result.to_dict()
        
        assert d['timestep'] == 100
        assert d['n_hbonds'] == 15
        assert d['betti_0'] == 2


class TestTrajectoryAnalysisResult:
    """Tests for the TrajectoryAnalysisResult class."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results."""
        frames = [
            FrameAnalysisResult(
                timestep=i * 100,
                n_water_molecules=10,
                n_hbonds=15 + i,
                n_nodes=10,
                n_edges=15 + i,
                n_triangles=3,
                betti_0=2,
                betti_1=1,
                betti_2=0,
                euler_characteristic=-2,
                mean_hbond_distance=2.8,
                mean_hbond_angle=165.0
            )
            for i in range(5)
        ]
        return TrajectoryAnalysisResult(frame_results=frames)
    
    def test_n_frames(self, sample_results):
        """Test frame count."""
        assert sample_results.n_frames == 5
    
    def test_timesteps_array(self, sample_results):
        """Test getting timesteps as array."""
        timesteps = sample_results.timesteps
        
        assert len(timesteps) == 5
        np.testing.assert_array_equal(timesteps, [0, 100, 200, 300, 400])
    
    def test_n_hbonds_array(self, sample_results):
        """Test getting H-bond counts as array."""
        n_hbonds = sample_results.n_hbonds
        
        assert len(n_hbonds) == 5
        np.testing.assert_array_equal(n_hbonds, [15, 16, 17, 18, 19])
    
    def test_to_numpy(self, sample_results):
        """Test converting to numpy dictionary."""
        arrays = sample_results.to_numpy()
        
        assert 'timesteps' in arrays
        assert 'n_hbonds' in arrays
        assert 'betti_0' in arrays
        assert 'betti_1' in arrays
    
    def test_statistics(self, sample_results):
        """Test computing statistics."""
        stats = sample_results.statistics()
        
        assert 'n_hbonds' in stats
        assert 'mean' in stats['n_hbonds']
        assert 'std' in stats['n_hbonds']
        assert 'min' in stats['n_hbonds']
        assert 'max' in stats['n_hbonds']
        
        assert stats['n_hbonds']['mean'] == 17.0
        assert stats['n_hbonds']['min'] == 15.0
        assert stats['n_hbonds']['max'] == 19.0
    
    def test_json_roundtrip(self, sample_results):
        """Test saving and loading JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                          delete=False) as f:
            filepath = f.name
        
        try:
            sample_results.to_json(filepath)
            loaded = TrajectoryAnalysisResult.from_json(filepath)
            
            assert loaded.n_frames == sample_results.n_frames
            np.testing.assert_array_equal(loaded.timesteps, sample_results.timesteps)
        finally:
            os.unlink(filepath)


class TestDynamicsAnalyzer:
    """Tests for the DynamicsAnalyzer class."""
    
    def test_analyzer_creation(self):
        """Test creating an analyzer."""
        analyzer = DynamicsAnalyzer(verbose=False)
        
        assert analyzer.detector is not None
        assert analyzer.builder is not None
        assert analyzer.invariants is not None
    
    def test_compute_autocorrelation(self):
        """Test autocorrelation computation."""
        frames = [
            FrameAnalysisResult(
                timestep=i,
                n_water_molecules=10,
                n_hbonds=10 + np.sin(i * 0.1) * 5,  # Periodic signal
                n_nodes=10,
                n_edges=10,
                n_triangles=0,
                betti_0=1,
                betti_1=0,
                betti_2=0,
                euler_characteristic=0,
                mean_hbond_distance=2.8,
                mean_hbond_angle=165.0
            )
            for i in range(100)
        ]
        results = TrajectoryAnalysisResult(frame_results=frames)
        
        analyzer = DynamicsAnalyzer(verbose=False)
        autocorr = analyzer.compute_autocorrelation(results, 'n_hbonds', max_lag=20)
        
        assert len(autocorr) == 20
        assert autocorr[0] == pytest.approx(1.0)  # Autocorrelation at lag 0 is 1
