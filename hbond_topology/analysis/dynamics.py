"""
Dynamics Analyzer

Analyzes time evolution of hydrogen bond network topology.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..io.trajectory_parser import TrajectoryParser, Frame
from ..detection.hbond_detector import HBondDetector, HBond
from ..topology.complex_builder import HBondComplexBuilder
from ..topology.invariants import TopologicalInvariants


@dataclass
class FrameAnalysisResult:
    """Results from analyzing a single frame."""
    
    timestep: int
    n_water_molecules: int
    n_hbonds: int
    n_nodes: int  # O atoms involved in H-bond network
    n_edges: int  # H-bonds (undirected)
    n_triangles: int
    betti_0: int  # Connected components
    betti_1: int  # Loops
    betti_2: int  # Voids
    euler_characteristic: int
    mean_hbond_distance: float
    mean_hbond_angle: float
    spectral_gap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestep': self.timestep,
            'n_water_molecules': self.n_water_molecules,
            'n_hbonds': self.n_hbonds,
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'n_triangles': self.n_triangles,
            'betti_0': self.betti_0,
            'betti_1': self.betti_1,
            'betti_2': self.betti_2,
            'euler_characteristic': self.euler_characteristic,
            'mean_hbond_distance': self.mean_hbond_distance,
            'mean_hbond_angle': self.mean_hbond_angle,
            'spectral_gap': self.spectral_gap
        }


@dataclass
class TrajectoryAnalysisResult:
    """Results from analyzing entire trajectory."""
    
    frame_results: List[FrameAnalysisResult] = field(default_factory=list)
    
    @property
    def n_frames(self) -> int:
        return len(self.frame_results)
    
    @property
    def timesteps(self) -> np.ndarray:
        return np.array([r.timestep for r in self.frame_results])
    
    @property
    def n_hbonds(self) -> np.ndarray:
        return np.array([r.n_hbonds for r in self.frame_results])
    
    @property
    def betti_0(self) -> np.ndarray:
        return np.array([r.betti_0 for r in self.frame_results])
    
    @property
    def betti_1(self) -> np.ndarray:
        return np.array([r.betti_1 for r in self.frame_results])
    
    @property
    def betti_2(self) -> np.ndarray:
        return np.array([r.betti_2 for r in self.frame_results])
    
    @property
    def n_triangles(self) -> np.ndarray:
        return np.array([r.n_triangles for r in self.frame_results])
    
    @property
    def euler_characteristic(self) -> np.ndarray:
        return np.array([r.euler_characteristic for r in self.frame_results])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_frames': self.n_frames,
            'frames': [r.to_dict() for r in self.frame_results]
        }
    
    def to_json(self, filepath: str | Path) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> "TrajectoryAnalysisResult":
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        frame_results = [
            FrameAnalysisResult(**frame_data)
            for frame_data in data['frames']
        ]
        return cls(frame_results=frame_results)
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary of numpy arrays."""
        return {
            'timesteps': self.timesteps,
            'n_hbonds': self.n_hbonds,
            'betti_0': self.betti_0,
            'betti_1': self.betti_1,
            'betti_2': self.betti_2,
            'n_triangles': self.n_triangles,
            'euler_characteristic': self.euler_characteristic,
            'n_nodes': np.array([r.n_nodes for r in self.frame_results]),
            'n_edges': np.array([r.n_edges for r in self.frame_results]),
            'mean_hbond_distance': np.array([r.mean_hbond_distance for r in self.frame_results]),
            'mean_hbond_angle': np.array([r.mean_hbond_angle for r in self.frame_results]),
        }
    
    def statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all time series."""
        arrays = self.to_numpy()
        stats = {}
        
        for name, arr in arrays.items():
            if name == 'timesteps':
                continue
            stats[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
        
        return stats


class DynamicsAnalyzer:
    """
    Analyze hydrogen bond network dynamics over trajectory.
    
    Parameters
    ----------
    hbond_detector : HBondDetector, optional
        Custom H-bond detector (uses defaults if not provided)
    complex_builder : HBondComplexBuilder, optional
        Custom complex builder
    compute_spectral : bool
        Whether to compute spectral features (slower)
    verbose : bool
        Print progress during analysis
    """
    
    def __init__(
        self,
        hbond_detector: Optional[HBondDetector] = None,
        complex_builder: Optional[HBondComplexBuilder] = None,
        compute_spectral: bool = False,
        verbose: bool = True
    ):
        self.detector = hbond_detector or HBondDetector()
        self.builder = complex_builder or HBondComplexBuilder()
        self.invariants = TopologicalInvariants()
        self.compute_spectral = compute_spectral
        self.verbose = verbose
    
    def analyze_frame(self, frame: Frame) -> FrameAnalysisResult:
        """
        Analyze a single frame.
        
        Parameters
        ----------
        frame : Frame
            Trajectory frame to analyze
            
        Returns
        -------
        FrameAnalysisResult
            Analysis results for the frame
        """
        # Detect H-bonds
        hbonds = self.detector.detect_hbonds(frame)
        n_water = len(self.detector.identify_water_molecules(frame))
        
        # Handle empty H-bond network
        if len(hbonds) == 0:
            return FrameAnalysisResult(
                timestep=frame.timestep,
                n_water_molecules=n_water,
                n_hbonds=0,
                n_nodes=0,
                n_edges=0,
                n_triangles=0,
                betti_0=0,
                betti_1=0,
                betti_2=0,
                euler_characteristic=0,
                mean_hbond_distance=0.0,
                mean_hbond_angle=0.0,
                spectral_gap=None
            )
        
        # Build simplicial complex
        sc = self.builder.build_from_frame(frame, hbonds)
        
        # Compute invariants
        invariants = self.invariants.compute_all_invariants(sc)
        
        # Compute spectral features if requested
        spectral_gap = None
        if self.compute_spectral and sc.shape[0] > 1:
            try:
                spectral = self.invariants.compute_spectral_features(sc, rank=0)
                spectral_gap = spectral.get('spectral_gap')
            except Exception:
                pass
        
        # Compute H-bond statistics
        mean_distance = np.mean([hb.distance_da for hb in hbonds])
        mean_angle = np.mean([hb.angle_dha for hb in hbonds])
        
        return FrameAnalysisResult(
            timestep=frame.timestep,
            n_water_molecules=n_water,
            n_hbonds=len(hbonds),
            n_nodes=invariants['shape'][0],
            n_edges=invariants['shape'][1] if len(invariants['shape']) > 1 else 0,
            n_triangles=invariants['shape'][2] if len(invariants['shape']) > 2 else 0,
            betti_0=invariants['betti_numbers'].get(0, 0),
            betti_1=invariants['betti_numbers'].get(1, 0),
            betti_2=invariants['betti_numbers'].get(2, 0),
            euler_characteristic=invariants['euler_characteristic'],
            mean_hbond_distance=mean_distance,
            mean_hbond_angle=mean_angle,
            spectral_gap=spectral_gap
        )
    
    def analyze_trajectory(
        self, 
        trajectory: TrajectoryParser | str | Path,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1
    ) -> TrajectoryAnalysisResult:
        """
        Analyze entire trajectory.
        
        Parameters
        ----------
        trajectory : TrajectoryParser or str or Path
            Trajectory to analyze
        start : int
            First frame index
        stop : int, optional
            Last frame index (exclusive)
        step : int
            Frame stride
            
        Returns
        -------
        TrajectoryAnalysisResult
            Complete analysis results
        """
        # Load trajectory if needed
        if isinstance(trajectory, (str, Path)):
            trajectory = TrajectoryParser(trajectory)
            trajectory.parse()
        
        # Get frame range
        if stop is None:
            stop = len(trajectory)
        
        frame_indices = range(start, stop, step)
        n_frames = len(frame_indices)
        
        results = []
        for i, idx in enumerate(frame_indices):
            if self.verbose and (i % 10 == 0 or i == n_frames - 1):
                print(f"Analyzing frame {i+1}/{n_frames} (timestep {trajectory[idx].timestep})")
            
            result = self.analyze_frame(trajectory[idx])
            results.append(result)
        
        return TrajectoryAnalysisResult(frame_results=results)
    
    def compute_autocorrelation(
        self, 
        results: TrajectoryAnalysisResult,
        property_name: str,
        max_lag: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute autocorrelation function for a property.
        
        Parameters
        ----------
        results : TrajectoryAnalysisResult
            Analysis results
        property_name : str
            Name of property (e.g., 'n_hbonds', 'betti_1')
        max_lag : int, optional
            Maximum lag for autocorrelation
            
        Returns
        -------
        np.ndarray
            Autocorrelation values
        """
        data = getattr(results, property_name)
        n = len(data)
        
        if max_lag is None:
            max_lag = n // 2
        
        # Normalize data
        data_normalized = data - np.mean(data)
        
        # Compute autocorrelation
        autocorr = np.correlate(data_normalized, data_normalized, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        
        return autocorr
