"""
Hydrogen Bond Network Topology Analysis Package

This package provides tools for analyzing hydrogen bond network topology
from LAMMPS molecular dynamics trajectories using TopoNetX, TopoModelX,
and TopoEmbedX libraries.
"""

__version__ = "0.1.0"
__author__ = "SIQI TANG"

from .io.trajectory_parser import TrajectoryParser
from .detection.hbond_detector import HBondDetector
from .topology.complex_builder import HBondComplexBuilder
from .topology.invariants import TopologicalInvariants
from .embedding.embedder import HBondEmbedder
from .analysis.dynamics import DynamicsAnalyzer
from .analysis.visualization import TopologyVisualizer

__all__ = [
    "TrajectoryParser",
    "HBondDetector", 
    "HBondComplexBuilder",
    "TopologicalInvariants",
    "HBondEmbedder",
    "DynamicsAnalyzer",
    "TopologyVisualizer",
]
