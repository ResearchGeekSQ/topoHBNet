"""Topology construction and analysis module."""

from .complex_builder import HBondComplexBuilder
from .invariants import TopologicalInvariants

# Persistence module requires gudhi (optional)
try:
    from .persistence import PersistenceAnalyzer, plot_persistence_diagram
    __all__ = ["HBondComplexBuilder", "TopologicalInvariants", "PersistenceAnalyzer", "plot_persistence_diagram"]
except ImportError:
    __all__ = ["HBondComplexBuilder", "TopologicalInvariants"]
