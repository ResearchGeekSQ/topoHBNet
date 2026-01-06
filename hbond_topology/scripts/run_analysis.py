#!/usr/bin/env python
"""
run_analysis.py

Main script for analyzing hydrogen bond network topology from LAMMPS trajectory.

Usage:
    python run_analysis.py trajectory.lammpstrj --output results/
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hbond_topology.io.trajectory_parser import TrajectoryParser
from hbond_topology.detection.hbond_detector import HBondDetector
from hbond_topology.topology.complex_builder import HBondComplexBuilder
from hbond_topology.topology.invariants import TopologicalInvariants
from hbond_topology.analysis.dynamics import DynamicsAnalyzer
from hbond_topology.analysis.visualization import TopologyVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze hydrogen bond network topology from LAMMPS trajectory"
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to LAMMPS trajectory file (.lammpstrj)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index (default: 0)"
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Last frame index (default: all)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame stride (default: 1)"
    )
    parser.add_argument(
        "--r-da-max",
        type=float,
        default=3.5,
        help="Max donor-acceptor distance for H-bond (default: 3.5 Å)"
    )
    parser.add_argument(
        "--r-ha-max",
        type=float,
        default=2.5,
        help="Max hydrogen-acceptor distance for H-bond (default: 2.5 Å)"
    )
    parser.add_argument(
        "--angle-min",
        type=float,
        default=120.0,
        help="Min D-H-A angle for H-bond (default: 120°)"
    )
    parser.add_argument(
        "--o-type",
        type=int,
        default=1,
        help="Atom type for oxygen (default: 1)"
    )
    parser.add_argument(
        "--h-type",
        type=int,
        default=2,
        help="Atom type for hydrogen (default: 2)"
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Compute spectral features (slower)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (no progress output)"
    )
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.quiet:
        print(f"=" * 60)
        print(f"Hydrogen Bond Network Topology Analysis")
        print(f"=" * 60)
        print(f"\nTrajectory: {args.trajectory}")
        print(f"Output: {output_dir}")
        print(f"\nH-bond criteria:")
        print(f"  D-A distance: < {args.r_da_max} Å")
        print(f"  H-A distance: < {args.r_ha_max} Å")
        print(f"  D-H-A angle: > {args.angle_min}°")
        print(f"\nAtom types: O={args.o_type}, H={args.h_type}")
        print()
    
    # Load trajectory
    if not args.quiet:
        print("Loading trajectory...")
    
    parser = TrajectoryParser(args.trajectory)
    frames = parser.parse()
    
    if not args.quiet:
        print(f"  Loaded {len(frames)} frames")
        print(f"  Atoms per frame: {frames[0].n_atoms}")
        print()
    
    # Configure H-bond detector
    detector = HBondDetector(
        r_da_max=args.r_da_max,
        r_ha_max=args.r_ha_max,
        angle_min=args.angle_min,
        o_atom_type=args.o_type,
        h_atom_type=args.h_type
    )
    
    # Run analysis
    if not args.quiet:
        print("Analyzing trajectory...")
    
    analyzer = DynamicsAnalyzer(
        hbond_detector=detector,
        compute_spectral=args.spectral,
        verbose=not args.quiet
    )
    
    results = analyzer.analyze_trajectory(
        parser,
        start=args.start,
        stop=args.stop,
        step=args.step
    )
    
    # Save results
    if not args.quiet:
        print(f"\nSaving results to {output_dir}...")
    
    # Save JSON
    results.to_json(output_dir / "analysis_results.json")
    
    # Save statistics
    stats = results.statistics()
    with open(output_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    if not args.quiet:
        print(f"\n" + "=" * 60)
        print("Analysis Summary")
        print("=" * 60)
        print(f"\nFrames analyzed: {results.n_frames}")
        print(f"\nProperty statistics:")
        for name, values in stats.items():
            print(f"  {name}:")
            print(f"    mean: {values['mean']:.2f} ± {values['std']:.2f}")
            print(f"    range: [{values['min']:.0f}, {values['max']:.0f}]")
    
    # Generate plots
    if not args.no_plots:
        try:
            if not args.quiet:
                print("\nGenerating plots...")
            
            viz = TopologyVisualizer()
            
            # Main dynamics plot
            viz.plot_dynamics(
                results,
                save_path=output_dir / "topology_dynamics.png",
                show=False
            )
            
            # H-bond statistics plot
            viz.plot_hbond_statistics(
                results,
                save_path=output_dir / "hbond_statistics.png",
                show=False
            )
            
            # Summary box plots
            viz.plot_statistics_summary(
                results,
                save_path=output_dir / "statistics_summary.png",
                show=False
            )
            
            if not args.quiet:
                print(f"  Saved plots to {output_dir}")
                
        except Exception as e:
            if not args.quiet:
                print(f"  Warning: Could not generate plots - {e}")
    
    if not args.quiet:
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
