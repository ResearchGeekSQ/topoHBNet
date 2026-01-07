#!/usr/bin/env python
"""

Example script demonstrating basic usage of the H-bond topology analysis package.

"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbond_topology.io.trajectory_parser import TrajectoryParser
from hbond_topology.detection.hbond_detector import HBondDetector
from hbond_topology.topology.complex_builder import HBondComplexBuilder
from hbond_topology.topology.invariants import TopologicalInvariants


def main():
    """Run example analysis on trajectory."""
    
    # Path to trajectory
    trajectory_path = Path(__file__).parent / "trajectory.lammpstrj"
    
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        return 1
    
    print("=" * 60)
    print("Hydrogen Bond Network Topology Analysis - Example")
    print("=" * 60)
    
    # Step 1: Load trajectory
    print("\n1. Loading trajectory...")
    parser = TrajectoryParser(trajectory_path)
    frames = parser.parse()
    print(f"   Loaded {len(frames)} frames")
    print(f"   Atoms per frame: {frames[0].n_atoms}")
    print(f"   Box size: {frames[0].box_lengths}")
    
    # Step 2: Analyze first frame
    print("\n2. Analyzing first frame...")
    frame = frames[0]
    
    # Detect water molecules and H-bonds
    detector = HBondDetector()
    water_mols = detector.identify_water_molecules(frame)
    print(f"   Water molecules: {len(water_mols)}")
    
    hbonds = detector.detect_hbonds(frame)
    print(f"   Hydrogen bonds: {len(hbonds)}")
    
    if len(hbonds) > 0:
        # Show sample H-bonds
        print("\n   Sample H-bonds:")
        for i, hb in enumerate(hbonds[:5]):
            print(f"   [{i+1}] O{hb.donor_o_idx} -> O{hb.acceptor_o_idx}: "
                  f"d={hb.distance_da:.2f}Å, angle={hb.angle_dha:.1f}°")
    
    # Step 3: Build simplicial complex
    print("\n3. Building simplicial complex...")
    builder = HBondComplexBuilder()
    sc = builder.build_from_frame(frame, hbonds)
    print(f"   Shape: {sc.shape}")
    n_nodes = sc.shape[0] if len(sc.shape) > 0 else 0
    print(f"   - Nodes (water molecules): {n_nodes}")
    if len(sc.shape) > 1:
        print(f"   - Edges (H-bonds): {sc.shape[1]}")
    if len(sc.shape) > 2:
        print(f"   - Triangles (3-water rings): {sc.shape[2]}")
    
    # Step 4: Compute topological invariants
    print("\n4. Computing topological invariants...")
    invariants = TopologicalInvariants()
    result = invariants.compute_all_invariants(sc)
    
    print(f"   Betti numbers:")
    for k, v in result['betti_numbers'].items():
        if k == 0:
            print(f"     β₀ = {v} (connected components)")
        elif k == 1:
            print(f"     β₁ = {v} (loops/holes)")
        elif k == 2:
            print(f"     β₂ = {v} (voids)")
    
    print(f"   Euler characteristic: χ = {result['euler_characteristic']}")
    
    # Step 5: Analyze a few more frames
    print("\n5. Analyzing multiple frames...")
    n_sample = min(5, len(frames))
    
    print(f"\n   {'Frame':<8} {'Timestep':<12} {'H-bonds':<10} {'Triangles':<12} {'β₀':<6} {'β₁':<6}")
    print(f"   {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*6} {'-'*6}")
    
    for i in range(n_sample):
        frame = frames[i]
        hbonds = detector.detect_hbonds(frame)
        
        if len(hbonds) > 0:
            sc = builder.build_from_frame(frame, hbonds)
            result = invariants.compute_all_invariants(sc)
            n_triangles = sc.shape[2] if len(sc.shape) > 2 else 0
            b0 = result['betti_numbers'].get(0, 0)
            b1 = result['betti_numbers'].get(1, 0)
        else:
            n_triangles = 0
            b0 = 0
            b1 = 0
        
        print(f"   {i:<8} {frame.timestep:<12} {len(hbonds):<10} {n_triangles:<12} {b0:<6} {b1:<6}")
    
    print("\n" + "=" * 60)
    print("Example analysis complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
