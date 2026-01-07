#!/usr/bin/env python3
"""

This script demonstrates comprehensive hydrogen bond network analysis from
CP2K AIMD XYZ trajectory files using the hbond_topology package.

Features:
- Basic H-bond detection and counting
- Topological invariants (Betti-0, Betti-1, Betti-2, Euler characteristic)
- H-bond lifetime and autocorrelation analysis
- Coordination, degree, and clustering distributions
- Radial distribution functions (RDF) for multiple species
- H-bond strength classification
- Persistent homology barcode and diagrams (TDA)
- Topological Machine Learning (Embedding, TNN feature extraction, PCA)

Usage:
    python example_analysis.py                        # Use default parameters
    python example_analysis.py --run-ml               # Run with topological ML
    python example_analysis.py --sample-interval 5    # Analyze every 5 frames

Output Files:
- analysis_results.json: Per-frame analysis data
- statistics_summary.json: Overall statistics (includes TML data)
- hbond_dynamics.png, betti_dynamics.png: Time-series analysis plots
- hbond_distributions.png, coordination_degree.png: Statistical distributions
- hbond_lifetime.png, autocorrelation.png: Dynamical properties
- rdf_*.png: Radial distribution functions for species pairs
- clustering_strength.png: Clustering and H-bond strength classification
- persistence_barcode.png, persistence_diagram.png: TDA visualizations
- similarity_heatmap.png, embedding_pca.png, pca_time_series.png: TML visualizations
- frame_embeddings.npy, tnn_features.npy: Raw topological ML features
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hbond_topology import (
    HBondDetector,
    HBondComplexBuilder,
    TopologicalInvariants,
)
from hbond_topology.analysis import (
    plot_persistence_barcode,
    plot_persistence_diagram,
)
from hbond_topology.io.trajectory_parser import TrajectoryParser, Frame
from hbond_topology.embedding import HBondEmbedder
from hbond_topology.learning import HBondTNN, prepare_tnn_data

# Check for optional dependencies
HAS_GUDHI = False
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CP2K AIMD H-bond Topology Analysis (Extended)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Trajectory parameters
    parser.add_argument('--trajectory', '-t', type=str, default='trajectory.xyz',
                        help='Path to trajectory file')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='MD timestep in femtoseconds')
    parser.add_argument('--sample-interval', type=int, default=1,
                        help='Analyze every N frames')
    
    # H-bond detection criteria
    parser.add_argument('--r-da-max', type=float, default=3.5,
                        help='Max donor-acceptor distance (Angstrom)')
    parser.add_argument('--r-ha-max', type=float, default=2.5,
                        help='Max hydrogen-acceptor distance (Angstrom)')
    parser.add_argument('--angle-min', type=float, default=120.0,
                        help='Min D-H-A angle (degrees)')
    
    # Output settings
    parser.add_argument('--dpi', type=int, default=600,
                        help='Figure DPI')
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Output directory')
    
    # Machine Learning parameters
    parser.add_argument('--run-ml', action='store_true',
                        help='Run topological machine learning analysis')
    parser.add_argument('--ml-dim', type=int, default=32,
                        help='Embedding/Hidden dimension for ML')
    
    return parser.parse_args()


# =============================================================================
# Core Analysis Functions
# =============================================================================

def analyze_frame(frame: Frame, detector: HBondDetector, builder: HBondComplexBuilder, 
                  invariants: TopologicalInvariants) -> Dict:
    """Perform full analysis on a single frame."""
    hbonds = detector.detect_hbonds(frame)
    
    # Build H-bond set for tracking
    hbond_set = set()
    for hb in hbonds:
        # Use sorted tuple for undirected edge
        hbond_set.add(tuple(sorted([hb.donor_o_idx, hb.acceptor_o_idx])))
    
    result = {
        'timestep': frame.timestep,
        'n_hbonds': len(hbonds),
        'hbond_set': hbond_set,
        'hbonds': hbonds,
        'distances_da': [hb.distance_da for hb in hbonds],
        'distances_ha': [hb.distance_ha for hb in hbonds],
        'angles_dha': [hb.angle_dha for hb in hbonds],
        'betti_0': 0,
        'betti_1': 0,
        'betti_2': 0,
        'euler_char': 0,
    }
    
    if len(hbonds) > 0:
        try:
            sc = builder.build_from_frame(frame, hbonds)
            inv = invariants.compute_all_invariants(sc)
            result['betti_0'] = inv['betti_numbers'][0]
            result['betti_1'] = inv['betti_numbers'][1] if len(inv['betti_numbers']) > 1 else 0
            result['betti_2'] = inv['betti_numbers'][2] if len(inv['betti_numbers']) > 2 else 0
            result['euler_char'] = inv['euler_characteristic']
        except Exception as e:
            # Note: We don't want to crash the whole analysis if one frame fails
            # but we should at least know it happened.
            if 'sc' in locals():
                print(f"    Warning: Invariant computation failed for frame {frame.timestep}: {e}")
    
    return result


# =============================================================================
# Advanced Analysis Functions
# =============================================================================

def compute_coordination_numbers(results: List[Dict]) -> Dict:
    """
    Compute coordination number distribution.
    Coordination number = number of H-bonds per water molecule.
    """
    all_coordination = []
    
    for r in results:
        # Count how many H-bonds each oxygen participates in
        coord_count = defaultdict(int)
        for hb in r['hbonds']:
            coord_count[hb.donor_o_idx] += 1
            coord_count[hb.acceptor_o_idx] += 1
        
        # Collect all coordination numbers
        all_coordination.extend(list(coord_count.values()))
    
    if not all_coordination:
        return {'mean': 0, 'std': 0, 'distribution': {}}
    
    # Distribution
    unique, counts = np.unique(all_coordination, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return {
        'mean': float(np.mean(all_coordination)),
        'std': float(np.std(all_coordination)),
        'distribution': distribution,
        'raw_data': all_coordination
    }


def compute_degree_distribution(results: List[Dict]) -> Dict:
    """
    Compute node degree distribution of H-bond network.
    Degree = number of connections per node (oxygen atom).
    """
    all_degrees = []
    
    for r in results:
        degree_count = defaultdict(int)
        for hb in r['hbonds']:
            degree_count[hb.donor_o_idx] += 1
            degree_count[hb.acceptor_o_idx] += 1
        all_degrees.extend(list(degree_count.values()))
    
    if not all_degrees:
        return {'mean': 0, 'std': 0, 'distribution': {}}
    
    unique, counts = np.unique(all_degrees, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    
    return {
        'mean': float(np.mean(all_degrees)),
        'std': float(np.std(all_degrees)),
        'distribution': distribution,
        'raw_data': all_degrees
    }


def compute_hbond_lifetime(results: List[Dict], timestep_fs: float) -> Dict:
    """
    Compute H-bond lifetime distribution.
    Tracks continuous existence of each H-bond pair across frames.
    """
    if len(results) < 2:
        return {'mean': 0, 'lifetimes': []}
    
    # Track all unique H-bonds ever observed
    all_hbond_pairs = set()
    for r in results:
        all_hbond_pairs.update(r['hbond_set'])
    
    lifetimes = []
    
    for pair in all_hbond_pairs:
        # Find continuous stretches where this pair exists
        exists = [pair in r['hbond_set'] for r in results]
        
        current_lifetime = 0
        for e in exists:
            if e:
                current_lifetime += 1
            else:
                if current_lifetime > 0:
                    lifetimes.append(current_lifetime * timestep_fs)
                current_lifetime = 0
        if current_lifetime > 0:
            lifetimes.append(current_lifetime * timestep_fs)
    
    if not lifetimes:
        return {'mean': 0, 'std': 0, 'max': 0, 'lifetimes': []}
    
    return {
        'mean': float(np.mean(lifetimes)),
        'std': float(np.std(lifetimes)),
        'max': float(np.max(lifetimes)),
        'lifetimes': lifetimes
    }


def compute_autocorrelation(results: List[Dict], max_lag: int = None) -> Dict:
    """
    Compute H-bond existence autocorrelation function C(t).
    C(t) = <h(0)h(t)> / <h(0)^2>
    where h(t) = 1 if H-bond exists at time t, 0 otherwise.
    """
    if len(results) < 2:
        return {'lags': [], 'acf': []}
    
    if max_lag is None:
        max_lag = min(len(results) // 2, 100)
    
    # Get all unique H-bond pairs
    all_pairs = set()
    for r in results:
        all_pairs.update(r['hbond_set'])
    
    if not all_pairs:
        return {'lags': [], 'acf': []}
    
    # Build existence matrix: (n_pairs, n_frames)
    pair_list = list(all_pairs)
    n_pairs = len(pair_list)
    n_frames = len(results)
    
    existence = np.zeros((n_pairs, n_frames), dtype=float)
    for t, r in enumerate(results):
        for i, pair in enumerate(pair_list):
            if pair in r['hbond_set']:
                existence[i, t] = 1.0
    
    # Compute autocorrelation
    acf = []
    lags = list(range(max_lag))
    
    for lag in lags:
        if lag >= n_frames:
            break
        # C(lag) = mean over all pairs and times of h(t)*h(t+lag)
        numerator = 0
        denominator = 0
        for i in range(n_pairs):
            for t in range(n_frames - lag):
                numerator += existence[i, t] * existence[i, t + lag]
                denominator += existence[i, t] * existence[i, t]
        
        if denominator > 0:
            acf.append(numerator / denominator)
        else:
            acf.append(0)
    
    return {
        'lags': lags[:len(acf)],
        'acf': acf
    }


def compute_clustering_coefficient(results: List[Dict]) -> Dict:
    """
    Compute network clustering coefficient for each frame.
    C = (3 * number of triangles) / (number of connected triples)
    """
    clustering_coeffs = []
    
    for r in results:
        if len(r['hbonds']) < 3:
            clustering_coeffs.append(0.0)
            continue
        
        # Build adjacency from H-bonds
        neighbors = defaultdict(set)
        for hb in r['hbonds']:
            neighbors[hb.donor_o_idx].add(hb.acceptor_o_idx)
            neighbors[hb.acceptor_o_idx].add(hb.donor_o_idx)
        
        # Count triangles and connected triples
        triangles = 0
        triples = 0
        
        for node in neighbors:
            node_neighbors = list(neighbors[node])
            k = len(node_neighbors)
            if k < 2:
                continue
            
            # Count pairs of neighbors that are connected (triangles)
            for i in range(k):
                for j in range(i + 1, k):
                    triples += 1
                    if node_neighbors[j] in neighbors[node_neighbors[i]]:
                        triangles += 1
        
        if triples > 0:
            clustering_coeffs.append(triangles / triples)
        else:
            clustering_coeffs.append(0.0)
    
    return {
        'mean': float(np.mean(clustering_coeffs)),
        'std': float(np.std(clustering_coeffs)),
        'per_frame': clustering_coeffs
    }


def compute_rdf(frames: List[Frame], symbol1: str, symbol2: str, r_max: float = 8.0, dr: float = 0.1) -> Dict:
    """
    General Radial Distribution Function g(r) computation for two elements.
    """
    bins = np.arange(0, r_max + dr, dr)
    hist = np.zeros(len(bins) - 1)
    n_frames = len(frames)
    total_n1 = 0
    total_n2 = 0
    
    for frame in frames:
        idx1 = np.where(frame.symbols == symbol1)[0]
        idx2 = np.where(frame.symbols == symbol2)[0]
        
        n1 = len(idx1)
        n2 = len(idx2)
        total_n1 += n1
        total_n2 += n2
        
        if n1 == 0 or n2 == 0:
            continue
            
        pos1 = frame.positions[idx1]
        pos2 = frame.positions[idx2]
        box_lengths = frame.box_lengths
        
        # Calculate all-pairs distances using broadcasting/vectorization if possible
        # For simplicity and to handle PBC, we use a loop but optimized
        for i in range(n1):
            diff = pos2 - pos1[i]
            # Minimum image convention
            diff = diff - box_lengths * np.round(diff / box_lengths)
            dist = np.linalg.norm(diff, axis=1)
            
            # If same species, avoid self-counting
            if symbol1 == symbol2:
                dist = dist[dist > 0.001]
                
            in_range = dist[dist < r_max]
            for d in in_range:
                idx = int(d / dr)
                if idx < len(hist):
                    hist[idx] += 1

    r_centers = (bins[:-1] + bins[1:]) / 2
    if n_frames > 0 and total_n1 > 0 and total_n2 > 0:
        avg_n1 = total_n1 / n_frames
        avg_n2 = total_n2 / n_frames
        avg_vol = np.mean([np.prod(f.box_lengths) for f in frames])
        rho2 = avg_n2 / avg_vol
        
        shell_volumes = 4 * np.pi * r_centers**2 * dr
        # Normalization factor: hist / (n_frames * n1 * rho2 * shell_vol)
        ideal_count = n_frames * avg_n1 * rho2 * shell_volumes
        
        with np.errstate(divide='ignore', invalid='ignore'):
            g_r = hist / ideal_count
            g_r = np.nan_to_num(g_r, nan=0.0, posinf=0.0)
    else:
        g_r = np.zeros_like(r_centers)
        
    return {'r': r_centers.tolist(), 'g_r': g_r.tolist()}


def classify_hbond_strength(results: List[Dict]) -> Dict:
    """
    Classify H-bonds by strength based on D-A distance.
    Strong: D-A < 2.8 Å
    Moderate: 2.8 Å <= D-A < 3.2 Å
    Weak: D-A >= 3.2 Å
    """
    strong = 0
    moderate = 0
    weak = 0
    
    for r in results:
        for d in r['distances_da']:
            if d < 2.8:
                strong += 1
            elif d < 3.2:
                moderate += 1
            else:
                weak += 1
    
    total = strong + moderate + weak
    if total == 0:
        return {
            'strong': 0, 'moderate': 0, 'weak': 0, 'total': 0,
            'strong_pct': 0.0, 'moderate_pct': 0.0, 'weak_pct': 0.0
        }
    
    return {
        'strong': strong,
        'moderate': moderate,
        'weak': weak,
        'total': total,
        'strong_pct': 100 * strong / total,
        'moderate_pct': 100 * moderate / total,
        'weak_pct': 100 * weak / total
    }


def compute_persistent_homology(results: List[Dict], frames: List[Frame]) -> Dict:
    """
    Compute persistent homology using GUDHI (if available).
    Uses Rips complex on oxygen positions with H-bond edges.
    """
    if not HAS_GUDHI:
        return {'available': False, 'message': 'GUDHI not installed'}
    
    # Use middle frame for demonstration
    mid_idx = len(frames) // 2
    frame = frames[mid_idx]
    r = results[mid_idx]
    
    o_indices = np.where(frame.symbols == 'O')[0]
    if len(o_indices) < 3:
        return {'available': True, 'barcodes': []}
    
    o_positions = frame.positions[o_indices]
    
    # Build Rips complex
    rips = gudhi.RipsComplex(points=o_positions, max_edge_length=5.0)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    
    # Extract barcodes
    barcodes = {
        'H0': [],  # Connected components
        'H1': []   # Loops/holes
    }
    
    for dim, (birth, death) in st.persistence():
        if dim == 0:
            barcodes['H0'].append([birth, death if death < float('inf') else 5.0])
        elif dim == 1:
            barcodes['H1'].append([birth, death if death < float('inf') else 5.0])
    
    return {
        'available': True,
        'frame_index': mid_idx,
        'barcodes': barcodes
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_basic_plots(results: List[Dict], output_dir: Path, timestep_fs: float, dpi: int):
    """Generate basic visualization plots."""
    import matplotlib.pyplot as plt
    
    # Correct time calculation: r['timestep'] is the absolute frame index
    times = [r['timestep'] * timestep_fs for r in results]
    n_hbonds = [r['n_hbonds'] for r in results]
    betti_0 = [r['betti_0'] for r in results]
    betti_1 = [r['betti_1'] for r in results]
    betti_2 = [r['betti_2'] for r in results]
    
    all_distances_da = []
    all_distances_ha = []
    all_angles = []
    for r in results:
        all_distances_da.extend(r['distances_da'])
        all_distances_ha.extend(r['distances_ha'])
        all_angles.extend(r['angles_dha'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. H-bond Dynamics
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, n_hbonds, 'b-', linewidth=1.5, alpha=0.8)
    ax.fill_between(times, n_hbonds, alpha=0.3)
    ax.set_xlabel('Simulation time (fs)', fontsize=12)
    ax.set_ylabel('Number of H-bonds', fontsize=12)
    ax.set_title('Hydrogen Bond Network Dynamics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "hbond_dynamics.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: hbond_dynamics.png")
    
    # 2. Betti Number Dynamics
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, betti_0, 'g-', linewidth=1.5, label=r'$\beta_0$ (components)')
    axes[0].fill_between(times, betti_0, alpha=0.3, color='green')
    axes[0].set_ylabel(r'$\beta_0$', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Topological Invariants Dynamics', fontsize=14, fontweight='bold')
    
    axes[1].plot(times, betti_1, 'r-', linewidth=1.5, label=r'$\beta_1$ (loops)')
    axes[1].fill_between(times, betti_1, alpha=0.3, color='red')
    axes[1].set_ylabel(r'$\beta_1$', fontsize=12)
    axes[1].legend(loc='upper right')
    
    axes[2].plot(times, betti_2, 'b-', linewidth=1.5, label=r'$\beta_2$ (voids)')
    axes[2].fill_between(times, betti_2, alpha=0.3, color='blue')
    axes[2].set_xlabel('Simulation time (fs)', fontsize=12)
    axes[2].set_ylabel(r'$\beta_2$', fontsize=12)
    axes[2].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(output_dir / "betti_dynamics.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: betti_dynamics.png")
    
    # 3. H-bond Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    if all_distances_da:
        axes[0].hist(all_distances_da, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(np.mean(all_distances_da), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_distances_da):.2f} A')
        axes[0].set_xlabel('D-A Distance (A)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Donor-Acceptor Distance', fontsize=12, fontweight='bold')
        axes[0].legend()
    
    if all_distances_ha:
        axes[1].hist(all_distances_ha, bins=50, color='coral', edgecolor='white', alpha=0.8)
        axes[1].axvline(np.mean(all_distances_ha), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_distances_ha):.2f} A')
        axes[1].set_xlabel('H-A Distance (A)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Hydrogen-Acceptor Distance', fontsize=12, fontweight='bold')
        axes[1].legend()
    
    if all_angles:
        axes[2].hist(all_angles, bins=50, color='mediumseagreen', edgecolor='white', alpha=0.8)
        axes[2].axvline(np.mean(all_angles), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_angles):.1f} deg')
        axes[2].set_xlabel('D-H-A Angle (deg)', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=12)
        axes[2].set_title('H-bond Angle', fontsize=12, fontweight='bold')
        axes[2].legend()
    
    fig.suptitle('Hydrogen Bond Geometry Distributions', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "hbond_distributions.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: hbond_distributions.png")


def generate_advanced_plots(results: List[Dict], frames: List[Frame], 
                           advanced_stats: Dict, output_dir: Path, 
                           timestep_fs: float, sample_interval: int, dpi: int):
    """Generate advanced analysis plots."""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 4. Coordination and Degree Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    coord_data = advanced_stats['coordination']
    if coord_data['raw_data']:
        axes[0].hist(coord_data['raw_data'], bins=range(0, max(coord_data['raw_data'])+2), 
                    color='teal', edgecolor='white', alpha=0.8, align='left')
        axes[0].axvline(coord_data['mean'], color='red', linestyle='--',
                       label=f'Mean: {coord_data["mean"]:.2f}')
        axes[0].set_xlabel('Coordination Number', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Coordination Number Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
    
    degree_data = advanced_stats['degree']
    if degree_data['raw_data']:
        axes[1].hist(degree_data['raw_data'], bins=range(0, max(degree_data['raw_data'])+2),
                    color='purple', edgecolor='white', alpha=0.8, align='left')
        axes[1].axvline(degree_data['mean'], color='red', linestyle='--',
                       label=f'Mean: {degree_data["mean"]:.2f}')
        axes[1].set_xlabel('Node Degree', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Degree Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
    
    fig.tight_layout()
    fig.savefig(output_dir / "coordination_degree.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: coordination_degree.png")
    
    # 5. H-bond Lifetime Distribution
    lifetime_data = advanced_stats['lifetime']
    if lifetime_data['lifetimes']:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lifetime_data['lifetimes'], bins=50, color='darkorange', 
               edgecolor='white', alpha=0.8)
        ax.axvline(lifetime_data['mean'], color='red', linestyle='--',
                  label=f'Mean: {lifetime_data["mean"]:.2f} fs')
        ax.set_xlabel('H-bond Lifetime (fs)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Hydrogen Bond Lifetime Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "hbond_lifetime.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: hbond_lifetime.png")
    
    # 6. Autocorrelation Function
    acf_data = advanced_stats['autocorrelation']
    if acf_data['acf']:
        fig, ax = plt.subplots(figsize=(8, 4))
        # ACF lags are in sampled frames, so multiply by (timestep * sample_interval)
        lags_time = [l * timestep_fs * sample_interval for l in acf_data['lags']]
        ax.plot(lags_time, acf_data['acf'], 'b-', linewidth=2)
        ax.axhline(1/np.e, color='red', linestyle='--', alpha=0.5, label='1/e')
        ax.set_xlabel('Time lag (fs)', fontsize=12)
        ax.set_ylabel('C(t)', fontsize=12)
        ax.set_title('H-bond Autocorrelation Function', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        fig.savefig(output_dir / "autocorrelation.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: autocorrelation.png")
    
    # 7. RDF Individual Pairs
    rdf_all = advanced_stats['rdf']
    if rdf_all:
        for pair_name, data in rdf_all.items():
            if data['r'] and data['g_r']:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(data['r'], data['g_r'], 'b-', linewidth=2)
                ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('r (A)', fontsize=12)
                ax.set_ylabel('g(r)', fontsize=12)
                ax.set_title(f'Radial Distribution Function: {pair_name}', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 8)
                fig.tight_layout()
                
                # Save as rdf_PairName.png, replacing '-' with '_' for filename consistency if desired
                filename = f"rdf_{pair_name.replace('-', '_')}.png"
                fig.savefig(output_dir / filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved: {filename}")
    
    # 8. Clustering Coefficient and H-bond Strength
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Clustering coefficient over time
    clustering_data = advanced_stats['clustering']
    # r['timestep'] is the absolute frame index
    times = [r['timestep'] * timestep_fs for r in results]
    axes[0].plot(times, clustering_data['per_frame'], 'g-', linewidth=1.5)
    axes[0].axhline(clustering_data['mean'], color='red', linestyle='--',
                   label=f'Mean: {clustering_data["mean"]:.3f}')
    axes[0].set_xlabel('Simulation time (fs)', fontsize=12)
    axes[0].set_ylabel('Clustering Coefficient', fontsize=12)
    axes[0].set_title('Network Clustering Coefficient', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # H-bond strength pie chart
    strength_data = advanced_stats['strength']
    if strength_data['total'] > 0:
        labels = ['Strong\n(<2.8 A)', 'Moderate\n(2.8-3.2 A)', 'Weak\n(>3.2 A)']
        sizes = [strength_data['strong'], strength_data['moderate'], strength_data['weak']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0, 0)
        axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('H-bond Strength Classification', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_strength.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: clustering_strength.png")
    
    # 9. Persistent Homology (Barcode and Diagram)
    persistence_data = advanced_stats.get('persistence', {})
    if persistence_data.get('available') and persistence_data.get('barcodes'):
        barcodes = persistence_data['barcodes']
        
        # Plot Barcode
        plot_persistence_barcode(
            barcodes, 
            title=f"Persistence Barcode (Frame {persistence_data['frame_index']})",
            save_path=output_dir / "persistence_barcode.png",
            dpi=dpi
        )
        print(f"    Saved: persistence_barcode.png")
        
        # Plot Diagram
        plot_persistence_diagram(
            barcodes,
            title=f"Persistence Diagram (Frame {persistence_data['frame_index']})",
            save_path=output_dir / "persistence_diagram.png",
            dpi=dpi
        )
        print(f"    Saved: persistence_diagram.png")


# =============================================================================
# Topological Machine Learning Analysis
# =============================================================================

def perform_topological_ml_analysis(sc_list: List, ml_dim: int) -> Dict:
    """
    Perform topological machine learning analysis on a list of simplicial complexes.
    
    Includes:
    - Topological embeddings (Cell2Vec)
    - TNN feature extraction (SAN)
    - Dimensionality reduction (PCA)
    """
    if not sc_list:
        return {}
    
    print(f"    Running TML on {len(sc_list)} frames...")
    
    # 1. Topological Embeddings
    print("        Generating Cell2Vec embeddings...")
    embedder = HBondEmbedder(method='cell2vec', dimensions=ml_dim)
    frame_embeddings = []
    
    for sc in sc_list:
        try:
            # Try specified method first, fallback to HOPE if it fails or returns zeros
            emb = embedder.fit_transform(sc)
            
            # If zeros or None, try HOPE (deterministic)
            if emb is None or len(emb) == 0 or np.ptp(emb) == 0:
                fallback_embedder = HBondEmbedder(method='hope', dimensions=ml_dim)
                emb = fallback_embedder.fit_transform(sc)
                
            if emb is not None and len(emb) > 0:
                frame_embeddings.append(np.mean(emb, axis=0))
            else:
                frame_embeddings.append(np.zeros(ml_dim))
        except Exception:
            frame_embeddings.append(np.zeros(ml_dim))
    
    frame_embeddings = np.array(frame_embeddings)
    
    # Debug: Check variance
    ptp = np.ptp(frame_embeddings)
    if ptp == 0:
        print("        Warning: Frame embeddings have zero variance. Try a different rank or check data.")
    
    # 2. TNN Feature Extraction (Demonstration)
    print("        Extracting TNN features (SAN forward pass)...")
    tnn_features = None
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"        Using device: {device}")
        
        model = HBondTNN(in_channels=1, hidden_channels=ml_dim, out_channels=ml_dim)
        model = model.to(device)
        model.eval()
        
        tnn_feats_list = []
        for sc in sc_list:
            if sc.dim >= 1:
                data = prepare_tnn_data(sc)
                # Move data to device
                edge_features = data['edge_features'].to(device)
                laplacian_up = data['laplacian_up'].to(device)
                laplacian_down = data['laplacian_down'].to(device)
                
                with torch.no_grad():
                    edge_feats, _ = model(edge_features, laplacian_up, laplacian_down)
                    # Move result back to CPU for numpy conversion
                    tnn_feats_list.append(edge_feats.mean(dim=0).cpu().numpy())
            else:
                tnn_feats_list.append(np.zeros(ml_dim))
        tnn_features = np.array(tnn_feats_list)
    except Exception as e:
        print(f"        Warning: TNN feature extraction failed: {e}")
    
    # 3. PCA on embeddings
    pca_results = None
    explained_variance = []
    try:
        from sklearn.decomposition import PCA
        # Check if we have enough variance to run PCA
        if len(frame_embeddings) > 1 and ptp > 0:
            pca = PCA(n_components=min(2, len(frame_embeddings)))
            pca_results = pca.fit_transform(frame_embeddings)
            explained_variance = pca.explained_variance_ratio_.tolist()
        else:
            # If zero variance, we can still "mock" PCA data for visualization if we want
            # but better to just report it.
            print("        Skipping PCA: zero variance in embeddings.")
    except Exception as e:
         print(f"        Warning: PCA analysis failed: {e}")
        
    return {
        'frame_embeddings': frame_embeddings,
        'tnn_features': tnn_features,
        'pca_results': pca_results,
        'explained_variance': explained_variance
    }


def generate_ml_plots(ml_results: Dict, output_dir: Path, dpi: int):
    """Generate plots for TML analysis."""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Similarity Heatmap (Doesn't depend on PCA)
    embeddings = ml_results.get('frame_embeddings')
    if embeddings is not None and len(embeddings) > 1:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(similarity_matrix, cmap='magma', origin='lower')
            cbar = plt.colorbar(im)
            cbar.set_label('Cosine Similarity', fontsize=12)
            ax.set_xlabel('Frame Index', fontsize=12)
            ax.set_ylabel('Frame Index', fontsize=12)
            ax.set_title('Inter-Frame Topological Similarity', fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            fig.savefig(output_dir / "similarity_heatmap.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: similarity_heatmap.png")
        except Exception as e:
            print(f"    Warning: Similarity heatmap failed: {e}")

    # 2. PCA Results (Scatter and Time-Series)
    pca_data = ml_results.get('pca_results')
    if pca_data is not None and len(pca_data) >= 1:
        try:
            # PCA Scatter Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=range(len(pca_data)), 
                                 cmap='viridis', alpha=0.7, s=50)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Frame Index', fontsize=12)
            
            if len(ml_results.get('explained_variance', [])) >= 2:
                ax.set_xlabel(f'PC1 ({ml_results["explained_variance"][0]*100:.1f}%)', fontsize=12)
                ax.set_ylabel(f'PC2 ({ml_results["explained_variance"][1]*100:.1f}%)', fontsize=12)
            else:
                ax.set_xlabel('PC1', fontsize=12)
                ax.set_ylabel('PC2', fontsize=12)
                
            ax.set_title('PCA of Topological Embeddings', fontsize=14, fontweight='bold')
            
            fig.tight_layout()
            fig.savefig(output_dir / "embedding_pca.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: embedding_pca.png")
        except Exception as e:
            print(f"    Warning: PCA scatter plot failed: {e}")
        
        try:
            # PCA vs Time
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(len(pca_data)), pca_data[:, 0], label='PC1', marker='.')
            if pca_data.shape[1] > 1:
                ax.plot(range(len(pca_data)), pca_data[:, 1], label='PC2', marker='.')
            ax.set_xlabel('Frame Index', fontsize=12)
            ax.set_ylabel('Principal Component Value', fontsize=12)
            ax.set_title('PCA Components Over Time', fontsize=14, fontweight='bold')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "pca_time_series.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: pca_time_series.png")
        except Exception as e:
            print(f"    Warning: PCA time-series plot failed: {e}")


# =============================================================================
# Results Saving
# =============================================================================

def save_results(results: List[Dict], advanced_stats: Dict, output_dir: Path, ml_results: Optional[Dict] = None):
    """Save analysis results to JSON files."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        return obj
    
    # Per-frame results (simplified)
    simplified_results = []
    for r in results:
        simplified_results.append({
            'timestep': convert(r['timestep']),
            'n_hbonds': convert(r['n_hbonds']),
            'betti_0': convert(r['betti_0']),
            'betti_1': convert(r['betti_1']),
            'betti_2': convert(r['betti_2']),
            'euler_char': convert(r['euler_char']),
            'mean_distance_da': float(np.mean(r['distances_da'])) if r['distances_da'] else 0,
            'mean_angle_dha': float(np.mean(r['angles_dha'])) if r['angles_dha'] else 0,
        })
    
    with open(output_dir / "analysis_results.json", 'w') as f:
        json.dump(simplified_results, f, indent=2)
    print(f"    Saved: analysis_results.json")
    
    # Comprehensive statistics
    n_hbonds = [r['n_hbonds'] for r in results]
    betti_0 = [r['betti_0'] for r in results]
    betti_1 = [r['betti_1'] for r in results]
    betti_2 = [r['betti_2'] for r in results]
    
    all_distances_da = []
    all_angles = []
    for r in results:
        all_distances_da.extend(r['distances_da'])
        all_angles.extend(r['angles_dha'])
    
    stats = {
        'n_frames_analyzed': len(results),
        'basic_statistics': {
            'hbond_count': {
                'mean': float(np.mean(n_hbonds)),
                'std': float(np.std(n_hbonds)),
                'min': int(np.min(n_hbonds)),
                'max': int(np.max(n_hbonds)),
            },
            'betti_0': {
                'mean': float(np.mean(betti_0)),
                'std': float(np.std(betti_0)),
            },
            'betti_1': {
                'mean': float(np.mean(betti_1)),
                'std': float(np.std(betti_1)),
            },
            'betti_2': {
                'mean': float(np.mean(betti_2)),
                'std': float(np.std(betti_2)),
            },
            'geometry': {
                'mean_distance_da': float(np.mean(all_distances_da)) if all_distances_da else 0,
                'std_distance_da': float(np.std(all_distances_da)) if all_distances_da else 0,
                'mean_angle_dha': float(np.mean(all_angles)) if all_angles else 0,
                'std_angle_dha': float(np.std(all_angles)) if all_angles else 0,
            }
        },
        'advanced_statistics': {
            'coordination': {
                'mean': advanced_stats['coordination']['mean'],
                'std': advanced_stats['coordination']['std'],
                'distribution': advanced_stats['coordination']['distribution'],
            },
            'degree': {
                'mean': advanced_stats['degree']['mean'],
                'std': advanced_stats['degree']['std'],
                'distribution': advanced_stats['degree']['distribution'],
            },
            'lifetime': {
                'mean_fs': advanced_stats['lifetime']['mean'],
                'std_fs': advanced_stats['lifetime']['std'],
                'max_fs': advanced_stats['lifetime']['max'],
            },
            'clustering': {
                'mean': advanced_stats['clustering']['mean'],
                'std': advanced_stats['clustering']['std'],
            },
            'hbond_strength': advanced_stats['strength'],
            'persistent_homology': {
                'available': advanced_stats['persistence']['available'],
            }
        }
    }
    
    if ml_results:
        stats['topological_machine_learning'] = {
            'pca_explained_variance': convert(ml_results.get('explained_variance')),
            'feature_extraction': 'Cell2Vec (Embedding) + SAN (TNN)'
        }
        
        # Save embeddings separately
        if ml_results.get('frame_embeddings') is not None:
            np.save(output_dir / "frame_embeddings.npy", ml_results['frame_embeddings'])
            print(f"    Saved: frame_embeddings.npy")
        if ml_results.get('tnn_features') is not None:
            np.save(output_dir / "tnn_features.npy", ml_results['tnn_features'])
            print(f"    Saved: tnn_features.npy")
    
    with open(output_dir / "statistics_summary.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"    Saved: statistics_summary.json")


# =============================================================================
# Main Analysis Script
# =============================================================================

def main():
    """Run comprehensive H-bond topology analysis on CP2K AIMD trajectory."""
    
    args = parse_args()
    traj_file = Path(__file__).parent / args.trajectory
    
    if not traj_file.exists():
        print(f"Error: Trajectory file not found: {traj_file}")
        return
    
    print("=" * 70)
    print("CP2K AIMD Hydrogen Bond Topology Analysis")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"    Timestep: {args.timestep} fs")
    print(f"    Sample interval: every {args.sample_interval} frames")
    print(f"    H-bond criteria: D-A < {args.r_da_max} A, H-A < {args.r_ha_max} A, angle > {args.angle_min} deg")
    print(f"    Output DPI: {args.dpi}")
    print(f"    GUDHI available: {HAS_GUDHI}")
    
    # Parse trajectory
    print("\n[1] Parsing trajectory with ASE backend...")
    parser = TrajectoryParser(traj_file, format='xyz')
    frames = parser.parse()
    print(f"    Loaded {len(frames)} frames")
    print(f"    Atoms per frame: {frames[0].n_atoms}")
    print(f"    Simulation time: {len(frames) * args.timestep:.1f} fs")
    
    n_oxygen = np.sum(frames[0].symbols == 'O')
    n_hydrogen = np.sum(frames[0].symbols == 'H')
    print(f"    Oxygen atoms: {n_oxygen}, Hydrogen atoms: {n_hydrogen}")
    
    unique_elements = np.unique(frames[0].symbols)
    print(f"    Elements: {', '.join(unique_elements)}")
    
    # Initialize analyzers
    print("\n[2] Initializing analyzers...")
    detector = HBondDetector(
        r_da_max=args.r_da_max,
        r_ha_max=args.r_ha_max,
        angle_min=args.angle_min,
        o_symbol='O',
        h_symbol='H'
    )
    builder = HBondComplexBuilder()
    invariants = TopologicalInvariants()
    
    # Analyze trajectory
    sample_interval = args.sample_interval
    print(f"\n[3] Analyzing trajectory (every {sample_interval} frames)...")
    sampled_frames = frames[::sample_interval]
    results = []
    sc_list = [] if args.run_ml else None
    
    for i, frame in enumerate(sampled_frames):
        result = analyze_frame(frame, detector, builder, invariants)
        result['timestep'] = i * sample_interval
        results.append(result)
        
        if args.run_ml:
            hbonds = detector.detect_hbonds(frame)
            if hbonds:
                sc_list.append(builder.build_from_frame(frame, hbonds))
            else:
                import toponetx as tnx
                sc_list.append(tnx.SimplicialComplex())
                
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(sampled_frames)} frames...")
    
    print(f"    Completed basic analysis of {len(results)} frames")
    
    # Advanced analysis
    print("\n[4] Running advanced analysis...")
    advanced_stats = {}
    
    print("    Computing coordination numbers...")
    advanced_stats['coordination'] = compute_coordination_numbers(results)
    
    print("    Computing degree distribution...")
    advanced_stats['degree'] = compute_degree_distribution(results)
    
    print("    Computing H-bond lifetimes...")
    advanced_stats['lifetime'] = compute_hbond_lifetime(results, args.timestep * sample_interval)
    
    print("    Computing autocorrelation function...")
    advanced_stats['autocorrelation'] = compute_autocorrelation(results)
    
    print("    Computing clustering coefficients...")
    advanced_stats['clustering'] = compute_clustering_coefficient(results)
    
    print("    Computing RDFs for multiple pairs...")
    rdfs = {}
    rdf_pairs = [('O', 'O'), ('O', 'H'), ('H', 'H'), ('La', 'O'), ('K', 'O'), ('P', 'O')]
    for s1, s2 in rdf_pairs:
        # Check if both elements exist in first frame
        if s1 in unique_elements and s2 in unique_elements:
            print(f"        RDF: {s1}-{s2}")
            rdfs[f"{s1}-{s2}"] = compute_rdf(sampled_frames, s1, s2)
    advanced_stats['rdf'] = rdfs
    
    print("    Classifying H-bond strength...")
    advanced_stats['strength'] = classify_hbond_strength(results)
    
    print("    Computing persistent homology...")
    advanced_stats['persistence'] = compute_persistent_homology(results, sampled_frames)
    
    # Print summary statistics
    print("\n[5] Summary Statistics:")
    n_hbonds = [r['n_hbonds'] for r in results]
    print(f"    H-bonds: {np.mean(n_hbonds):.1f} +/- {np.std(n_hbonds):.1f}")
    print(f"    Coordination: {advanced_stats['coordination']['mean']:.2f} +/- {advanced_stats['coordination']['std']:.2f}")
    print(f"    H-bond lifetime: {advanced_stats['lifetime']['mean']:.2f} fs")
    print(f"    Clustering coeff: {advanced_stats['clustering']['mean']:.3f}")
    print(f"    H-bond strength: Strong {advanced_stats['strength']['strong_pct']:.1f}%, "
          f"Moderate {advanced_stats['strength']['moderate_pct']:.1f}%, "
          f"Weak {advanced_stats['strength']['weak_pct']:.1f}%")
    
    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # ML Analysis
    ml_results = {}
    if args.run_ml:
        print("\n[5.5] Running Topological Machine Learning analysis...")
        ml_results = perform_topological_ml_analysis(sc_list, args.ml_dim)
        generate_ml_plots(ml_results, output_dir, args.dpi)
    
    # Save results
    print("\n[6] Saving results...")
    save_results(results, advanced_stats, output_dir, ml_results if args.run_ml else None)
    
    # Generate plots
    print("\n[7] Generating plots...")
    try:
        generate_basic_plots(results, output_dir, args.timestep, args.dpi)
        generate_advanced_plots(results, sampled_frames, advanced_stats, output_dir, 
                               args.timestep, args.sample_interval, args.dpi)
    except ImportError as e:
        print(f"    Warning: matplotlib not available: {e}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
