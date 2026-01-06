"""
Hydrogen Bond Detector

Detects hydrogen bonds in molecular dynamics frames using geometric criteria.
Supports periodic boundary conditions for proper distance calculations.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..io.trajectory_parser import Frame


@dataclass
class HBond:
    """Represents a hydrogen bond."""
    
    donor_o_idx: int      # Index of donor oxygen atom
    donor_h_idx: int      # Index of donor hydrogen atom  
    acceptor_o_idx: int   # Index of acceptor oxygen atom
    distance_da: float    # Donor-Acceptor distance
    distance_ha: float    # Hydrogen-Acceptor distance
    angle_dha: float      # D-H-A angle in degrees


@dataclass
class WaterMolecule:
    """Represents a water molecule."""
    
    o_idx: int           # Index of oxygen atom
    h1_idx: int          # Index of first hydrogen
    h2_idx: int          # Index of second hydrogen
    o_position: np.ndarray
    h1_position: np.ndarray
    h2_position: np.ndarray


class HBondDetector:
    """
    Hydrogen bond detector using geometric criteria.
    
    Default criteria (based on common literature values):
    - D-A distance < 3.5 Å (donor oxygen to acceptor oxygen)
    - H-A distance < 2.5 Å (hydrogen to acceptor oxygen)
    - D-H-A angle > 120° (angle at hydrogen)
    
    Parameters
    ----------
    r_oh_max : float
        Maximum O-H distance for water molecule assignment (default: 1.2 Å)
    r_da_max : float
        Maximum donor-acceptor distance for H-bond (default: 3.5 Å)
    r_ha_max : float
        Maximum hydrogen-acceptor distance for H-bond (default: 2.5 Å)
    angle_min : float
        Minimum D-H-A angle in degrees (default: 120°)
    o_atom_type : int
        Atom type for oxygen (default: 1)
    h_atom_type : int
        Atom type for hydrogen (default: 2)
    """
    
    def __init__(
        self,
        r_oh_max: float = 1.2,
        r_da_max: float = 3.5,
        r_ha_max: float = 2.5,
        angle_min: float = 120.0,
        o_atom_type: int = 1,
        h_atom_type: int = 2
    ):
        self.r_oh_max = r_oh_max
        self.r_da_max = r_da_max
        self.r_ha_max = r_ha_max
        self.angle_min = angle_min
        self.o_atom_type = o_atom_type
        self.h_atom_type = h_atom_type
    
    def minimum_image_distance(
        self, 
        pos1: np.ndarray, 
        pos2: np.ndarray, 
        box_lengths: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate minimum image distance with periodic boundary conditions.
        
        Returns
        -------
        distance : float
            Minimum image distance
        delta : np.ndarray
            Vector from pos1 to pos2 (minimum image)
        """
        delta = pos2 - pos1
        # Apply minimum image convention
        delta = delta - box_lengths * np.round(delta / box_lengths)
        distance = np.linalg.norm(delta)
        return distance, delta
    
    def identify_water_molecules(self, frame: Frame) -> List[WaterMolecule]:
        """
        Identify water molecules by finding O-H pairs.
        
        Each oxygen should have exactly 2 hydrogens within r_oh_max distance.
        """
        o_indices = frame.get_atoms_by_type(self.o_atom_type)
        h_indices = frame.get_atoms_by_type(self.h_atom_type)
        
        water_molecules = []
        box_lengths = frame.box_lengths
        
        for o_idx in o_indices:
            o_pos = frame.positions[o_idx]
            
            # Find nearby hydrogens
            nearby_h = []
            for h_idx in h_indices:
                h_pos = frame.positions[h_idx]
                dist, _ = self.minimum_image_distance(o_pos, h_pos, box_lengths)
                if dist < self.r_oh_max:
                    nearby_h.append((h_idx, dist))
            
            # Sort by distance and take closest 2
            nearby_h.sort(key=lambda x: x[1])
            
            if len(nearby_h) >= 2:
                h1_idx = nearby_h[0][0]
                h2_idx = nearby_h[1][0]
                
                water = WaterMolecule(
                    o_idx=o_idx,
                    h1_idx=h1_idx,
                    h2_idx=h2_idx,
                    o_position=o_pos,
                    h1_position=frame.positions[h1_idx],
                    h2_position=frame.positions[h2_idx]
                )
                water_molecules.append(water)
        
        return water_molecules
    
    def calculate_angle(
        self, 
        pos_d: np.ndarray, 
        pos_h: np.ndarray, 
        pos_a: np.ndarray,
        box_lengths: np.ndarray
    ) -> float:
        """Calculate D-H-A angle in degrees using minimum image convention."""
        _, vec_hd = self.minimum_image_distance(pos_h, pos_d, box_lengths)
        _, vec_ha = self.minimum_image_distance(pos_h, pos_a, box_lengths)
        
        # Normalize vectors
        vec_hd_norm = vec_hd / np.linalg.norm(vec_hd)
        vec_ha_norm = vec_ha / np.linalg.norm(vec_ha)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(vec_hd_norm, vec_ha_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def detect_hbonds(self, frame: Frame) -> List[HBond]:
        """
        Detect hydrogen bonds in a frame.
        
        Parameters
        ----------
        frame : Frame
            Trajectory frame to analyze
            
        Returns
        -------
        List[HBond]
            List of detected hydrogen bonds
        """
        water_molecules = self.identify_water_molecules(frame)
        box_lengths = frame.box_lengths
        hbonds = []
        
        for donor in water_molecules:
            for acceptor in water_molecules:
                # Skip self-interaction
                if donor.o_idx == acceptor.o_idx:
                    continue
                
                # Check donor-acceptor distance
                dist_da, _ = self.minimum_image_distance(
                    donor.o_position, acceptor.o_position, box_lengths
                )
                
                if dist_da > self.r_da_max:
                    continue
                
                # Check each hydrogen of the donor
                for h_idx, h_pos in [(donor.h1_idx, donor.h1_position), 
                                      (donor.h2_idx, donor.h2_position)]:
                    
                    # Check H-A distance
                    dist_ha, _ = self.minimum_image_distance(
                        h_pos, acceptor.o_position, box_lengths
                    )
                    
                    if dist_ha > self.r_ha_max:
                        continue
                    
                    # Check D-H-A angle
                    angle = self.calculate_angle(
                        donor.o_position, h_pos, acceptor.o_position, box_lengths
                    )
                    
                    if angle >= self.angle_min:
                        hbond = HBond(
                            donor_o_idx=donor.o_idx,
                            donor_h_idx=h_idx,
                            acceptor_o_idx=acceptor.o_idx,
                            distance_da=dist_da,
                            distance_ha=dist_ha,
                            angle_dha=angle
                        )
                        hbonds.append(hbond)
        
        return hbonds
    
    def get_hbond_edges(self, hbonds: List[HBond]) -> List[Tuple[int, int]]:
        """
        Convert hydrogen bonds to edge list (donor_O, acceptor_O pairs).
        
        Returns unique edges suitable for graph construction.
        """
        edges = set()
        for hb in hbonds:
            # Use tuple to represent directed edge (donor -> acceptor)
            edges.add((hb.donor_o_idx, hb.acceptor_o_idx))
        return list(edges)
    
    def get_undirected_edges(self, hbonds: List[HBond]) -> List[Tuple[int, int]]:
        """
        Convert hydrogen bonds to undirected edge list.
        
        Edges are canonicalized (smaller index first) to avoid duplicates.
        """
        edges = set()
        for hb in hbonds:
            edge = tuple(sorted([hb.donor_o_idx, hb.acceptor_o_idx]))
            edges.add(edge)
        return list(edges)
