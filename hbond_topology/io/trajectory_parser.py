"""
LAMMPS Trajectory Parser

Parses LAMMPS dump files (.lammpstrj) to extract atomic positions
and simulation box information for each timestep.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Generator
from pathlib import Path


@dataclass
class Frame:
    """A single trajectory frame containing atomic data."""
    
    timestep: int
    n_atoms: int
    box_bounds: np.ndarray  # Shape: (3, 2) - [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    atom_ids: np.ndarray
    atom_types: np.ndarray
    positions: np.ndarray  # Shape: (n_atoms, 3)
    
    @property
    def box_lengths(self) -> np.ndarray:
        """Return box lengths in x, y, z directions."""
        return self.box_bounds[:, 1] - self.box_bounds[:, 0]
    
    def get_atoms_by_type(self, atom_type: int) -> np.ndarray:
        """Get indices of atoms with specified type."""
        return np.where(self.atom_types == atom_type)[0]
    
    def get_positions_by_type(self, atom_type: int) -> np.ndarray:
        """Get positions of atoms with specified type."""
        mask = self.atom_types == atom_type
        return self.positions[mask]


class TrajectoryParser:
    """
    Parser for LAMMPS trajectory dump files.
    
    Supports the standard LAMMPS dump format with columns:
    id type x y z
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .lammpstrj file
    atom_type_map : dict, optional
        Mapping from atom type integers to element symbols.
        Default: {1: 'O', 2: 'H', 3: 'Si', 4: 'Si'}
    """
    
    def __init__(
        self, 
        filepath: str | Path,
        atom_type_map: Optional[Dict[int, str]] = None
    ):
        self.filepath = Path(filepath)
        self.atom_type_map = atom_type_map or {1: 'O', 2: 'H', 3: 'Si', 4: 'Si'}
        self._frames: List[Frame] = []
        self._parsed = False
    
    def parse(self) -> List[Frame]:
        """
        Parse the entire trajectory file.
        
        Returns
        -------
        List[Frame]
            List of Frame objects, one for each timestep
        """
        if self._parsed:
            return self._frames
        
        self._frames = list(self._parse_generator())
        self._parsed = True
        return self._frames
    
    def _parse_generator(self) -> Generator[Frame, None, None]:
        """Generator that yields frames one at a time."""
        with open(self.filepath, 'r') as f:
            while True:
                frame = self._read_frame(f)
                if frame is None:
                    break
                yield frame
    
    def _read_frame(self, file) -> Optional[Frame]:
        """Read a single frame from the file."""
        # Read ITEM: TIMESTEP
        line = file.readline()
        if not line:
            return None
        
        if 'ITEM: TIMESTEP' not in line:
            return None
        
        timestep = int(file.readline().strip())
        
        # Read ITEM: NUMBER OF ATOMS
        file.readline()  # ITEM: NUMBER OF ATOMS
        n_atoms = int(file.readline().strip())
        
        # Read ITEM: BOX BOUNDS
        file.readline()  # ITEM: BOX BOUNDS pp pp pp
        box_bounds = np.zeros((3, 2))
        for i in range(3):
            parts = file.readline().split()
            box_bounds[i, 0] = float(parts[0])
            box_bounds[i, 1] = float(parts[1])
        
        # Read ITEM: ATOMS
        file.readline()  # ITEM: ATOMS id type x y z
        
        atom_ids = np.zeros(n_atoms, dtype=int)
        atom_types = np.zeros(n_atoms, dtype=int)
        positions = np.zeros((n_atoms, 3))
        
        for i in range(n_atoms):
            parts = file.readline().split()
            atom_ids[i] = int(parts[0])
            atom_types[i] = int(parts[1])
            positions[i, 0] = float(parts[2])
            positions[i, 1] = float(parts[3])
            positions[i, 2] = float(parts[4])
        
        return Frame(
            timestep=timestep,
            n_atoms=n_atoms,
            box_bounds=box_bounds,
            atom_ids=atom_ids,
            atom_types=atom_types,
            positions=positions
        )
    
    def __len__(self) -> int:
        """Return number of frames in trajectory."""
        if not self._parsed:
            self.parse()
        return len(self._frames)
    
    def __getitem__(self, idx: int) -> Frame:
        """Get frame by index."""
        if not self._parsed:
            self.parse()
        return self._frames[idx]
    
    def __iter__(self):
        """Iterate over frames."""
        if not self._parsed:
            self.parse()
        return iter(self._frames)
    
    def get_element_symbol(self, atom_type: int) -> str:
        """Get element symbol for atom type."""
        return self.atom_type_map.get(atom_type, 'X')
