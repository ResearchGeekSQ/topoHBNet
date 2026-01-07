"""
Unified Trajectory Parser

Parses molecular dynamics trajectory files from various sources
(LAMMPS, CP2K, XYZ, VASP, etc.) using ASE as the unified backend.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Union
from pathlib import Path

# ASE imports
from ase.io import read, iread
from ase import Atoms
from ase.data import atomic_numbers


# Use atomic numbers from periodic table for element-to-type mapping
# H=1, He=2, Li=3, ... O=8, ... (all 118 elements supported)
DEFAULT_ELEMENT_TO_TYPE = atomic_numbers


@dataclass
class Frame:
    """
    A single trajectory frame containing atomic data.
    
    This is a unified frame structure that can be created from either:
    - ASE Atoms objects (recommended)
    - Legacy LAMMPS-style numeric data (backward compatible)
    
    Attributes
    ----------
    timestep : int
        Frame index or timestep number
    n_atoms : int
        Number of atoms in the frame
    box_bounds : np.ndarray
        Box boundaries, shape (3, 2) - [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    symbols : np.ndarray
        Element symbols for each atom (e.g., ['O', 'H', 'H', ...])
    positions : np.ndarray
        Atomic positions, shape (n_atoms, 3)
    atom_types : np.ndarray, optional
        Numeric atom types (for backward compatibility)
    atom_ids : np.ndarray, optional
        Atom IDs (for backward compatibility)
    """
    
    timestep: int
    n_atoms: int
    box_bounds: np.ndarray
    symbols: np.ndarray
    positions: np.ndarray
    atom_types: Optional[np.ndarray] = None
    atom_ids: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Ensure atom_types and atom_ids are set for backward compatibility."""
        if self.atom_ids is None:
            self.atom_ids = np.arange(1, self.n_atoms + 1)
        if self.atom_types is None:
            self.atom_types = np.array([
                DEFAULT_ELEMENT_TO_TYPE.get(s, 999) for s in self.symbols
            ])
    
    @property
    def box_lengths(self) -> np.ndarray:
        """Return box lengths in x, y, z directions."""
        return self.box_bounds[:, 1] - self.box_bounds[:, 0]
    
    def get_atoms_by_type(self, atom_type: int) -> np.ndarray:
        """Get indices of atoms with specified numeric type."""
        return np.where(self.atom_types == atom_type)[0]
    
    def get_atoms_by_symbol(self, symbol: str) -> np.ndarray:
        """Get indices of atoms with specified element symbol."""
        return np.where(self.symbols == symbol)[0]
    
    def get_positions_by_type(self, atom_type: int) -> np.ndarray:
        """Get positions of atoms with specified numeric type."""
        mask = self.atom_types == atom_type
        return self.positions[mask]
    
    def get_positions_by_symbol(self, symbol: str) -> np.ndarray:
        """Get positions of atoms with specified element symbol."""
        mask = self.symbols == symbol
        return self.positions[mask]
    
    @classmethod
    def from_ase_atoms(
        cls, 
        atoms: Atoms, 
        timestep: int = 0,
        element_to_type: Optional[Dict[str, int]] = None
    ) -> "Frame":
        """
        Create Frame from ASE Atoms object.
        
        Parameters
        ----------
        atoms : ase.Atoms
            ASE Atoms object
        timestep : int
            Frame index/timestep
        element_to_type : dict, optional
            Mapping from element symbols to numeric types
        
        Returns
        -------
        Frame
            Unified frame object
        """
        element_map = element_to_type or DEFAULT_ELEMENT_TO_TYPE
        
        # Extract cell/box information
        cell = atoms.get_cell()
        if cell.any():
            # Use cell diagonal as box bounds (works for orthorhombic cells)
            box_bounds = np.array([
                [0.0, cell[0, 0]],
                [0.0, cell[1, 1]],
                [0.0, cell[2, 2]]
            ])
        else:
            # No cell defined, use atom extent
            positions = atoms.get_positions()
            min_pos = positions.min(axis=0) - 1.0
            max_pos = positions.max(axis=0) + 1.0
            box_bounds = np.column_stack([min_pos, max_pos])
        
        symbols = np.array(atoms.get_chemical_symbols())
        atom_types = np.array([element_map.get(s, 999) for s in symbols])
        
        return cls(
            timestep=timestep,
            n_atoms=len(atoms),
            box_bounds=box_bounds,
            symbols=symbols,
            positions=atoms.get_positions(),
            atom_types=atom_types,
            atom_ids=np.arange(1, len(atoms) + 1)
        )


class TrajectoryParser:
    """
    Unified parser for molecular dynamics trajectories.
    
    Uses ASE as backend to support multiple trajectory formats:
    - LAMMPS dump (.lammpstrj, .dump)
    - CP2K (.xyz, .dcd)
    - XYZ (.xyz)
    - Extended XYZ (.extxyz)
    - VASP (POSCAR, CONTCAR, XDATCAR)
    - And 80+ other formats supported by ASE
    
    Parameters
    ----------
    filepath : str or Path
        Path to trajectory file
    format : str, optional
        File format (auto-detected if not specified).
        See https://wiki.fysik.dtu.dk/ase/ase/io/io.html for options.
    element_to_type : dict, optional
        Mapping from element symbols to numeric atom types.
        Default: {'O': 1, 'H': 2, ...}
    
    Examples
    --------
    # Auto-detect format
    >>> parser = TrajectoryParser("trajectory.lammpstrj")
    >>> frames = parser.parse()
    
    # Specify format explicitly
    >>> parser = TrajectoryParser("pos.xyz", format="xyz")
    >>> frames = parser.parse()
    
    # Iterate without loading all frames
    >>> for frame in TrajectoryParser("large_traj.xyz"):
    ...     process(frame)
    """
    
    def __init__(
        self, 
        filepath: Union[str, Path],
        format: Optional[str] = None,
        element_to_type: Optional[Dict[str, int]] = None
    ):
        self.filepath = Path(filepath)
        self.format = format
        # ASE sometimes needs explicit format for LAMMPS dump files
        if self.format is None:
            if self.filepath.suffix in ['.lammpstrj', '.dump']:
                self.format = 'lammps-dump-text'
        
        self.element_to_type = element_to_type or DEFAULT_ELEMENT_TO_TYPE
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
        
        # Read all frames using ASE
        atoms_list = read(str(self.filepath), index=':', format=self.format)
        
        # Handle single Atoms object (not a list)
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        
        self._frames = [
            Frame.from_ase_atoms(
                atoms, 
                atoms.info.get('timestep', atoms.info.get('time', i)), 
                self.element_to_type
            )
            for i, atoms in enumerate(atoms_list)
        ]
        self._parsed = True
        return self._frames
    
    def _parse_generator(self) -> Generator[Frame, None, None]:
        """Generator that yields frames one at a time (memory efficient)."""
        for i, atoms in enumerate(iread(str(self.filepath), format=self.format)):
            timestep = atoms.info.get('timestep', atoms.info.get('time', i))
            yield Frame.from_ase_atoms(atoms, timestep, self.element_to_type)
    
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
        """Iterate over frames (uses generator for memory efficiency if not parsed)."""
        if self._parsed:
            return iter(self._frames)
        return self._parse_generator()
    
    def get_element_symbol(self, atom_type: int) -> str:
        """Get element symbol for numeric atom type (reverse lookup)."""
        type_to_element = {v: k for k, v in self.element_to_type.items()}
        return type_to_element.get(atom_type, 'X')
