#!/usr/bin/env python3
"""
AtomForge Crystal v1.1 - Phase 0 Implementation

This module implements Phase 0 of the AtomForge Crystal MVP Plan:
- Crystal v1.1 + JSON Schema + typed adapters
- canonicalize - identity_hash - validate
- CI: CIF/POSCAR round-trip invariance
- symmetry equivalence - same hash

"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Literal, Any
from datetime import datetime
import hashlib
import json
import numpy as np
from pathlib import Path

# Type aliases
Vec3 = Tuple[float, float, float]            # fractional coords in [0,1)
Mat3 = Tuple[Tuple[int,int,int], Tuple[int,int,int], Tuple[int,int,int]]

# ============================================================================
# CORE CRYSTAL V1.1 DATA STRUCTURES (IMMUTABLE)
# ============================================================================

@dataclass(frozen=True)
class Lattice:
    """Lattice parameters defining the crystal unit cell"""
    a: float      # A
    b: float      # A
    c: float      # A
    alpha: float  # deg
    beta: float   # deg
    gamma: float  # deg

@dataclass(frozen=True)
class Symmetry:
    """Crystallographic symmetry information"""
    space_group: str                           # e.g., "P6_3/mmc"
    number: int                                # 1..230
    hall_symbol: Optional[str] = None
    origin_choice: Optional[str] = None
    symmetry_source: Optional[Literal["provided","inferred"]] = None
    crystal_system: Optional[str] = None       # e.g., "cubic", "monoclinic"

@dataclass(frozen=True)
class Site:
    """Atomic site in the crystal structure"""
    species: Dict[str, float]                  # {"Li":0.5,"La":0.5} (sum = 1.0)
    frac: Vec3                                 # fractional coord in [0,1)
    wyckoff: Optional[str] = None              # e.g., "4e" (filled by canonicalizer when possible)
    multiplicity: Optional[int] = None
    label: Optional[str] = None
    magnetic_moment: Optional[Tuple[float,float,float]] = None
    charge: Optional[float] = None
    disorder_group: Optional[str] = None       # correlate partial occupancies

@dataclass(frozen=True)
class Composition:
    """Chemical composition information"""
    reduced: Dict[str, int]                    # integer formula
    atomic_fractions: Dict[str, float]         # normalized to 1.0

@dataclass(frozen=True)
class ConstraintSet:
    """Physical and computational constraints"""
    min_interatomic_distance: Optional[float] = None   # A
    charge_neutrality: Optional[bool] = None
    symmetry_locked: Optional[bool] = None             # forbid symmetry-breaking edits

@dataclass(frozen=True)
class Provenance:
    """Complete provenance tracking information"""
    database: Optional[str] = None             # "MP","ICSD","COD","user"
    id: Optional[str] = None                   # e.g., "mp-12345"
    doi: Optional[str] = None
    retrieved_at: Optional[str] = None         # ISO 8601
    generator: Optional[str] = None            # if produced by a generator
    schema_version: str = "AtomForge/Crystal/1.1"
    hash: Optional[str] = None                 # canonical identity hash (set by canonicalizer)
    external_ids: Dict[str, str] = field(default_factory=dict)  # e.g., {"task_id":"mp-...","icsd":"..."} 

@dataclass(frozen=True)
class Crystal:
    """Complete crystal structure representation"""
    lattice: Lattice
    symmetry: Symmetry
    sites: Tuple[Site, ...]                    # symmetry-unique sites only
    composition: Composition
    oxidation_states: Optional[Dict[str, float]] = None  # species -> ox state (e.g., {"O":-2})
    constraints: Optional[ConstraintSet] = None
    provenance: Provenance = field(default_factory=Provenance)
    notes: Optional[str] = None

# ============================================================================
# PHASE 0 OPERATIONS: INGEST, CANONICALIZE, VALIDATE
# ============================================================================

@dataclass
class CanonReport:
    """Report from canonicalization process"""
    actions_taken: List[str]
    epsilon_used: float
    spglib_settings: Dict[str, Any]
    original_hash: Optional[str] = None
    canonical_hash: Optional[str] = None

@dataclass
class ValidationReport:
    """Report from validation process"""
    ok: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

# ============================================================================
# OPERATION 1: from_cif / from_poscar - Crystal
# ============================================================================

def from_cif(cif_path: str) -> Crystal:
    """
    Parse CIF file - minimal Crystal (symmetry may be inferred later).
    
    Args:
        cif_path: Path to CIF file
        
    Returns:
        Crystal structure
    """
    try:
        import pymatgen.io.cif as cif_io
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        raise ImportError("pymatgen is required for CIF parsing. Install with: pip install pymatgen")
    
    # Parse CIF file using pymatgen with increased occupancy tolerance
    # Suppress warnings about missing symmetry operations
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No _symmetry_equiv_pos_as_xyz type key found')
        warnings.filterwarnings('ignore', message='Issues encountered while parsing CIF')
        warnings.filterwarnings('ignore', message='Some occupancies')
        warnings.filterwarnings('ignore', message='The default value of primitive')
        
        cif_parser = cif_io.CifParser(cif_path, occupancy_tolerance=2.0)
        structures = cif_parser.parse_structures()
        if not structures:
            raise ValueError(f'No structures found in CIF file: {cif_path}')
        structure = structures[0]  # Get first structure
    
    # Extract lattice parameters
    lattice = Lattice(
        a=structure.lattice.a,
        b=structure.lattice.b,
        c=structure.lattice.c,
        alpha=structure.lattice.alpha,
        beta=structure.lattice.beta,
        gamma=structure.lattice.gamma
    )
    
    # Extract symmetry information using SpacegroupAnalyzer
    analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
    space_group_symbol = analyzer.get_space_group_symbol()
    space_group_number = analyzer.get_space_group_number()
    
    symmetry = Symmetry(
        space_group=space_group_symbol,
        number=space_group_number,
        symmetry_source="inferred"
    )
    
    # Extract sites
    sites = []
    for site in structure:
        # Convert species to occupancy dict
        species_dict = {}
        for species, occupancy in site.species.items():
            species_dict[str(species)] = occupancy
        
        # Get fractional coordinates
        frac_coords = tuple(site.frac_coords)
        
        # Create Site object
        site_obj = Site(
            species=species_dict,
            frac=frac_coords,
            wyckoff=None,  # Will be inferred during canonicalization
            multiplicity=None,  # Will be inferred during canonicalization
            label=site.label if hasattr(site, 'label') else None
        )
        sites.append(site_obj)
    
    # Compute composition
    composition_dict = structure.composition.as_dict()
    reduced_formula = structure.composition.reduced_formula
    
    # Convert to integer formula
    reduced = {}
    for element, count in composition_dict.items():
        reduced[element] = int(count) if count.is_integer() else count
    
    composition = Composition(
        reduced=reduced,
        atomic_fractions=composition_dict
    )
    
    # Create provenance
    provenance = Provenance(
        database="CIF",
        id=Path(cif_path).stem,
        retrieved_at=datetime.now().isoformat(),
        schema_version="AtomForge/Crystal/1.1"
    )
    
    return Crystal(
        lattice=lattice,
        symmetry=symmetry,
        sites=tuple(sites),
        composition=composition,
        provenance=provenance
    )

def from_poscar(poscar_path: str) -> Crystal:
    """
    Parse POSCAR file - minimal Crystal (symmetry may be inferred later).
    
    Args:
        poscar_path: Path to POSCAR file
        
    Returns:
        Crystal structure
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.io.vasp import Poscar
    except ImportError:
        raise ImportError("pymatgen is required for POSCAR parsing. Install with: pip install pymatgen")
    
    # Parse POSCAR file using pymatgen
    poscar = Poscar.from_file(poscar_path)
    structure = poscar.structure
    
    # Extract lattice parameters
    lattice = Lattice(
        a=structure.lattice.a,
        b=structure.lattice.b,
        c=structure.lattice.c,
        alpha=structure.lattice.alpha,
        beta=structure.lattice.beta,
        gamma=structure.lattice.gamma
    )
    
    # Extract symmetry information (will be inferred during canonicalization)
    symmetry = Symmetry(
        space_group="P1",  # Default, will be determined by spglib
        number=1,
        symmetry_source="inferred"
    )
    
    # Extract sites
    sites = []
    for site in structure:
        # Convert species to occupancy dict
        species_dict = {}
        for species, occupancy in site.species.items():
            species_dict[str(species)] = occupancy
        
        # Get fractional coordinates
        frac_coords = tuple(site.frac_coords)
        
        # Create Site object
        site_obj = Site(
            species=species_dict,
            frac=frac_coords,
            wyckoff=None,  # Will be inferred during canonicalization
            multiplicity=None,  # Will be inferred during canonicalization
            label=site.label if hasattr(site, 'label') else None
        )
        sites.append(site_obj)
    
    # Compute composition
    composition_dict = structure.composition.as_dict()
    
    # Convert to integer formula
    reduced = {}
    for element, count in composition_dict.items():
        reduced[element] = int(count) if count.is_integer() else count
    
    composition = Composition(
        reduced=reduced,
        atomic_fractions=composition_dict
    )
    
    # Create provenance
    provenance = Provenance(
        database="POSCAR",
        id=Path(poscar_path).stem,
        retrieved_at=datetime.now().isoformat(),
        schema_version="AtomForge/Crystal/1.1"
    )
    
    return Crystal(
        lattice=lattice,
        symmetry=symmetry,
        sites=tuple(sites),
        composition=composition,
        provenance=provenance
    )

# ============================================================================
# OPERATION 2: canonicalize(crystal, policy) - Crystal, CanonReport
# ============================================================================

class CrystalCanonicalizer:
    """Canonicalization engine for crystal structures"""
    
    def __init__(self, epsilon_lattice: float = 1e-6, epsilon_coords: float = 1e-8):
        self.epsilon_lattice = epsilon_lattice
        self.epsilon_coords = epsilon_coords
    
    def canonicalize(self, crystal: Crystal, policy: str = "conventional") -> Tuple[Crystal, CanonReport]:
        """
        Canonicalize a crystal structure.
        
        Steps:
        1. Reduce to primitive cell (spglib), then convert to conventional setting
        2. Wrap fractional coords to [0,1) and quantize lattice and coords with fixed epsilons
        3. Infer wyckoff and multiplicity where possible; sort sites by (species tuple - wyckoff - frac)
        4. Strip non-physical labels; normalize element symbols consistently
        5. Set provenance.hash
        
        Args:
            crystal: Input crystal structure
            policy: Canonicalization policy ("primitive", "conventional", "standard")
            
        Returns:
            Tuple of (canonicalized_crystal, canonicalization_report)
        """
        actions = []
        
        # Step 1: Reduce to primitive cell (if needed)
        if policy in ["primitive", "conventional"]:
            crystal = self._reduce_to_primitive(crystal)
            actions.append("reduced_to_primitive")
        
        # Step 2: Convert to conventional setting (if needed)
        if policy == "conventional":
            crystal = self._to_conventional(crystal)
            actions.append("converted_to_conventional")
        
        # Step 3: Wrap fractional coordinates to [0,1)
        crystal = self._wrap_coordinates(crystal)
        actions.append("wrapped_coordinates")
        
        # Step 4: Quantize lattice and coordinates
        crystal = self._quantize_structure(crystal)
        actions.append("quantized_structure")
        
        # Step 5: Infer Wyckoff positions and multiplicity
        crystal = self._infer_wyckoff(crystal)
        actions.append("inferred_wyckoff")
        
        # Step 6: Sort sites by (species tuple - wyckoff - frac)
        crystal = self._sort_sites(crystal)
        actions.append("sorted_sites")
        
        # Step 7: Compute identity hash and set provenance.hash
        identity_hash = self._compute_identity_hash(crystal)
        crystal = self._update_provenance_hash(crystal, identity_hash)
        actions.append("computed_identity_hash")
        
        # Create report
        report = CanonReport(
            actions_taken=actions,
            epsilon_used=self.epsilon_coords,
            spglib_settings={"policy": policy},
            canonical_hash=identity_hash
        )
        
        return crystal, report
    
    def _reduce_to_primitive(self, crystal: Crystal) -> Crystal:
        """Reduce to primitive cell using spglib"""
        try:
            import spglib
            from pymatgen.core import Structure, Lattice as PMGLattice
            from pymatgen.core.periodic_table import Element
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError:
            # If spglib/pymatgen not available, return as-is
            return crystal
        
        # Convert Crystal to pymatgen Structure for spglib analysis
        # Create lattice
        pmg_lattice = PMGLattice.from_parameters(
            crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
            crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma
        )
        
        # Create species and coords lists
        species = []
        coords = []
        for site in crystal.sites:
            for element, occupancy in site.species.items():
                species.append(element)
                coords.append(list(site.frac))
        
        # Create pymatgen structure
        structure = Structure(pmg_lattice, species, coords)
        
        # Use spglib to find primitive cell
        spglib_cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
        primitive_cell = spglib.find_primitive(spglib_cell, symprec=1e-5)
        
        if primitive_cell is None:
            # If spglib fails, return original structure
            return crystal
        
        # Convert back to Crystal format
        primitive_lattice_matrix, primitive_coords, primitive_numbers = primitive_cell
        
        # Convert lattice matrix to parameters
        primitive_lattice = PMGLattice(primitive_lattice_matrix)
        
        # Create new lattice
        new_lattice = Lattice(
            a=primitive_lattice.a,
            b=primitive_lattice.b,
            c=primitive_lattice.c,
            alpha=primitive_lattice.alpha,
            beta=primitive_lattice.beta,
            gamma=primitive_lattice.gamma
        )
        
        # Create new sites
        new_sites = []
        for i, (coords, atomic_number) in enumerate(zip(primitive_coords, primitive_numbers)):
            # Get element symbol from atomic number returned by spglib
            try:
                element = Element.from_Z(int(atomic_number)).symbol
            except Exception:
                # Fallback to original species list if Element lookup fails
                element = structure.species[i].symbol
            
            site = Site(
                species={element: 1.0},
                frac=tuple(coords),
                wyckoff=None,  # Will be inferred later
                multiplicity=None,
                label=None
            )
            new_sites.append(site)
        
        # Update symmetry information using SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
        
        new_symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            symmetry_source="inferred"
        )
        
        # Recompute composition
        composition_dict = {}
        for site in new_sites:
            for element, occupancy in site.species.items():
                composition_dict[element] = composition_dict.get(element, 0) + occupancy
        
        composition = Composition(
            reduced=composition_dict,
            atomic_fractions=composition_dict
        )
        
        return Crystal(
            lattice=new_lattice,
            symmetry=new_symmetry,
            sites=tuple(new_sites),
            composition=composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _to_conventional(self, crystal: Crystal) -> Crystal:
        """Convert to conventional setting using spglib"""
        try:
            import spglib
            from pymatgen.core import Structure, Lattice as PMGLattice
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError:
            # If spglib/pymatgen not available, return as-is
            return crystal
        
        # Convert Crystal to pymatgen Structure
        pmg_lattice = PMGLattice.from_parameters(
            crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
            crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma
        )
        
        species = []
        coords = []
        for site in crystal.sites:
            for element, occupancy in site.species.items():
                species.append(element)
                coords.append(list(site.frac))
        
        structure = Structure(pmg_lattice, species, coords)
        
        # Build conventional standard structure via pymatgen
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
        conventional = analyzer.get_conventional_standard_structure()
        if conventional is None:
            # Fallback: return original crystal
            return crystal
        
        # Build new Crystal from conventional structure
        new_lattice = Lattice(
            a=conventional.lattice.a,
            b=conventional.lattice.b,
            c=conventional.lattice.c,
            alpha=conventional.lattice.alpha,
            beta=conventional.lattice.beta,
            gamma=conventional.lattice.gamma
        )
        
        new_sites: List[Site] = []
        for site in conventional:
            species_dict: Dict[str, float] = {}
            for sp, occ in site.species.items():
                species_dict[str(sp)] = occ
            new_sites.append(
                Site(
                    species=species_dict,
                    frac=tuple(site.frac_coords),
                    wyckoff=None,
                    multiplicity=None,
                    label=site.label if hasattr(site, 'label') else None
                )
            )
        
        # Recompute composition from conventional structure
        composition_dict = conventional.composition.as_dict()
        reduced: Dict[str, Any] = {}
        for element, count in composition_dict.items():
            try:
                reduced[element] = int(count) if float(count).is_integer() else float(count)
            except Exception:
                reduced[element] = count
        new_composition = Composition(
            reduced=reduced,
            atomic_fractions=composition_dict
        )
        
        # Update symmetry info from analyzer
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
        new_symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            symmetry_source="inferred"
        )
        
        return Crystal(
            lattice=new_lattice,
            symmetry=new_symmetry,
            sites=tuple(new_sites),
            composition=new_composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _wrap_coordinates(self, crystal: Crystal) -> Crystal:
        """Wrap fractional coordinates to [0,1)"""
        new_sites = []
        for site in crystal.sites:
            wrapped_frac = tuple(coord % 1.0 for coord in site.frac)
            new_site = Site(
                species=site.species,
                frac=wrapped_frac,
                wyckoff=site.wyckoff,
                multiplicity=site.multiplicity,
                label=site.label,
                magnetic_moment=site.magnetic_moment,
                charge=site.charge,
                disorder_group=site.disorder_group
            )
            new_sites.append(new_site)
        
        return Crystal(
            lattice=crystal.lattice,
            symmetry=crystal.symmetry,
            sites=tuple(new_sites),
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _quantize_structure(self, crystal: Crystal) -> Crystal:
        """Quantize lattice and coordinates with fixed epsilons"""
        # Quantize lattice parameters
        quantized_lattice = Lattice(
            a=round(crystal.lattice.a / self.epsilon_lattice) * self.epsilon_lattice,
            b=round(crystal.lattice.b / self.epsilon_lattice) * self.epsilon_lattice,
            c=round(crystal.lattice.c / self.epsilon_lattice) * self.epsilon_lattice,
            alpha=round(crystal.lattice.alpha / self.epsilon_coords) * self.epsilon_coords,
            beta=round(crystal.lattice.beta / self.epsilon_coords) * self.epsilon_coords,
            gamma=round(crystal.lattice.gamma / self.epsilon_coords) * self.epsilon_coords
        )
        
        # Quantize site coordinates
        new_sites = []
        for site in crystal.sites:
            quantized_frac = tuple(
                round(coord / self.epsilon_coords) * self.epsilon_coords 
                for coord in site.frac
            )
            new_site = Site(
                species=site.species,
                frac=quantized_frac,
                wyckoff=site.wyckoff,
                multiplicity=site.multiplicity,
                label=site.label,
                magnetic_moment=site.magnetic_moment,
                charge=site.charge,
                disorder_group=site.disorder_group
            )
            new_sites.append(new_site)
        
        return Crystal(
            lattice=quantized_lattice,
            symmetry=crystal.symmetry,
            sites=tuple(new_sites),
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _infer_wyckoff(self, crystal: Crystal) -> Crystal:
        """Infer Wyckoff positions and multiplicity using spglib"""
        try:
            import spglib
            from pymatgen.core import Structure, Lattice as PMGLattice
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError:
            # If spglib/pymatgen not available, return as-is
            return crystal
        
        # Convert Crystal to pymatgen Structure
        pmg_lattice = PMGLattice.from_parameters(
            crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
            crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma
        )
        
        species = []
        coords = []
        for site in crystal.sites:
            for element, occupancy in site.species.items():
                species.append(element)
                coords.append(list(site.frac))
        
        structure = Structure(pmg_lattice, species, coords)
        
        # Use spglib to get symmetry dataset
        spglib_cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
        symmetry_dataset = spglib.get_symmetry_dataset(spglib_cell, symprec=1e-5)
        
        if symmetry_dataset is None:
            # If spglib fails, return original structure
            return crystal
        
        # Get Wyckoff positions and multiplicities using attribute interface
        wyckoff_letters = getattr(symmetry_dataset, 'wyckoffs', [])
        equivalent_atoms = getattr(symmetry_dataset, 'equivalent_atoms', [])
        
        # Create new sites with Wyckoff information
        new_sites = []
        
        # Map from original sites to spglib atoms (handle multiple species per site)
        site_to_atoms = {}
        atom_index = 0
        for site_idx, site in enumerate(crystal.sites):
            site_atoms = []
            for element, occupancy in site.species.items():
                site_atoms.append(atom_index)
                atom_index += 1
            site_to_atoms[site_idx] = site_atoms
        
        for i, site in enumerate(crystal.sites):
            # Get the atom indices for this site
            atom_indices = site_to_atoms[i]
            
            if atom_indices and atom_indices[0] < len(wyckoff_letters):
                # Get Wyckoff position for the first atom of this site
                wyckoff = wyckoff_letters[atom_indices[0]]
                
                # Calculate multiplicity based on equivalent atoms
                if atom_indices[0] < len(equivalent_atoms):
                    equivalent_group = equivalent_atoms[atom_indices[0]]
                    multiplicity = sum(1 for eq in equivalent_atoms if eq == equivalent_group)
                else:
                    multiplicity = 1
                
                # Create formatted Wyckoff position (e.g., "2a", "4b")
                wyckoff_formatted = f"{multiplicity}{wyckoff}"
            else:
                wyckoff_formatted = None
                multiplicity = None
            
            new_site = Site(
                species=site.species,
                frac=site.frac,
                wyckoff=wyckoff_formatted,
                multiplicity=multiplicity,
                label=site.label,
                magnetic_moment=site.magnetic_moment,
                charge=site.charge,
                disorder_group=site.disorder_group
            )
            new_sites.append(new_site)
        
        # Update symmetry information with more details using SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
        
        new_symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            hall_symbol=str(getattr(symmetry_dataset, 'hall_number', '')),
            origin_choice=str(getattr(symmetry_dataset, 'origin_choice', '')),
            symmetry_source="inferred"
        )
        
        return Crystal(
            lattice=crystal.lattice,
            symmetry=new_symmetry,
            sites=tuple(new_sites),
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _sort_sites(self, crystal: Crystal) -> Crystal:
        """Sort sites by (species tuple - wyckoff - frac)"""
        def site_key(site: Site) -> Tuple[Tuple[str, ...], str, Vec3]:
            species_tuple = tuple(sorted(site.species.keys()))
            wyckoff = site.wyckoff or ""
            return (species_tuple, wyckoff, site.frac)
        
        sorted_sites = sorted(crystal.sites, key=site_key)
        
        return Crystal(
            lattice=crystal.lattice,
            symmetry=crystal.symmetry,
            sites=tuple(sorted_sites),
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
    
    def _compute_identity_hash(self, crystal: Crystal) -> str:
        """Compute SHA-256 identity hash over canonical byte string"""
        # Create canonical string representation
        canonical_parts = [
            f"{crystal.symmetry.number}|{crystal.symmetry.space_group}",
            f"lattice:{crystal.lattice.a:.8f},{crystal.lattice.b:.8f},{crystal.lattice.c:.8f},{crystal.lattice.alpha:.8f},{crystal.lattice.beta:.8f},{crystal.lattice.gamma:.8f}"
        ]
        
        # Add sorted sites
        for site in crystal.sites:
            species_str = ",".join(f"{k}:{v:.8f}" for k, v in sorted(site.species.items()))
            frac_str = f"{site.frac[0]:.8f},{site.frac[1]:.8f},{site.frac[2]:.8f}"
            wyckoff_str = site.wyckoff or ""
            multiplicity_str = str(site.multiplicity) if site.multiplicity else ""
            
            site_str = f"site:{species_str}|{frac_str}|{wyckoff_str}|{multiplicity_str}"
            canonical_parts.append(site_str)
        
        canonical_string = "|".join(canonical_parts)
        return hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
    
    def _update_provenance_hash(self, crystal: Crystal, identity_hash: str) -> Crystal:
        """Update provenance with identity hash"""
        new_provenance = Provenance(
            database=crystal.provenance.database,
            id=crystal.provenance.id,
            doi=crystal.provenance.doi,
            retrieved_at=crystal.provenance.retrieved_at,
            generator=crystal.provenance.generator,
            schema_version=crystal.provenance.schema_version,
            hash=identity_hash,
            external_ids=crystal.provenance.external_ids
        )
        
        return Crystal(
            lattice=crystal.lattice,
            symmetry=crystal.symmetry,
            sites=crystal.sites,
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=new_provenance,
            notes=crystal.notes
        )

# ============================================================================
# OPERATION 3: validate(crystal, rules) - ValidationReport
# ============================================================================

class CrystalValidator:
    """Validation engine for crystal structures"""
    
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
    
    def validate(self, crystal: Crystal, rules: Dict[str, Any] = None) -> ValidationReport:
        """
        Validate a crystal structure against all rules.
        
        Validation checks:
        - Metric PD: cell metric tensor must be positive definite; angles in valid ranges
        - Species occupancies: for every Site, sum(occupancy) == 1 ± tolerance
        - Composition consistency: recompute formula from (sites times multiplicity) and assert match with composition within tolerance
        - Disorder groups: if disorder_group used, correlated occupancies per group obey domain rule (<= 1.0; exactly 1.0 if mutually exclusive)
        - Charge neutrality: if constraints.charge_neutrality == True, validate using oxidation_states
        - Symmetry coherence: reported Symmetry agrees with spglib analysis of the canonical structure
        - Fractional bounds: no value equals 1.0 after wrapping; negatives corrected by wrap rule
        
        Args:
            crystal: Crystal structure to validate
            rules: Validation rules configuration
            
        Returns:
            Validation report with results
        """
        if rules is None:
            rules = self._get_default_rules()
        
        errors = []
        warnings = []
        details = {}
        
        # Run all validation checks
        checks = [
            self._check_metric_pd,
            self._check_species_occupancies,
            self._check_composition_consistency,
            self._check_disorder_groups,
            self._check_charge_neutrality,
            self._check_symmetry_coherence,
            self._check_fractional_bounds,
        ]
        
        for check in checks:
            try:
                result = check(crystal, rules)
                if result:
                    if result.get('error'):
                        errors.append(result['error'])
                    if result.get('warning'):
                        warnings.append(result['warning'])
                    if result.get('details'):
                        details.update(result['details'])
            except Exception as e:
                errors.append(f"Validation check failed: {str(e)}")
        
        return ValidationReport(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default validation rules"""
        return {
            'check_metric_pd': True,
            'check_occupancies': True,
            'check_composition': True,
            'check_disorder': True,
            'check_charge_neutrality': True,
            'check_symmetry': True,
            'check_bounds': True,
            'tolerance': self.tolerance
        }
    
    def _check_metric_pd(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check that cell metric tensor is positive definite"""
        # TODO: Implement metric tensor validation
        # For now, basic angle checks
        angles = [crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma]
        for angle in angles:
            if angle <= 0 or angle >= 180:
                return {
                    'error': f"Invalid lattice angle: {angle} degrees",
                    'details': {'invalid_angle': angle}
                }
        return None
    
    def _check_species_occupancies(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check that species occupancies sum to 1.0 ± tolerance"""
        tolerance = rules.get('tolerance', self.tolerance)
        
        for i, site in enumerate(crystal.sites):
            occupancy_sum = sum(site.species.values())
            if abs(occupancy_sum - 1.0) > tolerance:
                return {
                    'error': f"Site {i}: occupancy sum {occupancy_sum:.8f} != 1.0",
                    'details': {'site_index': i, 'occupancy_sum': occupancy_sum}
                }
        return None
    
    def _check_composition_consistency(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check composition consistency between sites and composition field"""
        # TODO: Implement composition consistency check
        return None
    
    def _check_disorder_groups(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check disorder group correlations"""
        # TODO: Implement disorder group validation
        return None
    
    def _check_charge_neutrality(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check charge neutrality if enforced"""
        if not crystal.constraints or not crystal.constraints.charge_neutrality:
            return None
        
        if not crystal.oxidation_states:
            return {
                'error': "Charge neutrality enforced but no oxidation states provided",
                'details': {'constraint': 'charge_neutrality'}
            }
        
        # TODO: Implement charge neutrality calculation
        return None
    
    def _check_symmetry_coherence(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check that reported symmetry agrees with spglib analysis"""
        # TODO: Implement symmetry coherence check using spglib
        return None
    
    def _check_fractional_bounds(self, crystal: Crystal, rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check that fractional coordinates are properly bounded"""
        tolerance = rules.get('tolerance', self.tolerance)
        
        for i, site in enumerate(crystal.sites):
            for j, coord in enumerate(site.frac):
                if coord < 0 or coord >= 1.0:
                    return {
                        'error': f"Site {i}: coordinate {j} = {coord} not in [0,1)",
                        'details': {'site_index': i, 'coordinate_index': j, 'value': coord}
                    }
        return None

# ============================================================================
# OPERATION 4: identity_hash(crystal) - str
# ============================================================================

def identity_hash(crystal: Crystal) -> str:
    """
    Compute SHA-256 over canonical byte string; used as cache key across all ops.
    
    Args:
        crystal: Crystal structure
        
    Returns:
        SHA-256 hash string
    """
    canonicalizer = CrystalCanonicalizer()
    return canonicalizer._compute_identity_hash(crystal)

# ============================================================================
# JSON SCHEMA VALIDATION
# ============================================================================

class CrystalSchemaValidator:
    """JSON Schema validation for Crystal v1.1"""
    
    def __init__(self):
        self.schema_path = Path(__file__).parent.parent / "atomforge_crystal_schema.json"
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema"""
        with open(self.schema_path, 'r') as f:
            return json.load(f)
    
    def validate_json(self, data: Dict[str, Any]) -> ValidationReport:
        """Validate JSON data against the schema"""
        # TODO: Implement JSON schema validation using jsonschema library
        # For now, basic structure validation
        errors = []
        warnings = []
        details = {}
        
        required_fields = ["lattice", "symmetry", "sites", "composition", "provenance"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationReport(ok=False, errors=errors, warnings=warnings, details=details)
        
        # Basic type checks
        if not isinstance(data.get("lattice"), dict):
            errors.append("lattice must be an object")
        
        if not isinstance(data.get("sites"), list):
            errors.append("sites must be an array")
        
        return ValidationReport(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details=details
        )

# ============================================================================
# TYPED ADAPTERS
# ============================================================================

class CrystalAdapter:
    """Adapter for converting between Crystal objects and other formats"""
    
    @staticmethod
    def to_dict(crystal: Crystal) -> Dict[str, Any]:
        """Convert Crystal to dictionary"""
        return asdict(crystal)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Crystal:
        """Convert dictionary to Crystal"""
        # Convert lattice
        lattice_data = data["lattice"]
        lattice = Lattice(
            a=lattice_data["a"],
            b=lattice_data["b"],
            c=lattice_data["c"],
            alpha=lattice_data["alpha"],
            beta=lattice_data["beta"],
            gamma=lattice_data["gamma"]
        )
        
        # Convert symmetry
        symmetry_data = data["symmetry"]
        symmetry = Symmetry(
            space_group=symmetry_data["space_group"],
            number=symmetry_data["number"],
            hall_symbol=symmetry_data.get("hall_symbol"),
            origin_choice=symmetry_data.get("origin_choice"),
            symmetry_source=symmetry_data.get("symmetry_source")
        )
        
        # Convert sites
        sites = []
        for site_data in data["sites"]:
            site = Site(
                species=site_data["species"],
                frac=tuple(site_data["frac"]),
                wyckoff=site_data.get("wyckoff"),
                multiplicity=site_data.get("multiplicity"),
                label=site_data.get("label"),
                magnetic_moment=tuple(site_data["magnetic_moment"]) if site_data.get("magnetic_moment") else None,
                charge=site_data.get("charge"),
                disorder_group=site_data.get("disorder_group")
            )
            sites.append(site)
        
        # Convert composition
        comp_data = data["composition"]
        composition = Composition(
            reduced=comp_data["reduced"],
            atomic_fractions=comp_data["atomic_fractions"]
        )
        
        # Convert provenance
        prov_data = data["provenance"]
        provenance = Provenance(
            database=prov_data.get("database"),
            id=prov_data.get("id"),
            doi=prov_data.get("doi"),
            retrieved_at=prov_data.get("retrieved_at"),
            generator=prov_data.get("generator"),
            schema_version=prov_data.get("schema_version", "AtomForge/Crystal/1.1"),
            hash=prov_data.get("hash"),
            external_ids=prov_data.get("external_ids", {})
        )
        
        # Create crystal
        return Crystal(
            lattice=lattice,
            symmetry=symmetry,
            sites=tuple(sites),
            composition=composition,
            oxidation_states=data.get("oxidation_states"),
            constraints=data.get("constraints"),
            provenance=provenance,
            notes=data.get("notes")
        )
    
    @staticmethod
    def to_json(crystal: Crystal) -> str:
        """Convert Crystal to JSON string"""
        return json.dumps(CrystalAdapter.to_dict(crystal), indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> Crystal:
        """Convert JSON string to Crystal"""
        data = json.loads(json_str)
        return CrystalAdapter.from_dict(data)

# ============================================================================
# MAIN PHASE 0 OPERATIONS (API)
# ============================================================================

def canonicalize(crystal: Crystal, policy: str = "conventional") -> Tuple[Crystal, CanonReport]:
    """
    Canonicalize a crystal structure.
    
    Args:
        crystal: Input crystal structure
        policy: Canonicalization policy ("primitive", "conventional", "standard")
        
    Returns:
        Tuple of (canonicalized_crystal, canonicalization_report)
    """
    canonicalizer = CrystalCanonicalizer()
    return canonicalizer.canonicalize(crystal, policy)

def validate(crystal: Crystal, rules: Dict[str, Any] = None) -> ValidationReport:
    """
    Validate a crystal structure.
    
    Args:
        crystal: Crystal structure to validate
        rules: Validation rules configuration
        
    Returns:
        Validation report with results
    """
    validator = CrystalValidator()
    return validator.validate(crystal, rules)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_simple_crystal(
    lattice_params: Tuple[float, float, float, float, float, float],
    sites: List[Tuple[str, Vec3]],
    space_group: str = "P1",
    space_group_number: int = 1
) -> Crystal:
    """Create a simple crystal structure for testing"""
    
    # Create lattice
    a, b, c, alpha, beta, gamma = lattice_params
    lattice = Lattice(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    
    # Create symmetry
    symmetry = Symmetry(
        space_group=space_group,
        number=space_group_number
    )
    
    # Create sites
    crystal_sites = []
    for element, frac_coords in sites:
        site = Site(
            species={element: 1.0},
            frac=frac_coords
        )
        crystal_sites.append(site)
    
    # Create composition
    elements = [site[0] for site in sites]
    element_counts = {}
    for element in elements:
        element_counts[element] = element_counts.get(element, 0) + 1
    
    composition = Composition(
        reduced=element_counts,
        atomic_fractions={element: count/len(elements) for element, count in element_counts.items()}
    )
    
    # Create provenance
    provenance = Provenance(
        schema_version="AtomForge/Crystal/1.1"
    )
    
    return Crystal(
        lattice=lattice,
        symmetry=symmetry,
        sites=tuple(crystal_sites),
        composition=composition,
        provenance=provenance
    ) 