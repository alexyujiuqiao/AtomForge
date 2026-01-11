#!/usr/bin/env python3
"""
Format Conversion Utilities

Converts between AtomForge, CIF, POSCAR formats.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

# Import AtomForge parser
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
    from atomforge.src.atomforge_ir import Length, Angle
except ImportError:
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser
    from atomforge_ir import Length, Angle


def get_val(x, length_unit: Optional[str] = None, angle_unit: Optional[str] = None) -> float:
    """
    Extract value from Length/Angle objects with unit conversion.
    
    Args:
        x: Value to extract (Length, Angle, or numeric)
        length_unit: Unit for length conversion (from program.units.length or Length.unit)
        angle_unit: Unit for angle conversion (from program.units.angle or Angle.unit)
    
    Returns:
        Converted value in Angstroms (for Length) or degrees (for Angle)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Unit conversion factors to Angstroms
    LENGTH_CONVERSIONS = {
        "angstrom": 1.0,
        "nm": 10.0,
        "pm": 0.01,
        "bohr": 0.529177,
        "a0": 0.529177,  # Bohr radius
    }
    
    # Unit conversion factors to degrees
    ANGLE_CONVERSIONS = {
        "degree": 1.0,
        "deg": 1.0,
        "radian": 57.29577951308232,  # 180/pi
        "rad": 57.29577951308232,
    }
    
    if isinstance(x, Length):
        value = float(x.value)
        # Check if Length object has its own unit
        unit = getattr(x, "unit", None) or length_unit
        if unit:
            unit_lower = unit.lower()
            if unit_lower in LENGTH_CONVERSIONS:
                return value * LENGTH_CONVERSIONS[unit_lower]
            else:
                logger.warning(f"Unknown length unit '{unit}', assuming Angstroms")
                return value
        else:
            # No unit specified, assume Angstroms (pymatgen default)
            return value
    elif isinstance(x, Angle):
        value = float(x.value)
        # Check if Angle object has its own unit
        unit = getattr(x, "unit", None) or angle_unit
        if unit:
            unit_lower = unit.lower()
            if unit_lower in ANGLE_CONVERSIONS:
                return value * ANGLE_CONVERSIONS[unit_lower]
            else:
                logger.warning(f"Unknown angle unit '{unit}', assuming degrees")
                return value
        else:
            # No unit specified, assume degrees (pymatgen default)
            return value
    return float(x)


def atomforge_to_structure(program, expand_symmetry: bool = True, symprec: float = 0.2, auto_detect_expanded: bool = True) -> tuple:
    """
    Convert AtomForge program to pymatgen Structure.
    
    Args:
        program: Parsed AtomForgeProgram
        expand_symmetry: Whether to expand symmetry (default: True). If False, never expand.
            If True and auto_detect_expanded=True, may skip expansion for already-expanded structures.
        symprec: Symmetry tolerance for expansion (default: 0.2)
        auto_detect_expanded: If True, auto-detect already-expanded structures and skip expansion
    
    Returns:
        (structure, metadata) tuple where metadata is a dict with:
        - n_input_sites: Number of input sites
        - used_symmetry_expansion: Whether symmetry expansion was used
        - auto_skipped: Whether expansion was auto-skipped
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen required")
    
    from pymatgen.core import Lattice as PmgLattice
    
    lat = program.lattice.bravais if program.lattice else None
    if lat is None:
        raise ValueError("Missing lattice bravais parameters")
    
    # Get units from program if available
    length_unit = None
    angle_unit = None
    if program.units:
        length_unit = getattr(program.units, "length", None)
        angle_unit = getattr(program.units, "angle", None)
    
    lattice = PmgLattice.from_parameters(
        get_val(lat.a, length_unit=length_unit),
        get_val(lat.b, length_unit=length_unit),
        get_val(lat.c, length_unit=length_unit),
        get_val(lat.alpha, angle_unit=angle_unit),
        get_val(lat.beta, angle_unit=angle_unit),
        get_val(lat.gamma, angle_unit=angle_unit),
    )
    
    species_all = []
    coords_all = []
    
    if not program.basis or not program.basis.sites:
        raise ValueError("Missing basis sites")
    
    n_input_sites = len(program.basis.sites)
    
    # Calculate total occupancy-weighted sites for auto-detection
    total_weighted_sites = 0
    for site in program.basis.sites:
        if site.species:
            for sp in site.species:
                total_weighted_sites += float(sp.occupancy)
    
    # Determine if we should expand
    actually_expand = False
    auto_skipped = False
    
    if not expand_symmetry:
        # Explicitly disabled
        actually_expand = False
        sites_to_use = program.basis.sites
    elif auto_detect_expanded and (n_input_sites >= 20 or total_weighted_sites >= 40):
        # Auto-detect: likely already expanded
        actually_expand = False
        auto_skipped = True
        sites_to_use = program.basis.sites
        logger.warning(f"Auto-skipping symmetry expansion: n_sites={n_input_sites}, weighted_sites={total_weighted_sites:.1f}")
    else:
        # Expand: deduplicate by Wyckoff position
        actually_expand = True
        # Group sites by Wyckoff position
        wyckoff_groups = {}
        for site in program.basis.sites:
            wyckoff = getattr(site, "wyckoff", None)
            if wyckoff:
                if wyckoff not in wyckoff_groups:
                    wyckoff_groups[wyckoff] = []
                wyckoff_groups[wyckoff].append(site)
        
        # If we have multiple sites with the same Wyckoff label, they're symmetry mates
        # Use only the first one for expansion
        if wyckoff_groups and any(len(sites) > 1 for sites in wyckoff_groups.values()):
            sites_to_use = []
            for wyckoff, sites in wyckoff_groups.items():
                sites_to_use.append(sites[0])  # Use first site of each Wyckoff group
        else:
            # All sites have unique Wyckoff positions (or no Wyckoff labels)
            sites_to_use = program.basis.sites
    
    for site in sites_to_use:
        pos = site.position
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except Exception:
            raise ValueError(f"Invalid position for site {site.name}: {pos}")
        
        if site.frame == "cartesian":
            fx, fy, fz = lattice.get_fractional_coords([x, y, z])
        else:
            fx, fy, fz = x, y, z
        
        if not site.species:
            raise ValueError(f"Site {site.name} has no species")
        
        if len(site.species) == 1 and abs(site.species[0].occupancy - 1.0) < 1e-6:
            species_all.append(site.species[0].element)
            coords_all.append((fx, fy, fz))
        else:
            # Partial occupancy
            for sp in site.species:
                occ = float(sp.occupancy)
                if occ > 1e-6:
                    species_all.append({sp.element: occ})
                    coords_all.append((fx, fy, fz))
    
    sg = program.symmetry.space_group if program.symmetry else None
    if sg is None:
        raise ValueError("Missing symmetry space_group")
    
    if actually_expand:
        # Try with symprec, fallback to tol if needed
        try:
            structure = Structure.from_spacegroup(
                sg,
                lattice,
                species_all,
                coords_all,
                symprec=symprec,
            )
        except TypeError:
            # Older pymatgen versions use tol instead of symprec
            structure = Structure.from_spacegroup(
                sg,
                lattice,
                species_all,
                coords_all,
                tol=symprec,
            )
    else:
        # Create structure without expansion (just asymmetric unit)
        structure = Structure(lattice, species_all, coords_all, coords_are_cartesian=False)
    
    metadata = {
        "n_input_sites": n_input_sites,
        "used_symmetry_expansion": actually_expand,
        "auto_skipped": auto_skipped,
    }
    
    return structure, metadata


def cif_to_structure(cif_path: Path) -> Structure:
    """Load structure from CIF file."""
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen required")
    return Structure.from_file(str(cif_path))


def poscar_to_structure(poscar_path: Path) -> Structure:
    """Load structure from POSCAR file."""
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen required")
    return Structure.from_file(str(poscar_path))


def parse_atomforge_file(file_path: Path) -> tuple:
    """
    Parse AtomForge file.
    
    Returns:
        (success, result) where result is program or error message
    """
    parser = AtomForgeParser()
    try:
        text = file_path.read_text(encoding='utf-8')
        program = parser.parse_and_transform(text)
        program.validate()
        return True, program
    except Exception as e:
        return False, str(e)

