#!/usr/bin/env python3
"""
Format Conversion Utilities

Converts between AtomForge, CIF, POSCAR formats.
"""

from pathlib import Path
from typing import Optional

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


def get_val(x) -> float:
    """Extract numeric value from Length/Angle objects."""
    if isinstance(x, (Length, Angle)):
        return float(x.value)
    return float(x)


def atomforge_to_structure(program, expand_symmetry: bool = True, symprec: float = 0.2) -> Structure:
    """
    Convert AtomForge program to pymatgen Structure.
    
    Args:
        program: Parsed AtomForgeProgram
        expand_symmetry: Whether to expand symmetry (default: True)
        symprec: Symmetry tolerance (not used if expand_symmetry=False)
    """
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen required")
    
    from pymatgen.core import Lattice as PmgLattice
    
    lat = program.lattice.bravais if program.lattice else None
    if lat is None:
        raise ValueError("Missing lattice bravais parameters")
    
    lattice = PmgLattice.from_parameters(
        get_val(lat.a),
        get_val(lat.b),
        get_val(lat.c),
        get_val(lat.alpha),
        get_val(lat.beta),
        get_val(lat.gamma),
    )
    
    species_all = []
    coords_all = []
    
    if not program.basis or not program.basis.sites:
        raise ValueError("Missing basis sites")
    
    # If expand_symmetry is True, we need to deduplicate sites by Wyckoff position
    # because Structure.from_spacegroup() expects the asymmetric unit, not the full cell
    if expand_symmetry:
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
    else:
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
    
    if expand_symmetry:
        structure = Structure.from_spacegroup(
            sg,
            lattice,
            species_all,
            coords_all,
        )
    else:
        # Create structure without expansion (just asymmetric unit)
        structure = Structure(lattice, species_all, coords_all, coords_are_cartesian=False)
    
    return structure


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

