#!/usr/bin/env python3
"""
Compile Generated AtomForge Programs to CIF and Structure Objects

This script converts generated .atomforge programs into CIF files and pymatgen Structure
objects for downstream evaluation. It handles partial occupancies as disordered sites.

Usage:
    python -m experiments.compile_generated --in_dir outputs/uncond_200/programs --out_dir outputs/uncond_200/compiled

Output:
    - outputs/uncond_200/compiled/*.cif (CIF files)
    - outputs/uncond_200/compiled/*.pkl (pymatgen Structure objects, optional)
    - outputs/uncond_200/compile_metrics.jsonl (per-file success/failure logs)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import AtomForge components
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
    from atomforge.src.atomforge_ir import AtomForgeProgram
except ImportError:
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser
    from atomforge_ir import AtomForgeProgram

# Import pymatgen
try:
    from pymatgen.core import Lattice as PmgLattice, Structure, Composition
    from pymatgen.io.cif import CifWriter
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Install with: pip install pymatgen")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_value(obj: Any) -> float:
    """Extract numeric value from Length/Angle objects or direct floats."""
    if hasattr(obj, "value"):
        return float(getattr(obj, "value"))
    return float(obj)


def atomforge_to_structure(program: AtomForgeProgram) -> Structure:
    """
    Convert an AtomForge program to a pymatgen Structure.
    
    Handles partial occupancies by creating disordered sites.
    
    Args:
        program: Parsed AtomForgeProgram object
        
    Returns:
        pymatgen Structure object
    """
    if not PYMAGEN_AVAILABLE:
        raise ImportError("pymatgen is required for structure conversion")
    
    # Build lattice
    if not program.lattice or not program.lattice.bravais:
        raise ValueError("Lattice information is required")
    
    bravais = program.lattice.bravais
    lattice = PmgLattice.from_parameters(
        get_value(bravais.a),
        get_value(bravais.b),
        get_value(bravais.c),
        get_value(bravais.alpha),
        get_value(bravais.beta),
        get_value(bravais.gamma)
    )
    
    # Build species and coordinates
    species_list: List[Any] = []
    frac_coords: List[Tuple[float, float, float]] = []
    
    if not program.basis or not program.basis.sites:
        raise ValueError("Basis with sites is required")
    
    for site in program.basis.sites:
        pos = site.position
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"Invalid position for site {site.name}: {pos}")
        
        # Handle frame conversion if needed
        if site.frame == "cartesian":
            # Convert cartesian to fractional
            frac_pos = lattice.get_fractional_coords([x, y, z])[0]
            x, y, z = frac_pos[0], frac_pos[1], frac_pos[2]
        
        # Handle species with occupancies
        if not site.species:
            raise ValueError(f"Site {site.name} has no species")
        
        # If single species with occupancy 1.0, use simple element
        if len(site.species) == 1 and abs(site.species[0].occupancy - 1.0) < 1e-6:
            species_list.append(site.species[0].element)
            frac_coords.append((x, y, z))
        else:
            # Multiple species or partial occupancy: create disordered site
            species_dict: Dict[str, float] = {}
            total_occ = 0.0
            for sp in site.species:
                occ = float(sp.occupancy)
                if occ > 1e-6:  # Only include non-zero occupancies
                    species_dict[sp.element] = occ
                    total_occ += occ
            
            if abs(total_occ - 1.0) > 1e-3:
                logger.warning(
                    f"Site {site.name} has occupancies summing to {total_occ:.4f}, "
                    f"normalizing to 1.0"
                )
                # Normalize
                for key in species_dict:
                    species_dict[key] /= total_occ
            
            # Create Composition for disordered site
            comp = Composition(species_dict)
            species_list.append(comp)
            frac_coords.append((x, y, z))
    
    # Create Structure
    struct = Structure(
        lattice,
        species_list,
        frac_coords,
        coords_are_cartesian=False
    )
    
    return struct


def compile_program(
    program_path: Path,
    out_dir: Path,
    parser: AtomForgeParser
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Compile a single AtomForge program to CIF.
    
    Args:
        program_path: Path to .atomforge file
        out_dir: Output directory for CIF files
        parser: AtomForgeParser instance
        
    Returns:
        Tuple of (success, error_message, metadata)
    """
    program_id = program_path.stem
    metadata: Dict[str, Any] = {
        'program_id': program_id,
        'input_file': str(program_path),
        'timestamp': datetime.now().isoformat(),
    }
    
    try:
        # Read and parse
        with open(program_path, 'r', encoding='utf-8') as f:
            program_text = f.read()
        
        program = parser.parse_and_transform(program_text)
        program.validate()
        
        # Convert to structure
        if not PYMAGEN_AVAILABLE:
            raise ImportError("pymatgen is required")
        
        struct = atomforge_to_structure(program)
        
        # Write CIF
        cif_path = out_dir / f"{program_id}.cif"
        writer = CifWriter(struct)
        with open(cif_path, 'w', encoding='utf-8') as f:
            f.write(str(writer))
        
        metadata.update({
            'status': 'success',
            'cif_file': str(cif_path),
            'num_sites': len(struct),
            'num_species': len(struct.composition),
            'density': struct.density,
            'formula': struct.formula,
        })
        
        return True, None, metadata
        
    except Exception as e:
        error_msg = str(e)
        metadata.update({
            'status': 'error',
            'error_message': error_msg,
        })
        return False, error_msg, metadata


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compile generated AtomForge programs to CIF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--in_dir',
        type=str,
        default='outputs/uncond_200/programs',
        help='Input directory containing .atomforge files'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs/uncond_200/compiled',
        help='Output directory for compiled CIF files'
    )
    
    args = parser.parse_args()
    
    in_path = Path(args.in_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not in_path.exists():
        raise ValueError(f"Input directory does not exist: {in_path}")
    
    # Find all .atomforge files
    atomforge_files = sorted(in_path.glob("*.atomforge"))
    if not atomforge_files:
        logger.warning(f"No .atomforge files found in {in_path}")
        return
    
    logger.info(f"Found {len(atomforge_files)} AtomForge programs to compile")
    
    # Initialize parser
    atomforge_parser = AtomForgeParser()
    
    # Metrics file
    metrics_file = out_path.parent / "compile_metrics.jsonl"
    
    # Compile each program
    success_count = 0
    error_count = 0
    
    for i, program_file in enumerate(atomforge_files, 1):
        logger.info(f"[{i}/{len(atomforge_files)}] Compiling {program_file.name}...")
        
        success, error_msg, metadata = compile_program(
            program_file,
            out_path,
            atomforge_parser
        )
        
        if success:
            success_count += 1
            logger.info(f"  ✓ Success: {metadata.get('formula', 'N/A')}")
        else:
            error_count += 1
            logger.warning(f"  ✗ Failed: {error_msg}")
        
        # Log metric
        with open(metrics_file, 'a', encoding='utf-8') as f:
            json.dump(metadata, f, default=str)
            f.write('\n')
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("COMPILATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total programs: {len(atomforge_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {error_count}")
    logger.info(f"Success rate: {success_count/len(atomforge_files)*100:.1f}%")
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

