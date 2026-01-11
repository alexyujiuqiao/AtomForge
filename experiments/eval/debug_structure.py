#!/usr/bin/env python3
"""
Debug helper to print structure conversion details for a single .atomforge file.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval.convert import parse_atomforge_file, atomforge_to_structure


def debug_structure(file_path: Path, expand_symmetry: bool = True, symprec: float = 0.2):
    """Print detailed structure conversion information."""
    print(f"=== Parsing {file_path} ===\n")
    
    ok, result = parse_atomforge_file(file_path)
    if not ok:
        print(f"Parse failed: {result}")
        return
    
    print(f"Parse successful!")
    print(f"Program type: {type(result)}")
    print(f"Header: dsl_version={result.header.dsl_version}, title={result.header.title}")
    
    if result.units:
        print(f"Units: length={result.units.length}, angle={result.units.angle}")
    
    if result.lattice and result.lattice.bravais:
        lat = result.lattice.bravais
        print(f"Lattice: type={lat.type}")
        print(f"  a={lat.a}, b={lat.b}, c={lat.c}")
        print(f"  alpha={lat.alpha}, beta={lat.beta}, gamma={lat.gamma}")
    
    if result.symmetry:
        print(f"Symmetry: space_group={result.symmetry.space_group}, origin_choice={result.symmetry.origin_choice}")
    
    if result.basis:
        n_sites = len(result.basis.sites)
        print(f"Basis: {n_sites} sites")
        total_weighted = 0
        for i, site in enumerate(result.basis.sites[:5]):  # Show first 5
            weighted = sum(sp.occupancy for sp in site.species) if site.species else 0
            total_weighted += weighted
            print(f"  Site {i+1}: {site.name}, wyckoff={site.wyckoff}, position={site.position}, weighted={weighted:.2f}")
        for i, site in enumerate(result.basis.sites[5:], 6):
            weighted = sum(sp.occupancy for sp in site.species) if site.species else 0
            total_weighted += weighted
        if n_sites > 5:
            print(f"  ... ({n_sites - 5} more sites)")
        print(f"Total weighted sites: {total_weighted:.2f}")
    
    print(f"\n=== Converting to Structure ===\n")
    print(f"expand_symmetry={expand_symmetry}, symprec={symprec}")
    
    try:
        struct, metadata = atomforge_to_structure(
            result, 
            expand_symmetry=expand_symmetry,
            symprec=symprec,
            auto_detect_expanded=True
        )
        
        print(f"Conversion successful!")
        print(f"  n_input_sites: {metadata['n_input_sites']}")
        print(f"  used_symmetry_expansion: {metadata['used_symmetry_expansion']}")
        print(f"  auto_skipped: {metadata['auto_skipped']}")
        print(f"\nStructure properties:")
        print(f"  natoms: {len(struct)}")
        print(f"  formula: {struct.composition.reduced_formula}")
        print(f"  volume: {struct.volume:.4f} Å³")
        print(f"  density: {struct.density:.4f} g/cm³")
        print(f"  lattice (Å): a={struct.lattice.a:.4f}, b={struct.lattice.b:.4f}, c={struct.lattice.c:.4f}")
        print(f"  lattice angles: α={struct.lattice.alpha:.2f}°, β={struct.lattice.beta:.2f}°, γ={struct.lattice.gamma:.2f}°")
        
        # Sanity checks
        if len(struct) > 2000:
            print(f"\n⚠️  WARNING: natoms={len(struct)} > 2000 (expansion explosion detected)")
        if struct.density > 50.0:
            print(f"⚠️  WARNING: density={struct.density:.2f} g/cm³ > 50 (density explosion detected)")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Debug structure conversion for a single .atomforge file")
    parser.add_argument("file", type=str, help="Path to .atomforge file")
    parser.add_argument("--expand_symmetry", type=int, default=1, help="Expand symmetry (1) or not (0)")
    parser.add_argument("--symprec", type=float, default=0.2, help="Symmetry tolerance")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    debug_structure(file_path, expand_symmetry=bool(args.expand_symmetry), symprec=args.symprec)


if __name__ == "__main__":
    main()

