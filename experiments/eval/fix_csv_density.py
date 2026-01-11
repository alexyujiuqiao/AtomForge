#!/usr/bin/env python3
"""
Fix density values in per_sample.csv by recalculating with corrected code.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval.convert import parse_atomforge_file, atomforge_to_structure


def fix_csv_densities(csv_path: Path, gen_dir: Path, expand_symmetry: bool = True, symprec: float = 0.2):
    """Fix density values in CSV by recalculating."""
    # Read existing CSV
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows in CSV")
    print(f"Recalculating densities with expand_symmetry={expand_symmetry}, symprec={symprec}...")
    
    fixed_count = 0
    error_count = 0
    
    for i, row in enumerate(rows):
        file_path = Path(row.get('file', ''))
        if not file_path.exists():
            # Try relative to gen_dir
            file_path = gen_dir / file_path.name
        
        # Only fix .atomforge files
        if file_path.suffix != '.atomforge' or not file_path.exists():
            continue
        
        try:
            # Parse and convert
            ok, program = parse_atomforge_file(file_path)
            if not ok:
                print(f"  Row {i+1}: Parse failed - {program}")
                error_count += 1
                continue
            
            struct, metadata = atomforge_to_structure(
                program,
                expand_symmetry=expand_symmetry,
                symprec=symprec,
                auto_detect_expanded=True
            )
            
            # Update row with corrected values
            row['density'] = f"{struct.density:.6f}"
            row['natoms'] = str(len(struct))
            
            # Add volume if field exists or add it to fieldnames
            if 'volume' in fieldnames:
                row['volume'] = f"{struct.volume:.6f}"
            else:
                row['volume'] = f"{struct.volume:.6f}"
                fieldnames = list(fieldnames) + ['volume']
            
            # Update metadata fields if they exist
            if 'n_input_sites' in fieldnames:
                row['n_input_sites'] = str(metadata.get('n_input_sites', ''))
            elif 'n_input_sites' not in row:
                row['n_input_sites'] = str(metadata.get('n_input_sites', ''))
                fieldnames = list(fieldnames) + ['n_input_sites']
            
            if 'used_symmetry_expansion' in fieldnames:
                row['used_symmetry_expansion'] = str(metadata.get('used_symmetry_expansion', False))
            elif 'used_symmetry_expansion' not in row:
                row['used_symmetry_expansion'] = str(metadata.get('used_symmetry_expansion', False))
                fieldnames = list(fieldnames) + ['used_symmetry_expansion']
            
            if 'auto_skipped' in fieldnames:
                row['auto_skipped'] = str(metadata.get('auto_skipped', False))
            elif 'auto_skipped' not in row:
                row['auto_skipped'] = str(metadata.get('auto_skipped', False))
                fieldnames = list(fieldnames) + ['auto_skipped']
            
            # Update formula if changed
            row['formula'] = struct.composition.reduced_formula
            
            fixed_count += 1
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(rows)} rows...")
                
        except Exception as e:
            print(f"  Row {i+1} ({file_path.name}): Error - {e}")
            error_count += 1
            continue
    
    # Write fixed CSV
    output_path = csv_path.parent / f"{csv_path.stem}_fixed.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nDone! Fixed {fixed_count} rows, {error_count} errors")
    print(f"Output saved to: {output_path}")
    print(f"\nTo replace the original, run:")
    print(f"  mv {output_path} {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix density values in per_sample.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to per_sample.csv")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated .atomforge files")
    parser.add_argument("--expand_symmetry", type=int, default=1, help="Expand symmetry (1) or not (0)")
    parser.add_argument("--symprec", type=float, default=0.2, help="Symmetry tolerance")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    gen_dir = Path(args.gen_dir)
    if not gen_dir.exists():
        print(f"Error: Generation directory not found: {gen_dir}")
        sys.exit(1)
    
    fix_csv_densities(
        csv_path,
        gen_dir,
        expand_symmetry=bool(args.expand_symmetry),
        symprec=args.symprec
    )


if __name__ == "__main__":
    main()

