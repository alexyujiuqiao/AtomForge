#!/usr/bin/env python3
"""
Fix formula values in per_sample.csv by extracting from program files.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def extract_formula_from_program(file_path: Path) -> Optional[str]:
    """Extract formula from program file (atom_spec name or title)."""
    try:
        text = file_path.read_text()
        
        # Try to extract from atom_spec name first (e.g., atom_spec "ZnAl2O4_Spinel")
        atom_spec_match = re.search(r'atom_spec\s+"([^"]+)"', text)
        if atom_spec_match:
            name = atom_spec_match.group(1)
            # Remove suffixes like "_Spinel", "_Perovskite", etc.
            formula = re.sub(r'_[A-Z][a-z]*$', '', name)
            # Remove common suffixes
            formula = re.sub(r'_(Spinel|Perovskite|Feldspathoid|Melilite|Hypothetical)$', '', formula, flags=re.IGNORECASE)
            if formula:
                return formula
        
        # Fallback to title (e.g., title = "ZnAl2O4 (normal spinel)")
        title_match = re.search(r'title\s*=\s*"([^"]+)"', text)
        if title_match:
            title = title_match.group(1)
            # Extract formula before parentheses
            formula_match = re.match(r'([A-Za-z0-9]+)', title)
            if formula_match:
                return formula_match.group(1)
        
        return None
    except Exception:
        return None


def fix_csv_formulas(csv_path: Path, gen_dir: Path):
    """Fix formula values in CSV by extracting from program files."""
    # Read existing CSV
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows in CSV")
    print(f"Extracting formulas from program files...")
    
    fixed_count = 0
    error_count = 0
    
    for i, row in enumerate(rows):
        file_path_str = row.get('file', '')
        if not file_path_str:
            continue
        
        # Get file path
        file_path = Path(file_path_str)
        if not file_path.exists():
            # Try relative to gen_dir
            file_path = gen_dir / Path(file_path_str).name
        
        # Only fix .atomforge files
        if file_path.suffix != '.atomforge' or not file_path.exists():
            continue
        
        try:
            formula = extract_formula_from_program(file_path)
            if formula:
                row['formula'] = formula
                fixed_count += 1
            else:
                print(f"  Row {i+1} ({file_path.name}): Could not extract formula")
                error_count += 1
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
    parser = argparse.ArgumentParser(description="Fix formula values in per_sample.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to per_sample.csv")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated .atomforge files")
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    gen_dir = Path(args.gen_dir)
    if not gen_dir.exists():
        print(f"Error: Generation directory not found: {gen_dir}")
        sys.exit(1)
    
    fix_csv_formulas(csv_path, gen_dir)


if __name__ == "__main__":
    main()

