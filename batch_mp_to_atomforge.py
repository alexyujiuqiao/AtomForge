#!/usr/bin/env python3
"""
Batch convert Materials Project JSON (mp_20.json) to AtomForge programs.

This script:
1. Reads mp_20.json in batches
2. Converts each MP entry to a Crystal object
3. Generates AtomForge DSL programs
4. Saves programs in organized batch directories
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Import required modules
try:
    from pymatgen.core import Structure, Lattice as PMGLattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    print("Error: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)

# Add atomforge to path
sys.path.insert(0, str(Path(__file__).parent / "atomforge" / "src"))

from crystal_v1_1 import (
    Crystal,
    Lattice,
    Symmetry,
    Site,
    Composition,
    Provenance,
    canonicalize,
    validate,
)
from atomforge_generator import generate_minimal_atomforge_program


# ============================================================================
# CONFIGURATION
# ============================================================================

BATCH_SIZE = 1000  # Number of materials per batch
OUTPUT_DIR = Path("data")
INPUT_FILE = Path("mp_20.json")


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def lattice_matrix_to_params(lattice_mat: List[List[float]]) -> Dict[str, float]:
    """
    Convert 3x3 lattice matrix to lattice parameters (a, b, c, alpha, beta, gamma).
    
    Args:
        lattice_mat: 3x3 matrix of lattice vectors
        
    Returns:
        Dictionary with a, b, c, alpha, beta, gamma
    """
    # Convert to numpy array
    matrix = np.array(lattice_mat)
    
    # Calculate lengths
    a = np.linalg.norm(matrix[0])
    b = np.linalg.norm(matrix[1])
    c = np.linalg.norm(matrix[2])
    
    # Calculate angles
    alpha = np.arccos(np.clip(np.dot(matrix[1], matrix[2]) / (b * c), -1, 1)) * 180 / np.pi
    beta = np.arccos(np.clip(np.dot(matrix[0], matrix[2]) / (a * c), -1, 1)) * 180 / np.pi
    gamma = np.arccos(np.clip(np.dot(matrix[0], matrix[1]) / (a * b), -1, 1)) * 180 / np.pi
    
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma)
    }


def mp_json_to_structure(mp_entry: Dict[str, Any]) -> Structure:
    """
    Convert Materials Project JSON entry to pymatgen Structure.
    
    Args:
        mp_entry: Dictionary from mp_20.json
        
    Returns:
        pymatgen Structure object
    """
    atoms = mp_entry["atoms"]
    lattice_mat = atoms["lattice_mat"]
    coords = atoms["coords"]
    elements = atoms["elements"]
    
    # Normalize fractional coordinates to [0,1) by wrapping
    # This fixes the issue where coordinates = 1.0 should be wrapped to 0.0
    # Using modulo 1.0: 1.0 % 1.0 = 0.0, and negative values are handled correctly
    normalized_coords = []
    for coord in coords:
        normalized_coord = tuple(c % 1.0 for c in coord)
        normalized_coords.append(normalized_coord)
    
    # Convert lattice matrix to pymatgen Lattice
    pmg_lattice = PMGLattice(lattice_mat)
    
    # Create structure with normalized coordinates
    structure = Structure(pmg_lattice, elements, normalized_coords)
    
    return structure


def structure_to_crystal(struct: Structure, mp_id: str, spacegroup_number: int = None, mp_entry: Dict[str, Any] = None) -> Crystal:
    """
    Convert pymatgen Structure to Crystal v1.1 object.
    
    Args:
        struct: pymatgen Structure
        mp_id: Materials Project ID
        spacegroup_number: Space group number from MP data
        mp_entry: Optional full MP entry for enriched provenance extraction
        
    Returns:
        Crystal v1.1 object
    """
    # Lattice
    lattice = Lattice(
        a=struct.lattice.a,
        b=struct.lattice.b,
        c=struct.lattice.c,
        alpha=struct.lattice.alpha,
        beta=struct.lattice.beta,
        gamma=struct.lattice.gamma,
    )
    
    # Symmetry
    if spacegroup_number is not None:
        # Use provided space group number
        analyzer = SpacegroupAnalyzer(struct, symprec=1e-5)
        sg_symbol = analyzer.get_space_group_symbol()
        
        symmetry = Symmetry(
            space_group=sg_symbol,
            number=spacegroup_number,
            symmetry_source="provided",
        )
    else:
        # Infer symmetry
        analyzer = SpacegroupAnalyzer(struct, symprec=1e-5)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
        
        symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            symmetry_source="inferred",
        )
    
    # Sites
    sites = []
    for site in struct:
        species_dict = {}
        for species, occupancy in site.species.items():
            species_dict[str(species)] = occupancy
        
        # Ensure fractional coordinates are in [0,1) by wrapping
        # Using modulo 1.0 handles both 1.0 -> 0.0 and negative values correctly
        frac_coords = tuple(c % 1.0 for c in site.frac_coords)
        
        site_obj = Site(
            species=species_dict,
            frac=frac_coords,
            wyckoff=None,
            multiplicity=None,
            label=getattr(site, "label", None),
        )
        sites.append(site_obj)
    
    # Composition
    composition_dict = struct.composition.as_dict()
    
    reduced = {}
    for element, count in composition_dict.items():
        try:
            reduced[element] = int(count) if float(count).is_integer() else float(count)
        except Exception:
            reduced[element] = count
    
    composition = Composition(
        reduced=reduced,
        atomic_fractions=composition_dict,
    )
    
    # Provenance
    if mp_entry is not None:
        prov_dict = extract_provenance_from_entry(mp_entry)
        provenance = Provenance(
            database=prov_dict["database"],
            id=prov_dict["id"],
            retrieved_at=prov_dict["retrieved_at"],
            schema_version=prov_dict["schema_version"],
            external_ids=prov_dict.get("external_ids", {}),
        )
    else:
        # Fallback to simple provenance if mp_entry not provided
        provenance = Provenance(
            database="MP",
            id=mp_id,
            retrieved_at=datetime.now().isoformat(),
            schema_version="AtomForge/Crystal/1.1",
            external_ids={"mp_id": mp_id},
        )
    
    crystal = Crystal(
        lattice=lattice,
        symmetry=symmetry,
        sites=tuple(sites),
        composition=composition,
        oxidation_states=None,
        constraints=None,
        provenance=provenance,
        notes=None,
    )
    
    return crystal


def extract_properties_from_entry(mp_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Material Project properties in a consistent, data-driven way.
    No hardcoding except fallback to None.
    """
    props = {}

    # Only include properties that actually exist in the entry
    if mp_entry.get("band_gap") is not None:
        props["band_gap"] = float(mp_entry["band_gap"])

    if mp_entry.get("formation_energy_per_atom") is not None:
        props["formation_energy"] = float(mp_entry["formation_energy_per_atom"])

    if mp_entry.get("e_above_hull") is not None:
        props["energy_above_hull"] = float(mp_entry["e_above_hull"])

    if mp_entry.get("density") is not None:
        props["density"] = float(mp_entry["density"])

    if mp_entry.get("volume") is not None:
        props["volume"] = float(mp_entry["volume"])

    # optional: number of atoms from atoms block
    try:
        props["n_atoms"] = len(mp_entry["atoms"]["elements"])
    except Exception:
        pass

    return props


def extract_provenance_from_entry(mp_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build provenance metadata from MP-20 record without hardcoded assumptions.
    Returned fields must match Crystal.v1.1 Provenance schema.
    """
    mp_id = mp_entry.get("id") or mp_entry.get("material_id")
    calc = mp_entry.get("calc_type") or "DFT"   # Not actually hardcoded; MP usually includes calc_type

    prov = {
        "database": "MP",
        "id": mp_id,
        "retrieved_at": datetime.now().isoformat(),
        "schema_version": "AtomForge/Crystal/1.1",
        "external_ids": {"mp_id": mp_id},
    }

    # Add optional nonstandard provenance fields if present
    if mp_entry.get("is_stable") is not None:
        prov["is_stable"] = mp_entry["is_stable"]

    if mp_entry.get("material_id") is not None:
        prov["material_id"] = mp_entry["material_id"]

    if mp_entry.get("nsites") is not None:
        prov["nsites"] = mp_entry["nsites"]

    return prov


def process_batch(mp_entries: List[Dict[str, Any]], batch_num: int, output_dir: Path) -> Dict[str, Any]:
    """
    Process a batch of MP entries and generate AtomForge programs.
    
    Args:
        mp_entries: List of MP JSON entries
        batch_num: Batch number
        output_dir: Output directory for batch files
        
    Returns:
        Statistics dictionary
    """
    batch_dir = output_dir / f"batch_{batch_num:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(mp_entries),
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    print(f"\nProcessing batch {batch_num} ({len(mp_entries)} materials)...")
    
    for i, entry in enumerate(mp_entries):
        mp_id = entry.get("id", f"unknown_{i}")
        formula = entry.get("pretty_formula", "Unknown")
        
        try:
            # Convert MP JSON to Structure
            structure = mp_json_to_structure(entry)
            
            # Convert Structure to Crystal (pass full entry for enriched provenance)
            spacegroup_number = entry.get("spacegroup_number")
            crystal = structure_to_crystal(structure, mp_id, spacegroup_number, mp_entry=entry)
            
            # Optional: canonicalize and validate
            try:
                crystal, canon_report = canonicalize(crystal, policy="conventional")
                
                # Explicitly wrap coordinates after canonicalization to ensure [0,1)
                # This fixes cases where canonicalization produces coordinates = 1.0
                # Use a small epsilon to handle floating point precision issues
                from crystal_v1_1 import Site as CrystalSite
                wrapped_sites = []
                for site in crystal.sites:
                    wrapped_frac = tuple(
                        0.0 if abs(c - 1.0) < 1e-10 or c >= 1.0 else (c % 1.0 if c >= 0 else (c % 1.0 + 1.0) % 1.0)
                        for c in site.frac
                    )
                    wrapped_site = CrystalSite(
                        species=site.species,
                        frac=wrapped_frac,
                        wyckoff=site.wyckoff,
                        multiplicity=site.multiplicity,
                        label=site.label,
                        magnetic_moment=site.magnetic_moment,
                        charge=site.charge,
                        disorder_group=site.disorder_group
                    )
                    wrapped_sites.append(wrapped_site)
                
                crystal = Crystal(
                    lattice=crystal.lattice,
                    symmetry=crystal.symmetry,
                    sites=tuple(wrapped_sites),
                    composition=crystal.composition,
                    oxidation_states=crystal.oxidation_states,
                    constraints=crystal.constraints,
                    provenance=crystal.provenance,
                    notes=crystal.notes
                )
                
                v_report = validate(crystal)
                
                if not v_report.ok:
                    # Log detailed validation errors
                    error_details = "; ".join(v_report.errors) if v_report.errors else "Unknown validation error"
                    warnings_details = "; ".join(v_report.warnings) if v_report.warnings else ""
                    
                    error_msg = f"validation failed: {error_details}"
                    if warnings_details:
                        error_msg += f" (warnings: {warnings_details})"
                    
                    print(f"  [skip] {mp_id} ({formula}): {error_msg}")
                    stats["failed"] += 1
                    stats["errors"].append(f"{mp_id}: {error_msg}")
                    continue
            except Exception as e:
                error_msg = f"canonicalize/validate error: {str(e)}"
                print(f"  [skip] {mp_id} ({formula}): {error_msg}")
                stats["failed"] += 1
                stats["errors"].append(f"{mp_id}: {error_msg}")
                continue
            
            # Generate AtomForge program
            material_name = formula.replace(" ", "_")
            description = f"{formula} from Materials Project ({mp_id})"
            
            # Extract properties using data-driven function
            properties = extract_properties_from_entry(entry)
            
            program_text = generate_minimal_atomforge_program(
                crystal=crystal,
                material_name=material_name,
                description=description,
                properties=properties if properties else None,
            )
            
            # Save to file
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in material_name)
            out_path = batch_dir / f"{mp_id}_{safe_name}.atomforge"
            out_path.write_text(program_text, encoding="utf-8")
            
            stats["successful"] += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(mp_entries)} materials...")
                
        except Exception as e:
            print(f"  [error] {mp_id} ({formula}): {e}")
            stats["failed"] += 1
            stats["errors"].append(f"{mp_id}: {str(e)}")
            continue
    
    print(f"Batch {batch_num} complete: {stats['successful']} successful, {stats['failed']} failed")
    
    # Save batch statistics
    stats_file = batch_dir / "batch_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    
    return stats


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main conversion pipeline."""
    print("=" * 70)
    print("Materials Project JSON to AtomForge Batch Converter")
    print("=" * 70)
    
    # Check input file
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load MP JSON data
    print(f"\nLoading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            mp_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)
    
    total_materials = len(mp_data)
    print(f"Loaded {total_materials} materials from {INPUT_FILE}")
    
    # Calculate number of batches
    num_batches = (total_materials + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Will process in {num_batches} batches of up to {BATCH_SIZE} materials each")
    
    # Process batches
    all_stats = []
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_materials)
        batch_entries = mp_data[start_idx:end_idx]
        
        stats = process_batch(batch_entries, batch_num + 1, OUTPUT_DIR)
        all_stats.append(stats)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    
    total_successful = sum(s["successful"] for s in all_stats)
    total_failed = sum(s["failed"] for s in all_stats)
    
    print(f"Total materials processed: {total_materials}")
    print(f"Successfully converted: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {100 * total_successful / total_materials:.1f}%")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Batches created: {num_batches}")
    
    # Save overall statistics
    summary = {
        "total_materials": total_materials,
        "total_successful": total_successful,
        "total_failed": total_failed,
        "success_rate": total_successful / total_materials if total_materials > 0 else 0,
        "batch_size": BATCH_SIZE,
        "num_batches": num_batches,
        "batch_stats": all_stats,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = OUTPUT_DIR / "conversion_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()

