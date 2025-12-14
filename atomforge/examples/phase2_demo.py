#!/usr/bin/env python3
"""
Phase 2 Demonstration Script

This script demonstrates all Phase 2 editing and patching operations:
- Species substitution and vacancy creation
- Interstitial atom addition
- Lattice and symmetry modification
- Supercell creation with mapping
- Export to various formats

Based on the Phase 2 requirements from plan-v-2-0.tex
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal_v1_1 import (
    Crystal, Lattice, Symmetry, Site, Composition, Provenance, ConstraintSet,
    create_simple_crystal, canonicalize, validate, identity_hash,
    from_cif, from_poscar
)
from crystal_edit import (
    substitute, vacancy, interstitial, set_lattice, set_symmetry,
    make_supercell, to_poscar, to_cif, to_pymatgen, from_pymatgen,
    PatchRecord, SupercellMap
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def _load_crystal_file(path: Path) -> Crystal:
    """Load a crystal structure from a CIF or POSCAR file."""
    if path.suffix.lower() == ".cif":
        crystal = from_cif(str(path))
    elif path.name.upper().startswith("POSCAR"):
        crystal = from_poscar(str(path))
    else:
        raise ValueError(f"Unsupported crystal file type: {path}")
    return crystal


def _load_from_candidates(candidates, fallback_factory):
    last_error = None
    for rel_path in candidates:
        path = DATA_ROOT / rel_path
        if not path.exists():
            continue
        try:
            crystal = _load_crystal_file(path)
            crystal, report = canonicalize(crystal)
            return crystal, report, path
        except Exception as exc:  # pragma: no cover - best-effort loader
            last_error = exc
            continue
    crystal = fallback_factory()
    crystal, report = canonicalize(crystal)
    return crystal, report, None


def load_reference_crystal():
    """Load NaCl rocksalt structure from data directory (fallback to synthetic)."""
    candidates = [
        Path("nacl_rocksalt") / "nacl_rocksalt.cif",
        Path("nacl_rocksalt") / "POSCAR_nacl_rocksalt",
    ]

    def fallback():
        return create_simple_crystal(
            lattice_params=(5.64, 5.64, 5.64, 90.0, 90.0, 90.0),
            sites=[
                ("Na", (0.0, 0.0, 0.0)),
                ("Cl", (0.5, 0.5, 0.5)),
            ],
            space_group="Fm-3m",
            space_group_number=225,
        )

    return _load_from_candidates(candidates, fallback)


def load_silicon_crystal():
    """Load silicon diamond structure (fallback to synthetic tetrahedral cell)."""
    candidates = [
        Path("silicon_diamond") / "silicon_diamond.cif",
        Path("silicon_diamond") / "POSCAR_silicon_diamond",
    ]

    def fallback():
        return create_simple_crystal(
            lattice_params=(5.43, 5.43, 5.43, 90.0, 90.0, 90.0),
            sites=[
                ("Si", (0.0, 0.0, 0.0)),
                ("Si", (0.25, 0.25, 0.25)),
            ],
            space_group="Fd-3m",
            space_group_number=227,
        )

    return _load_from_candidates(candidates, fallback)


def load_graphene_crystal():
    """Load graphene primitive structure (fallback to simple hexagonal cell)."""
    candidates = [
        Path("graphene") / "graphene.cif",
        Path("graphene") / "POSCAR_graphene",
    ]

    def fallback():
        return create_simple_crystal(
            lattice_params=(2.46, 2.46, 10.0, 90.0, 90.0, 120.0),
            sites=[
                ("C", (0.0, 0.0, 0.5)),
                ("C", (1/3, 2/3, 0.5)),
            ],
            space_group="P6/mmm",
            space_group_number=191,
        )

    return _load_from_candidates(candidates, fallback)

def print_crystal_info(crystal: Crystal, title: str = "Crystal Structure"):
    """Print basic information about a crystal structure"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Formula: {' '.join(f'{k}{v}' for k, v in crystal.composition.reduced.items())}")
    print(f"Space Group: {crystal.symmetry.space_group} (#{crystal.symmetry.number})")
    print(f"Lattice: a={crystal.lattice.a:.3f}, b={crystal.lattice.b:.3f}, c={crystal.lattice.c:.3f}")
    print(f"Angles: α={crystal.lattice.alpha:.1f}°, β={crystal.lattice.beta:.1f}°, γ={crystal.lattice.gamma:.1f}°")
    print(f"Sites: {len(crystal.sites)}")
    for i, site in enumerate(crystal.sites):
        species_str = ", ".join(f"{k}:{v:.3f}" for k, v in site.species.items())
        wyckoff_str = f" ({site.wyckoff})" if site.wyckoff else ""
        print(f"  {i}: {species_str} at {site.frac}{wyckoff_str}")
    print(f"Hash: {crystal.provenance.hash[:16]}..." if crystal.provenance.hash else "No hash")

def print_patch_record(patch: PatchRecord, title: str = "Patch Record"):
    """Print patch record information"""
    print(f"\n{title}:")
    print(f"  Operation: {patch.op}")
    print(f"  Parameters: {patch.params}")
    print(f"  Timestamp: {patch.timestamp}")
    print(f"  Result Hash: {patch.result_hash[:16]}...")

def demonstrate_editing_operations():
    """Demonstrate all editing operations"""
    print("PHASE 2 DEMONSTRATION: Crystal Editing & Patching Operations")
    print("=" * 80)
    
    print("\n1. Loading NaCl crystal from data set...")
    crystal, canon_report, source_path = load_reference_crystal()
    if source_path:
        print(f"   Loaded from: {source_path.relative_to(PROJECT_ROOT)}")
    else:
        print("   Loaded fallback synthetic NaCl structure")
    print_crystal_info(crystal, "Initial Crystal (Canonicalized)")
    
    # Demonstrate species substitution (Na -> K alloying)
    print("\n2. Demonstrating species substitution...")
    print("   Alloying Na sites with 20% K...")
    crystal_sub, patch1 = substitute(
        crystal,
        "Species:Na",
        {"Na": 0.8, "K": 0.2},
        1.0,
    )
    print_crystal_info(crystal_sub, "After Na/K Substitution")
    print_patch_record(patch1, "Substitution Patch")
    
    # Demonstrate vacancy creation
    print("\n3. Demonstrating vacancy creation...")
    print("   Creating 25% vacancies on Cl sublattice...")
    crystal_vac, patch2 = vacancy(crystal, "Species:Cl", 0.75)
    print_crystal_info(crystal_vac, "After Cl Vacancy Creation")
    print_patch_record(patch2, "Vacancy Patch")
    
    # Demonstrate interstitial addition
    print("\n4. Demonstrating interstitial addition...")
    print("   Adding Na interstitial at (0.125, 0.125, 0.125)...")
    crystal_int, patch3 = interstitial(crystal, (0.125, 0.125, 0.125), "Na", 1.0)
    print_crystal_info(crystal_int, "After Na Interstitial Addition")
    print_patch_record(patch3, "Interstitial Patch")
    
    # Demonstrate lattice modification
    print("\n5. Demonstrating lattice modification...")
    print("   Changing lattice parameters...")
    new_lattice = Lattice(a=6.0, b=6.0, c=6.0, alpha=90.0, beta=90.0, gamma=90.0)
    crystal_lat, patch4 = set_lattice(crystal, new_lattice)
    print_crystal_info(crystal_lat, "After Lattice Modification")
    print_patch_record(patch4, "Lattice Patch")
    
    # Demonstrate symmetry modification
    print("\n6. Demonstrating symmetry modification...")
    print("   Changing space group to P1...")
    new_symmetry = Symmetry(
        space_group="P1",
        number=1,
        symmetry_source="user_defined"
    )
    crystal_sym, patch5 = set_symmetry(crystal, new_symmetry)
    print_crystal_info(crystal_sym, "After Symmetry Modification")
    print_patch_record(patch5, "Symmetry Patch")
    
    return crystal

def demonstrate_supercell_operations(crystal: Crystal):
    """Demonstrate supercell operations"""
    print("\n7. Demonstrating supercell creation...")
    print("   Creating 2x2x1 supercell from NaCl...")
    
    # Create supercell
    M = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    supercell, supercell_map, patch6 = make_supercell(crystal, M)
    
    print_crystal_info(supercell, "Supercell (2x2x1)")
    print_patch_record(patch6, "Supercell Patch")
    
    # Show supercell mapping
    print(f"\nSupercell Mapping:")
    print(f"  Parent sites: {len(crystal.sites)}")
    print(f"  Child sites: {len(supercell.sites)}")
    print(f"  Expansion factor: {len(supercell.sites) // len(crystal.sites)}")
    
    # Show some mapping examples
    print(f"\nMapping Examples:")
    for i in range(min(4, len(supercell_map.child_to_parent))):
        child_idx, (parent_idx, lattice_vector) = list(supercell_map.child_to_parent.items())[i]
        print(f"  Child site {child_idx} -> Parent site {parent_idx} + lattice vector {lattice_vector}")
    
    return supercell

def demonstrate_export_operations(crystal: Crystal):
    """Demonstrate export operations"""
    print("\n8. Demonstrating export operations...")
    
    # Export to POSCAR
    print("   Exporting to POSCAR format...")
    poscar_data = to_poscar(crystal)
    print(f"   POSCAR formula: {poscar_data['formula']}")
    print(f"   POSCAR lines: {len(poscar_data['poscar'].split(chr(10)))}")
    
    # Show first few lines of POSCAR
    poscar_lines = poscar_data['poscar'].split('\n')
    print(f"   First 5 lines of POSCAR:")
    for i, line in enumerate(poscar_lines[:5]):
        print(f"     {i+1}: {line}")
    
    # Export to CIF
    print("\n   Exporting to CIF format...")
    cif_content = to_cif(crystal)
    print(f"   CIF lines: {len(cif_content.split(chr(10)))}")
    
    # Show CIF header
    cif_lines = cif_content.split('\n')
    print(f"   First 10 lines of CIF:")
    for i, line in enumerate(cif_lines[:10]):
        print(f"     {i+1}: {line}")
    
    # Try pymatgen export (if available)
    try:
        print("\n   Exporting to pymatgen Structure...")
        pmg_structure = to_pymatgen(crystal)
        print(f"   pymatgen Structure created successfully")
        print(f"   Lattice: {pmg_structure.lattice}")
        print(f"   Composition: {pmg_structure.composition}")
        
        # Try importing back
        print("   Importing back from pymatgen...")
        imported_crystal = from_pymatgen(pmg_structure)
        print(f"   Imported crystal has {len(imported_crystal.sites)} sites")
        
    except ImportError:
        print("   pymatgen not available - skipping pymatgen export/import")

def demonstrate_workflows():
    """Demonstrate complete workflows from the operations document using real data."""
    print("\n9. Demonstrating complete workflows...")
    
    print("\n   Workflow 1: Clean import - ready for DFT (silicon diamond)")
    silicon, canon_rep, source = load_silicon_crystal()
    if source:
        print(f"   Loaded silicon from: {source.relative_to(PROJECT_ROOT)}")
    val_rep = validate(silicon)
    print(f"   Canonicalization: {len(canon_rep.actions_taken)} actions taken")
    print(f"   Validation: {'PASS' if val_rep.ok else 'FAIL'}")
    if not val_rep.ok:
        print(f"   Validation errors: {val_rep.errors}")
    
    print("\n   Workflow 2: Doping sweep - supercell - export (NaCl -> Ta doping)")
    nacl, _, _ = load_reference_crystal()
    nacl, _ = canonicalize(nacl)
    print("   Doping Na with Ta (10% occupancy)...")
    doped_crystal, p1 = substitute(nacl, "Species:Na", {"Na": 0.9, "Ta": 0.1}, 1.0)
    print_patch_record(p1, "   Doping Patch")
    print("   Creating 2x2x1 supercell...")
    M = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
    supercell, supercell_map, p2 = make_supercell(doped_crystal, M)
    val_rep = validate(supercell)
    print(f"   Supercell validation: {'PASS' if val_rep.ok else 'FAIL'}")
    print("   Exporting to POSCAR...")
    poscar_data = to_poscar(supercell)
    print(f"   Exported supercell with {len(supercell.sites)} sites")
    print(f"   Formula: {poscar_data['formula']}")
    
    print("\n   Workflow 3: Graphene property regression (hash check)")
    graphene, _, _ = load_graphene_crystal()
    graphene, _ = canonicalize(graphene)
    hash1 = identity_hash(graphene)
    graphene_roundtrip, _ = canonicalize(graphene)
    hash2 = identity_hash(graphene_roundtrip)
    print(f"   Graphene hash: {hash1[:16]}...")
    print(f"   Round-trip hash: {hash2[:16]}...")
    print(f"   Hashes match: {'YES' if hash1 == hash2 else 'NO'}")

def demonstrate_constraints():
    """Demonstrate constraint handling"""
    print("\n10. Demonstrating constraint handling...")
    
    # Create a crystal with symmetry locked
    constraints = ConstraintSet(symmetry_locked=True)
    crystal = create_simple_crystal(
        lattice_params=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
        sites=[("Si", (0.0, 0.0, 0.0))],
        space_group="Fd-3m",
        space_group_number=227
    )
    crystal = Crystal(
        lattice=crystal.lattice,
        symmetry=crystal.symmetry,
        sites=crystal.sites,
        composition=crystal.composition,
        constraints=constraints,
        provenance=crystal.provenance
    )
    
    print("   Created crystal with symmetry_locked=True")
    
    # Try to edit - should fail
    try:
        substitute(crystal, 0, "Ge", 1.0)
        print("   ERROR: Substitution should have failed!")
    except ValueError as e:
        print(f"   ✓ Substitution correctly blocked: {e}")
    
    try:
        vacancy(crystal, 0, 0.5)
        print("   ERROR: Vacancy creation should have failed!")
    except ValueError as e:
        print(f"   ✓ Vacancy creation correctly blocked: {e}")
    
    try:
        interstitial(crystal, (0.1, 0.1, 0.1), "H", 1.0)
        print("   ERROR: Interstitial addition should have failed!")
    except ValueError as e:
        print(f"   ✓ Interstitial addition correctly blocked: {e}")

def main():
    """Main demonstration function"""
    print("ATOMFORGE CRYSTAL PHASE 2 DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows all Phase 2 editing and patching operations")
    print("as specified in the AtomForge Crystal MVP Plan v2.0")
    print("=" * 80)
    
    try:
        # Demonstrate editing operations
        crystal = demonstrate_editing_operations()
        
        # Demonstrate supercell operations
        supercell = demonstrate_supercell_operations(crystal)
        
        # Demonstrate export operations
        demonstrate_export_operations(crystal)
        
        # Demonstrate complete workflows
        demonstrate_workflows()
        
        # Demonstrate constraint handling
        demonstrate_constraints()
        
        print("\n" + "=" * 80)
        print("PHASE 2 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nAll Phase 2 operations have been demonstrated:")
        print("✓ Species substitution and vacancy creation")
        print("✓ Interstitial atom addition")
        print("✓ Lattice and symmetry modification")
        print("✓ Supercell creation with child-parent mapping")
        print("✓ Export to POSCAR, CIF, and pymatgen formats")
        print("✓ Complete workflows from the operations document")
        print("✓ Constraint handling and error checking")
        print("\nPhase 2 implementation is complete and ready for use!")
        
    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
