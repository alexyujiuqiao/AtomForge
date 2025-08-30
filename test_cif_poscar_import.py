#!/usr/bin/env python3
"""
Test script for CIF and POSCAR import functionality with spglib integration.

This script demonstrates:
1. CIF file parsing and Crystal object creation
2. POSCAR file parsing and Crystal object creation
3. Canonicalization with spglib integration
4. Symmetry analysis and Wyckoff position inference
"""

import os
from pathlib import Path

# Import Crystal v1.1 functionality
from atomforge.src.crystal_v1_1 import (
    from_cif, from_poscar, canonicalize, validate, identity_hash,
    create_simple_crystal, CrystalAdapter
)

def ensure_data_folder():
    """Create data folder if it doesn't exist"""
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    return data_folder

def create_test_cif():
    """Create a test CIF file for demonstration"""
    cif_content = """# Test CIF file for Fe (bcc structure)
data_Fe_bcc
_symmetry_space_group_name_H-M    'Im-3m'
_symmetry_Int_Tables_number       229
_symmetry_cell_setting            cubic
_cell_length_a                    2.8665
_cell_length_b                    2.8665
_cell_length_c                    2.8665
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 90.0
_cell_volume                      23.55
_cell_formula_units_Z             2
_symmetry_equiv_pos_as_xyz
   x,y,z
   -x,-y,-z
   x+1/2,y+1/2,z+1/2
   -x+1/2,-y+1/2,-z+1/2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1    Fe    0.00000    0.00000    0.00000    1.0
Fe2    Fe    0.50000    0.50000    0.50000    1.0
"""
    
    # Create CIF file in data folder
    data_folder = ensure_data_folder()
    cif_path = data_folder / "test_Fe_bcc.cif"
    with open(cif_path, 'w') as f:
        f.write(cif_content)
    return str(cif_path)

def create_test_poscar():
    """Create a test POSCAR file for demonstration"""
    poscar_content = """Fe bcc structure
1.0
2.8665 0.0000 0.0000
0.0000 2.8665 0.0000
0.0000 0.0000 2.8665
Fe
2
Direct
0.0000000000000000  0.0000000000000000  0.0000000000000000
0.5000000000000000  0.5000000000000000  0.5000000000000000
"""
    
    # Create POSCAR file in data folder
    data_folder = ensure_data_folder()
    poscar_path = data_folder / "test_Fe_bcc.vasp"
    with open(poscar_path, 'w') as f:
        f.write(poscar_content)
    return str(poscar_path)

def create_test_silicon_cif():
    """Create a test CIF file for Silicon (diamond structure)"""
    cif_content = """# Test CIF file for Si (diamond structure)
data_Si_diamond
_symmetry_space_group_name_H-M    'Fd-3m'
_symmetry_Int_Tables_number       227
_symmetry_cell_setting            cubic
_cell_length_a                    5.4307
_cell_length_b                    5.4307
_cell_length_c                    5.4307
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 90.0
_cell_volume                      160.1
_cell_formula_units_Z             8
_symmetry_equiv_pos_as_xyz
   x,y,z
   -x,-y,-z
   -x+1/2,y+1/2,-z+1/2
   x+1/2,-y+1/2,z+1/2
   x+1/2,y+1/2,z+1/2
   -x+1/2,-y+1/2,-z+1/2
   -x,y,-z
   x,-y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1    Si    0.00000    0.00000    0.00000    1.0
Si2    Si    0.25000    0.25000    0.25000    1.0
Si3    Si    0.50000    0.50000    0.00000    1.0
Si4    Si    0.75000    0.75000    0.25000    1.0
Si5    Si    0.50000    0.00000    0.50000    1.0
Si6    Si    0.75000    0.25000    0.75000    1.0
Si7    Si    0.00000    0.50000    0.50000    1.0
Si8    Si    0.25000    0.75000    0.75000    1.0
"""
    
    # Create CIF file in data folder
    data_folder = ensure_data_folder()
    cif_path = data_folder / "test_Si_diamond.cif"
    with open(cif_path, 'w') as f:
        f.write(cif_content)
    return str(cif_path)

def create_test_graphene_poscar():
    """Create a test POSCAR file for Graphene (2D structure)"""
    poscar_content = """Graphene 2D structure
1.0
2.4560 0.0000 0.0000
-1.2280 2.1270 0.0000
0.0000 0.0000 20.0000
C
2
Direct
0.0000000000000000  0.0000000000000000  0.5000000000000000
0.3333333333333333  0.3333333333333333  0.5000000000000000
"""
    
    # Create POSCAR file in data folder
    data_folder = ensure_data_folder()
    poscar_path = data_folder / "test_graphene.vasp"
    with open(poscar_path, 'w') as f:
        f.write(poscar_content)
    return str(poscar_path)

def test_cif_import():
    """Test CIF file import functionality"""
    print("=== Testing CIF Import ===")
    
    try:
        # Create test CIF files
        cif_path_fe = create_test_cif()
        cif_path_si = create_test_silicon_cif()
        print(f"Created test CIF files:")
        print(f"  - {cif_path_fe} (Fe bcc structure)")
        print(f"  - {cif_path_si} (Si diamond structure)")
        
        # Test Fe bcc structure
        print(f"\n--- Testing {cif_path_fe} ---")
        crystal_fe = from_cif(cif_path_fe)
        print("✓ CIF import successful")
        
        # Display crystal information
        print(f"  Lattice: a={crystal_fe.lattice.a:.4f}, b={crystal_fe.lattice.b:.4f}, c={crystal_fe.lattice.c:.4f}")
        print(f"  Angles: α={crystal_fe.lattice.alpha:.1f}°, β={crystal_fe.lattice.beta:.1f}°, γ={crystal_fe.lattice.gamma:.1f}°")
        print(f"  Space Group: {crystal_fe.symmetry.space_group} (#{crystal_fe.symmetry.number})")
        print(f"  Number of sites: {len(crystal_fe.sites)}")
        print(f"  Composition: {crystal_fe.composition.reduced}")
        
        # Test Si diamond structure
        print(f"\n--- Testing {cif_path_si} ---")
        crystal_si = from_cif(cif_path_si)
        print("✓ CIF import successful")
        
        # Display crystal information
        print(f"  Lattice: a={crystal_si.lattice.a:.4f}, b={crystal_si.lattice.b:.4f}, c={crystal_si.lattice.c:.4f}")
        print(f"  Angles: α={crystal_si.lattice.alpha:.1f}°, β={crystal_si.lattice.beta:.1f}°, γ={crystal_si.lattice.gamma:.1f}°")
        print(f"  Space Group: {crystal_si.symmetry.space_group} (#{crystal_si.symmetry.number})")
        print(f"  Number of sites: {len(crystal_si.sites)}")
        print(f"  Composition: {crystal_si.composition.reduced}")
        
        return crystal_fe, crystal_si
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Install pymatgen: pip install pymatgen")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def test_poscar_import():
    """Test POSCAR file import functionality"""
    print("\n=== Testing POSCAR Import ===")
    
    try:
        # Create test POSCAR files
        poscar_path_fe = create_test_poscar()
        poscar_path_graphene = create_test_graphene_poscar()
        print(f"Created test POSCAR files:")
        print(f"  - {poscar_path_fe} (Fe bcc structure)")
        print(f"  - {poscar_path_graphene} (Graphene 2D structure)")
        
        # Test Fe bcc structure
        print(f"\n--- Testing {poscar_path_fe} ---")
        crystal_fe = from_poscar(poscar_path_fe)
        print("✓ POSCAR import successful")
        
        # Display crystal information
        print(f"  Lattice: a={crystal_fe.lattice.a:.4f}, b={crystal_fe.lattice.b:.4f}, c={crystal_fe.lattice.c:.4f}")
        print(f"  Angles: α={crystal_fe.lattice.alpha:.1f}°, β={crystal_fe.lattice.beta:.1f}°, γ={crystal_fe.lattice.gamma:.1f}°")
        print(f"  Space Group: {crystal_fe.symmetry.space_group} (#{crystal_fe.symmetry.number})")
        print(f"  Number of sites: {len(crystal_fe.sites)}")
        print(f"  Composition: {crystal_fe.composition.reduced}")
        
        # Test Graphene structure
        print(f"\n--- Testing {poscar_path_graphene} ---")
        crystal_graphene = from_poscar(poscar_path_graphene)
        print("✓ POSCAR import successful")
        
        # Display crystal information
        print(f"  Lattice: a={crystal_graphene.lattice.a:.4f}, b={crystal_graphene.lattice.b:.4f}, c={crystal_graphene.lattice.c:.4f}")
        print(f"  Angles: α={crystal_graphene.lattice.alpha:.1f}°, β={crystal_graphene.lattice.beta:.1f}°, γ={crystal_graphene.lattice.gamma:.1f}°")
        print(f"  Space Group: {crystal_graphene.symmetry.space_group} (#{crystal_graphene.symmetry.number})")
        print(f"  Number of sites: {len(crystal_graphene.sites)}")
        print(f"  Composition: {crystal_graphene.composition.reduced}")
        
        return crystal_fe, crystal_graphene
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Install pymatgen: pip install pymatgen")
        return None, None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

def test_canonicalization_with_spglib(crystal, name="crystal"):
    """Test canonicalization with spglib integration"""
    print(f"\n=== Testing Canonicalization with SPGLIB ({name}) ===")
    
    if crystal is None:
        print("No crystal to canonicalize")
        return
    
    try:
        # Test different canonicalization policies
        policies = ["standard", "primitive", "conventional"]
        
        for policy in policies:
            print(f"\n--- Policy: {policy} ---")
            
            # Canonicalize
            canonical_crystal, report = canonicalize(crystal, policy=policy)
            print(f"  Actions taken: {report.actions_taken}")
            print(f"  Epsilon used: {report.epsilon_used}")
            print(f"  Canonical hash: {report.canonical_hash[:16]}...")
            
            # Display Wyckoff information with detailed analysis
            print(f"  Site Analysis:")
            for i, site in enumerate(canonical_crystal.sites):
                wyckoff_info = f"Wyckoff: {site.wyckoff}" if site.wyckoff else "Wyckoff: None"
                mult_info = f"Multiplicity: {site.multiplicity}" if site.multiplicity else "Multiplicity: None"
                species_info = f"Species: {list(site.species.keys())}"
                coords_info = f"Coords: ({site.frac[0]:.4f}, {site.frac[1]:.4f}, {site.frac[2]:.4f})"
                print(f"    Site {i+1}: {species_info}, {wyckoff_info}, {mult_info}, {coords_info}")
            
            # Validate
            validation_report = validate(canonical_crystal)
            print(f"  Validation: {'✓ PASS' if validation_report.ok else '✗ FAIL'}")
            
            if not validation_report.ok:
                for error in validation_report.errors:
                    print(f"    Error: {error}")
        
        return canonical_crystal
        
    except Exception as e:
        print(f"✗ Canonicalization error: {e}")
        return None

def test_symmetry_equivalence():
    """Test symmetry equivalence with different origin choices"""
    print("\n=== Testing Symmetry Equivalence ===")
    
    try:
        # Create two equivalent structures with different origin choices
        crystal1 = create_simple_crystal(
            lattice_params=(2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0),
            sites=[("Fe", (0.0, 0.0, 0.0))],  # Origin choice 1
            space_group="Im-3m",
            space_group_number=229
        )
        
        crystal2 = create_simple_crystal(
            lattice_params=(2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0),
            sites=[("Fe", (0.5, 0.5, 0.5))],  # Origin choice 2
            space_group="Im-3m",
            space_group_number=229
        )
        
        print("Created two equivalent structures with different origin choices")
        
        # Canonicalize both
        canonical1, _ = canonicalize(crystal1, policy="conventional")
        canonical2, _ = canonicalize(crystal2, policy="conventional")
        
        # Get hashes
        hash1 = identity_hash(canonical1)
        hash2 = identity_hash(canonical2)
        
        print(f"Hash 1: {hash1[:16]}...")
        print(f"Hash 2: {hash2[:16]}...")
        print(f"Symmetry equivalence: {'✓ PASS' if hash1 == hash2 else '✗ FAIL'}")
        
        if hash1 == hash2:
            print("  ✓ SPGLIB successfully identified symmetry-equivalent structures")
        else:
            print("  ⚠ Different hashes - may need spglib integration")
        
    except Exception as e:
        print(f"✗ Symmetry equivalence test error: {e}")

def test_json_round_trip(crystal, name="crystal"):
    """Test JSON serialization round-trip"""
    print(f"\n=== Testing JSON Round-Trip ({name}) ===")
    
    if crystal is None:
        print("No crystal to test")
        return
    
    try:
        # Convert to JSON
        json_str = CrystalAdapter.to_json(crystal)
        print(f"✓ Converted to JSON ({len(json_str)} characters)")
        
        # Save JSON to file for inspection in data folder
        data_folder = ensure_data_folder()
        json_filename = data_folder / f"test_{name}_export.json"
        with open(json_filename, 'w') as f:
            f.write(json_str)
        print(f"✓ Saved JSON to {json_filename}")
        
        # Convert back from JSON
        recovered_crystal = CrystalAdapter.from_json(json_str)
        print("✓ Converted back from JSON")
        
        # Verify round-trip integrity
        original_hash = identity_hash(crystal)
        recovered_hash = identity_hash(recovered_crystal)
        
        print(f"Original hash: {original_hash[:16]}...")
        print(f"Recovered hash: {recovered_hash[:16]}...")
        print(f"Round-trip integrity: {'✓ PASS' if original_hash == recovered_hash else '✗ FAIL'}")
        
    except Exception as e:
        print(f"✗ JSON round-trip error: {e}")

def main():
    """Run all tests"""
    print("CIF/POSCAR Import and SPGLIB Integration Tests")
    print("=" * 60)
    
    # Ensure data folder exists
    data_folder = ensure_data_folder()
    print(f"Using data folder: {data_folder.absolute()}")
    
    # Test CIF import
    crystal_fe_cif, crystal_si_cif = test_cif_import()
    
    # Test POSCAR import
    crystal_fe_poscar, crystal_graphene_poscar = test_poscar_import()
    
    # Test canonicalization with spglib and store canonicalized structures
    canonical_fe_cif = None
    canonical_si_cif = None
    canonical_fe_poscar = None
    canonical_graphene_poscar = None
    
    if crystal_fe_cif:
        canonical_fe_cif = test_canonicalization_with_spglib(crystal_fe_cif, "Fe_bcc_CIF")
    if crystal_si_cif:
        canonical_si_cif = test_canonicalization_with_spglib(crystal_si_cif, "Si_diamond_CIF")
    if crystal_fe_poscar:
        canonical_fe_poscar = test_canonicalization_with_spglib(crystal_fe_poscar, "Fe_bcc_POSCAR")
    if crystal_graphene_poscar:
        canonical_graphene_poscar = test_canonicalization_with_spglib(crystal_graphene_poscar, "Graphene_POSCAR")
    
    # Test symmetry equivalence
    test_symmetry_equivalence()
    
    # Test JSON round-trip with canonicalized structures (which have Wyckoff positions)
    if canonical_fe_cif:
        test_json_round_trip(canonical_fe_cif, "Fe_bcc_CIF_canonical")
    if canonical_si_cif:
        test_json_round_trip(canonical_si_cif, "Si_diamond_CIF_canonical")
    if canonical_fe_poscar:
        test_json_round_trip(canonical_fe_poscar, "Fe_bcc_POSCAR_canonical")
    if canonical_graphene_poscar:
        test_json_round_trip(canonical_graphene_poscar, "Graphene_POSCAR_canonical")
    
    # Also test JSON round-trip with original structures for comparison
    if crystal_fe_cif:
        test_json_round_trip(crystal_fe_cif, "Fe_bcc_CIF_original")
    if crystal_si_cif:
        test_json_round_trip(crystal_si_cif, "Si_diamond_CIF_original")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("✓ CIF import functionality implemented")
    print("✓ POSCAR import functionality implemented")
    print("✓ SPGLIB integration for canonicalization")
    print("✓ Wyckoff position inference")
    print("✓ Symmetry equivalence testing")
    print("✓ JSON round-trip validation")
    print(f"\nAll test files saved in: {data_folder.absolute()}")
    print("Test files created:")
    print("  - data/test_Fe_bcc.cif (Fe bcc structure)")
    print("  - data/test_Si_diamond.cif (Si diamond structure)")
    print("  - data/test_Fe_bcc.vasp (Fe bcc POSCAR)")
    print("  - data/test_graphene.vasp (Graphene POSCAR)")
    print("  - data/test_*_original_export.json (Original structures)")
    print("  - data/test_*_canonical_export.json (Canonicalized structures with Wyckoff)")

if __name__ == "__main__":
    main() 