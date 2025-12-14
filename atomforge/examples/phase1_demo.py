#!/usr/bin/env python3
"""
Phase 1 Demonstration Script

This script demonstrates all Phase 1 functionality:
- Database matching and variant selection
- CIF/POSCAR parsing with provenance
- Variant cards with space group, energy hull, site count
- Selection policies and ranking
- Complete Phase 1 workflows

Based on Phase 1 requirements from plan-v-2-0.tex
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal_v1_1 import (
    Crystal, Lattice, Symmetry, Site, Composition, Provenance,
    create_simple_crystal, canonicalize, validate, identity_hash, CrystalAdapter
)
from atomforge_interop import (
    from_cif, from_poscar, match_database, select_variant,
    match_database_with_provenance, select_variant_with_policy,
    create_variant_cards_ui, VariantCard
)
from atomforge_database_connector import (
    MaterialsProjectConnector, CODConnector, ICSDConnector,
    DatabaseMatch, MatchReport, SelectionReport
)

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

def print_match_report(report: MatchReport, title: str = "Database Search Results"):
    """Print database search results"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Query Formula: {report.query_formula}")
    print(f"Total Matches: {report.total_matches}")
    print(f"Search Time: {report.search_time:.2f}s")
    print(f"Databases Searched: {', '.join(report.databases_searched)}")
    
    if report.errors:
        print(f"Errors: {len(report.errors)}")
        for error in report.errors:
            print(f"  - {error}")
    
    print(f"\nMatches by Database:")
    for db_name, matches in report.matches_found.items():
        print(f"  {db_name}: {len(matches)} matches")
        for i, match in enumerate(matches[:3]):  # Show first 3
            print(f"    {i+1}. {match.material_id}: {match.formula} (similarity: {match.similarity_score:.3f})")

def print_variant_cards(cards: list, title: str = "Variant Cards"):
    """Print variant cards information"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total Cards: {len(cards)}")
    
    for i, card in enumerate(cards[:5]):  # Show first 5
        print(f"\n{i+1}. {card.get_display_summary()}")
        print(f"   Database: {card.database_name}")
        print(f"   Material ID: {card.material_id}")
        print(f"   Formula: {card.formula}")
        print(f"   Space Group: {card.space_group or 'Unknown'}")
        print(f"   Energy Hull: {card.energy_hull or 'N/A'} eV/atom")
        print(f"   Site Count: {card.site_count or 'Unknown'}")
        print(f"   Experimental: {'Yes' if card.is_experimental else 'No'}")
        print(f"   Stable: {'Yes' if card.is_stable else 'No' if card.is_stable is not None else 'Unknown'}")

def print_selection_report(report: SelectionReport, title: str = "Selection Report"):
    """Print variant selection report"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Selected Variant: {report.selected_variant.database_name}:{report.selected_variant.material_id}")
    print(f"Policy Used: {report.policy_used}")
    print(f"Selection Reason: {report.selection_reason}")
    print(f"Ranking Criteria: {', '.join(report.ranking_criteria)}")
    print(f"Alternatives Considered: {len(report.alternatives_considered)}")
    print(f"Timestamp: {report.timestamp}")

def demonstrate_cif_poscar_parsing():
    """Demonstrate CIF and POSCAR parsing functionality from existing files."""
    print("PHASE 1 DEMONSTRATION: Database Matching & Variant Selection")
    print("=" * 80)
    
    print("\n1. Demonstrating CIF/POSCAR parsing...")
    
    # Prefer parsing from existing CIF/POSCAR under project data directory
    project_root = Path(__file__).parents[2]
    data_dir = project_root / "data"
    cif_candidates = [
        data_dir / "nacl_rocksalt" / "nacl_rocksalt.cif",
        data_dir / "iron_bcc" / "iron_bcc.cif",
        data_dir / "quartz" / "quartz.cif",
        data_dir / "silicon_diamond" / "silicon_diamond.cif",
    ]
    poscar_candidates = [
        data_dir / "iron_bcc" / "POSCAR_iron_bcc",
        data_dir / "silicon_diamond" / "POSCAR_silicon_diamond",
        data_dir / "graphene" / "POSCAR_graphene",
    ]
    
    crystal = None
    # Try CIF first
    for path in cif_candidates:
        if path.exists():
            print(f"   Parsing CIF file: {path}")
            crystal = from_cif(str(path))
            break
    
    # Fallback to POSCAR
    if crystal is None:
        for path in poscar_candidates:
            if path.exists():
                print(f"   Parsing POSCAR file: {path}")
                crystal = from_poscar(str(path))
                break
    
    # Final fallback: create simple crystal (should rarely be needed)
    if crystal is None:
        print("   No CIF/POSCAR found. Falling back to generated test crystal...")
        crystal = create_simple_crystal(
            lattice_params=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
            sites=[
                ("Si", (0.0, 0.0, 0.0)),
                ("Si", (0.5, 0.5, 0.5))
            ],
            space_group="Fd-3m",
            space_group_number=227
        )
    
    # Canonicalize the crystal
    crystal, canon_report = canonicalize(crystal)
    print_crystal_info(crystal, "Test Crystal (Canonicalized)")
    
    # Store parsed result (JSON) first
    try:
        project_root = Path(__file__).parents[2]
        runs_dir = project_root / "docs" / "phase1_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parsed_path_ts = runs_dir / f"parsed_crystal_{ts}.json"
        parsed_path_latest = runs_dir / "parsed_crystal_latest.json"
        json_str = CrystalAdapter.to_json(crystal)
        parsed_path_ts.write_text(json_str)
        parsed_path_latest.write_text(json_str)
        print(f"   ✓ Stored parsed crystal JSON: {parsed_path_ts}")
    except Exception as e:
        print(f"   ⚠ Failed to store parsed crystal JSON: {e}")

    # Export to AtomForge DSL program
    try:
        out_dir = project_root / "atomforge_materials"
        out_dir.mkdir(parents=True, exist_ok=True)
        formula_str = " ".join(f"{k}{v}" for k, v in crystal.composition.reduced.items())
        out_path = out_dir / "nacl_rocksalt.atomforge"
        _export_atomforge_program(crystal, title=f"NaCl Rocksalt ({formula_str})", out_path=out_path)
        print(f"   Wrote AtomForge program: {out_path}")
    except Exception as e:
        print(f"   Failed to write AtomForge program: {e}")

    return crystal

def demonstrate_database_matching(crystal: Crystal):
    """Demonstrate database matching functionality"""
    print("\n2. Demonstrating database matching...")
    
    # Extract formula for matching
    formula = "".join(f"{k}{v}" for k, v in crystal.composition.reduced.items())
    print(f"   Searching for formula: {formula}")
    
    # Search databases
    print("   Searching Materials Project and COD databases...")
    print("   Note: COD requires known 7-digit IDs (e.g., https://www.crystallography.net/cod/1000000.cif)")
    try:
        match_report = match_database(formula, sources=["MP", "COD"], tolerance=0.1)
        print_match_report(match_report, "Database Search Results")
        
        # Reduced verbosity: skip detailed per-match listing
        
        return match_report
        
    except Exception as e:
        print(f"   Error in database matching: {e}")
        print("   This is expected if API keys are not configured")
        return None

def demonstrate_variant_cards(crystal: Crystal):
    """Demonstrate variant cards creation and display"""
    print("\n3. Demonstrating variant cards creation...")
    
    try:
        # Create variant cards
        match_report, variant_cards = match_database_with_provenance(
            crystal, sources=["MP", "COD"], tolerance=0.1
        )
        
        print(f"   Created {len(variant_cards)} variant cards")
        print_variant_cards(variant_cards, "Variant Cards")
        
        # Create UI-ready cards
        print("\n   Creating UI-ready cards...")
        ui_cards = create_variant_cards_ui(variant_cards, str(crystal.composition.reduced))
        print(f"   Created {len(ui_cards)} UI cards (showing top 1)")
        if ui_cards:
            top = ui_cards[0]
            print(f"   Top: {top['display_summary']} (sim {top['similarity_score']:.3f})")
        
        return variant_cards, ui_cards
        
    except Exception as e:
        print(f"   Error creating variant cards: {e}")
        print("   This is expected if API keys are not configured")
        return [], []

def demonstrate_variant_selection(variant_cards: list, reference_space_group: str = None):
    """Demonstrate variant selection with different policies"""
    print("\n4. Demonstrating variant selection...")
    
    if not variant_cards:
        print("   No variant cards available for selection")
        return
    
    # Test different selection policies
    policies = [
        "prefer_low_hull_then_experimental",
        "prefer_experimental",
        "prefer_completeness",
        "prefer_explicit_space_group",
        "low_hull_experimental_completeness_explicit_space_group"
    ]
    
    for policy in policies:
        print(f"\n   Testing policy: {policy}")
        try:
            selected_card, report = select_variant_with_policy(variant_cards, policy, reference_space_group=reference_space_group)
            print(f"   Selected: {selected_card.get_display_summary()}")
            print(f"   Reason: {report.selection_reason}")
            print(f"   Policy: {report.policy_used}")
            
        except Exception as e:
            print(f"   Error with policy {policy}: {e}")

def demonstrate_workflows(crystal: Crystal):
    """Demonstrate complete Phase 1 workflows using the parsed crystal"""
    print("\n5. Demonstrating complete workflows...")
    
    # Workflow 1: Database-pinned variant
    print("\n   Workflow 1: Database-pinned variant (match + select)")
    
    try:
        # Match to database
        match_report, variant_cards = match_database_with_provenance(crystal, sources=["MP", "COD"])
        
        if variant_cards:
            # Select best variant
            ref_sg = crystal.symmetry.space_group if hasattr(crystal, 'symmetry') else None
            selected_card, report = select_variant_with_policy(variant_cards, "prefer_explicit_space_group", reference_space_group=ref_sg)
            
            print(f"   ✓ Matches: {len(variant_cards)}; Selected: {selected_card.get_display_summary()}")
        else:
            print("   ⚠ No database matches found (API keys may not be configured)")
    
    except Exception as e:
        print(f"   ⚠ Database workflow failed: {e}")
        print("   This is expected if API keys are not configured")
    
    # Workflow 2: Provenance tracking
    print("\n   Workflow 2: Provenance tracking (before/after)")
    print(f"   Before: db={crystal.provenance.database or 'None'}, id={crystal.provenance.id or 'None'}")
    if variant_cards:
        selected_card = variant_cards[0]
        print(f"   After:  db={selected_card.database_name}, id={selected_card.material_id}")

# Removed verbose helper demos (error handling, API integration) for concise output

def demonstrate_ui_integration(variant_cards: list = None, query_formula: str = None):
    """Demonstrate UI integration capabilities using real cards if available."""
    print("\n8. Demonstrating UI integration...")
    
    cards_for_ui = []
    if variant_cards:
        cards_for_ui = variant_cards
    else:
        # Fallback to mock variant cards
        mock_cards = [
            VariantCard(
                database_name="MP",
                material_id="mp-149",
                formula="Si",
                space_group="Fd-3m",
                energy_hull=0.0,
                site_count=2,
                is_experimental=True,
                is_stable=True,
                provenance={"source": "materials_project"},
                properties={"energy_hull": 0.0, "space_group": "Fd-3m"},
                metadata={"nsites": 2, "is_experimental": True}
            ),
            VariantCard(
                database_name="COD",
                material_id="cod-12345",
                formula="Si",
                space_group="Fd-3m",
                energy_hull=0.1,
                site_count=2,
                is_experimental=False,
                is_stable=True,
                provenance={"source": "crystallography_open_database"},
                properties={"energy_hull": 0.1, "space_group": "Fd-3m"},
                metadata={"nsites": 2, "is_experimental": False}
            )
        ]
        cards_for_ui = mock_cards
        if query_formula is None:
            query_formula = "Si"
    
    if query_formula is None:
        query_formula = ""  # no similarity scoring if unknown
    
    print("   Creating UI-ready cards...")
    ui_cards = create_variant_cards_ui(cards_for_ui, query_formula)
    
    print(f"   Created {len(ui_cards)} UI cards (showing top 1)")
    if ui_cards:
        top = ui_cards[0]
        print(f"   Top: {top['display_summary']} (sim {top['similarity_score']:.3f})")
    
    # Reduced verbosity: omit integration target list

def _export_atomforge_program(crystal: Crystal, title: str, out_path: Path):
    """Export a minimal AtomForge DSL program from a Crystal v1.1 object."""
    # Lattice
    lat = crystal.lattice
    # Symmetry
    sg_num = crystal.symmetry.number
    # Build basis sites blocks with wyckoff/multiplicity if available
    site_blocks = []
    for i, site in enumerate(crystal.sites, start=1):
        wyck = site.wyckoff or ""
        pos = site.frac
        # Single-species assumption per site for Phase 1 demo
        elem, occ = next(iter(site.species.items()))
        block = (
            f"    site \"S{i}\" {{\n"
            f"      wyckoff = \"{wyck}\",\n"
            f"      position = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}),\n"
            f"      frame = fractional,\n"
            f"      species = ({{ element = \"{elem}\", occupancy = {occ:.3f} }})\n"
            f"    }}"
        )
        site_blocks.append(block)
    site_section = "\n".join(site_blocks)
    # Compose program
    program = (
        f"atom_spec \"{out_path.stem}\" {{\n"
        f"  header {{ dsl_version = \"2.1\", content_schema_version = \"materials_science_v1.0\", title = \"{title}\" }}\n"
        f"  lattice {{ a = {lat.a:.3f} angstrom, b = {lat.b:.3f} angstrom, c = {lat.c:.3f} angstrom, "
        f"alpha = {lat.alpha:.1f} degree, beta = {lat.beta:.1f} degree, gamma = {lat.gamma:.1f} degree }}\n"
        f"  symmetry {{ space_group = {sg_num} }}\n"
        f"  basis {{\n{site_section}\n  }}\n"
        f"}}\n"
    )
    out_path.write_text(program)

def main():
    """Main demonstration function"""
    print("ATOMFORGE CRYSTAL PHASE 1 DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows all Phase 1 functionality:")
    print("- Database matching and variant selection")
    print("- CIF/POSCAR parsing with provenance")
    print("- Variant cards with space group, energy hull, site count")
    print("- Selection policies and ranking")
    print("- Complete Phase 1 workflows")
    print("=" * 80)
    
    try:
        # Demonstrate core functionality
        crystal = demonstrate_cif_poscar_parsing()
        match_report = demonstrate_database_matching(crystal)
        variant_cards, ui_cards = demonstrate_variant_cards(crystal)
        ref_sg = crystal.symmetry.space_group if hasattr(crystal, 'symmetry') else None
        demonstrate_variant_selection(variant_cards, reference_space_group=ref_sg)
        
        # Demonstrate workflows using the parsed crystal
        demonstrate_workflows(crystal)
        
        # (Omitted) error handling and API integration sections for concise demo output
        
        # Demonstrate UI integration
        # Demonstrate UI integration (use real cards if available)
        demonstrate_ui_integration(variant_cards, "".join(f"{k}{v}" for k, v in (crystal.composition.reduced or {}).items()))
        
        print("\n" + "=" * 80)
        print("PHASE 1 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
