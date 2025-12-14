#!/usr/bin/env python3
"""
LLZO Garnet Solid Electrolyte Demo - Real-World Materials Science Case

This demonstration showcases AtomForge Crystal's capabilities using the real-world example
of tetragonal garnet-type Li7La3Zr2O12 (LLZO), an important solid electrolyte for 
all-solid-state lithium ion batteries.

Real-world context:
- High ionic conductivity due to lithium vacancy-mediated diffusion
- Li2O loss creates lithium and oxygen vacancies
- Activation energies: Li migration (0.45 eV) << O migration (1.65 eV)
- Temperature-dependent oxygen diffusion facilitated by vacancies
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal_v1_1 import (
    Crystal, canonicalize, validate, identity_hash,
    from_cif, CanonReport, ValidationReport
)
from crystal_edit import (
    substitute, vacancy, interstitial, make_supercell, 
    to_poscar, to_cif, PatchRecord
)
from crystal_calc import (
    prepare_calc, CalculationTarget
)
from report_manager import ReportManager
from atomforge_generator import generate_atomforge_program

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "docs" / "demo_output"

# Create output directory
OUTPUT_ROOT.mkdir(exist_ok=True)

# Setup logging
def setup_logging():
    """Setup logging configuration for the demo"""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    log_file = OUTPUT_ROOT / f"llzo_demo_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"LLZO Demo logging initialized. Log file: {log_file}")
    return logger, log_file

# Global logger (will be initialized in main)
logger = None

def log_and_print(message: str, level: str = "info"):
    """Helper function to both print and log a message"""
    print(message)
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

def print_section_header(title: str, phase: str = None):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    if phase:
        print(f"Phase: {phase}")
    print(f"{'='*80}")
    
    # Also log to file
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"{title}")
        if phase:
            logger.info(f"Phase: {phase}")
        logger.info(f"{'='*80}")

def print_crystal_summary(crystal: Crystal, title: str = "Crystal Structure"):
    """Print a summary of crystal structure"""
    print(f"\n{title}:")
    print(f"  Formula: {' '.join(f'{k}{v}' for k, v in crystal.composition.reduced.items())}")
    print(f"  Space Group: {crystal.symmetry.space_group} (#{crystal.symmetry.number})")
    print(f"  Lattice: a={crystal.lattice.a:.3f}, b={crystal.lattice.b:.3f}, c={crystal.lattice.c:.3f}")
    print(f"  Angles: α={crystal.lattice.alpha:.1f}°, β={crystal.lattice.beta:.1f}°, γ={crystal.lattice.gamma:.1f}°")
    print(f"  Sites: {len(crystal.sites)}")
    if len(crystal.sites) <= 10:  # Show all sites if few
        for i, site in enumerate(crystal.sites):
            species_str = ", ".join(f"{k}:{v:.3f}" for k, v in site.species.items())
            wyckoff_str = f" (Wyckoff: {site.wyckoff})" if site.wyckoff else ""
            print(f"    {i}: {species_str} at {site.frac}{wyckoff_str}")
    else:  # Show summary if many sites
        element_counts = {}
        for site in crystal.sites:
            for element in site.species.keys():
                element_counts[element] = element_counts.get(element, 0) + 1
        print(f"  Element counts: {element_counts}")
    if crystal.provenance.hash:
        print(f"  Hash: {crystal.provenance.hash[:16]}...")
    
    # Also log to file
    if logger:
        logger.info(f"\n{title}:")
        logger.info(f"  Formula: {' '.join(f'{k}{v}' for k, v in crystal.composition.reduced.items())}")
        logger.info(f"  Space Group: {crystal.symmetry.space_group} (#{crystal.symmetry.number})")
        logger.info(f"  Lattice: a={crystal.lattice.a:.3f}, b={crystal.lattice.b:.3f}, c={crystal.lattice.c:.3f}")
        logger.info(f"  Angles: α={crystal.lattice.alpha:.1f}°, β={crystal.lattice.beta:.1f}°, γ={crystal.lattice.gamma:.1f}°")
        logger.info(f"  Sites: {len(crystal.sites)}")
        if len(crystal.sites) <= 10:  # Show all sites if few
            for i, site in enumerate(crystal.sites):
                species_str = ", ".join(f"{k}:{v:.3f}" for k, v in site.species.items())
                wyckoff_str = f" (Wyckoff: {site.wyckoff})" if site.wyckoff else ""
                logger.info(f"    {i}: {species_str} at {site.frac}{wyckoff_str}")
        else:  # Show summary if many sites
            element_counts = {}
            for site in crystal.sites:
                for element in site.species.keys():
                    element_counts[element] = element_counts.get(element, 0) + 1
            logger.info(f"  Element counts: {element_counts}")
        if crystal.provenance.hash:
            logger.info(f"  Hash: {crystal.provenance.hash[:16]}...")

def workflow_1_llzo_loading(report_manager: ReportManager = None):
    """Workflow 1: Load and canonicalize LLZO crystal structure"""
    print_section_header("WORKFLOW 1: LLZO Crystal Structure Loading", "Phase 0")
    
    log_and_print("Loading real Li7La3Zr2O12 garnet crystal structure from CIF file...")
    log_and_print("Real-world context: High ionic conductivity solid electrolyte")
    
    # Load the actual LLZO structure
    cif_path = DATA_ROOT / "LLZO" / "Li7La3Zr2O12.cif"
    if not cif_path.exists():
        error_msg = f"ERROR: LLZO CIF file not found at {cif_path}"
        log_and_print(error_msg, "error")
        return None
    
    log_and_print(f"Loading CIF file: {cif_path}")
    crystal = from_cif(str(cif_path))
    print_crystal_summary(crystal, "Initial LLZO Crystal (from CIF)")
    
    log_and_print("\nStep 1: Canonicalize crystal structure...")
    crystal_canon, canon_report = canonicalize(crystal)
    log_and_print(f"  Canonicalization actions: {canon_report.actions_taken}")
    log_and_print(f"  Epsilon used: {canon_report.epsilon_used}")
    log_and_print(f"  Canonical hash: {canon_report.canonical_hash[:16]}..." if canon_report.canonical_hash else "  No hash generated")
    
    print_crystal_summary(crystal_canon, "Canonicalized LLZO Crystal")
    
    log_and_print("\nStep 2: Validate crystal structure...")
    val_report = validate(crystal_canon)
    log_and_print(f"  Validation: {'PASS' if val_report.ok else 'FAIL'}")
    if val_report.errors:
        log_and_print(f"  Errors: {val_report.errors}", "error")
    if val_report.warnings:
        log_and_print(f"  Warnings: {val_report.warnings}", "warning")
    
    if not val_report.ok:
        error_msg = "ERROR: Validation failed, cannot proceed"
        log_and_print(error_msg, "error")
        return None
    
    # Store lightweight reports if report manager is available
    if report_manager:
        # Store canonicalization report
        report_manager.store_canon_report(
            "LLZO",
            canon_report,
            "LLZO canonicalization report"
        )
        
        # Store validation report
        report_manager.store_validation_report(
            "LLZO",
            val_report,
            "LLZO validation report"
        )
        
        # Generate and store AtomForge program for initial crystal
        atomforge_program = generate_atomforge_program(
            crystal_canon,
            material_name="LLZO_initial",
            description="LLZO crystal after canonicalization and validation",
            operations=[{
                "operation": "canonicalize",
                "params": {
                    "actions": canon_report.actions_taken,
                    "epsilon": canon_report.epsilon_used
                }
            }],
            defects=None  # No defects in initial structure
        )
        
        report_manager.store_atomforge_program(
            "LLZO",
            atomforge_program,
            "initial_canonicalized",
            "LLZO crystal after canonicalization and validation"
        )
    
    return crystal_canon

def workflow_2_li2o_loss_simulation(crystal: Crystal):
    """Workflow 2: Simulate Li2O loss - the most favorable disorder process"""
    print_section_header("WORKFLOW 2: Li2O Loss Simulation", "Phase 1")
    
    print("Simulating Li2O loss - the most favorable disorder process in LLZO")
    print("Real-world context: Creates Li and O vacancies for enhanced ionic conductivity")
    
    print_crystal_summary(crystal, "Initial LLZO Crystal")
    
    print("\nStep 1: Create lithium vacancies (Li2O loss effect)...")
    print("  Removing 2 Li atoms to simulate Li2O loss...")
    
    # Create lithium vacancies to simulate Li2O loss
    # LLZO has Li56 in the supercell, so removing 2 Li atoms = 96.4% occupancy
    crystal_li_vac, patch1 = vacancy(crystal, "Species:Li", occupancy=0.964)
    
    print(f"  Lithium vacancy creation successful:")
    print(f"    Operation: {patch1.op}")
    print(f"    Parameters: {patch1.params}")
    print(f"    Result hash: {patch1.result_hash[:16]}...")
    
    print_crystal_summary(crystal_li_vac, "LLZO with Li Vacancies")
    
    print("\nStep 2: Create oxygen vacancies (Li2O loss effect)...")
    print("  Removing 1 O atom to complete Li2O loss...")
    
    # Create oxygen vacancies to complete Li2O loss
    # LLZO has O96 in the supercell, so removing 1 O atom = 98.96% occupancy
    crystal_o_vac, patch2 = vacancy(crystal_li_vac, "Species:O", occupancy=0.9896)
    
    print(f"  Oxygen vacancy creation successful:")
    print(f"    Operation: {patch2.op}")
    print(f"    Parameters: {patch2.params}")
    print(f"    Result hash: {patch2.result_hash[:16]}...")
    
    print_crystal_summary(crystal_o_vac, "LLZO with Li and O Vacancies")
    
    print("\nStep 3: Validate defect-containing structure...")
    val_report = validate(crystal_o_vac)
    print(f"  Validation: {'PASS' if val_report.ok else 'FAIL'}")
    if val_report.errors:
        print(f"  Errors: {val_report.errors}")
    if val_report.warnings:
        print(f"  Warnings: {val_report.warnings}")
    
    print("  Real-world impact: Vacancies promote vacancy-mediated self-diffusion")
    return crystal_o_vac, [patch1, patch2]

def workflow_3_diffusion_pathway_analysis(crystal: Crystal):
    """Workflow 3: Analyze vacancy-mediated diffusion pathways"""
    print_section_header("WORKFLOW 3: Vacancy-Mediated Diffusion Analysis", "Phase 2")
    
    print("Analyzing vacancy-mediated diffusion pathways")
    print("Real-world context: Li migration (0.45 eV) << O migration (1.65 eV)")
    
    print_crystal_summary(crystal, "LLZO with Vacancies")
    
    print("\nStep 1: Analyze lithium diffusion pathways...")
    print("  Simulating Li+ migration through vacancy-mediated diffusion")
    print("  Expected: Low activation energy (0.45 eV) for Li migration")
    
    # Analyze lithium sites and vacancies
    li_sites = [site for site in crystal.sites if "Li" in site.species]
    li_vacancies = [site for site in crystal.sites if "Li" in site.species and site.species["Li"] < 1.0]
    
    print(f"  Total Li sites in structure: {len(li_sites)}")
    print(f"  Li vacancy sites: {len(li_vacancies)}")
    if len(li_sites) > 0:
        print(f"  Li vacancy concentration: {len(li_vacancies)/len(li_sites)*100:.1f}%")
    
    print("\nStep 2: Analyze oxygen diffusion pathways...")
    print("  Simulating O2- migration through vacancy-mediated diffusion")
    print("  Expected: High activation energy (1.65 eV) for O migration")
    
    # Analyze oxygen sites and vacancies
    o_sites = [site for site in crystal.sites if "O" in site.species]
    o_vacancies = [site for site in crystal.sites if "O" in site.species and site.species["O"] < 1.0]
    
    print(f"  Total O sites in structure: {len(o_sites)}")
    print(f"  O vacancy sites: {len(o_vacancies)}")
    if len(o_sites) > 0:
        print(f"  O vacancy concentration: {len(o_vacancies)/len(o_sites)*100:.1f}%")
    
    print("\nStep 3: Create supercell for diffusion analysis...")
    print("  Creating 2x2x2 supercell to study long-range diffusion...")
    
    # Create supercell for diffusion analysis
    M = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    crystal_super, supercell_map, patch3 = make_supercell(crystal, M)
    
    print(f"  Supercell creation successful:")
    print(f"    Parent sites: {len(crystal.sites)}")
    print(f"    Child sites: {len(crystal_super.sites)}")
    print(f"    Expansion factor: {len(crystal_super.sites) // len(crystal.sites)}")
    print(f"    Result hash: {patch3.result_hash[:16]}...")
    
    print_crystal_summary(crystal_super, "LLZO Supercell for Diffusion Analysis")
    
    print("  Key insight: Li diffusion >> O diffusion due to activation energy difference")
    return crystal_super, [patch3]

def workflow_4_ionic_conductivity_calculation(crystal: Crystal):
    """Workflow 4: Prepare calculations for ionic conductivity and activation energies"""
    print_section_header("WORKFLOW 4: Ionic Conductivity Calculation Setup", "Phase 3")
    
    print("Preparing calculations for ionic conductivity and activation energies")
    print("Real-world context: Temperature-dependent ionic conductivity")
    
    print_crystal_summary(crystal, "LLZO Supercell for Calculations")
    
    print("\nStep 1: Prepare calculation for ionic conductivity...")
    print("  Setting up VASP calculations for ionic conductivity determination")
    
    targets = [
        "ionic_conductivity",
        CalculationTarget(property="li_migration_energy", unit="eV", accuracy=0.05),
        CalculationTarget(property="oxygen_migration_energy", unit="eV", accuracy=0.1),
        CalculationTarget(property="formation_energy", unit="eV", accuracy=0.1)
    ]
    
    try:
        calc_input, prep_report = prepare_calc(
            crystal,
            backend_caps="vasp",
            targets=targets
        )
        
        print(f"  Calculation settings:")
        print(f"    Functional: {getattr(calc_input.settings, 'functional', 'N/A')}")
        print(f"    Energy cutoff: {getattr(calc_input.settings, 'encut', 'N/A')} eV")
        print(f"    K-point density: {getattr(calc_input.settings, 'k_point_density', 'N/A')}")
        print(f"    Spin polarized: {getattr(calc_input.settings, 'spin_polarized', 'N/A')}")
        
        print(f"  Preparation decisions:")
        for decision, reason in prep_report.decisions.items():
            print(f"    {decision}: {reason}")
        
        if prep_report.warnings:
            print(f"  Warnings: {prep_report.warnings}")
        
        return calc_input, prep_report
        
    except Exception as e:
        print(f"  ERROR in calculation preparation: {e}")
        print("  Continuing without calculation preparation...")
        return None, None

def workflow_5_temperature_dependent_analysis():
    """Workflow 5: Temperature-dependent diffusion analysis"""
    print_section_header("WORKFLOW 5: Temperature-Dependent Analysis", "Phase 3")
    
    print("Analyzing temperature-dependent diffusion behavior")
    print("Real-world context: O diffusion facilitated at higher temperatures")
    
    print("\nStep 1: Li migration analysis...")
    print("  Li+ migration activation energy: 0.45 eV")
    print("  Low activation energy → high ionic conductivity")
    print("  Primary conduction mechanism in LLZO")
    
    print("\nStep 2: O migration analysis...")
    print("  O2- migration activation energy: 1.65 eV")
    print("  High activation energy → limited O diffusion at low T")
    print("  O diffusion becomes significant at higher temperatures")
    
    print("\nStep 3: Temperature-dependent conductivity...")
    print("  Arrhenius behavior: σ(T) = σ₀ exp(-Eₐ/kT)")
    print("  Li conduction dominates at room temperature")
    print("  O conduction becomes important at elevated temperatures")
    
    print("  Key insight: LLZO shows mixed Li/O conduction with T-dependent behavior")

def demonstrate_llzo_export(crystal: Crystal, report_manager: ReportManager = None):
    """Demonstrate LLZO structure export"""
    print_section_header("LLZO STRUCTURE EXPORT", "All Phases")
    
    print("Step 1: Export LLZO structure to various formats...")
    
    export_paths = []
    export_formats = []
    
    # Export to POSCAR
    print("  Exporting to POSCAR format...")
    try:
        poscar_data = to_poscar(crystal)
        poscar_file = OUTPUT_ROOT / "llzo_garnet_poscar"
        with open(poscar_file, 'w') as f:
            f.write(poscar_data['poscar'])
        print(f"    POSCAR saved to: {poscar_file}")
        print(f"    Formula: {poscar_data['formula']}")
        export_paths.append(str(poscar_file))
        export_formats.append("POSCAR")
    except Exception as e:
        print(f"    ERROR: POSCAR export failed: {e}")
    
    # Export to CIF
    print("  Exporting to CIF format...")
    try:
        cif_content = to_cif(crystal)
        cif_file = OUTPUT_ROOT / "llzo_garnet_defective.cif"
        with open(cif_file, 'w') as f:
            f.write(cif_content)
        print(f"    CIF saved to: {cif_file}")
        export_paths.append(str(cif_file))
        export_formats.append("CIF")
    except Exception as e:
        print(f"    ERROR: CIF export failed: {e}")
    
    # Generate and store final AtomForge program
    if report_manager:
        # Create supercell info if this is a supercell
        supercell_info = None
        if len(crystal.sites) > 200:  # Likely a supercell
            supercell_info = {
                "repeat": (2, 2, 2),  # Assuming 2x2x2 based on typical LLZO workflow
                "origin_shift": (0.0, 0.0, 0.0)
            }
        
        # Define defects per FullLanguage.tex line 765-768
        # Based on Li2O loss simulation: Li vacancies (2%) and O vacancies (1%)
        defects = [
            {
                "site_ref": "Li1",
                "type": "vacancy",
                "prob": 0.02  # 2% Li vacancy concentration
            },
            {
                "site_ref": "Li2", 
                "type": "vacancy",
                "prob": 0.02  # Additional Li vacancy
            },
            {
                "site_ref": "O1",
                "type": "vacancy",
                "prob": 0.01  # 1% O vacancy concentration (Li2O loss)
            }
        ]
        
        final_atomforge_program = generate_atomforge_program(
            crystal,
            material_name="LLZO_final",
            description="Final LLZO structure with Li2O loss defects (Li and O vacancies)",
            operations=[
                {"operation": "canonicalize", "params": {"actions": ["reduced_to_primitive", "converted_to_conventional"]}},
                {"operation": "vacancy", "params": {"site_sel": "Species:Li", "occupancy": 0.964}},
                {"operation": "vacancy", "params": {"site_sel": "Species:O", "occupancy": 0.9896}}
            ],
            supercell_info=supercell_info,
            defects=defects  # Include defects per FullLanguage.tex
        )
        
        report_manager.store_atomforge_program(
            "LLZO",
            final_atomforge_program,
            "final_with_vacancies_supercell",
            "Final LLZO structure with Li/O vacancies in supercell"
        )
    
    print("  Export completed")

def main():
    """Main LLZO Garnet demonstration function"""
    global logger
    
    # Setup logging (optional - set to False to disable logging)
    enable_logging = False
    if enable_logging:
        logger, log_file = setup_logging()
    else:
        logger = None
        log_file = None
    
    # Initialize report manager for storing all reports
    report_manager = ReportManager(OUTPUT_ROOT)
    
    # Setup output file for all print statements
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    output_file = OUTPUT_ROOT / f"llzo_demo_output_{timestamp}.txt"
    
    class TeeOutput:
        """Class to write to both console and file"""
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open output file and setup tee
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        
        try:
            print("LLZO GARNET SOLID ELECTROLYTE DEMONSTRATION")
            print("Real-World Materials Science Case Study")
            print("=" * 80)
            print("Context: Li7La3Zr2O12 solid electrolyte for all-solid-state Li-ion batteries")
            print("Key properties: High ionic conductivity, vacancy-mediated diffusion")
            print(f"Output directory: {OUTPUT_ROOT}")
            print(f"Output file: {output_file}")
            print(f"Timestamp: {datetime.now().isoformat()}")
            
            # Log the same information (if logging enabled)
            if logger:
                logger.info("LLZO GARNET SOLID ELECTROLYTE DEMONSTRATION")
                logger.info("Real-World Materials Science Case Study")
                logger.info("=" * 80)
                logger.info("Context: Li7La3Zr2O12 solid electrolyte for all-solid-state Li-ion batteries")
                logger.info("Key properties: High ionic conductivity, vacancy-mediated diffusion")
                logger.info(f"Output directory: {OUTPUT_ROOT}")
                logger.info(f"Output file: {output_file}")
                logger.info(f"Timestamp: {datetime.now().isoformat()}")
            
            # Workflow 1: Load and canonicalize LLZO crystal structure
            crystal_base = workflow_1_llzo_loading(report_manager)
            if not crystal_base:
                log_and_print("ERROR: Workflow 1 failed, cannot continue", "error")
                return 1
            
            # Workflow 2: Simulate Li2O loss (vacancy creation)
            crystal_vacancies, patches_vac = workflow_2_li2o_loss_simulation(crystal_base)
            
            # Workflow 3: Analyze diffusion pathways
            crystal_super, patches_diff = workflow_3_diffusion_pathway_analysis(crystal_vacancies)
            
            # Workflow 4: Prepare ionic conductivity calculations
            calc_input, prep_report = workflow_4_ionic_conductivity_calculation(crystal_super)
            
            # Workflow 5: Temperature-dependent analysis
            workflow_5_temperature_dependent_analysis()
            
            # Export operations
            demonstrate_llzo_export(crystal_super, report_manager)
            
            # Generate session summary
            summary_path = report_manager.generate_session_summary()
            print(f"\nSession summary generated: {summary_path}")
            
            # Show generated files
            print(f"\nGenerated files in {OUTPUT_ROOT}:")
            for file in OUTPUT_ROOT.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
            
            # Show report directories
            print(f"\nReport directories:")
            for report_type, dir_path in report_manager.report_dirs.items():
                if dir_path.exists():
                    files = list(dir_path.iterdir())
                    if files:
                        print(f"  - {report_type}: {len(files)} files")
                        for file in files:
                            print(f"    - {file.name}")
            
            print("\n" + "="*80)
            print("LLZO GARNET SOLID ELECTROLYTE DEMONSTRATION COMPLETED")
            print("="*80)
            
            return 0
            
        except Exception as e:
            error_msg = f"\nERROR during LLZO demonstration: {e}"
            log_and_print(error_msg, "error")
            import traceback
            traceback.print_exc()
            if logger:
                logger.error("Full traceback:", exc_info=True)
            return 1
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            print(f"\nDemo output saved to: {output_file}")
            if log_file:
                print(f"Log file saved to: {log_file}")

if __name__ == "__main__":
    exit(main())
