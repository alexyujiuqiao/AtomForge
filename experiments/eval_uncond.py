#!/usr/bin/env python3
"""
Evaluate Unconditional AtomForge Generation

This script computes structure-level evaluation metrics on compiled AtomForge programs:
- Validity: min interatomic distance, charge neutrality
- Uniqueness: structure-based fingerprinting
- Novelty: comparison against reference dataset
- Distribution matching: density, element counts

Usage:
    python -m experiments.eval_uncond --compiled_dir outputs/uncond_200/compiled --reference_dir data

Output:
    - outputs/uncond_200/eval/metrics.jsonl (per-structure metrics)
    - outputs/uncond_200/eval/summary.json (aggregated statistics)
    - outputs/uncond_200/eval/plots/*.png (distribution plots)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import pymatgen
try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_analyzer import StructureAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Install with: pip install pymatgen")

# Import scipy for Wasserstein distance
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

# Import local utilities
from experiments.utils import program_fingerprint, load_seed_programs
from experiments.compile_generated import atomforge_to_structure, get_value

# Import AtomForge components for fingerprinting
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
except ImportError:
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_min_distance(structure: Structure) -> float:
    """Compute minimum interatomic distance in Angstrom."""
    distances = structure.distance_matrix
    # Exclude diagonal (self-distances)
    mask = np.eye(len(distances), dtype=bool)
    distances_masked = np.ma.masked_array(distances, mask=mask)
    return float(np.min(distances_masked))


def check_charge_neutrality(structure: Structure) -> Tuple[bool, Optional[float], str]:
    """
    Check charge neutrality using pymatgen oxidation state guessing.
    
    Returns:
        Tuple of (is_neutral, net_charge, status)
        status: "neutral", "charged", "unknown_charge"
    """
    try:
        # Try to guess oxidation states
        structure.add_oxidation_state_by_guess()
        total_charge = sum(site.specie.oxi_state for site in structure)
        
        if abs(total_charge) < 0.01:
            return True, 0.0, "neutral"
        else:
            return False, total_charge, "charged"
    except Exception:
        # Oxidation state guessing failed
        return False, None, "unknown_charge"


def structure_fingerprint(structure: Structure) -> str:
    """
    Generate a structure-based fingerprint from a pymatgen Structure.
    
    This is similar to program_fingerprint but works from the compiled structure.
    """
    # Get lattice parameters
    lattice = structure.lattice
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    
    signature: Dict[str, Any] = {
        "bravais": {
            "a": round(a, 4),
            "b": round(b, 4),
            "c": round(c, 4),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "gamma": round(gamma, 4),
        },
        "sites": []
    }
    
    # Extract sites
    for site in structure:
        coords = tuple(round(x, 4) for x in site.frac_coords)
        
        # Extract species and occupancies
        species_entries: List[Tuple[str, float]] = []
        if hasattr(site.specie, 'elements'):
            # Disordered site
            for elem, occ in site.specie.items():
                species_entries.append((str(elem), round(occ, 4)))
        else:
            # Ordered site
            species_entries.append((str(site.specie), 1.0))
        
        species_entries.sort(key=lambda t: (t[0], t[1]))
        
        signature["sites"].append({
            "position": coords,
            "species": species_entries,
        })
    
    # Sort sites deterministically
    signature["sites"].sort(key=lambda s: (tuple(s["species"]), s["position"]))
    
    # Hash
    import hashlib
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def evaluate_structure(
    cif_path: Path,
    reference_fingerprints: Set[str],
    parser: Optional[AtomForgeParser] = None
) -> Dict[str, Any]:
    """
    Evaluate a single structure.
    
    Args:
        cif_path: Path to CIF file
        reference_fingerprints: Set of reference structure fingerprints
        parser: Optional parser for computing program-based fingerprint
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics: Dict[str, Any] = {
        'structure_id': cif_path.stem,
        'cif_file': str(cif_path),
    }
    
    try:
        # Load structure from CIF
        structure = Structure.from_file(str(cif_path))
        
        # A) Validity checks
        min_dist = compute_min_distance(structure)
        metrics['min_interatomic_distance'] = min_dist
        metrics['valid_min_distance'] = min_dist > 0.5
        
        is_neutral, net_charge, charge_status = check_charge_neutrality(structure)
        metrics['charge_status'] = charge_status
        metrics['net_charge'] = net_charge
        metrics['charge_neutral'] = is_neutral
        
        # B) Uniqueness (structure fingerprint)
        struct_fp = structure_fingerprint(structure)
        metrics['structure_fingerprint'] = struct_fp
        
        # C) Novelty (check against reference)
        metrics['novel'] = struct_fp not in reference_fingerprints
        
        # D) Distribution metrics
        metrics['density'] = structure.density
        metrics['num_elements'] = len(structure.composition.elements)
        metrics['num_sites'] = len(structure)
        metrics['formula'] = structure.formula
        
        metrics['status'] = 'success'
        
    except Exception as e:
        metrics['status'] = 'error'
        metrics['error_message'] = str(e)
    
    return metrics


def load_reference_fingerprints(
    reference_dir: str,
    max_samples: Optional[int] = None,
    parser: AtomForgeParser
) -> Set[str]:
    """
    Load and fingerprint reference structures from the dataset.
    
    Args:
        reference_dir: Directory containing reference .atomforge files
        max_samples: Maximum number of reference files to process (None = all)
        parser: AtomForgeParser instance
        
    Returns:
        Set of structure fingerprints
    """
    logger.info(f"Loading reference fingerprints from {reference_dir}...")
    
    # Load seed programs
    seed_programs = load_seed_programs(reference_dir, limit=max_samples or 10000)
    
    fingerprints: Set[str] = set()
    
    for file_path, program_text in seed_programs:
        try:
            program = parser.parse_and_transform(program_text)
            program.validate()
            
            # Convert to structure and fingerprint
            if PYMAGEN_AVAILABLE:
                struct = atomforge_to_structure(program)
                fp = structure_fingerprint(struct)
                fingerprints.add(fp)
        except Exception as e:
            logger.debug(f"Failed to process reference {file_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(fingerprints)} unique reference fingerprints")
    return fingerprints


def plot_distributions(
    generated_metrics: List[Dict[str, Any]],
    reference_metrics: List[Dict[str, Any]],
    out_dir: Path
) -> None:
    """Generate distribution comparison plots."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract densities
    gen_densities = [m.get('density', 0) for m in generated_metrics if m.get('status') == 'success' and 'density' in m]
    ref_densities = [m.get('density', 0) for m in reference_metrics if 'density' in m]
    
    # Extract element counts
    gen_elements = [m.get('num_elements', 0) for m in generated_metrics if m.get('status') == 'success' and 'num_elements' in m]
    ref_elements = [m.get('num_elements', 0) for m in reference_metrics if 'num_elements' in m]
    
    # Plot density distribution
    if gen_densities and ref_densities:
        plt.figure(figsize=(10, 6))
        plt.hist(gen_densities, bins=30, alpha=0.5, label='Generated', density=True)
        plt.hist(ref_densities, bins=30, alpha=0.5, label='Reference', density=True)
        plt.xlabel('Density (g/cm³)')
        plt.ylabel('Probability Density')
        plt.title('Density Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'density_distribution.png', dpi=150)
        plt.close()
    
    # Plot element count distribution
    if gen_elements and ref_elements:
        plt.figure(figsize=(10, 6))
        bins = range(min(min(gen_elements, default=1), min(ref_elements, default=1)),
                     max(max(gen_elements, default=10), max(ref_elements, default=10)) + 2)
        plt.hist(gen_elements, bins=bins, alpha=0.5, label='Generated', align='left')
        plt.hist(ref_elements, bins=bins, alpha=0.5, label='Reference', align='left')
        plt.xlabel('Number of Unique Elements')
        plt.ylabel('Count')
        plt.title('Element Count Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'element_count_distribution.png', dpi=150)
        plt.close()
    
    logger.info(f"Plots saved to {plots_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate unconditional AtomForge generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--compiled_dir',
        type=str,
        default='outputs/uncond_200/compiled',
        help='Directory containing compiled CIF files'
    )
    
    parser.add_argument(
        '--reference_dir',
        type=str,
        default='data',
        help='Directory containing reference .atomforge files'
    )
    
    parser.add_argument(
        '--max_reference',
        type=int,
        default=None,
        help='Maximum number of reference structures to load (None = all)'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: parent of compiled_dir + /eval)'
    )
    
    args = parser.parse_args()
    
    compiled_path = Path(args.compiled_dir)
    if not compiled_path.exists():
        raise ValueError(f"Compiled directory does not exist: {compiled_path}")
    
    # Determine output directory
    if args.out_dir:
        out_path = Path(args.out_dir)
    else:
        out_path = compiled_path.parent / "eval"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find CIF files
    cif_files = sorted(compiled_path.glob("*.cif"))
    if not cif_files:
        logger.warning(f"No CIF files found in {compiled_path}")
        return
    
    logger.info(f"Found {len(cif_files)} CIF files to evaluate")
    
    # Load reference fingerprints
    atomforge_parser = AtomForgeParser()
    reference_fps = load_reference_fingerprints(
        args.reference_dir,
        max_samples=args.max_reference,
        parser=atomforge_parser
    )
    
    # Evaluate each structure
    metrics_file = out_path / "metrics.jsonl"
    all_metrics: List[Dict[str, Any]] = []
    
    for i, cif_file in enumerate(cif_files, 1):
        logger.info(f"[{i}/{len(cif_files)}] Evaluating {cif_file.name}...")
        
        metrics = evaluate_structure(cif_file, reference_fps, atomforge_parser)
        all_metrics.append(metrics)
        
        # Log to JSONL
        with open(metrics_file, 'a', encoding='utf-8') as f:
            json.dump(metrics, f, default=str)
            f.write('\n')
    
    # Compute summary statistics
    successful = [m for m in all_metrics if m.get('status') == 'success']
    
    summary: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'total_structures': len(all_metrics),
        'successful_evaluations': len(successful),
        'validity': {
            'min_distance_valid': sum(1 for m in successful if m.get('valid_min_distance', False)),
            'min_distance_valid_rate': sum(1 for m in successful if m.get('valid_min_distance', False)) / max(len(successful), 1),
            'charge_neutral': sum(1 for m in successful if m.get('charge_neutral', False)),
            'charge_neutral_rate': sum(1 for m in successful if m.get('charge_neutral', False)) / max(len(successful), 1),
            'unknown_charge_count': sum(1 for m in successful if m.get('charge_status') == 'unknown_charge'),
        },
        'uniqueness': {
            'unique_fingerprints': len(set(m.get('structure_fingerprint') for m in successful if 'structure_fingerprint' in m)),
            'unique_rate': len(set(m.get('structure_fingerprint') for m in successful if 'structure_fingerprint' in m)) / max(len(successful), 1),
        },
        'novelty': {
            'novel_structures': sum(1 for m in successful if m.get('novel', False)),
            'novelty_rate': sum(1 for m in successful if m.get('novel', False)) / max(len(successful), 1),
        },
        'distribution': {},
    }
    
    # Distribution statistics
    if successful:
        gen_densities = [m.get('density', 0) for m in successful if 'density' in m]
        gen_elements = [m.get('num_elements', 0) for m in successful if 'num_elements' in m]
        
        if gen_densities:
            summary['distribution']['density'] = {
                'mean': float(np.mean(gen_densities)),
                'std': float(np.std(gen_densities)),
                'min': float(np.min(gen_densities)),
                'max': float(np.max(gen_densities)),
            }
        
        if gen_elements:
            summary['distribution']['num_elements'] = {
                'mean': float(np.mean(gen_elements)),
                'std': float(np.std(gen_elements)),
                'min': int(np.min(gen_elements)),
                'max': int(np.max(gen_elements)),
            }
    
    # Save summary
    summary_file = out_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Load reference metrics for distribution comparison
    reference_metrics: List[Dict[str, Any]] = []
    if args.reference_dir and PYMAGEN_AVAILABLE:
        logger.info("Computing reference distribution metrics...")
        ref_files = sorted(Path(args.reference_dir).glob("batch_*/*.atomforge"))
        if args.max_reference:
            ref_files = ref_files[:args.max_reference]
        
        for ref_file in ref_files[:min(1000, len(ref_files))]:  # Limit for speed
            try:
                with open(ref_file, 'r', encoding='utf-8') as f:
                    program_text = f.read()
                program = atomforge_parser.parse_and_transform(program_text)
                program.validate()
                struct = atomforge_to_structure(program)
                reference_metrics.append({
                    'density': struct.density,
                    'num_elements': len(struct.composition.elements),
                })
            except Exception:
                continue
    
    # Generate plots
    plot_distributions(all_metrics, reference_metrics, out_path)
    
    # Compute Wasserstein distances if scipy available
    if SCIPY_AVAILABLE and successful and reference_metrics:
        gen_densities = [m.get('density', 0) for m in successful if 'density' in m]
        ref_densities = [m.get('density', 0) for m in reference_metrics if 'density' in m]
        
        if gen_densities and ref_densities:
            wasserstein_density = wasserstein_distance(gen_densities, ref_densities)
            summary['distribution']['wasserstein_density'] = float(wasserstein_density)
            
            gen_elements = [m.get('num_elements', 0) for m in successful if 'num_elements' in m]
            ref_elements = [m.get('num_elements', 0) for m in reference_metrics if 'num_elements' in m]
            if gen_elements and ref_elements:
                wasserstein_elements = wasserstein_distance(gen_elements, ref_elements)
                summary['distribution']['wasserstein_num_elements'] = float(wasserstein_elements)
        
        # Re-save summary with Wasserstein distances
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total structures: {summary['total_structures']}")
    logger.info(f"Successful evaluations: {summary['successful_evaluations']}")
    logger.info(f"Min distance valid: {summary['validity']['min_distance_valid']} ({summary['validity']['min_distance_valid_rate']:.1%})")
    logger.info(f"Charge neutral: {summary['validity']['charge_neutral']} ({summary['validity']['charge_neutral_rate']:.1%})")
    logger.info(f"Unique structures: {summary['uniqueness']['unique_fingerprints']} ({summary['uniqueness']['unique_rate']:.1%})")
    logger.info(f"Novel structures: {summary['novelty']['novel_structures']} ({summary['novelty']['novelty_rate']:.1%})")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

