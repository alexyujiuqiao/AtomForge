#!/usr/bin/env python3
"""
Evaluate Baseline AtomForge Generation

This script evaluates baseline-generated crystal structures using AtomForge's metrics:
- Validity: minimum interatomic distance (bond length realism) and charge neutrality, 
  including a check for likely VASP failures.
- Uniqueness: fingerprint-based identification of duplicate structures.
- Novelty: whether structures are new compared to a reference dataset (e.g., MP-20).
- Distribution: summary statistics (density, element count) and comparison to reference distribution.

It reads baseline outputs (AtomForge programs or structure files like CIF/POSCAR), computes metrics for each, 
and saves:
  - per-structure metrics in JSONL format,
  - an aggregated summary JSON,
  - plots comparing baseline vs. reference distributions (if matplotlib is available).

Usage:
    python -m experiments.eval_baseline --baseline_dir <path_to_baseline_outputs> \\
                                       --reference_dir <path_to_reference_data> \\
                                       --out_dir outputs/baseline/eval

Args:
    --baseline_dir      Directory with baseline-generated structures (AtomForge DSL files or CIFs).
    --reference_dir     Directory with reference .atomforge files for novelty comparison.
    --max_reference     Max number of reference files to use (defaults to all).
    --out_dir           Output directory for evaluation results (default: baseline_dir/../eval).
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from datetime import datetime

# Add project root to PYTHONPATH to import AtomForge parser/IR
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import AtomForge parser (and IR if needed)
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
    from atomforge.src import atomforge_ir  # ensure IR types are loaded
except ImportError:
    # Fallback to direct path if running as script
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser
    import atomforge_ir

# Import pymatgen for structure handling
try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Install with: pip install pymatgen")

# Import scipy for distance metrics
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Import matplotlib for plotting distributions
try:
    import matplotlib
    matplotlib.use('Agg')  # use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_min_distance(structure: Structure) -> float:
    """Compute the minimum interatomic distance in the structure (Å)."""
    distances = structure.distance_matrix
    # Mask out zero distances (self-distances on diagonal)
    mask = np.eye(len(distances), dtype=bool)
    dist_masked = np.ma.masked_array(distances, mask=mask)
    return float(np.min(dist_masked))

def check_charge_neutrality(structure: Structure) -> Dict[str, Any]:
    """
    Check if the structure is charge-neutral using oxidation state guessing.
    Returns a dict with:
      - charge_neutral (bool)
      - net_charge (float or None)
      - charge_status (str): "neutral", "charged", or "unknown_charge"
    """
    result = {"charge_neutral": False, "net_charge": None, "charge_status": "unknown_charge"}
    try:
        structure.add_oxidation_state_by_guess()
        total_charge = sum(site.specie.oxi_state for site in structure)
        result["net_charge"] = total_charge
        if abs(total_charge) < 1e-2:
            result["charge_neutral"] = True
            result["charge_status"] = "neutral"
        else:
            result["charge_neutral"] = False
            result["charge_status"] = "charged"
    except Exception:
        # Could not determine oxidation states
        result["charge_neutral"] = False
        result["charge_status"] = "unknown_charge"
    return result

def structure_fingerprint(structure: Structure) -> str:
    """
    Compute a deterministic fingerprint for a structure based on lattice and fractional coordinates.
    Used for identifying duplicates and novelty (via set membership).
    """
    # Create a canonical representation of lattice parameters and site positions/species
    lattice = structure.lattice
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    signature: Dict[str, Any] = {
        "lattice": {
            "a": round(a, 4), "b": round(b, 4), "c": round(c, 4),
            "alpha": round(alpha, 4), "beta": round(beta, 4), "gamma": round(gamma, 4),
        },
        "sites": []
    }
    for site in structure:
        coords = tuple(round(x, 4) for x in site.frac_coords)
        # Handle disordered sites vs ordered
        species_entries: List[tuple] = []
        try:
            # site.specie may be an element or Composition (for disordered)
            if hasattr(site.specie, "items"):
                # disordered site, iterate elements
                for elem, occ in site.specie.items():
                    species_entries.append((str(elem), round(float(occ), 4)))
            else:
                species_entries.append((str(site.specie), 1.0))
        except Exception:
            species_entries.append((str(site.specie), 1.0))
        species_entries.sort()
        signature["sites"].append({"position": coords, "species": tuple(species_entries)})
    # Sort sites list to ensure order invariance
    signature["sites"].sort(key=lambda x: (x["species"], x["position"]))
    # Hash the signature
    import hashlib
    sig_str = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:16]

def evaluate_structure(file_path: Path, reference_fps: Set[str], parser: Optional[AtomForgeParser]) -> Dict[str, Any]:
    """
    Evaluate a single structure file (baseline sample).
    If the file is an AtomForge program, it will be parsed and compiled; if it's a structure file (CIF/POSCAR), it will be read directly.
    Returns a dictionary of computed metrics for the structure.
    """
    metrics: Dict[str, Any] = {
        "structure_id": file_path.stem,
        "source_file": str(file_path)
    }
    try:
        # Obtain a pymatgen Structure from the file
        if file_path.suffix.lower() in [".atomforge", ".af"]:
            # Parse AtomForge DSL and compile to structure
            if parser is None:
                raise ValueError("AtomForge parser is required for .atomforge files")
            text = file_path.read_text(encoding="utf-8")
            program = parser.parse_and_transform(text)
            program.validate()
            if not PYMATGEN_AVAILABLE:
                raise ImportError("pymatgen required to convert AtomForge program to structure")
            # Use AtomForge IR to build Structure
            from atomforge.src.atomforge_ir import AtomForgeProgram  # ensure class is loaded
            # The program variable is an AtomForgeProgram after parse_and_transform
            # Construct Structure via known routine (assuming similar to compile_generated)
            # (We reuse the atomforge_to_structure logic here inline for simplicity)
            lattice = program.lattice.bravais
            if lattice is None:
                raise ValueError("No lattice information in program")
            from pymatgen.core import Lattice as PmgLattice
            lat = PmgLattice.from_parameters(
                float(lattice.a.value), float(lattice.b.value), float(lattice.c.value),
                float(lattice.alpha.value), float(lattice.beta.value), float(lattice.gamma.value)
            )
            # Build sites
            if not program.basis or not program.basis.sites:
                raise ValueError("No atomic sites in program basis")
            species_list = []
            frac_coords = []
            for site in program.basis.sites:
                # Determine fractional coordinates (convert from cartesian if needed)
                x, y, z = site.position
                x = float(x); y = float(y); z = float(z)
                if site.frame == "cartesian":
                    # convert to fractional coordinates
                    coord = lat.get_fractional_coords([x, y, z])
                    x, y, z = coord[0], coord[1], coord[2]
                # Determine species (handle partial occupancy)
                if not site.species:
                    raise ValueError(f"No species specified for site {site.name}")
                if len(site.species) == 1 and abs(float(site.species[0].occupancy) - 1.0) < 1e-6:
                    # single species fully occupied
                    species_list.append(site.species[0].element)
                    frac_coords.append((x, y, z))
                else:
                    # multiple species or partial occupancy
                    comp_dict: Dict[str, float] = {}
                    total_occ = 0.0
                    for sp in site.species:
                        occ_val = float(sp.occupancy)
                        if occ_val <= 0: 
                            continue
                        comp_dict[sp.element] = occ_val
                        total_occ += occ_val
                    if total_occ < 1e-6:
                        raise ValueError(f"No significant occupancy at site {site.name}")
                    # normalize occupancy if needed
                    if abs(total_occ - 1.0) > 1e-3:
                        for el in comp_dict:
                            comp_dict[el] /= total_occ
                    from pymatgen.core import Composition
                    species_list.append(Composition(comp_dict))
                    frac_coords.append((x, y, z))
            structure = Structure(lat, species_list, frac_coords)
        else:
            # Try reading as a structure (CIF, POSCAR, etc.)
            if not PYMATGEN_AVAILABLE:
                raise ImportError("pymatgen is required to read structure files")
            structure = Structure.from_file(str(file_path))
        # Compute metrics
        # A) Validity: min distance and charge
        min_dist = compute_min_distance(structure)
        metrics["min_interatomic_distance"] = min_dist
        metrics["valid_min_distance"] = (min_dist > 0.5)  # basic validity threshold
        # Check charge neutrality
        charge_info = check_charge_neutrality(structure)
        metrics.update(charge_info)  # adds charge_neutral, net_charge, charge_status
        # Mark likely VASP failure if very short bond or non-neutral
        metrics["likely_vasp_fail"] = (min_dist < 1.0 or (not metrics.get("charge_neutral", False)))
        # B) Uniqueness: structure fingerprint
        fp = structure_fingerprint(structure) if PYMATGEN_AVAILABLE else None
        metrics["structure_fingerprint"] = fp
        # C) Novelty: compare fingerprint to reference set
        metrics["novel"] = (fp not in reference_fps) if fp is not None else None
        # D) Distribution: basic properties
        metrics["density"] = structure.density if hasattr(structure, "density") else None
        metrics["num_elements"] = len(structure.composition.elements) if structure.composition else 0
        metrics["num_sites"] = len(structure)
        metrics["formula"] = structure.formula
        metrics["status"] = "success"
    except Exception as e:
        metrics["status"] = "error"
        metrics["error_message"] = str(e)
        logger.warning(f"Failed to evaluate {file_path.name}: {e}")
    return metrics

def load_reference_fingerprints(reference_dir: str, max_samples: Optional[int], parser: AtomForgeParser) -> Set[str]:
    """
    Load reference .atomforge files, convert to structure fingerprints.
    Returns a set of fingerprints for novelty checking.
    """
    ref_dir_path = Path(reference_dir)
    if not ref_dir_path.exists():
        logger.warning(f"Reference directory not found: {reference_dir}")
        return set()
    files = sorted(ref_dir_path.rglob("*.atomforge"))
    if max_samples:
        files = files[:max_samples]
    logger.info(f"Loading reference structures from '{reference_dir}' ({len(files)} files)")
    ref_fps: Set[str] = set()
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8")
            program = parser.parse_and_transform(text)
            program.validate()
            if PYMATGEN_AVAILABLE:
                # Convert to Structure and fingerprint it
                # (Reuse evaluate_structure logic partially)
                struct = None
                try:
                    struct = evaluate_structure(fpath, set(), parser).get("structure")  # not ideal, could parse directly
                except Exception:
                    # Fallback: directly use parser output to build struct
                    struct = None
                if struct is None:
                    continue
                fp = structure_fingerprint(struct)
                ref_fps.add(fp)
        except Exception as e:
            logger.debug(f"Skipping reference file {fpath.name}: {e}")
            continue
    logger.info(f"Loaded {len(ref_fps)} unique reference fingerprints")
    return ref_fps

def plot_distributions(generated_metrics: List[Dict[str, Any]], reference_metrics: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate density and element-count distribution plots comparing baseline vs reference."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plots")
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Density distribution
    gen_densities = [m["density"] for m in generated_metrics if m.get("status") == "success" and m.get("density") is not None]
    ref_densities = [m["density"] for m in reference_metrics if m.get("density") is not None]
    if gen_densities and ref_densities:
        plt.figure(figsize=(6,4))
        plt.hist(gen_densities, bins=30, alpha=0.5, label='Baseline', density=True)
        plt.hist(ref_densities, bins=30, alpha=0.5, label='Reference', density=True)
        plt.xlabel('Density (g/cm³)')
        plt.ylabel('Probability Density')
        plt.title('Density Distribution: Baseline vs Reference')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "density_distribution.png", dpi=150)
        plt.close()
    # Element count distribution
    gen_elem_counts = [m["num_elements"] for m in generated_metrics if m.get("status") == "success"]
    ref_elem_counts = [m["num_elements"] for m in reference_metrics if "num_elements" in m]
    if gen_elem_counts and ref_elem_counts:
        plt.figure(figsize=(6,4))
        min_bin = min(min(gen_elem_counts), min(ref_elem_counts))
        max_bin = max(max(gen_elem_counts), max(ref_elem_counts))
        bins = range(min_bin, max_bin + 2)
        plt.hist(gen_elem_counts, bins=bins, alpha=0.5, label='Baseline', align='left')
        plt.hist(ref_elem_counts, bins=bins, alpha=0.5, label='Reference', align='left')
        plt.xlabel('Number of Unique Elements')
        plt.ylabel('Count')
        plt.title('Element Count Distribution: Baseline vs Reference')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "element_count_distribution.png", dpi=150)
        plt.close()
    logger.info(f"Distribution plots saved to {plots_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline AtomForge generation outputs",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help="Directory containing baseline-generated structures (programs or CIFs)")
    parser.add_argument('--reference_dir', type=str, default='data',
                        help="Directory of reference .atomforge files for novelty and distribution comparison")
    parser.add_argument('--max_reference', type=int, default=None,
                        help="Maximum number of reference structures to load (None = all)")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Output directory for evaluation results (default: baseline_dir/../eval)")
    args = parser.parse_args()

    base_path = Path(args.baseline_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline directory not found: {base_path}")
    # Determine output directory
    if args.out_dir:
        out_path = Path(args.out_dir)
    else:
        out_path = base_path.parent / "eval"
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect baseline files (AtomForge or CIF)
    baseline_files = sorted(base_path.rglob("*.atomforge")) + sorted(base_path.rglob("*.cif"))
    # Also allow common VASP output names (POSCAR/CONTCAR) if present
    for fname in ["POSCAR", "CONTCAR"]:
        fpath = base_path / fname
        if fpath.exists():
            baseline_files.append(fpath)
    if not baseline_files:
        logger.error(f"No baseline structure files found in {base_path}")
        return
    logger.info(f"Found {len(baseline_files)} baseline files to evaluate in '{args.baseline_dir}'")

    # Load reference fingerprints for novelty checking
    atomforge_parser = AtomForgeParser()
    reference_fps = load_reference_fingerprints(args.reference_dir, max_samples=args.max_reference, parser=atomforge_parser)

    # Evaluate each baseline structure
    metrics_path = out_path / "metrics.jsonl"
    all_metrics: List[Dict[str, Any]] = []
    for i, fpath in enumerate(baseline_files, 1):
        logger.info(f"[{i}/{len(baseline_files)}] Evaluating {fpath.name}...")
        m = evaluate_structure(fpath, reference_fps, parser=atomforge_parser)
        all_metrics.append(m)
        # Append to JSONL file
        with open(metrics_path, 'a', encoding='utf-8') as mf:
            json.dump(m, mf, default=str)
            mf.write("\n")

    # Aggregate summary statistics
    successful = [m for m in all_metrics if m.get("status") == "success"]
    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_structures": len(all_metrics),
        "successful_evaluations": len(successful),
        "validity": {
            "min_distance_valid_count": sum(1 for m in successful if m.get("valid_min_distance")),
            "min_distance_valid_rate": sum(1 for m in successful if m.get("valid_min_distance")) / max(len(successful), 1),
            "charge_neutral_count": sum(1 for m in successful if m.get("charge_neutral")),
            "charge_neutral_rate": sum(1 for m in successful if m.get("charge_neutral")) / max(len(successful), 1),
            "unknown_charge_count": sum(1 for m in successful if m.get("charge_status") == "unknown_charge"),
            "likely_vasp_fail_count": sum(1 for m in successful if m.get("likely_vasp_fail")),
            "likely_vasp_fail_rate": sum(1 for m in successful if m.get("likely_vasp_fail")) / max(len(successful), 1)
        },
        "uniqueness": {
            "unique_structures": len(set(m.get("structure_fingerprint") for m in successful if m.get("structure_fingerprint"))),
            "unique_fraction":  len(set(m.get("structure_fingerprint") for m in successful if m.get("structure_fingerprint"))) / max(len(successful), 1)
        },
        "novelty": {
            "novel_structures": sum(1 for m in successful if m.get("novel") is True),
            "novelty_rate": sum(1 for m in successful if m.get("novel") is True) / max(len(successful), 1)
        },
        "distribution": {}
    }
    # Distribution stats for generated set
    if successful:
        densities = [m["density"] for m in successful if m.get("density") is not None]
        elem_counts = [m["num_elements"] for m in successful if m.get("num_elements") is not None]
        if densities:
            summary["distribution"]["density"] = {
                "mean": float(np.mean(densities)),
                "std": float(np.std(densities)),
                "min": float(np.min(densities)),
                "max": float(np.max(densities))
            }
        if elem_counts:
            summary["distribution"]["num_elements"] = {
                "mean": float(np.mean(elem_counts)),
                "std": float(np.std(elem_counts)),
                "min": int(np.min(elem_counts)),
                "max": int(np.max(elem_counts))
            }
    # Save summary to JSON
    summary_path = out_path / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=2)
    # If reference data available, collect reference distribution metrics for comparison
    reference_metrics: List[Dict[str, Any]] = []
    if reference_fps and PYMATGEN_AVAILABLE:
        # We will sample up to 1000 reference structures for distribution comparison
        ref_files = sorted(Path(args.reference_dir).rglob("*.atomforge"))
        if args.max_reference:
            ref_files = ref_files[:args.max_reference]
        ref_files = ref_files[:min(1000, len(ref_files))]
        for ref_file in ref_files:
            try:
                # Parse and compile reference structure
                program_text = ref_file.read_text(encoding="utf-8")
                prog = atomforge_parser.parse_and_transform(program_text)
                prog.validate()
                struct = evaluate_structure(ref_file, set(), atomforge_parser).get("structure")
                if struct is None:
                    continue
                reference_metrics.append({
                    "density": struct.density,
                    "num_elements": len(struct.composition.elements)
                })
            except Exception:
                continue
    # Generate distribution plots
    plot_distributions(all_metrics, reference_metrics, out_path)
    # Compute Wasserstein distances if possible
    if SCIPY_AVAILABLE and successful and reference_metrics:
        gen_densities = [m["density"] for m in successful if m.get("density") is not None]
        ref_densities = [m["density"] for m in reference_metrics if m.get("density") is not None]
        if gen_densities and ref_densities:
            summary["distribution"]["wasserstein_density"] = float(wasserstein_distance(gen_densities, ref_densities))
        gen_elems = [m["num_elements"] for m in successful if m.get("num_elements") is not None]
        ref_elems = [m["num_elements"] for m in reference_metrics if m.get("num_elements") is not None]
        if gen_elems and ref_elems:
            summary["distribution"]["wasserstein_num_elements"] = float(wasserstein_distance(gen_elems, ref_elems))
        # Update summary file with Wasserstein metrics
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump(summary, sf, indent=2)
    # Log key results
    logger.info("\n===== Baseline Evaluation Summary =====")
    logger.info(f"Total structures evaluated: {summary['total_structures']}")
    logger.info(f"Successful evaluations: {summary['successful_evaluations']}")
    if summary['successful_evaluations'] > 0:
        val = summary["validity"]
        logger.info(f"Min distance valid: {val['min_distance_valid_count']} ({val['min_distance_valid_rate']:.1%})")
        logger.info(f"Charge neutral: {val['charge_neutral_count']} ({val['charge_neutral_rate']:.1%})")
        logger.info(f"Likely VASP failures: {val['likely_vasp_fail_count']} ({val['likely_vasp_fail_rate']:.1%})")
        uniq = summary["uniqueness"]
        logger.info(f"Unique structures: {uniq['unique_structures']} ({uniq['unique_fraction']:.1%})")
        nov = summary["novelty"]
        logger.info(f"Novel structures: {nov['novel_structures']} ({nov['novelty_rate']:.1%})")
    logger.info(f"Results saved to: {out_path}")
    logger.info("======================================")

if __name__ == "__main__":
    main()
